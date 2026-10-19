from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from .events import push_by_setting_velocity_with_record

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def push_velocity(
  env: ManagerBasedRlEnv, event_name: str = "push_robot"
) -> torch.Tensor:
  """Latest push velocity kick, expressed in the root body frame.

  The value persists until the next push or episode reset. SRMA's teacher uses
  :func:`previous_push_velocity` instead.

  Returns:
    Tensor of shape [B, 6] with linear xyz and angular xyz kick components.
  """
  term_names = env.event_manager.active_terms.get("interval", [])
  if event_name not in term_names:
    return torch.zeros(env.num_envs, 6, device=env.device)
  event = env.event_manager.get_term_cfg(event_name).func
  if not isinstance(event, push_by_setting_velocity_with_record):
    raise TypeError(
      f"Event '{event_name}' must be a push_by_setting_velocity_with_record "
      f"to observe the disturbance, got {type(event).__name__}."
    )
  return event.push_velocity.clone()


def previous_push_velocity(
  env: ManagerBasedRlEnv, event_name: str = "push_robot"
) -> torch.Tensor:
  """Return the kick that was applied exactly one control step ago."""
  term_names = env.event_manager.active_terms.get("interval", [])
  if event_name not in term_names:
    return torch.zeros(env.num_envs, 6, device=env.device)
  event = env.event_manager.get_term_cfg(event_name).func
  if not isinstance(event, push_by_setting_velocity_with_record):
    raise TypeError(
      f"Event '{event_name}' must be a push_by_setting_velocity_with_record, "
      f"got {type(event).__name__}."
    )
  age = env.common_step_counter - event.last_push_step
  valid = (event.last_push_step >= 0) & (age == 1)
  return torch.where(valid.unsqueeze(-1), event.push_velocity, 0.0)


class local_height_map(ManagerTermBase):
  """Body-local terrain map normalized by the current support height."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    params = cfg.params
    self._robot: Entity = env.scene[params["asset_cfg"].name]
    self._height_sensor: RayCastSensor = env.scene[params["height_sensor_name"]]
    self._contact_sensor: ContactSensor = env.scene[params["contact_sensor_name"]]
    self._foot_height_sensor: TerrainHeightSensor = env.scene[
      params["foot_height_sensor_name"]
    ]
    self._height_shape = tuple(params["height_shape"])
    self._height_scale = float(params["height_scale"])
    site_ids, matched_sites = self._robot.find_sites(
      params["site_names"], preserve_order=True
    )
    if len(site_ids) != len(params["site_names"]):
      raise ValueError(
        f"Expected foot sites {params['site_names']}, got {matched_sites}."
      )
    self._site_ids = torch.tensor(site_ids, dtype=torch.long, device=env.device)
    self._z_ref = torch.zeros(env.num_envs, device=env.device)
    self._has_z_ref = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(
    self,
    env_ids: Sequence[int] | torch.Tensor | slice | None = None,
  ) -> None:
    ids: Sequence[int] | torch.Tensor | slice = (
      slice(None) if env_ids is None else env_ids
    )
    self._z_ref[ids] = 0.0
    self._has_z_ref[ids] = False

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    sensor_data = self._height_sensor.data
    miss = sensor_data.distances < 0
    hit_z = sensor_data.hit_pos_w[..., 2]

    contact_data = self._contact_sensor.data
    found = contact_data.found
    assert found is not None
    found = found > 0

    foot_z = self._robot.data.site_pos_w[:, self._site_ids, 2]
    foot_clearance = self._foot_height_sensor.data.heights
    terrain_z = foot_z - foot_clearance

    contact_count = found.sum(dim=-1)
    ordered_z = torch.sort(
      torch.where(found, terrain_z, torch.full_like(terrain_z, torch.inf)),
      dim=-1,
    ).values
    lower_idx = (contact_count.clamp_min(1) - 1) // 2
    upper_idx = contact_count.clamp_min(1) // 2
    lower = ordered_z.gather(1, lower_idx.unsqueeze(-1)).squeeze(-1)
    upper = ordered_z.gather(1, upper_idx.unsqueeze(-1)).squeeze(-1)
    contact_reference = 0.5 * (lower + upper)
    fallback_reference = terrain_z.median(dim=-1).values

    has_contact = contact_count > 0
    first_reference = ~self._has_z_ref
    reference = torch.where(
      has_contact,
      contact_reference,
      torch.where(first_reference, fallback_reference, self._z_ref),
    )
    self._z_ref = torch.where(has_contact | first_reference, reference, self._z_ref)
    self._has_z_ref = self._has_z_ref | has_contact | first_reference

    height_map = (hit_z - self._z_ref.unsqueeze(-1)) / self._height_scale
    height_map = height_map.clamp(-3.0, 3.0)
    height_map = torch.where(miss, torch.zeros_like(height_map), height_map)
    expected_shape = (self.num_envs, math.prod(self._height_shape))
    if height_map.shape != expected_shape:
      raise ValueError(
        f"Expected local height map {expected_shape}, got {tuple(height_map.shape)}."
      )
    return height_map.flatten(start_dim=1)


class foot_friction(ManagerTermBase):
  """Shared foot friction in log space, normalized to roughly [-1, 1]."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    params = cfg.params
    self._robot: Entity = env.scene[params["asset_cfg"].name]
    geom_ids, matched_geoms = self._robot.find_geoms(
      params["geom_names"], preserve_order=True
    )
    if not geom_ids:
      raise ValueError(f"No foot geoms matched {params['geom_names']}.")
    if len(geom_ids) != len(params["geom_names"]):
      raise ValueError(
        f"Expected foot geoms {params['geom_names']}, got {matched_geoms}."
      )
    self._geom_id = int(self._robot.indexing.geom_ids[geom_ids[0]])
    ranges = torch.tensor(params["log_ranges"], dtype=torch.float, device=env.device)
    if ranges.shape != (3, 2):
      raise ValueError(f"Expected three log ranges, got {tuple(ranges.shape)}.")
    self._log_min = ranges[:, 0]
    self._log_max = ranges[:, 1]

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del kwargs
    friction = env.sim.model.geom_friction[:, self._geom_id, :]
    log_friction = torch.log(friction.clamp_min(1.0e-8))
    normalized = (
      2.0 * (log_friction - self._log_min) / (self._log_max - self._log_min) - 1.0
    )
    return normalized.clamp(-1.0, 1.0)
