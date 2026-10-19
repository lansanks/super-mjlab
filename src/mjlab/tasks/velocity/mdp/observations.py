from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from .events import push_by_setting_velocity_with_record

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
  """Return the most recently applied push velocity in the root body frame."""
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
