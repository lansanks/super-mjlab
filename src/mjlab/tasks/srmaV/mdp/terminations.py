from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class command_stall_reset:
  """Reset an environment that stalls along its commanded direction too long."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self._robot: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._step_dt = float(env.step_dt)
    self._stall_steps = torch.zeros(
      env.num_envs,
      dtype=torch.long,
      device=env.device,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.2,
    stall_speed_threshold: float = 0.05,
    recover_speed_threshold: float = 0.1,
    command_off_threshold: float = 0.1,
    reset_time: float = 2.5,
  ) -> torch.Tensor:
    del asset_cfg
    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_xy = command[:, :2]
    command_norm = torch.norm(command_xy, dim=1)
    actual_xy = self._robot.data.root_link_lin_vel_b[:, :2]
    progress_speed = torch.sum(command_xy * actual_xy, dim=1) / command_norm.clamp_min(
      1.0e-6
    )

    stalled = (command_norm > command_threshold) & (
      progress_speed < stall_speed_threshold
    )
    recovered = (command_norm < command_off_threshold) | (
      progress_speed > recover_speed_threshold
    )
    self._stall_steps = torch.where(
      stalled,
      self._stall_steps + 1,
      torch.where(
        recovered,
        torch.zeros_like(self._stall_steps),
        self._stall_steps,
      ),
    )
    reset_steps = max(1, math.ceil(reset_time / self._step_dt))
    return self._stall_steps >= reset_steps

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._stall_steps[env_ids] = 0


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  # Use the generated grid shape (curriculum mode overrides cfg.num_cols with
  # len(sub_terrains)), and include the flat border around the patch grid.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def out_of_sub_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when a robot leaves its assigned sub-terrain patch.

  Patch bounds come from the curriculum row and terrain column, not from the
  environment origin. This keeps bridge spawns at the entry deck valid while
  still preventing robots from wandering into neighboring terrain types.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  size_x, size_y = terrain_generator.size
  patch_center_x = (
    -0.5 * num_rows * size_x + (terrain.terrain_levels.float() + 0.5) * size_x
  )
  patch_center_y = (
    -0.5 * num_cols * size_y + (terrain.terrain_types.float() + 0.5) * size_y
  )
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  outside_x = (root_xy_w[:, 0] - patch_center_x).abs() > size_x / 2.0 - margin
  outside_y = (root_xy_w[:, 1] - patch_center_y).abs() > size_y / 2.0 - margin
  outside = outside_x | outside_y
  outside &= env.episode_length_buf > 2
  return outside


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge
