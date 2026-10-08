"""Reward terms from Table I of the DreamWaQ paper.

mjlab already provides most of the DreamWaQ reward terms (velocity tracking,
orientation, joint acceleration, action rate/acceleration, feet clearance).
The three functions below cover the remaining ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _mechanical_power(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Per-joint mechanical power ``tau * qd`` (shape ``(num_envs, num_joints)``)."""
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return tau * qd


def lin_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize vertical base velocity ``v_z^2``. DreamWaQ weight: ``-2.0``."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def joint_power(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize total mechanical power ``sum_i |tau_i| * |qd_i|``.

  DreamWaQ weight: ``-2e-5``.
  """
  return torch.sum(torch.abs(_mechanical_power(env, asset_cfg)), dim=1)


def power_distribution(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize the variance of the mechanical power across joints.

  This is the paper's "power distribution" reward that discourages a single
  motor from doing most of the work (and overheating on hardware).
  DreamWaQ weight: ``-1e-5``.
  """
  power = torch.abs(_mechanical_power(env, asset_cfg))
  return torch.var(power, dim=1)


def base_height_penalty(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize ``(h - h_des)^2`` where ``h`` is the trunk height above terrain.

  The terrain height under the robot is estimated from the per-foot terrain
  height sensor, which is robust on stairs where a ray grid may miss.
  DreamWaQ weight: ``-1.0``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor = env.scene[height_sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"base_height_penalty expects a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  foot_clearance = sensor.data.heights  # (num_envs, num_feet)
  foot_z = asset.data.site_pose_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_feet)
  terrain_z = (foot_z - foot_clearance).mean(dim=1)
  base_height = asset.data.root_link_pose_w[:, 2] - terrain_z
  return torch.square(base_height - target_height)


def feet_clearance_penalty(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Foot clearance cost weighted by foot lateral speed (paper, Eq. Table I).

  ``cost = sum_k (p_f,z,k - p_des)^2 * |v_f,xy,k|`` for moving commands.
  DreamWaQ weight: ``-0.01``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor = env.scene[height_sensor_name]
  assert isinstance(sensor, TerrainHeightSensor)
  foot_height = sensor.data.heights  # (num_envs, num_feet)
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(torch.square(foot_height - target_height) * vel_norm, dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      active = (torch.norm(command[:, :2], dim=1) > command_threshold).float()
      cost = cost * active
  return cost
