"""Velocity-task environment metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def peak_joint_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Return the per-environment peak absolute joint velocity."""
  asset = env.scene[asset_cfg.name]
  return asset.data.joint_vel.abs().amax(dim=-1)
