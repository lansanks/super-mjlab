"""SRMAV environment metrics."""

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
  """Per-environment peak absolute joint velocity.

  Registered with ``reduce="max"`` so each episode reports its worst joint
  velocity. A spike here means the simulation produced a violent state, which is
  the failure mode that once blew up training through a single extreme
  observation sample (see ``doc/实验记录.md``, iteration 806).

  Returns:
    Per-environment scalar. Shape: ``(B,)``, unit rad/s.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.joint_vel.abs().amax(dim=-1)
