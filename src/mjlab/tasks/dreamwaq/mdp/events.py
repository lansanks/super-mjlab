"""DreamWaQ-specific MDP events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

# Reuse the base push event's sampling convention and env-id handling so the
# recorded disturbance is exactly the velocity kick that gets applied.
from mjlab.envs.mdp.events import _sample_se3_range, resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class push_by_setting_velocity_with_record:
  """Push the robot and record the applied velocity kick.

  Behaves like :func:`mjlab.envs.mdp.push_by_setting_velocity` but keeps the
  sampled SE(3) kick so it can be observed as the privileged disturbance
  ``d_t`` of DreamWaQ (paper Eq. 2). The kick is stored in the root body frame
  as it was applied, and rows are zeroed on episode reset by the event
  manager's class-term reset hook.

  Use with ``mode="interval"``.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.push_velocity = torch.zeros(env.num_envs, 6, device=env.device)
    """Applied kick (linear xyz, angular xyz) per environment, body frame."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    """Apply a sampled velocity kick and record it."""
    del asset_cfg  # Resolved once at construction time.
    env_ids = resolve_env_ids(env, env_ids)
    delta_w = _sample_se3_range(velocity_range, (len(env_ids), 6), env.device)
    self._asset.write_root_link_velocity_to_sim(
      self._asset.data.root_link_vel_w[env_ids] + delta_w, env_ids=env_ids
    )
    quat_w = self._asset.data.root_link_quat_w[env_ids]
    self.push_velocity[env_ids] = torch.cat(
      (
        quat_apply_inverse(quat_w, delta_w[:, :3]),
        quat_apply_inverse(quat_w, delta_w[:, 3:]),
      ),
      dim=-1,
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear the recorded kick for the environments that reset."""
    self.push_velocity[env_ids] = 0.0
