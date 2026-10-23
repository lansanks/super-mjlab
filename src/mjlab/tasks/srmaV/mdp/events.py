"""SRMAV-specific MDP events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

# Reuse the base push event's sampling convention and env-id handling so the
# recorded disturbance is exactly the velocity kick that gets applied.
from mjlab.envs.mdp.events import (
  _sample_se3_range,
  reset_root_state_uniform,
  resolve_env_ids,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .terrain_utils import terrain_type_mask

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def reset_root_state_with_bridge_spawn(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  terrain_name: str = "broken_bridge",
) -> None:
  """Reset normally, then place bridge envs at the deck entry facing +x."""
  env_ids = resolve_env_ids(env, env_ids)
  reset_root_state_uniform(
    env,
    env_ids,
    pose_range=pose_range,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )

  bridge_mask = terrain_type_mask(env, env_ids, terrain_name)
  bridge_ids = env_ids[bridge_mask]
  if bridge_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[bridge_ids].clone()
  positions = root_states[:, :3] + env.scene.env_origins[bridge_ids]
  orientations = torch.tensor(
    [1.0, 0.0, 0.0, 0.0],
    dtype=root_states.dtype,
    device=root_states.device,
  ).expand(len(bridge_ids), 4)
  velocities = root_states[:, 7:13]

  asset.write_root_link_pose_to_sim(
    torch.cat((positions, orientations), dim=-1),
    env_ids=bridge_ids,
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=bridge_ids)


class push_by_setting_velocity_with_record:
  """Push the robot and record the kick and its control-step index.

  Behaves like :func:`mjlab.envs.mdp.push_by_setting_velocity` but keeps the
  sampled SE(3) kick so it can be observed as a privileged disturbance. The
  kick is stored in the root body frame as it was applied. ``last_push_step``
  allows SRMAV to expose a strict one-step delayed impulse without modifying the
  existing persistent ``push_velocity`` observation.

  Use with ``mode="interval"``.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.push_velocity = torch.zeros(env.num_envs, 6, device=env.device)
    """Applied kick (linear xyz, angular xyz) per environment, body frame."""
    self.last_push_step = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    """Control step at which the latest kick was applied."""

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
    self.last_push_step[env_ids] = env.common_step_counter

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear the recorded kick for the environments that reset."""
    self.push_velocity[env_ids] = 0.0
    self.last_push_step[env_ids] = -1
