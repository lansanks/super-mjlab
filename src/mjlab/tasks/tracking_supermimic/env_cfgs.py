"""G1 LAFAN1 tracking environment trained by the SuperMimic loop."""

from __future__ import annotations

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION_WITHOUT_SELF,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import time_out

from . import terminations
from .constants import ANCHOR_BODY, DEFAULT_MOTION_FILE, TRACKED_BODIES
from .motion_command import LafanMotionCommand, LafanMotionCommandCfg


def unitree_g1_tracking_supermimic_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the multi-clip LAFAN1 tracking task used by SuperMimic."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 4096
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.collisions = (FULL_COLLISION_WITHOUT_SELF,)

  # The reference implementation disables self-collisions for speed.
  cfg.scene.sensors = ()

  motion_cfg = LafanMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range={
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
    motion_file=DEFAULT_MOTION_FILE,
    anchor_body_name=ANCHOR_BODY,
    body_names=TRACKED_BODIES,
    sampling_mode="adaptive",
  )
  cfg.commands = {"motion": motion_cfg}
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "joint_pos": actor_terms["joint_pos"],
        "joint_vel": actor_terms["joint_vel"],
        "actions": actor_terms["actions"],
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "joint_pos": critic_terms["joint_pos"],
        "joint_vel": critic_terms["joint_vel"],
        "base_lin_vel": critic_terms["base_lin_vel"],
        "actions": critic_terms["actions"],
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.events.pop("push_robot", None)
  cfg.events["encoder_bias"].mode = "reset"

  # SuperMimic computes its own analytic reward and termination. The manager
  # terms are kept only for episode boundaries and logging.
  cfg.rewards = {}
  cfg.terminations = {
    "time_out": TerminationTermCfg(
      func=time_out,
      time_out=True,
    ),
    "anchor_pos": TerminationTermCfg(
      func=terminations.bad_anchor_pos,
    ),
    "anchor_ori": TerminationTermCfg(
      func=terminations.bad_anchor_ori,
    ),
    "motion_clip_end": TerminationTermCfg(
      func=terminations.motion_clip_end,
      time_out=True,
    ),
  }
  cfg.episode_length_s = 10.0
  cfg.auto_reset = play

  if play:
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.sampling_mode = "start"

  return cfg


__all__ = [
  "LafanMotionCommand",
  "LafanMotionCommandCfg",
  "unitree_g1_tracking_supermimic_env_cfg",
]
