"""Unitree Go2 cycle-tracking environment trained with SuperMimic."""

from __future__ import annotations

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  FULL_COLLISION_WITHOUT_SELF,
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp import time_out
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.tracking_supermimic import terminations
from mjlab.tasks.tracking_supermimic.motion_command import (
  LafanMotionCommand,
  LafanMotionCommandCfg,
)

from .constants import (
  ANCHOR_BODY,
  DEFAULT_MOTION_FILE,
  TRACKED_BODIES,
)


def unitree_go2_tracking_supermimic_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Go2 cycle SuperMimic environment."""
  cfg = make_tracking_env_cfg()

  robot_cfg = get_go2_robot_cfg()
  robot_cfg.collisions = (FULL_COLLISION_WITHOUT_SELF,)
  cfg.scene.entities = {"robot": robot_cfg}
  cfg.scene.sensors = ()
  cfg.scene.num_envs = 1 if play else 4096
  cfg.sim.mujoco.impratio = 100.0
  cfg.sim.mujoco.cone = "elliptic"

  joint_action = cfg.actions["joint_pos"]
  assert isinstance(joint_action, JointPositionActionCfg)
  joint_action.scale = GO2_ACTION_SCALE

  motion_cfg = LafanMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    pose_range={
      "x": (-0.02, 0.02),
      "y": (-0.02, 0.02),
      "z": (-0.01, 0.01),
      "roll": (-0.03, 0.03),
      "pitch": (-0.03, 0.03),
      "yaw": (-0.1, 0.1),
    },
    velocity_range={
      "x": (-0.1, 0.1),
      "y": (-0.1, 0.1),
      "z": (-0.05, 0.05),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    joint_position_range=(-0.05, 0.05),
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
  foot_friction = cfg.events["foot_friction"]
  foot_friction.params["asset_cfg"].geom_names = r"^[FR][LR]_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  cfg.rewards = {}
  cfg.terminations = {
    "time_out": TerminationTermCfg(
      func=time_out,
      time_out=True,
    ),
    "anchor_pos": TerminationTermCfg(
      func=terminations.bad_anchor_pos,
      params={"threshold": 0.15},
    ),
    "anchor_ori": TerminationTermCfg(
      func=terminations.bad_anchor_ori,
      params={"threshold": 0.8},
    ),
    "motion_clip_end": TerminationTermCfg(
      func=terminations.motion_clip_end,
      time_out=True,
    ),
  }
  cfg.episode_length_s = 10.0
  cfg.auto_reset = play
  cfg.viewer.body_name = ANCHOR_BODY
  cfg.viewer.distance = 1.6
  cfg.viewer.elevation = -10.0

  if play:
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.sampling_mode = "all"
    cfg.episode_length_s = int(1e9)

  return cfg


__all__ = [
  "LafanMotionCommand",
  "LafanMotionCommandCfg",
  "unitree_go2_tracking_supermimic_env_cfg",
]
