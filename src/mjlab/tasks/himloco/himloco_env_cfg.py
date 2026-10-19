"""Shared environment configuration for HiMLoco tasks.

HiMLoco uses the robot-agnostic mjlab locomotion environment and adds the two
observation groups consumed by the hybrid internal model. Robot-specific
configuration lives in ``config/<robot>``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

HISTORY_LENGTH = 6
"""Frames in ``him_input``: the current frame plus ``H = 5`` past frames."""

ANG_VEL_SCALE = 0.25
"""Scale for body angular velocity, matching the released HiMLoco code."""

JOINT_VEL_SCALE = 0.05
"""Scale for joint velocity, matching the released HiMLoco code."""

COMMAND_SCALE = (2.0, 2.0, 0.25)
"""Scale for ``[linear_x, linear_y, angular_z]`` commands."""

LIN_VEL_SCALE = 2.0
"""Scale for the privileged base linear velocity and successor target."""


def make_himloco_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create a robot-independent HiMLoco environment configuration."""
  cfg = make_velocity_env_cfg()

  actor_terms = cfg.observations["actor"].terms
  actor_terms["base_ang_vel"].scale = ANG_VEL_SCALE
  actor_terms["joint_vel"].scale = JOINT_VEL_SCALE
  actor_terms["command"].scale = COMMAND_SCALE

  critic_terms = cfg.observations["critic"].terms
  critic_terms["base_ang_vel"].scale = ANG_VEL_SCALE
  critic_terms["joint_vel"].scale = JOINT_VEL_SCALE
  critic_terms["command"].scale = COMMAND_SCALE
  critic_terms["base_lin_vel"].scale = LIN_VEL_SCALE

  cfg.observations["him_input"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
    history_length=HISTORY_LENGTH,
    flatten_history_dim=False,
  )
  cfg.observations["him_state"] = ObservationGroupCfg(
    terms={
      "base_ang_vel": actor_terms["base_ang_vel"],
      "projected_gravity": actor_terms["projected_gravity"],
      "joint_pos": actor_terms["joint_pos"],
      "joint_vel": actor_terms["joint_vel"],
      "actions": actor_terms["actions"],
      "base_lin_vel": ObservationTermCfg(
        func=envs_mdp.base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-10.0, 10.0),
        scale=LIN_VEL_SCALE,
      ),
    },
    concatenate_terms=True,
    enable_corruption=True,
  )

  cfg.metrics["peak_joint_vel"] = MetricsTermCfg(func=mdp.peak_joint_vel, reduce="max")

  del cfg.events["base_com"]
  cfg.events["payload_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.body_payload,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=()),
      "mass_range": (-1.0, 2.0),
      "mount_offset": (0.0, 0.0, 0.06),
      "com_shift_range": (-0.05, 0.05),
    },
  )
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
    },
  )
  cfg.events["motor_strength"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "effort_limit_range": (0.9, 1.1),
    },
  )
  return cfg
