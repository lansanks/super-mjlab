"""DreamWaQ environment configuration (Unitree Go1, rough terrain).

This module adapts mjlab's Go1 rough-terrain velocity task to the DreamWaQ
recipe:

* the actor only sees proprioception: ``o_t = [omega, g, c, theta, theta_dot, a_{t-1}]``
  (45 dims) stacked over ``H = 5`` frames (225 dims) for the CENet,
* the critic additionally sees privileged information (base linear velocity,
  terrain height scan, foot contacts),
* rewards follow Table I of the paper,
* domain randomization follows Table II of the paper.
"""

from dataclasses import replace

from mjlab.asset_zoo.robots import GO1_ACTION_SCALE, get_go1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.dreamwaq import mdp as dreamwaq_mdp
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import (
  ROUGH_TERRAINS_CFG,
  discrete_obstacles,
  hf_pyramid_slope,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)

HISTORY_LENGTH = 5
"""Number of proprioceptive frames stacked for the CENet (paper: ``H = 5``)."""

PROPRIO_DIM = 45
"""Size of one proprioceptive frame: 3 + 3 + 3 + 12 + 12 + 12."""

BASE_HEIGHT_TARGET = 0.30
"""Desired trunk height above the terrain, in meters (tune per robot)."""

COMMAND_SCALE = (2.0, 2.0, 0.25)
"""legged_gym scales the commands by ``[lin_vel, lin_vel, ang_vel]``."""

ANG_VEL_SCALE = 0.25
"""legged_gym observation scale for the base angular velocity."""

DOF_VEL_SCALE = 0.05
"""legged_gym observation scale for the joint velocities."""


def dreamwaq_terrains_cfg():
  """Terrain mixture of the paper: smooth slope, rough slope, stairs, discrete."""
  return replace(
    ROUGH_TERRAINS_CFG,
    num_rows=10,
    num_cols=20,
    sub_terrains={
      "smooth_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 0.6)),
      "rough_slope": random_rough(proportion=0.1),
      "stairs_up": pyramid_stairs(
        proportion=0.35, step_height_range=(0.0, 0.2), step_width=0.30
      ),
      "stairs_down": pyramid_stairs_inv(
        proportion=0.35, step_height_range=(0.0, 0.2), step_width=0.30
      ),
      "discrete": discrete_obstacles(proportion=0.1),
    },
  )


def unitree_go1_dreamwaq_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the DreamWaQ Go1 rough-terrain velocity configuration."""
  cfg = unitree_go1_rough_env_cfg(play=play)

  # DreamWaQ reward weights are per-control-step (Isaac Gym convention).
  # mjlab multiplies rewards by the step duration (0.02 s at 50 Hz) by default.
  cfg.scale_rewards_by_dt = False

  # System delay randomization [0, 15] ms. mjlab models command delay in physics
  # timesteps; the Go1 task runs with a 5 ms physics step.
  delay_lag = round(0.015 / cfg.sim.mujoco.timestep)
  robot_cfg = get_go1_robot_cfg()
  articulation = robot_cfg.articulation
  assert articulation is not None
  robot_cfg = replace(
    robot_cfg,
    articulation=replace(
      articulation,
      actuators=tuple(
        replace(a, delay_min_lag=0, delay_max_lag=delay_lag)
        for a in articulation.actuators
      ),
    ),
  )
  cfg.scene.entities = {"robot": robot_cfg}

  ##
  # Observations.
  ##

  base_actor_terms = cfg.observations["actor"].terms
  base_critic_terms = cfg.observations["critic"].terms

  # Eq. (1): o_t = [omega_t, g_t, c_t, theta_t, theta_dot_t, a_{t-1}].
  # Scales and noise levels follow the released DreamWaQ/legged_gym config, so
  # the standard-normal VAE prior stays meaningful.
  proprio_terms = {
    "base_ang_vel": replace(base_actor_terms["base_ang_vel"], scale=ANG_VEL_SCALE),
    "projected_gravity": replace(base_actor_terms["projected_gravity"]),
    "command": replace(base_actor_terms["command"], scale=COMMAND_SCALE),
    "joint_pos": replace(base_actor_terms["joint_pos"]),
    "joint_vel": replace(base_actor_terms["joint_vel"], scale=DOF_VEL_SCALE),
    "actions": replace(base_actor_terms["actions"]),
  }
  priv_velocity_term = ObservationTermCfg(
    func=base_actor_terms["base_lin_vel"].func,
    params=base_actor_terms["base_lin_vel"].params,
    scale=2.0,  # legged_gym lin_vel observation scale.
  )

  cfg.observations = {
    # Temporal partial observation o^H_t for the CENet / actor. Named "actor"
    # so that mjlab tooling (ONNX metadata, play) picks it up.
    "actor": ObservationGroupCfg(
      terms=proprio_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=HISTORY_LENGTH,
      flatten_history_dim=False,  # -> (num_envs, H, 45), oldest first.
    ),
    # Privileged base linear velocity: L_est regression target, and the AdaBoot
    # teacher signal. Never used by the deployed policy.
    "priv": ObservationGroupCfg(
      terms={"base_lin_vel": priv_velocity_term},
      concatenate_terms=True,
      enable_corruption=False,
    ),
    # Full state s_t = [o_t, v_t, d_t, h_t] (paper Eq. 2) for the value network.
    "critic": ObservationGroupCfg(
      terms={
        "base_lin_vel": replace(base_critic_terms["base_lin_vel"], scale=2.0),
        "base_ang_vel": replace(base_critic_terms["base_ang_vel"], scale=ANG_VEL_SCALE),
        "projected_gravity": replace(base_critic_terms["projected_gravity"]),
        "joint_pos": replace(base_critic_terms["joint_pos"]),
        "joint_vel": replace(base_critic_terms["joint_vel"], scale=DOF_VEL_SCALE),
        "actions": replace(base_critic_terms["actions"]),
        "command": replace(base_critic_terms["command"], scale=COMMAND_SCALE),
        "height_scan": replace(base_critic_terms["height_scan"]),
        "foot_height": replace(base_critic_terms["foot_height"]),
        "foot_air_time": replace(base_critic_terms["foot_air_time"]),
        "foot_contact": replace(base_critic_terms["foot_contact"]),
        "foot_contact_forces": replace(base_critic_terms["foot_contact_forces"]),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Commands: uniform in [-1, 1] m/s and [-1, 1] rad/s, resampled every 10 s.
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.ranges.heading = None
  cfg.curriculum.pop("command_vel", None)

  ##
  # Terrain: 10 difficulty levels, paper terrain mixture.
  ##

  if cfg.scene.terrain is not None:
    terrain_cfg = dreamwaq_terrains_cfg()
    if play:
      # Keep the play-mode layout (small grid, random patches).
      terrain_cfg.curriculum = False
      terrain_cfg.num_rows = 5
      terrain_cfg.num_cols = 5
      terrain_cfg.border_width = 10.0
    else:
      terrain_cfg.curriculum = True
    cfg.scene.terrain.terrain_generator = terrain_cfg
    cfg.scene.terrain.max_init_terrain_level = 5

  ##
  # Rewards (paper Table I).
  ##

  foot_sites = SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))
  joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
  trunk_cfg = SceneEntityCfg("robot", body_names=("trunk",))

  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": 0.5},  # exp(-4 ||e||^2)
    ),
    "track_angular_velocity": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": 0.5},
    ),
    "lin_vel_z": RewardTermCfg(
      func=dreamwaq_mdp.lin_vel_z_penalty,
      weight=-2.0,
    ),
    "ang_vel_xy": RewardTermCfg(
      func=velocity_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": trunk_cfg},
    ),
    "orientation": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-0.2,
    ),
    "dof_acc": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": joint_cfg},
    ),
    "joint_power": RewardTermCfg(
      func=dreamwaq_mdp.joint_power,
      weight=-2e-5,
      params={"asset_cfg": joint_cfg},
    ),
    "power_distribution": RewardTermCfg(
      func=dreamwaq_mdp.power_distribution,
      weight=-1e-5,
      params={"asset_cfg": joint_cfg},
    ),
    "base_height": RewardTermCfg(
      func=dreamwaq_mdp.base_height_penalty,
      weight=-1.0,
      params={
        "target_height": BASE_HEIGHT_TARGET,
        "height_sensor_name": "foot_height_scan",
        "asset_cfg": foot_sites,
      },
    ),
    "feet_clearance": RewardTermCfg(
      func=dreamwaq_mdp.feet_clearance_penalty,
      weight=-0.01,
      params={
        "target_height": 0.08,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": foot_sites,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    "smoothness": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.01),
    # Not in the paper's table but present in the released DreamWaQ config.
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": joint_cfg},
    ),
  }

  ##
  # Domain randomization (paper Table II).
  ##

  events = cfg.events
  events["base_com"].params["asset_cfg"].body_names = ("trunk",)
  events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  events["payload"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
      "operation": "add",
      "ranges": (-1.0, 2.0),  # kg
    },
  )
  events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
    },
  )
  events["motor_strength"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "effort_limit_range": (0.9, 1.1),
    },
  )
  events["foot_friction_slide"].params["ranges"] = (0.2, 1.25)

  if play:
    events.pop("push_robot", None)

  ##
  # Actions.
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO1_ACTION_SCALE

  ##
  # Terminations: reset when the trunk hits the ground (DreamWaQ).
  ##

  cfg.terminations["trunk_contact"] = TerminationTermCfg(
    func=velocity_mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch", "force_threshold": 10.0},
  )

  return cfg
