"""Shared SRMAV environment configuration.

This is an independent copy of the DreamWaQ base environment. Robot-specific
configurations call this factory and add SRMAV teacher observations.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.srmaV import mdp
from mjlab.tasks.srmaV.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

SRMAV_TEACHER_HEIGHT_SHAPE = (17, 11)
"""Central 1.6 m x 1.0 m velocity scan at 0.1 m resolution."""


def make_srmaV_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # Set per-robot: frame and pattern.
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##

  actor_terms = {
    # "base_lin_vel": ObservationTermCfg(
    #   func=mdp.builtin_sensor,
    #   params={"sensor_name": "robot/imu_lin_vel"},
    #   noise=Unoise(n_min=-0.5, n_max=0.5),
    # ),
    # The clips bound the damage of a rare violent simulation state: a single
    # environment with, e.g., an exploding joint velocity can otherwise inject
    # a huge sample into a PPO mini-batch and destroy the policy in one step.
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      clip=(-30.0, 30.0),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      clip=(-3.0, 3.0),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      clip=(-50.0, 50.0),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action, clip=(-20.0, 20.0)),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  critic_terms = {
    **actor_terms,
    # Privileged ground-truth base linear velocity v_t (paper Eq. 2). The actor
    # observation deliberately excludes it, so it must be added back here.
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      clip=(-10.0, 10.0),
    ),
    # Privileged disturbance d_t (paper Eq. 2): the velocity kick applied by
    # the most recent push, in the root body frame. Zero until the first push
    # of an episode and while the push event is disabled (play mode).
    "push_velocity": ObservationTermCfg(func=mdp.push_velocity),
    # Critic sees the true (unbiased) joint positions as privileged information.
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "teacher_height_map": ObservationGroupCfg(
      terms={
        "height_map": ObservationTermCfg(
          func=mdp.local_height_map,
          params={
            "asset_cfg": SceneEntityCfg("robot", site_names=()),
            "height_sensor_name": "terrain_scan",
            "contact_sensor_name": "feet_ground_contact",
            "foot_height_sensor_name": "foot_height_scan",
            "site_names": (),
            "height_shape": SRMAV_TEACHER_HEIGHT_SHAPE,
            "height_scale": 0.15,
          },
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "teacher_friction": ObservationGroupCfg(
      terms={
        "friction": ObservationTermCfg(
          func=mdp.foot_friction,
          params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=()),
            "geom_names": (),
            "log_ranges": (
              (math.log(0.3), math.log(1.5)),
              (math.log(1.0e-4), math.log(2.0e-2)),
              (math.log(1.0e-5), math.log(5.0e-3)),
            ),
          },
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "teacher_prev_push": ObservationGroupCfg(
      terms={
        "previous_push": ObservationTermCfg(func=mdp.previous_push_velocity),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "teacher_velocity_gt": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(
          func=envs_mdp.base_lin_vel,
          params={"asset_cfg": SceneEntityCfg("robot")},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
    # Worst joint velocity of the episode: catches violent simulation states
    # before they can blow up training through an extreme observation sample.
    "peak_joint_vel": MetricsTermCfg(func=mdp.peak_joint_vel, reduce="max"),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      rel_forward_envs=0.2,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity_with_record,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    # Paper Table II: payload [-1, 2] kg and COM shift [-50, 50] mm. The two
    # are randomized jointly so mass, COM, and inertia always describe one
    # consistent rigid body (payload inertia via the parallel-axis theorem).
    "payload_inertia": EventTermCfg(
      mode="startup",
      func=dr.body_payload,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "mass_range": (-1.0, 2.0),
        "mount_offset": (0.0, 0.0, 0.06),  # Payload sits above the body COM.
        "com_shift_range": (-0.05, 0.05),
      },
    ),
    # Paper Table II: Kp and Kd factors [0.9, 1.1] (multiplicative).
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    ),
    # Paper Table II: motor strength factor [0.9, 1.1] (effort limit scale).
    "motor_strength": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "effort_limit_range": (0.9, 1.1),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "stair_climb_progress": RewardTermCfg(
      func=mdp.stair_climb_progress,
      weight=0.25,
      params={
        "contact_sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "max_step_height": 0.25,
        "min_height_gain": 0.02,
        "support_clearance_max": 0.05,
        "contact_force_threshold": 10.0,
        "command_threshold": 0.1,
        "first_step_multiplier": 5.0,
      },
    ),
    "command_stall_penalty": RewardTermCfg(
      func=mdp.command_stall_penalty,
      weight=1.0,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot"),
        "command_threshold": 0.2,
        "stall_speed_threshold": 0.05,
        "recover_speed_threshold": 0.1,
        "command_off_threshold": 0.1,
        "grace_time": 0.5,
        "ramp_time": 1.0,
        "penalty_start": 0.1,
        "penalty_end": 0.5,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "command_stall_reset": TerminationTermCfg(
      func=mdp.command_stall_reset,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot"),
        "command_threshold": 0.2,
        "stall_speed_threshold": 0.05,
        "recover_speed_threshold": 0.1,
        "command_off_threshold": 0.1,
        "reset_time": 2.5,
      },
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=deepcopy(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan, foot_height_scan),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
