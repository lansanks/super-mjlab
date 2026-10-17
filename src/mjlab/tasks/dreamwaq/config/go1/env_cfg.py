"""Unitree Go1 DreamWaQ environment configurations."""

import math
from dataclasses import replace

from mjlab.asset_zoo.robots import GO1_ACTION_SCALE, get_go1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.dreamwaq import mdp
from mjlab.tasks.dreamwaq.dreamwaq_env_cfg import make_dreamwaq_env_cfg
from mjlab.tasks.dreamwaq.mdp import UniformVelocityCommandCfg

# Paper Table II: system delay 0-15 ms. One physics substep is 5 ms, so the
# command lag spans 0-3 substeps, resampled once per 20 ms policy step.
GO1_COMMAND_DELAY_MIN_LAG = 0
GO1_COMMAND_DELAY_MAX_LAG = 3
GO1_COMMAND_DELAY_UPDATE_PERIOD = 4


def unitree_go1_dreamwaq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go1 rough-terrain DreamWaQ configuration."""
  cfg = make_dreamwaq_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  # The Go1 actuator configs are shared module-level objects, so make copies
  # before attaching the command delay to keep other Go1 tasks unaffected.
  robot_cfg = get_go1_robot_cfg()
  assert robot_cfg.articulation is not None
  robot_cfg = replace(
    robot_cfg,
    articulation=replace(
      robot_cfg.articulation,
      actuators=tuple(
        replace(
          actuator,
          delay_min_lag=GO1_COMMAND_DELAY_MIN_LAG,
          delay_max_lag=GO1_COMMAND_DELAY_MAX_LAG,
          delay_update_period=GO1_COMMAND_DELAY_UPDATE_PERIOD,
        )
        for actuator in robot_cfg.articulation.actuators
      ),
    ),
  )
  cfg.scene.entities = {"robot": robot_cfg}

  # Point the terrain raycast sensor at the Go1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # Wire the foot height scan to the per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_geom_names = tuple(
    f"{leg}_thigh_collision{i}" for leg in foot_names for i in (1, 2, 3)
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(mode="geom", entity="robot", pattern=thigh_geom_names),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  calf_geom_names = tuple(
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(mode="geom", entity="robot", pattern=calf_geom_names),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("trunk_collision", "head_collision"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO1_ACTION_SCALE

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Replace the base foot friction event with per-axis friction for condim 6.
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["payload_inertia"].params["asset_cfg"].body_names = ("trunk",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0
  # cfg.rewards["foot_clearance"].weight = 0.0

  # Per-body-group collision penalties.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
  )

  # On rough terrain the robot tilts significantly; do not terminate on
  # orientation alone. Let out_of_terrain_bounds handle the resets.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    # Keep both the policy observation and the CENet history clean at play time.
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["cenet_input"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go1_dreamwaq_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go1 flat-terrain DreamWaQ configuration."""
  cfg = unitree_go1_dreamwaq_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast and collision sensors not needed on flat ground.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in remove_sensors
  )
  # The policy and CENet observations never contain the height scan; only the
  # critic reads it, and the terrain sensor is gone on flat ground.
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears it).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
