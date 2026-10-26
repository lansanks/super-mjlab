"""Black SRMAV environment configurations."""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.asset_zoo.robots import BLACK_ACTION_SCALE, get_black_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sensor.camera_sensor import CameraDataType
from mjlab.tasks.srmaV import mdp
from mjlab.tasks.srmaV.mdp import UniformVelocityCommandCfg
from mjlab.tasks.srmaV.srmaV_env_cfg import (
  SRMAV_TEACHER_HEIGHT_SHAPE,
  make_srmaV_env_cfg,
)

from .terrain_cfg import BLACK_SRMAV_BRIDGE_TERRAIN, BLACK_SRMAV_TERRAINS_CFG

# Paper Table II: system delay 0-15 ms. One physics substep is 5 ms, so the
# command lag spans 0-3 substeps, resampled once per 20 ms policy step.
BLACK_COMMAND_DELAY_MIN_LAG = 0
BLACK_COMMAND_DELAY_MAX_LAG = 3
BLACK_COMMAND_DELAY_UPDATE_PERIOD = 4

BLACK_TEACHER_HEIGHT_SHAPE = SRMAV_TEACHER_HEIGHT_SHAPE

# 放在 Black 自带 D435 外壳前方，相机朝向车体前方并略微向下。
BLACK_DEPTH_CAMERA_POS = (0.10, 0.0, -0.066)
BLACK_DEPTH_CAMERA_QUAT = (
  0.6163438796401546,
  0.3465836436275126,
  -0.3465836436275126,
  -0.6163438796401546,
)


def black_depth_camera_cfg(
  data_types: tuple[CameraDataType, ...] = ("rgb", "depth"),
) -> CameraSensorCfg:
  """Create the camera mounted in Black's D435 housing."""
  return CameraSensorCfg(
    name="black_depth_camera",
    parent_body="robot/d435_link",
    pos=BLACK_DEPTH_CAMERA_POS,
    quat=BLACK_DEPTH_CAMERA_QUAT,
    fovy=58.0,
    width=160,
    height=90,
    data_types=data_types,
    enabled_geom_groups=(0, 1, 2),
    use_textures=True,
    use_shadows=False,
  )


def black_srmaV_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Black rough-terrain SRMAV configuration."""
  cfg = make_srmaV_env_cfg()
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(BLACK_SRMAV_TERRAINS_CFG)

  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 200

  # The Black actuator configs are shared module-level objects, so make copies
  # before attaching the command delay to keep other Black tasks unaffected.
  robot_cfg = get_black_robot_cfg()
  assert robot_cfg.articulation is not None
  robot_cfg = replace(
    robot_cfg,
    articulation=replace(
      robot_cfg.articulation,
      actuators=tuple(
        replace(
          actuator,
          delay_min_lag=BLACK_COMMAND_DELAY_MIN_LAG,
          delay_max_lag=BLACK_COMMAND_DELAY_MAX_LAG,
          delay_update_period=BLACK_COMMAND_DELAY_UPDATE_PERIOD,
        )
        for actuator in robot_cfg.articulation.actuators
      ),
    ),
  )
  cfg.scene.entities = {"robot": robot_cfg}

  # Point all body-mounted terrain sensors at the Black trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"

  # The Black MJCF names its legs FL, FR, RR, RL.
  foot_names = ("FL", "FR", "RR", "RL")
  site_names = ("FL", "FR", "RR", "RL")
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
  thigh_geom_names = tuple(f"{leg}_thigh_collision" for leg in foot_names)
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
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2, 3)
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
  # Black has no separate head body; the chassis and the depth-camera mount are
  # covered by the trunk collision boxes.
  trunk_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=tuple(f"trunk_collision{i}" for i in range(1, 7)),
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
    trunk_ground_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = BLACK_ACTION_SCALE

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.terrain_override_name = BLACK_SRMAV_BRIDGE_TERRAIN
  twist_cmd.terrain_override_lin_vel_x = (0.3, 1.0)
  twist_cmd.terrain_override_heading = 0.0

  reset_base = cfg.events["reset_base"]
  cfg.events["reset_base"] = EventTermCfg(
    func=mdp.reset_root_state_with_bridge_spawn,
    mode="reset",
    params={
      **reset_base.params,
      "terrain_name": BLACK_SRMAV_BRIDGE_TERRAIN,
    },
  )

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

  height_map_term = cfg.observations["teacher_height_map"].terms["height_map"]
  height_map_term.params["asset_cfg"].site_names = site_names
  height_map_term.params["site_names"] = site_names

  friction_term = cfg.observations["teacher_friction"].terms["friction"]
  friction_term.params["asset_cfg"].geom_names = geom_names
  friction_term.params["geom_names"] = geom_names

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

  for reward_name in ("foot_clearance", "foot_slip", "stair_climb_progress"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  for reward_name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards[reward_name].params["target_height"] = 0.10

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

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
  cfg.rewards["trunk_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_ground_cfg.name},
  )
  cfg.rewards["bridge_gap_crossing"] = RewardTermCfg(
    func=mdp.bridge_gap_crossing,
    weight=0.1,
    params={
      "contact_sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      "terrain_name": BLACK_SRMAV_BRIDGE_TERRAIN,
      "foot_bonus": 2.5,
      "far_margin": 0.02,
      "contact_force_threshold": 10.0,
      "command_threshold": 0.1,
    },
  )

  # On rough terrain the robot tilts significantly; do not terminate on
  # orientation alone. Let out_of_terrain_bounds handle the resets.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )
  cfg.terminations["out_of_sub_terrain_bounds"] = TerminationTermCfg(
    func=mdp.out_of_sub_terrain_bounds,
    params={"margin": 0.1},
    time_out=True,
  )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    # Let MuJoCo Warp derive the contact capacity from the sampled play
    # terrain, as the Black DreamWaQ and HiMLoco configurations do.
    cfg.sim.nconmax = None

    # 当前只用于 Viser 调试，避免训练时渲染未接入策略的画面。
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (black_depth_camera_cfg(),)

    # The teacher and critic both use the central terrain scan.
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        assert isinstance(sensor, RayCastSensorCfg)
        sensor.debug_vis = True

    # Keep the deployable policy observation clean at play time.
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("out_of_sub_terrain_bounds", None)
    cfg.curriculum = {}

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def black_srmaV_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Black flat-terrain SRMAV configuration."""
  cfg = black_srmaV_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.terrain_override_name = None

  # Remove raycast and collision sensors not needed on flat ground.
  remove_sensors = {
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in remove_sensors
  )
  # The critic does not need the terrain scan on flat ground, but the teacher
  # group still uses it to produce a support-normalized zero map.
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_collision"):
    cfg.rewards.pop(key, None)
  cfg.rewards.pop("stair_climb_progress", None)
  cfg.rewards.pop("bridge_gap_crossing", None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.pop("out_of_sub_terrain_bounds", None)
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
