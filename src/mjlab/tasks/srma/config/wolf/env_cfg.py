"""Wolf SRMA environment configurations.

Wolf 是 16 自由度轮足机器人（12 个腿关节 + 4 个驱动轮）。本文件在共享的
SRMA 环境骨架上做机器人侧适配，核心差异：

- **动作**：``leg_joint_pos``（12 维位置，缩放到 ±0.1875 rad）+ ``wheel_vel``
  （4 维速度，缩放到 ±10 rad/s，按 ``WOLF_WHEEL_FORWARD_SIGN`` 逐轮取符号）。
- **观测**：``joint_pos`` 只含 12 个腿关节（轮子位置在速度控制下自由积分，
  不能进观测），``joint_vel`` 拆成腿 ``joint_vel`` + ``wheel_vel`` 两项。
- **接触 / 足端**：feet 传感器与 foot-height 扫描换到轮子碰撞 geom / 轮心
  site；腿 / 车体的非法接触按 body 匹配（Wolf 的腿碰撞 geom 未命名）。
- **teacher 高度图**：扫描尺寸按 Wolf 默认站姿的轮距（0.571 m × 0.426 m）
  重新取，(0.60, 0.50) m / 0.05 m 分辨率 → (13, 11) 高度图。
- **奖励**：``pose`` 只约束腿关节；``foot_clearance`` 目标高度改为轮半径；
  ``foot_slip`` 关闭（轮心 site 速度在滚动时非零，不等于打滑）。
- **物理**：沿用 SRMA 的 solver 设定；轮子接触在 asset 层提升为 ``condim=6``，
  以便三轴摩擦域随机化在轮子上真实生效。
"""

import math
from dataclasses import replace

from mjlab.asset_zoo.robots.wolf.wolf_constants import (
  WOLF_LEG_ACTION_SCALE_BY_JOINT,
  WOLF_LEG_JOINT_ORDER,
  WOLF_ROOT_BODY_NAME,
  WOLF_SITE_NAMES,
  WOLF_STANDING_WHEEL_HEIGHT,
  WOLF_WHEEL_ACTION_SCALE_BY_JOINT,
  WOLF_WHEEL_COLLISION_GEOM_NAMES,
  WOLF_WHEEL_JOINT_NAMES,
  WOLF_WHEEL_VELOCITY_CLIP,
  get_wolf_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.srma import mdp
from mjlab.tasks.srma.mdp import UniformVelocityCommandCfg
from mjlab.tasks.srma.srma_env_cfg import make_srma_env_cfg
from mjlab.terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Paper Table II: system delay 0-15 ms. One physics substep is 5 ms, so the
# command lag spans 0-3 substeps, resampled once per 20 ms policy step.
# Wolf 部署侧同为 dt=5 ms、decimation=4（50 Hz 策略），保持同一延迟模型。
WOLF_COMMAND_DELAY_MIN_LAG = 0
WOLF_COMMAND_DELAY_MAX_LAG = 3
WOLF_COMMAND_DELAY_UPDATE_PERIOD = 4

# Wolf 默认站姿（INIT_STATE）下四个轮心 FK：x ±0.2856 m、y ±0.2129 m。
# 留出约 8-9 cm 余量，teacher 采样网格取 (0.60, 0.50) m / 0.05 m，
# 网格形状 = (size_x/res+1, size_y/res+1) = (13, 11)。
WOLF_TEACHER_SCAN_SIZE = (0.60, 0.50)
WOLF_TEACHER_SCAN_RESOLUTION = 0.05
WOLF_TEACHER_HEIGHT_SHAPE = (13, 11)

# 腿 / 躯干的 body 名：Wolf 的腿碰撞 geom 没有命名，接触按 body 匹配。
WOLF_THIGH_BODY_NAMES = tuple(f"{leg}_Link2" for leg in WOLF_SITE_NAMES)
WOLF_SHANK_BODY_NAMES = tuple(f"{leg}_Link3" for leg in WOLF_SITE_NAMES)


def wolf_srma_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Wolf rough-terrain SRMA configuration."""
  cfg = make_srma_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 200
  # Wolf 的四个轮子 + 腿碰撞比 Black 多：qpos0 参考站姿下（mjlab 侧把 XML 的
  # base_link 参考高度抬到零关节站立位）就有约 40 个接触点，Black 的 35 不够。
  # mujoco_warp 的接触缓冲按 nconmax × num_envs 线性吃显存（4096 环境下每 16
  # 个 nconmax 约 0.9 GB），96 在 8 GB 卡上会 OOM；64 既高于 qpos0 的 40，
  # 又能在 8 GB 卡上跑满 4096 环境。
  cfg.sim.nconmax = 64

  # 与 Black 一样，把通信延迟挂到 16 个 IdealPdActuator 上。actuator 配置是
  # 共享模块级对象，先 replace 出副本，避免污染其它 Wolf 任务。
  robot_cfg = get_wolf_robot_cfg()
  assert robot_cfg.articulation is not None
  robot_cfg = replace(
    robot_cfg,
    articulation=replace(
      robot_cfg.articulation,
      actuators=tuple(
        replace(
          actuator,
          delay_min_lag=WOLF_COMMAND_DELAY_MIN_LAG,
          delay_max_lag=WOLF_COMMAND_DELAY_MAX_LAG,
          delay_update_period=WOLF_COMMAND_DELAY_UPDATE_PERIOD,
        )
        for actuator in robot_cfg.articulation.actuators
      ),
    ),
  )
  cfg.scene.entities = {"robot": robot_cfg}

  # 两台机载扫描都挂到车体 root（Wolf 没有叫 trunk 的 body）。
  for sensor in cfg.scene.sensors or ():
    if sensor.name in ("terrain_scan", "srma_teacher_scan"):
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = WOLF_ROOT_BODY_NAME
    if sensor.name == "srma_teacher_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = GridPatternCfg(
        size=WOLF_TEACHER_SCAN_SIZE,
        resolution=WOLF_TEACHER_SCAN_RESOLUTION,
      )

  # Wolf 的腿顺序是 FL/FR/RL/RR（与 Black 的 FL/FR/RR/RL 不同）。
  site_names = WOLF_SITE_NAMES
  wheel_geom_names = WOLF_WHEEL_COLLISION_GEOM_NAMES

  # 轮心 site 的 foot-height 扫描：四向 ring 取轮下最低地形。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=wheel_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Wolf 的腿碰撞 geom 未命名，按 body 匹配；躯干用 root body 整体匹配。
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=WOLF_ROOT_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=WOLF_ROOT_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(mode="body", entity="robot", pattern=WOLF_THIGH_BODY_NAMES),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(mode="body", entity="robot", pattern=WOLF_SHANK_BODY_NAMES),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(mode="body", entity="robot", pattern=(WOLF_ROOT_BODY_NAME,)),
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

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = True
    stairs = generator.sub_terrains["pyramid_stairs"]
    inverse_stairs = generator.sub_terrains["pyramid_stairs_inv"]
    assert isinstance(stairs, BoxPyramidStairsTerrainCfg)
    assert isinstance(inverse_stairs, BoxInvertedPyramidStairsTerrainCfg)
    stairs.step_height_range = (0.0, 0.25)
    inverse_stairs.step_height_range = (0.0, 0.25)

  ##
  # Actions: 12 leg position targets + 4 wheel velocity targets.
  ##
  cfg.actions = {
    "leg_joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=WOLF_LEG_JOINT_ORDER,
      scale=WOLF_LEG_ACTION_SCALE_BY_JOINT,
      use_default_offset=True,
    ),
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=WOLF_WHEEL_JOINT_NAMES,
      # 符号 contract 直接编码在缩放里：正动作 = 驱动车体前进。
      scale=WOLF_WHEEL_ACTION_SCALE_BY_JOINT,
      use_default_offset=True,
      clip={".*_foot": WOLF_WHEEL_VELOCITY_CLIP},
    ),
  }

  cfg.viewer.body_name = WOLF_ROOT_BODY_NAME
  cfg.viewer.distance = 1.6
  cfg.viewer.elevation = -10.0

  ##
  # Observations: wheel joint positions are excluded (they integrate freely
  # under velocity control); wheel speeds get their own term.
  ##
  leg_asset_cfg = SceneEntityCfg(
    "robot", joint_names=WOLF_LEG_JOINT_ORDER, preserve_order=True
  )
  wheel_asset_cfg = SceneEntityCfg(
    "robot", joint_names=WOLF_WHEEL_JOINT_NAMES, preserve_order=True
  )
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    joint_pos_term = terms["joint_pos"]
    joint_pos_term.params = {**joint_pos_term.params, "asset_cfg": leg_asset_cfg}
    terms["joint_vel"].params = {"asset_cfg": leg_asset_cfg}
  # 轮速观测直接插在 joint_vel 后面，保证两组观测的布局一致。
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    rebuilt: dict[str, ObservationTermCfg] = {}
    for name, term in terms.items():
      rebuilt[name] = term
      if name == "joint_vel":
        rebuilt["wheel_vel"] = ObservationTermCfg(
          func=mdp.joint_vel_rel,
          params={"asset_cfg": wheel_asset_cfg},
          noise=Unoise(n_min=-1.5, n_max=1.5),
          clip=(-60.0, 60.0),
        )
    cfg.observations[group_name].terms = rebuilt

  ##
  # Domain randomization: 轮子三轴摩擦（需要 asset 层的 condim=6）。
  ##
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=wheel_geom_names),
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
      "asset_cfg": SceneEntityCfg("robot", geom_names=wheel_geom_names),
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
      "asset_cfg": SceneEntityCfg("robot", geom_names=wheel_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["payload_inertia"].params["asset_cfg"].body_names = (WOLF_ROOT_BODY_NAME,)

  height_map_term = cfg.observations["teacher_height_map"].terms["height_map"]
  height_map_term.params["asset_cfg"].site_names = site_names
  height_map_term.params["site_names"] = site_names
  height_map_term.params["height_shape"] = WOLF_TEACHER_HEIGHT_SHAPE

  friction_term = cfg.observations["teacher_friction"].terms["friction"]
  friction_term.params["asset_cfg"].geom_names = wheel_geom_names
  friction_term.params["geom_names"] = wheel_geom_names

  ##
  # Rewards.
  ##

  # 姿态惩罚只作用在 12 个腿关节：轮子位置没有物理意义（速度控制下自由积分）。
  pose_cfg = cfg.rewards["pose"]
  pose_cfg.params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=WOLF_LEG_JOINT_ORDER, preserve_order=True
  )
  pose_cfg.params["std_standing"] = {
    r".*_(hip|thigh)$": 0.05,
    r".*_calf$": 0.1,
  }
  pose_cfg.params["std_walking"] = {
    r".*_(hip|thigh)$": 0.3,
    r".*_calf$": 0.6,
  }
  pose_cfg.params["std_running"] = {
    r".*_(hip|thigh)$": 0.3,
    r".*_calf$": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (WOLF_ROOT_BODY_NAME,)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (WOLF_ROOT_BODY_NAME,)

  # Wolf 的训练 MJCF 没有 subtreeangmom 传感器；该惩罚在 SRMA 里本来就是
  # 0 权重，直接移除。
  cfg.rewards.pop("angular_momentum", None)

  for reward_name in ("foot_clearance", "foot_slip", "stair_climb_progress"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # 轮心站高就是轮半径：clearance 的“低于目标”只会在轮子陷进地形时触发，
  # 滚动 / 上下台阶时的正常抬轮不会被误罚。
  cfg.rewards["foot_clearance"].params["target_height"] = WOLF_STANDING_WHEEL_HEIGHT
  # 轮子抬离地面的目标峰值高度：站高 + 2 cm。
  cfg.rewards["foot_swing_height"].params["target_height"] = (
    WOLF_STANDING_WHEEL_HEIGHT + 0.02
  )
  # 轮足滚动时轮心 site 速度并不等于接触点打滑速度，原 slip 语义失效，关闭。
  cfg.rewards["foot_slip"].weight = 0.0

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Per-body-group collision penalties（与 Black 相同，只换传感器）。
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

  # On rough terrain the robot tilts significantly; do not terminate on
  # orientation alone. Let out_of_terrain_bounds handle the resets.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  # Official mjlab NaN workflow (FAQ "My training crashes with NaN errors"):
  # 轮-地 cylinder 接触在 mujoco_warp 的凸体/CCD 路径上偶发退化（见
  # nan_guard dump：摔倒后轮子侧面贴着地形台阶时出现重复/近似退化的接触，
  # 下一步 solver 出 NaN），单个 env 的 NaN 会被 rsl_rl 的全局检查放大成整
  # 个训练中断。这里把 NaN env 判 terminated 并 reset，让训练继续；出现频率
  # 约每 4M env-step 一次，重置代价可忽略。nan_guard 仍用于抓根因现场。
  cfg.terminations["nan_term"] = TerminationTermCfg(
    func=nan_detection,
    time_out=False,
  )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    # Let MuJoCo Warp derive the contact capacity from the sampled play terrain.
    cfg.sim.nconmax = None

    # Show the privileged teacher sampling grid in Viser.
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        assert isinstance(sensor, RayCastSensorCfg)
        sensor.debug_vis = False
      elif sensor.name == "srma_teacher_scan":
        assert isinstance(sensor, RayCastSensorCfg)
        sensor.debug_vis = True

    # Keep the deployable policy observation clean at play time.
    cfg.observations["actor"].enable_corruption = False
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


def wolf_srma_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Wolf flat-terrain SRMA configuration."""
  cfg = wolf_srma_env_cfg(play=play)

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
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_collision"):
    cfg.rewards.pop(key, None)
  cfg.rewards.pop("stair_climb_progress", None)

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
