"""Wolf wheel-legged robot constants (mjlab asset-zoo copy).

源资产：``XJTURoboCon_quadruped_assets/training_mjlab/wolf``（MJCF 是模型权威
来源，结构 / 惯量 / 关节层级改动应先在 URDF 源资产里做，再同步到这里）。

本文件在源资产包基础上做 mjlab 侧的适配，其余内容与源包保持一致：

- **轮子中心 site**：``xmls/wolf.xml`` 的四个 Link4 里新增 ``FL/FR/RL/RR``
  site（``pos="0 0 0.032"``，正好落在轮子铰链轴线上，因此世界位置就是轮心、
  不随轮子自转）。``TerrainHeightSensor``、``feet_clearance``、``foot_slip``、
  ``stair_climb_progress`` 等按 foot site 定义的 MDP 项需要它。
- **轮子接触参数**：轮子碰撞 geom 提升为 ``condim=6`` 并给出滑动 / 扭转 /
  滚动摩擦，使 SRMA 的三轴摩擦域随机化（以及 teacher 的 friction 特权观测）
  在轮子上真实生效。数值与 Black 的足端一致。
- **动作契约**：动作布局 = 12 个腿关节位置（FL→FR→RL→RR，腿内
  hip→thigh→calf）+ 4 个轮速（``*_foot``，FL→FR→RL→RR）；轮子每腿符号
  contract 见 ``WOLF_WHEEL_FORWARD_SIGN``，由 ``WOLF_WHEEL_ACTION_SCALE_BY_JOINT``
  直接编码成正 / 负缩放。

关节名保持 Wolf 原生命名（``FL_hip / FL_thigh / FL_calf / FL_foot``，其余腿
同理），不向 Black 的 ``*_joint`` 命名靠拢。

PD 参数为当前实际值，**实机电机规格仍待确认**：
- 腿部 hip/thigh/calf：position PD，Kp=80、Kd=3.0、effort_limit=60 N·m；
- 轮部 foot：velocity PD，Kp=0、Kd=1.0、effort_limit=17 N·m（速度目标由
  ``JointVelocityAction`` 提供；轮部不做 torque action，也不用 XML velocity
  servo）。
"""

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import GeomCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent
WOLF_XML: Path = _HERE / "xmls" / "wolf.xml"
WOLF_ASSETS_DIR: Path = _HERE / "xmls" / "assets"

assert WOLF_XML.exists(), f"Wolf MJCF not found: {WOLF_XML}"


##
# Joint / site contract.
##

# Policy / action / deployment 腿顺序：FL → FR → RL → RR。
# 该顺序与 ``xmls/wolf.xml`` 里四个腿 body 的声明顺序一致，因此 mjlab 的
# ``find_joints_by_actuator_names`` 返回的 natural order 恰好就是它。
WOLF_LEG_ORDER = ("FL", "FR", "RL", "RR")
# 每腿 joint 顺序：hip → thigh → calf → foot（foot = 驱动轮）。
WOLF_JOINTS_PER_LEG = ("hip", "thigh", "calf", "foot")
# 每条腿的 joint 名（按 leg 键查询；无序依赖）。
WOLF_LEG_JOINT_NAMES = {
  leg: (f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf", f"{leg}_foot")
  for leg in WOLF_LEG_ORDER
}
# 腿关节（12 个，动作顺序 = FL→FR→RL→RR，腿内 hip→thigh→calf）。
WOLF_LEG_JOINT_ORDER = tuple(
  f"{leg}_{joint}" for leg in WOLF_LEG_ORDER for joint in ("hip", "thigh", "calf")
)
# 轮关节（每腿第 4 个 joint，即 Link4 驱动轮），动作顺序 FL→FR→RL→RR。
WOLF_WHEEL_JOINT_NAMES = tuple(f"{leg}_foot" for leg in WOLF_LEG_ORDER)
# mjlab 动作布局（16 维）：12 个腿关节位置在前、4 个轮速在后。
# 与 mjlab ``ActionManager`` 的 action term 顺序一致（见 task 配置里
# ``actions = {"leg_joint_pos": ..., "wheel_vel": ...}``）。
WOLF_ACTION_LAYOUT = WOLF_LEG_JOINT_ORDER + WOLF_WHEEL_JOINT_NAMES
# 轮子刚体（Link4）名（观测 / DR 选择用）。
WOLF_WHEEL_BODY_NAMES = tuple(f"{leg}_Link4" for leg in WOLF_LEG_ORDER)
# 轮子碰撞 geom 名（URDF 转换时命名）；腿和躯干的碰撞 geom 沿用源 XML 的
# 匿名形式，任务侧按 body 匹配。
WOLF_WHEEL_COLLISION_GEOM_NAMES = tuple(
  f"{leg}_wheel_collision" for leg in WOLF_LEG_ORDER
)
# foot / height / climb 等 MDP 项使用的 site 名（XML 里新增的轮心 site）。
WOLF_SITE_NAMES = WOLF_LEG_ORDER
# 轮子符号 contract：策略正轮速动作 = 该轮为机身前进方向（base +x）驱动。
# 验证依据（默认姿态 FK）：FL/RL 的轮轴 world 方向为 +y（a × ẑ = +x̂），
# FR/RR 的轮轴 world 方向为 -y（a × ẑ = -x̂），因此 MuJoCo joint target
# 为 policy 目标乘以该符号。动作侧直接编码进 ``WOLF_WHEEL_ACTION_SCALE_BY_JOINT``。
WOLF_WHEEL_FORWARD_SIGN = {"FL": 1, "FR": -1, "RL": 1, "RR": -1}

# 轮几何（XML 冻结值；测试用于 lowest-point / rolling 验证）。
# 2026-10 更新：轮直径 200mm → 160mm（碰撞圆柱 size 0.1 0.0225 → 0.08 0.0225）。
WOLF_WHEEL_RADIUS = 0.08
WOLF_WHEEL_HALF_WIDTH = 0.0225
# 站立姿态下轮心离地高度（= 轮半径；foot_clearance 的目标高度用它）。
WOLF_STANDING_WHEEL_HEIGHT = WOLF_WHEEL_RADIUS

#
# IMU（imu_Link body 内原生 site + XML sensor；scene 中按 "<entity>/imu_*" 访问）
#

WOLF_IMU_BODY_NAME = "imu_Link"
WOLF_IMU_SITE_NAME = "imu_site"
# XML sensor 名（scene 访问时带实体名前缀，例如 "robot/imu_ang_vel"）。
WOLF_IMU_ANG_VEL_SENSOR = "imu_ang_vel"
WOLF_IMU_UPVECTOR_SENSOR = "imu_upvector"
# IMU 相对 base_link 的安装位置偏移（MJCF 冻结值）。
WOLF_IMU_BODY_OFFSET = (0.0, 0.0, 0.05975)

# 根 body（躯干）名：upright / payload / 观测等按 body 选择。
WOLF_ROOT_BODY_NAME = "base_link"


##
# PD actuator contract（与源资产一致）。
##

# 腿部（hip/thigh/calf）position PD
WOLF_LEG_STIFFNESS = 80.0
WOLF_LEG_DAMPING = 3.0
WOLF_LEG_EFFORT_LIMIT = 60.0
# 轮部（foot）velocity PD：Kp=0 → τ = Kd * (vel_target - dq)，clamp ±17。
WOLF_WHEEL_STIFFNESS = 0.0
WOLF_WHEEL_DAMPING = 1.0
WOLF_WHEEL_EFFORT_LIMIT = 17.0

_LEG_JOINTS = ("hip", "thigh", "calf")


def _actuator_cfg(joint_name: str) -> IdealPdActuatorCfg:
  if any(joint_name.endswith(f"_{joint}") for joint in _LEG_JOINTS):
    return IdealPdActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=WOLF_LEG_STIFFNESS,
      damping=WOLF_LEG_DAMPING,
      effort_limit=WOLF_LEG_EFFORT_LIMIT,
    )
  if joint_name in WOLF_WHEEL_JOINT_NAMES:
    return IdealPdActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=WOLF_WHEEL_STIFFNESS,
      damping=WOLF_WHEEL_DAMPING,
      effort_limit=WOLF_WHEEL_EFFORT_LIMIT,
    )
  raise KeyError(f"unknown Wolf joint: {joint_name}")


# 逐关节 16 个 IdealPdActuatorCfg，覆盖 WOLF_ACTION_LAYOUT。
# sort_actuators=True 时 MjLab 会按 model natural order 排列编译后的 actuator /
# ctrl 顺序，它与 policy action 顺序是两个概念（action mapping 按 joint 名）。
WOLF_ACTUATOR_CFGS = tuple(_actuator_cfg(name) for name in WOLF_ACTION_LAYOUT)

WOLF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=WOLF_ACTUATOR_CFGS,
  soft_joint_pos_limit_factor=0.9,
)


##
# Action contract.
##

# 腿：position action。缩放沿用 Black 的 0.25 * effort_limit / stiffness
# → 0.25 * 60 / 80 = 0.1875 rad per unit action。
WOLF_LEG_ACTION_SCALE: float = 0.25 * WOLF_LEG_EFFORT_LIMIT / WOLF_LEG_STIFFNESS
# 轮：velocity action。动作单位 = rad/s 的轮速目标；10 rad/s 约等于
# 0.8 m/s 的车速（轮半径 0.08 m）。
WOLF_WHEEL_ACTION_SCALE: float = 10.0
# 轮速目标裁剪（rad/s）：约 4 m/s 线速度上限，防止动作发散打穿物理。
WOLF_WHEEL_VELOCITY_CLIP: tuple[float, float] = (-50.0, 50.0)
# 腿动作缩放（按 joint 名）。
WOLF_LEG_ACTION_SCALE_BY_JOINT: dict[str, float] = {
  name: WOLF_LEG_ACTION_SCALE for name in WOLF_LEG_JOINT_ORDER
}
# 每个轮子的实际缩放：符号 contract × 速度缩放。
WOLF_WHEEL_ACTION_SCALE_BY_JOINT: dict[str, float] = {
  WOLF_LEG_JOINT_NAMES[leg][-1]: WOLF_WHEEL_FORWARD_SIGN[leg] * WOLF_WHEEL_ACTION_SCALE
  for leg in WOLF_LEG_ORDER
}


##
# Collision contract.
##

# 轮子接触：condim=6 才能同时随机化滑动 / 扭转 / 滚动三轴摩擦（SRMA teacher
# 的 friction 特权观测是 3 维）。摩擦数值与 Black 的足端一致。
# 其余 geom（匿名碰撞盒 / 圆柱、视觉 mesh）保持源 XML 编译结果不动。
WOLF_WHEEL_GEOM_CFG = GeomCfg(
  geom_names_expr=WOLF_WHEEL_COLLISION_GEOM_NAMES,
  condim=6,
  friction=(1.0, 5e-3, 5e-4),
)

##
# Initial state.
##

# 轮心偏移 0.3632 + 轮半径 0.08（calf=1.43 姿态 FK 实测几何接触高度）。
WOLF_DEFAULT_ROOT_Z = 0.4432

# 默认关节姿态（此前确认的 Wolf 值，单位 rad）。
# 注：thigh 与 calf 共用同一腿间符号映射（FL/RR 为正、FR/RL 为负）。
WOLF_DEFAULT_JOINT_POS = {
  "FL": (0.0, 0.82, 1.43, 0.0),
  "FR": (0.0, -0.82, -1.43, 0.0),
  "RL": (0.0, -0.82, -1.43, 0.0),
  "RR": (0.0, 0.82, 1.43, 0.0),
}

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, WOLF_DEFAULT_ROOT_Z),
  joint_pos={
    joint: value
    for leg in WOLF_LEG_ORDER
    for joint, value in zip(
      WOLF_LEG_JOINT_NAMES[leg], WOLF_DEFAULT_JOINT_POS[leg], strict=True
    )
  },
  joint_vel={".*": 0.0},
)


##
# Final robot config.
##


def _mesh_files() -> tuple[str, ...]:
  """MJCF <asset> 里引用的全部 mesh 相对文件名（按声明顺序）。"""
  import xml.etree.ElementTree as ET

  root = ET.parse(WOLF_XML).getroot()
  files: list[str] = []
  for mesh in root.iter("mesh"):
    filename = mesh.get("file")
    if filename:
      files.append(filename)
  return tuple(files)


def get_spec() -> mujoco.MjSpec:
  """返回 Wolf 的完整 MJCF spec（网格资产缺失时直接报错，不静默回退）。"""
  missing = [f for f in _mesh_files() if not (WOLF_ASSETS_DIR / f).is_file()]
  if missing:
    raise FileNotFoundError(
      f"Wolf mesh assets missing: {missing}（meshdir={WOLF_ASSETS_DIR}）"
    )
  return mujoco.MjSpec.from_file(str(WOLF_XML))


def get_wolf_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    init_state=INIT_STATE,
    articulation=WOLF_ARTICULATION,
    geoms=(WOLF_WHEEL_GEOM_CFG,),
    sort_actuators=True,
  )


if __name__ == "__main__":
  from mjlab.entity.entity import Entity

  robot = Entity(get_wolf_robot_cfg())
  model = robot.spec.compile()

  print("Wolf Entity compiled successfully.")
  print(f"nq: {model.nq}")
  print(f"nv: {model.nv}")
  print(f"nu: {model.nu}")
  print(f"nbody: {model.nbody}")

  print("\nJoints（MJCF natural order）:")
  for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

  print("\nActuators（编译后 ctrl 顺序）:")
  for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
