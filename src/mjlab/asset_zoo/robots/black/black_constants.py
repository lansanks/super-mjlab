"""Black quadruped constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

BLACK_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "black" / "black_description.xml"
)
assert BLACK_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(BLACK_XML))


##
# Actuator config.
##

# Rotor inertia. Black uses the same GO-M8010-6 motors as the Go1, so the
# rotor inertia is reused from the Go1 constants.
# Ref: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go1_description/urdf/go1.urdf#L515
# Extracted Ixx (rotation along x-axis).
ROTOR_INERTIA = 0.000111842

# Gearbox. The GO-M8010-6 datasheet lists a 6.33:1 rotor-to-output reduction
# while the Go1 config rounds it to 6; the resulting ~11% shift in reflected
# inertia sits inside the domain-randomization band, so Black stays aligned with
# the Go1. The calf uses the same 1.5:1 extra reduction as the Go1.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=30.1,
  effort_limit=23.7,
)
# The extra calf reduction trades speed for torque, scaling the motor's output
# ratings (23.7 N*m, 30 rad/s) by the belt ratio.
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=30.1 / 1.5,
  effort_limit=23.7 * 1.5,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

BLACK_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
BLACK_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

# Black's MJCF default pose, with the base height lowered from 0.5 m to 0.436 m
# so the feet rest on the ground at spawn (at 0.5 m the soles hang ~6 cm in the
# air). The right legs' thigh and calf joints are mirrored in the XML, hence the
# opposite signs.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.436),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*L_thigh_joint": 0.82,
    ".*R_thigh_joint": -0.82,
    ".*L_calf_joint": -1.5,
    ".*R_calf_joint": 1.5,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This enables all collisions.
# Foot collisions are given custom condim, friction.
# The Black XML numbers its collision geoms (trunk_collision1,
# FL_calf_collision2, ...), so the pattern allows a trailing index. This keeps
# the match set identical whether patterns are matched by prefix or in full.
_collision_regex = r".*_collision\d*"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  # Harden all collision geoms.
  solref=(0.01, 1),
  # Configure feet colliders. Other colliders are frictionless (condim=1).
  condim={_foot_regex: 6, _collision_regex: 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

BLACK_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BLACK_HIP_ACTUATOR_CFG,
    BLACK_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_black_robot_cfg() -> EntityCfg:
  """Get a fresh Black robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=BLACK_ARTICULATION,
  )


BLACK_ACTION_SCALE: dict[str, float] = {}
for a in BLACK_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    BLACK_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_black_robot_cfg())

  viewer.launch(robot.spec.compile())
