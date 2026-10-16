"""Tests for black_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.black import black_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def black_entity() -> Entity:
  return Entity(black_constants.get_black_robot_cfg())


@pytest.fixture(scope="module")
def black_model(black_entity: Entity) -> mujoco.MjModel:
  return black_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (black_constants.BLACK_HIP_ACTUATOR_CFG, black_constants.STIFFNESS_HIP, black_constants.DAMPING_HIP),
    (black_constants.BLACK_KNEE_ACTUATOR_CFG, black_constants.STIFFNESS_KNEE, black_constants.DAMPING_KNEE),
  ],
)
# fmt: on
def test_actuator_parameters(black_model, actuator_config, stiffness, damping):
  for i in range(black_model.nu):
    actuator = black_model.actuator(i)
    actuator_name = actuator.name
    matches = any(
      re.match(pattern, actuator_name) for pattern in actuator_config.target_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_knee_actuator_scales_with_belt_ratio() -> None:
  """The extra 2.5:1 calf reduction trades speed for torque."""
  hip, knee = black_constants.HIP_ACTUATOR, black_constants.KNEE_ACTUATOR
  assert knee.effort_limit == pytest.approx(hip.effort_limit * 2.5)
  assert knee.velocity_limit == pytest.approx(hip.velocity_limit / 2.5)
  assert knee.reflected_inertia == pytest.approx(hip.reflected_inertia * 2.5**2)


def test_keyframe_joint_positions(black_entity, black_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = black_model.key("init_state")
  expected_joint_pos = black_constants.INIT_STATE.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, black_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    black_entity.joint_names, expected_values, strict=True
  ):
    joint = black_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_default_pose_stands_on_the_ground(black_model) -> None:
  """The default pose must rest the feet on the ground, not in the air.

  The MJCF keyframe spawns at 0.5 m, where the feet hang ~6 cm above the
  ground; the initial state lowers the trunk to 0.436 m instead.
  """
  data = mujoco.MjData(black_model)
  data.qpos[:] = black_model.key("init_state").qpos
  mujoco.mj_forward(black_model, data)
  for site_name in ("FL", "FR", "RL", "RR"):
    foot_center_z = data.site_xpos[black_model.site(site_name).id][2]
    # Foot spheres have a 0.02 m radius, so the soles start on the ground.
    assert abs(foot_center_z - 0.02) < 5e-3, site_name


def test_foot_collision_geoms(black_model) -> None:
  """Foot collision geoms should have specific properties."""
  foot_pattern = r"^[FR][LR]_foot_collision$"
  for i in range(black_model.ngeom):
    geom = black_model.geom(i)
    if re.match(foot_pattern, geom.name):
      assert geom.condim == 6
      assert geom.priority == 1
      assert geom.friction[0] == 1.0


def test_all_collision_geoms_enabled(black_model) -> None:
  """Every *_collision geom, including numbered ones, must stay collidable.

  The black XML numbers its collision boxes and capsules (trunk_collision1,
  FL_calf_collision2, ...), which a pattern anchored at "_collision" would
  silently drop, letting disable_other_geoms zero their contype. Guard against
  that regression with an oracle independent of the config's own regex
  resolution.
  """
  collision_geoms = [
    black_model.geom(i)
    for i in range(black_model.ngeom)
    if "_collision" in black_model.geom(i).name
  ]
  assert len(collision_geoms) == 30
  for geom in collision_geoms:
    assert geom.contype == 1, f"{geom.name} lost its contype"
    assert geom.conaffinity == 1, f"{geom.name} lost its conaffinity"


def test_black_entity_creation(black_entity) -> None:
  """Test basic Black entity properties."""
  assert black_entity.num_actuators == 12
  assert black_entity.num_joints == 12
  assert black_entity.is_actuated
  assert not black_entity.is_fixed_base
