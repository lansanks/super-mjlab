"""Tests for the Wolf SRMA teacher and student configurations."""

from typing import cast

import mujoco
import numpy as np
import torch
from conftest import get_test_device

import mjlab.tasks.srma.config.wolf  # noqa: F401
from mjlab.asset_zoo.robots import get_wolf_robot_cfg
from mjlab.asset_zoo.robots.wolf.wolf_constants import (
  INIT_STATE,
  WOLF_ACTION_LAYOUT,
  WOLF_LEG_JOINT_ORDER,
  WOLF_WHEEL_COLLISION_GEOM_NAMES,
  WOLF_WHEEL_FORWARD_SIGN,
  WOLF_WHEEL_JOINT_NAMES,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.tasks.srma.config.wolf.env_cfg import (
  WOLF_TEACHER_HEIGHT_SHAPE,
  WOLF_TEACHER_SCAN_RESOLUTION,
  WOLF_TEACHER_SCAN_SIZE,
  wolf_srma_env_cfg,
)
from mjlab.tasks.srma.config.wolf.student_env_cfg import wolf_srma_student_env_cfg
from mjlab.tasks.srma.rl.config import RslRlSRMAStudentRunnerCfg

TEACHER_TASK = "Mjlab-SRMA-Teacher-Rough-Wolf"
STUDENT_TASK = "Mjlab-SRMA-Student-Rough-Wolf"


def test_wolf_tasks_are_registered_with_matching_rl_configs() -> None:
  task_ids = set(list_tasks())
  assert TEACHER_TASK in task_ids
  assert STUDENT_TASK in task_ids

  teacher_cfg = load_rl_cfg(TEACHER_TASK)
  assert isinstance(teacher_cfg, RslRlOnPolicyRunnerCfg)
  assert teacher_cfg.experiment_name == "wolf_srma_teacher"
  assert teacher_cfg.actor.srma_cfg is not None
  assert teacher_cfg.actor.srma_cfg["height_shape"] == WOLF_TEACHER_HEIGHT_SHAPE

  student_cfg = load_rl_cfg(STUDENT_TASK)
  assert isinstance(student_cfg, RslRlSRMAStudentRunnerCfg)
  assert student_cfg.experiment_name == "wolf_srma_student"
  assert student_cfg.teacher_checkpoint == ""
  assert student_cfg.teacher.srma_cfg is not None
  assert student_cfg.teacher.srma_cfg["height_shape"] == WOLF_TEACHER_HEIGHT_SHAPE


def test_wolf_action_contract_is_split_into_legs_and_wheels() -> None:
  cfg = wolf_srma_env_cfg()
  assert list(cfg.actions) == ["leg_joint_pos", "wheel_vel"]

  legs = cfg.actions["leg_joint_pos"]
  assert isinstance(legs, JointPositionActionCfg)
  assert legs.actuator_names == WOLF_LEG_JOINT_ORDER
  assert legs.use_default_offset

  wheels = cfg.actions["wheel_vel"]
  assert isinstance(wheels, JointVelocityActionCfg)
  assert wheels.actuator_names == WOLF_WHEEL_JOINT_NAMES
  assert wheels.use_default_offset
  # Positive policy action = wheel drives the chassis forward.
  scales = wheels.scale
  assert isinstance(scales, dict)
  for leg in WOLF_WHEEL_FORWARD_SIGN:
    scale = scales[f"{leg}_foot"]
    assert np.sign(scale) == WOLF_WHEEL_FORWARD_SIGN[leg]
    assert abs(scale) == 10.0


def test_wolf_joint_order_matches_action_layout() -> None:
  entity = Entity(get_wolf_robot_cfg())
  leg_ids, leg_names = entity.find_joints_by_actuator_names(WOLF_LEG_JOINT_ORDER)
  wheel_ids, wheel_names = entity.find_joints_by_actuator_names(WOLF_WHEEL_JOINT_NAMES)
  assert tuple(leg_names) + tuple(wheel_names) == WOLF_ACTION_LAYOUT
  assert len(leg_ids) == 12
  assert len(wheel_ids) == 4


def test_wolf_robot_specific_observations_and_rewards_are_wired() -> None:
  cfg = wolf_srma_env_cfg()
  actor_terms = cfg.observations["actor"].terms
  assert list(actor_terms) == [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "wheel_vel",
    "actions",
    "command",
  ]
  # Wheel joint positions integrate freely under velocity control, so they are
  # excluded from the position observation but present in the velocity ones.
  joint_pos_cfg = actor_terms["joint_pos"].params["asset_cfg"]
  assert joint_pos_cfg.joint_names == WOLF_LEG_JOINT_ORDER
  joint_vel_cfg = actor_terms["joint_vel"].params["asset_cfg"]
  assert joint_vel_cfg.joint_names == WOLF_LEG_JOINT_ORDER
  wheel_vel_cfg = actor_terms["wheel_vel"].params["asset_cfg"]
  assert wheel_vel_cfg.joint_names == WOLF_WHEEL_JOINT_NAMES

  sites = ("FL", "FR", "RL", "RR")
  height_term = cfg.observations["teacher_height_map"].terms["height_map"]
  assert height_term.params["site_names"] == sites
  assert height_term.params["asset_cfg"].site_names == sites
  assert height_term.params["height_shape"] == WOLF_TEACHER_HEIGHT_SHAPE

  friction_term = cfg.observations["teacher_friction"].terms["friction"]
  assert friction_term.params["geom_names"] == WOLF_WHEEL_COLLISION_GEOM_NAMES
  assert friction_term.params["asset_cfg"].geom_names == WOLF_WHEEL_COLLISION_GEOM_NAMES

  pose = cfg.rewards["pose"]
  assert pose.params["asset_cfg"].joint_names == WOLF_LEG_JOINT_ORDER
  for reward_name in ("foot_clearance", "foot_slip", "stair_climb_progress"):
    assert cfg.rewards[reward_name].params["asset_cfg"].site_names == sites
  # Rolling wheel-center sites move without slipping, so the slip term is off.
  assert cfg.rewards["foot_slip"].weight == 0.0
  # The Wolf training MJCF has no subtree angular-momentum sensor.
  assert "angular_momentum" not in cfg.rewards

  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  for actuator in articulation.actuators:
    assert actuator.delay_min_lag == 0
    assert actuator.delay_max_lag == 3
    assert actuator.delay_update_period == 4


def test_wolf_teacher_scan_covers_the_wheel_footprint() -> None:
  cfg = wolf_srma_env_cfg()
  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}
  teacher_scan = sensors["srma_teacher_scan"]
  assert isinstance(teacher_scan, RayCastSensorCfg)
  assert isinstance(teacher_scan.pattern, GridPatternCfg)
  assert teacher_scan.pattern.size == WOLF_TEACHER_SCAN_SIZE
  assert teacher_scan.pattern.resolution == 0.05
  expected_shape = tuple(
    int(round(size / WOLF_TEACHER_SCAN_RESOLUTION)) + 1
    for size in WOLF_TEACHER_SCAN_SIZE
  )
  assert WOLF_TEACHER_HEIGHT_SHAPE == expected_shape
  assert expected_shape == (13, 11)

  foot_scan = sensors["foot_height_scan"]
  assert isinstance(foot_scan, TerrainHeightSensorCfg)
  foot_frames = foot_scan.frame
  assert isinstance(foot_frames, tuple)
  typed_frames = cast(tuple[ObjRef, ...], foot_frames)
  assert tuple(frame.name for frame in typed_frames) == ("FL", "FR", "RL", "RR")

  play_sensors = {
    sensor.name: sensor for sensor in (wolf_srma_env_cfg(play=True).scene.sensors or ())
  }
  play_terrain_scan = play_sensors["terrain_scan"]
  play_teacher_scan = play_sensors["srma_teacher_scan"]
  assert isinstance(play_terrain_scan, RayCastSensorCfg)
  assert isinstance(play_teacher_scan, RayCastSensorCfg)
  assert not play_terrain_scan.debug_vis
  assert play_teacher_scan.debug_vis


def test_wolf_initial_state_stands_on_the_wheels() -> None:
  """The default stance must put every wheel center at the wheel radius."""
  entity = Entity(get_wolf_robot_cfg())
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  data.qpos[0:3] = INIT_STATE.pos
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  assert INIT_STATE.joint_pos is not None
  for joint_name, value in INIT_STATE.joint_pos.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[model.jnt_qposadr[joint_id]] = value
  mujoco.mj_forward(model, data)

  for leg in ("FL", "FR", "RL", "RR"):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
    np.testing.assert_allclose(data.site_xpos[site_id][2], 0.08, atol=1e-4)


def test_wolf_wheel_geoms_use_condim_6() -> None:
  """Friction DR randomizes three axes, which needs condim >= 6."""
  model = Entity(get_wolf_robot_cfg()).spec.compile()
  wheel_names = set(WOLF_WHEEL_COLLISION_GEOM_NAMES)
  seen = set()
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name in wheel_names:
      seen.add(name)
      assert model.geom_condim[geom_id] == 6, name
  assert seen == wheel_names


def test_wolf_teacher_environment_builds_with_expected_observations() -> None:
  cfg = wolf_srma_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    expected_shapes = {
      # 3 ang vel + 3 grav + 12 leg pos + 12 leg vel + 4 wheel vel
      # + 16 actions + 3 command
      "actor": (1, 53),
      "critic": (1, 273),
      "teacher_height_map": (1, 143),
      "teacher_friction": (1, 3),
      "teacher_prev_push": (1, 6),
      "teacher_velocity_gt": (1, 3),
    }
    for group, shape in expected_shapes.items():
      value = obs[group]
      assert isinstance(value, torch.Tensor)
      assert value.shape == shape, group

    terrain_scan = env.scene["terrain_scan"]
    teacher_scan = env.scene["srma_teacher_scan"]
    assert terrain_scan.num_rays_per_frame == 187
    assert teacher_scan.num_rays_per_frame == 143
    assert env.action_manager.total_action_dim == 16
  finally:
    env.close()


def test_wolf_student_play_keeps_only_actor_group() -> None:
  cfg = wolf_srma_student_env_cfg(play=True)
  assert set(cfg.observations) == {"actor"}
  assert "srma_teacher_scan" not in {
    sensor.name for sensor in (cfg.scene.sensors or ())
  }
  assert cfg.viewer.latent_bar_attr == "srma_latent_estimate"
  assert cfg.viewer.latent_bar_label == "SRMA student l_hat"


def test_wolf_nan_termination_is_registered() -> None:
  """The official mjlab NaN workflow: NaN environments reset, training survives."""
  from mjlab.envs.mdp.terminations import nan_detection

  for cfg in (
    wolf_srma_env_cfg(),
    wolf_srma_env_cfg(play=True),
    wolf_srma_student_env_cfg(play=True),
  ):
    term = cfg.terminations["nan_term"]
    assert term.func is nan_detection
    assert term.time_out is False


def test_foot_contact_forces_observation_is_nan_safe() -> None:
  """A transient non-finite contact-force reading must not poison PPO."""
  from types import SimpleNamespace

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.srma.mdp.observations import foot_contact_forces

  force = torch.tensor(
    [[[float("nan"), 1.0, -2.0], [0.0, float("inf"), float("-inf")]]]
  )
  fake_env = SimpleNamespace(
    scene={"feet": SimpleNamespace(data=SimpleNamespace(force=force))}
  )
  out = foot_contact_forces(cast(ManagerBasedRlEnv, fake_env), "feet")
  assert torch.isfinite(out).all(), out
  # Finite inputs are transformed as before.
  finite = torch.tensor([[[1.0, -2.0, 0.0]]])
  fake_env = SimpleNamespace(
    scene={"feet": SimpleNamespace(data=SimpleNamespace(force=finite))}
  )
  out = foot_contact_forces(cast(ManagerBasedRlEnv, fake_env), "feet")
  flat = finite.flatten(start_dim=1)
  expected = torch.sign(flat) * torch.log1p(torch.abs(flat))
  torch.testing.assert_close(out, expected)


def test_wolf_nan_detection_flags_corrupted_state() -> None:
  """The nan_term termination must catch a NaN physics state before reset."""
  from mjlab.envs.mdp.terminations import nan_detection
  from mjlab.tasks.srma.config.wolf.env_cfg import wolf_srma_flat_env_cfg

  cfg = wolf_srma_flat_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    assert not bool(nan_detection(env).any())
    env.sim.data.qpos[0, 0] = float("nan")
    mask = nan_detection(env)
    assert bool(mask[0])
    assert not bool(mask[1])
  finally:
    env.close()
