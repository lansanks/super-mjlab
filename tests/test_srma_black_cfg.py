"""Tests for the Black SRMA teacher and student configurations."""

import torch
from conftest import get_test_device

import mjlab.tasks.srma.config.black  # noqa: F401
from mjlab.asset_zoo.robots import BLACK_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.tasks.srma.config.black.env_cfg import (
  BLACK_TEACHER_HEIGHT_SHAPE,
  BLACK_TEACHER_SCAN_SIZE,
  black_srma_env_cfg,
  black_srma_flat_env_cfg,
)
from mjlab.tasks.srma.config.black.student_env_cfg import (
  black_srma_student_env_cfg,
)
from mjlab.tasks.srma.rl.config import RslRlSRMAStudentRunnerCfg


def test_black_tasks_are_registered_with_matching_rl_configs() -> None:
  task_ids = set(list_tasks())
  assert "Mjlab-SRMA-Teacher-Rough-Black" in task_ids
  assert "Mjlab-SRMA-Student-Rough-Black" in task_ids

  teacher_cfg = load_rl_cfg("Mjlab-SRMA-Teacher-Rough-Black")
  assert isinstance(teacher_cfg, RslRlOnPolicyRunnerCfg)
  assert teacher_cfg.experiment_name == "black_srma_teacher"
  assert teacher_cfg.actor.srma_cfg is not None
  assert teacher_cfg.actor.srma_cfg["height_shape"] == BLACK_TEACHER_HEIGHT_SHAPE

  student_cfg = load_rl_cfg("Mjlab-SRMA-Student-Rough-Black")
  assert isinstance(student_cfg, RslRlSRMAStudentRunnerCfg)
  assert student_cfg.experiment_name == "black_srma_student"
  assert student_cfg.teacher_checkpoint == ""
  assert student_cfg.teacher.srma_cfg is not None
  assert student_cfg.teacher.srma_cfg["height_shape"] == BLACK_TEACHER_HEIGHT_SHAPE


def test_black_robot_specific_names_are_wired() -> None:
  cfg = black_srma_env_cfg()
  sites = ("FL", "FR", "RR", "RL")
  foot_geoms = tuple(f"{site}_foot_collision" for site in sites)

  height_term = cfg.observations["teacher_height_map"].terms["height_map"]
  assert height_term.params["site_names"] == sites
  assert height_term.params["asset_cfg"].site_names == sites
  assert height_term.params["height_shape"] == BLACK_TEACHER_HEIGHT_SHAPE

  friction_term = cfg.observations["teacher_friction"].terms["friction"]
  assert friction_term.params["geom_names"] == foot_geoms
  assert friction_term.params["asset_cfg"].geom_names == foot_geoms

  for reward_name in ("foot_clearance", "foot_slip", "stair_climb_progress"):
    reward = cfg.rewards[reward_name]
    assert reward.params["asset_cfg"].site_names == sites

  joint_action = cfg.actions["joint_pos"]
  assert isinstance(joint_action, JointPositionActionCfg)
  assert joint_action.scale == BLACK_ACTION_SCALE
  assert "trunk_collision" in cfg.rewards
  assert "trunk_head_collision" not in cfg.rewards

  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  for actuator in articulation.actuators:
    assert actuator.delay_min_lag == 0
    assert actuator.delay_max_lag == 3
    assert actuator.delay_update_period == 4


def test_black_teacher_scan_uses_expanded_map_for_training_and_play() -> None:
  cfg = black_srma_env_cfg()
  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}
  terrain_scan = sensors["terrain_scan"]
  teacher_scan = sensors["srma_teacher_scan"]
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(teacher_scan, RayCastSensorCfg)
  assert isinstance(teacher_scan.pattern, GridPatternCfg)
  assert teacher_scan.pattern.size == BLACK_TEACHER_SCAN_SIZE
  assert teacher_scan.pattern.resolution == 0.05

  play_sensors = {
    sensor.name: sensor
    for sensor in (black_srma_env_cfg(play=True).scene.sensors or ())
  }
  play_terrain_scan = play_sensors["terrain_scan"]
  play_teacher_scan = play_sensors["srma_teacher_scan"]
  assert isinstance(play_terrain_scan, RayCastSensorCfg)
  assert isinstance(play_teacher_scan, RayCastSensorCfg)
  assert not play_terrain_scan.debug_vis
  assert play_teacher_scan.debug_vis


def test_black_teacher_environment_builds_with_expected_observations() -> None:
  cfg = black_srma_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    expected_shapes = {
      "actor": (1, 45),
      "critic": (1, 265),
      "teacher_height_map": (1, 144),
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
    assert teacher_scan.num_rays_per_frame == 144
  finally:
    env.close()


def test_black_student_play_keeps_only_actor_group() -> None:
  cfg = black_srma_student_env_cfg(play=True)
  assert set(cfg.observations) == {"actor"}
  assert "srma_teacher_scan" not in {
    sensor.name for sensor in (cfg.scene.sensors or ())
  }
  assert cfg.viewer.latent_bar_attr == "srma_latent_estimate"
  assert cfg.viewer.latent_bar_label == "SRMA student l_hat"


def test_black_flat_teacher_height_map_is_zero() -> None:
  cfg = black_srma_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    height_map = obs["teacher_height_map"]
    assert isinstance(height_map, torch.Tensor)
    assert height_map.shape == (1, 144)
    torch.testing.assert_close(height_map, torch.zeros_like(height_map))
  finally:
    env.close()
