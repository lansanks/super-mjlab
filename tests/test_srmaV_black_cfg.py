"""Tests for the Black split-encoder SRMAV teacher configuration."""

import math

import torch
from conftest import get_test_device

import mjlab.tasks.srmaV.config.black  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.tasks.srmaV.config.black.env_cfg import (
  BLACK_TEACHER_HEIGHT_SHAPE,
  black_srmaV_env_cfg,
  black_srmaV_flat_env_cfg,
)
from mjlab.tasks.srmaV.config.black.terrain_cfg import (
  BLACK_SRMAV_BRIDGE_TERRAIN,
  BLACK_SRMAV_TERRAINS_CFG,
)
from mjlab.tasks.srmaV.mdp import UniformVelocityCommandCfg
from mjlab.terrains import BoxBrokenBridgeTerrainCfg, HfRandomUniformTerrainCfg


def test_black_teacher_reuses_the_central_velocity_scan() -> None:
  cfg = black_srmaV_env_cfg()
  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}

  assert "srmaV_teacher_scan" not in sensors
  terrain_scan = sensors["terrain_scan"]
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)
  assert terrain_scan.pattern.size == (1.6, 1.0)
  assert terrain_scan.pattern.resolution == 0.1

  term = cfg.observations["teacher_height_map"].terms["height_map"]
  assert term.params["height_sensor_name"] == "terrain_scan"
  assert term.params["height_shape"] == BLACK_TEACHER_HEIGHT_SHAPE
  assert math.prod(BLACK_TEACHER_HEIGHT_SHAPE) == 187


def test_black_teacher_config_uses_split_latent_dimensions() -> None:
  cfg = load_rl_cfg("Mjlab-SRMAV-Teacher-Rough-Black")
  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
  assert cfg.actor.srmaV_cfg is not None
  assert cfg.actor.srmaV_cfg["height_latent_dim"] == 16
  assert cfg.actor.srmaV_cfg["privilege_latent_dim"] == 6
  assert cfg.actor.srmaV_cfg["height_shape"] == BLACK_TEACHER_HEIGHT_SHAPE


def test_black_terrain_has_four_curriculum_columns() -> None:
  assert BLACK_SRMAV_TERRAINS_CFG.curriculum
  assert BLACK_SRMAV_TERRAINS_CFG.num_rows == 10
  assert list(BLACK_SRMAV_TERRAINS_CFG.sub_terrains) == [
    "stairs_up",
    "stairs_down",
    "random_rough",
    BLACK_SRMAV_BRIDGE_TERRAIN,
  ]
  bridge = BLACK_SRMAV_TERRAINS_CFG.sub_terrains[BLACK_SRMAV_BRIDGE_TERRAIN]
  assert isinstance(bridge, BoxBrokenBridgeTerrainCfg)
  assert bridge.gap_width_range == (0.10, 0.35)
  assert bridge.num_gaps == 4
  assert bridge.deck_between_gaps == 1.2
  assert bridge.bridge_width is None
  rough = BLACK_SRMAV_TERRAINS_CFG.sub_terrains["random_rough"]
  assert isinstance(rough, HfRandomUniformTerrainCfg)
  assert rough.scale_with_difficulty is False


def test_black_training_uses_bridge_commands_and_spawn() -> None:
  cfg = black_srmaV_env_cfg()
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.terrain_override_name == BLACK_SRMAV_BRIDGE_TERRAIN
  assert twist.terrain_override_lin_vel_x == (0.3, 1.0)
  assert twist.terrain_override_heading == 0.0
  assert cfg.events["reset_base"].func.__name__ == "reset_root_state_with_bridge_spawn"
  assert cfg.terminations["out_of_sub_terrain_bounds"].time_out is True
  bridge_reward = cfg.rewards["bridge_gap_crossing"]
  assert bridge_reward.weight == 0.1
  assert bridge_reward.params["foot_bonus"] == 2.5
  assert bridge_reward.params["terrain_name"] == BLACK_SRMAV_BRIDGE_TERRAIN


def test_black_flat_teacher_keeps_only_the_shared_terrain_scan() -> None:
  cfg = black_srmaV_flat_env_cfg(play=True)
  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}

  assert "terrain_scan" in sensors
  assert "srmaV_teacher_scan" not in sensors
  assert "height_scan" not in cfg.observations["critic"].terms
  assert "bridge_gap_crossing" not in cfg.rewards


def test_black_flat_teacher_height_map_is_zero() -> None:
  cfg = black_srmaV_flat_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    height_map = obs["teacher_height_map"]
    assert isinstance(height_map, torch.Tensor)
    assert height_map.shape == (1, 187)
    torch.testing.assert_close(height_map, torch.zeros_like(height_map))
  finally:
    env.close()


def test_black_srmaV_tasks_are_registered() -> None:
  tasks = list_tasks()
  assert "Mjlab-SRMAV-Teacher-Rough-Black" in tasks
  assert "Mjlab-SRMAV-Student-Rough-Black" in tasks
  assert "Mjlab-SRMAV-Play-Bridge-Black" not in tasks
