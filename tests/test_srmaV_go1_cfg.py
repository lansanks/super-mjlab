"""Tests for the shared split-encoder SRMAV teacher on Go1."""

import math

import mjlab.tasks.srmaV.config.go1  # noqa: F401
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.srmaV.config.go1.env_cfg import unitree_go1_srmaV_env_cfg
from mjlab.tasks.srmaV.srmaV_env_cfg import SRMAV_TEACHER_HEIGHT_SHAPE


def test_go1_teacher_uses_the_shared_central_scan() -> None:
  cfg = unitree_go1_srmaV_env_cfg()
  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}
  term = cfg.observations["teacher_height_map"].terms["height_map"]

  assert "srmaV_teacher_scan" not in sensors
  assert term.params["height_sensor_name"] == "terrain_scan"
  assert term.params["height_shape"] == SRMAV_TEACHER_HEIGHT_SHAPE
  assert math.prod(SRMAV_TEACHER_HEIGHT_SHAPE) == 187


def test_go1_teacher_config_matches_the_shared_split_architecture() -> None:
  cfg = load_rl_cfg("Mjlab-SRMAV-Teacher-Rough-Unitree-Go1")
  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
  assert cfg.actor.srmaV_cfg is not None
  assert cfg.actor.srmaV_cfg["height_latent_dim"] == 16
  assert cfg.actor.srmaV_cfg["privilege_latent_dim"] == 6
  assert cfg.actor.srmaV_cfg["height_hidden_dim"] == 32
  assert cfg.actor.srmaV_cfg["privilege_hidden_dim"] == 32
