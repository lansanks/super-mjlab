"""Tests for the standalone HiMLoco environment configuration."""

from conftest import get_test_device
from torch import Tensor

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.himloco.config.black.env_cfg import (
  black_him_env_cfg,
  black_him_flat_env_cfg,
)
from mjlab.tasks.himloco.config.go1.env_cfg import unitree_go1_him_env_cfg
from mjlab.tasks.himloco.himloco_env_cfg import (
  ANG_VEL_SCALE,
  COMMAND_SCALE,
  JOINT_VEL_SCALE,
  LIN_VEL_SCALE,
)


def test_him_observations_match_released_observation_scales():
  configs = (
    unitree_go1_him_env_cfg(),
    black_him_env_cfg(),
    black_him_flat_env_cfg(),
  )
  for cfg in configs:
    for group in ("actor", "critic", "him_input", "him_state"):
      terms = cfg.observations[group].terms
      assert terms["base_ang_vel"].scale == ANG_VEL_SCALE
      assert terms["joint_vel"].scale == JOINT_VEL_SCALE
      if group != "him_state":
        assert terms["command"].scale == COMMAND_SCALE

    assert cfg.observations["him_input"].history_length == 6
    assert cfg.observations["him_state"].terms["base_lin_vel"].scale == LIN_VEL_SCALE
    assert cfg.observations["critic"].terms["base_lin_vel"].scale == LIN_VEL_SCALE


def test_him_play_mode_disables_all_observation_noise():
  configs = (
    unitree_go1_him_env_cfg(play=True),
    black_him_env_cfg(play=True),
    black_him_flat_env_cfg(play=True),
  )
  for cfg in configs:
    for group in ("actor", "him_input", "him_state"):
      assert not cfg.observations[group].enable_corruption
    assert cfg.episode_length_s >= 1e9
    assert "push_robot" not in cfg.events


def test_him_environment_builds_with_expected_observation_shapes():
  cfg = unitree_go1_him_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs = env.observation_manager.compute()
    for group, shape in (
      ("actor", (1, 45)),
      ("critic", (1, 265)),
      ("him_input", (1, 6, 45)),
      ("him_state", (1, 45)),
    ):
      value = obs[group]
      assert isinstance(value, Tensor)
      assert value.shape == shape
  finally:
    env.close()
