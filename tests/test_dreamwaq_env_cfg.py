"""Tests for the DreamWaQ observation clips and simulation-health metric."""

import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import MetricsManager
from mjlab.tasks.dreamwaq.config.go1.env_cfg import (
  unitree_go1_dreamwaq_env_cfg,
  unitree_go1_dreamwaq_flat_env_cfg,
)


def test_actor_observations_are_clipped():
  """The unbounded terms must be clipped against violent simulation states."""
  for cfg in (unitree_go1_dreamwaq_env_cfg(), unitree_go1_dreamwaq_flat_env_cfg()):
    for group in ("actor", "cenet_input", "cenet_target_o"):
      terms = cfg.observations[group].terms
      assert terms["joint_vel"].clip == (-50.0, 50.0), group
      assert terms["base_ang_vel"].clip == (-30.0, 30.0), group
      assert terms["actions"].clip == (-20.0, 20.0), group
    # The critic inherits the clipped terms and adds a bound on the ground-truth
    # base velocity it observes.
    critic_terms = cfg.observations["critic"].terms
    assert critic_terms["joint_vel"].clip == (-50.0, 50.0)
    assert critic_terms["base_ang_vel"].clip == (-30.0, 30.0)
    assert critic_terms["base_lin_vel"].clip == (-10.0, 10.0)


def test_peak_joint_vel_metric_is_registered():
  device = get_test_device()
  cfg = unitree_go1_dreamwaq_flat_env_cfg(play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    assert isinstance(env.metrics_manager, MetricsManager)
    assert env.metrics_manager.active_terms == ["mean_action_acc", "peak_joint_vel"]
    term_cfg = env.metrics_manager.cfg["peak_joint_vel"]
    value = term_cfg.func(env, **term_cfg.params)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    assert torch.all(value >= 0.0)
  finally:
    env.close()
