"""Tests for the DreamWaQ observation clips and simulation-health metric."""

import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots import get_go1_robot_cfg
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


def test_domain_randomization_follows_paper_table_ii():
  """Paper Table II ranges are wired while the previous extras stay intact."""
  cfg = unitree_go1_dreamwaq_env_cfg()

  payload = cfg.events["payload_inertia"]
  assert payload.mode == "startup"
  assert payload.params["mass_range"] == (-1.0, 2.0)
  assert payload.params["com_shift_range"] == (-0.05, 0.05)
  assert payload.params["asset_cfg"].body_names == ("trunk",)

  pd_gains = cfg.events["pd_gains"]
  assert pd_gains.mode == "startup"
  assert pd_gains.params["operation"] == "scale"
  assert pd_gains.params["kp_range"] == (0.9, 1.1)
  assert pd_gains.params["kd_range"] == (0.9, 1.1)

  motor_strength = cfg.events["motor_strength"]
  assert motor_strength.mode == "startup"
  assert motor_strength.params["operation"] == "scale"
  assert motor_strength.params["effort_limit_range"] == (0.9, 1.1)

  # The pre-existing extras remain untouched.
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.015, 0.015)
  assert cfg.events["push_robot"].mode == "interval"
  assert "foot_friction_slide" in cfg.events

  # Paper Table II: system delay 0-15 ms. One physics substep is 5 ms.
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  for actuator in articulation.actuators:
    assert actuator.delay_min_lag == 0
    assert actuator.delay_max_lag == 3

  # The shared asset-zoo actuator configs must not be mutated by the task.
  shared_articulation = get_go1_robot_cfg().articulation
  assert shared_articulation is not None
  for actuator in shared_articulation.actuators:
    assert actuator.delay_max_lag == 0


def test_paper_dr_is_applied_per_env():
  """Payload, gains, and effort limits vary per env within the paper ranges."""
  device = get_test_device()
  cfg = unitree_go1_dreamwaq_flat_env_cfg(play=True)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    robot = env.scene["robot"]
    model = env.sim.model

    body_id = robot.indexing.body_ids[robot.find_bodies("trunk")[0][0]]
    default_mass = env.sim.get_default_field("body_mass")[body_id]
    masses = model.body_mass[:, body_id]
    assert torch.all(masses >= default_mass - 1.0 - 1e-5)
    assert torch.all(masses <= default_mass + 2.0 + 1e-5)
    assert len(torch.unique(masses)) > 1

    default_ipos = env.sim.get_default_field("body_ipos")[body_id]
    # COM shift (<= 50 mm) plus the payload's own COM displacement (< 17 mm).
    assert torch.all((model.body_ipos[:, body_id] - default_ipos).abs() <= 0.07)

    ctrl = robot.indexing.ctrl_ids
    kp_default = env.sim.get_default_field("actuator_gainprm")[ctrl, 0]
    kd_default = env.sim.get_default_field("actuator_biasprm")[ctrl, 2]
    effort_default = env.sim.get_default_field("actuator_forcerange")[ctrl, 1]
    for ratio in (
      model.actuator_gainprm[:, ctrl, 0] / kp_default,
      model.actuator_biasprm[:, ctrl, 2] / kd_default,
      model.actuator_forcerange[:, ctrl, 1] / effort_default,
    ):
      assert torch.all(ratio >= 0.9 - 1e-5)
      assert torch.all(ratio <= 1.1 + 1e-5)

    assert all(actuator.has_delay for actuator in robot.actuators)
  finally:
    env.close()
