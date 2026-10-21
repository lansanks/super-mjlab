"""Tests for SRMA student environment and task configuration."""

from dataclasses import asdict

import torch
from conftest import get_test_device

import mjlab.tasks.srma.config.go1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.tasks.srma.config.go1.student_env_cfg import (
  unitree_go1_srma_student_env_cfg,
)
from mjlab.tasks.srma.rl.algorithm.student_dagger import SRMAStudentDagger
from mjlab.tasks.srma.rl.config import (
  DEFAULT_TEACHER_CHECKPOINT,
  RslRlSRMAStudentRunnerCfg,
)
from mjlab.tasks.srma.rl.runner import SRMAStudentRunner


def test_student_task_is_registered_with_final_protocol():
  assert "Mjlab-SRMA-Student-Rough-Unitree-Go1" in list_tasks()
  cfg = load_rl_cfg("Mjlab-SRMA-Student-Rough-Unitree-Go1")
  assert isinstance(cfg, RslRlSRMAStudentRunnerCfg)
  assert cfg.num_steps_per_env == 50
  assert cfg.max_iterations == 500
  assert cfg.teacher_checkpoint == DEFAULT_TEACHER_CHECKPOINT
  assert cfg.algorithm.pretrain_iterations == 20
  assert cfg.algorithm.teacher_fraction_min == 0.2
  assert asdict(cfg.student)["student_cfg"]["gru_hidden_dim"] == 64


def test_training_environment_keeps_all_teacher_groups():
  cfg = unitree_go1_srma_student_env_cfg()
  for group in (
    "critic",
    "teacher_height_map",
    "teacher_friction",
    "teacher_prev_push",
    "teacher_velocity_gt",
  ):
    assert group in cfg.observations
  assert "srma_teacher_scan" in {sensor.name for sensor in (cfg.scene.sensors or ())}


def test_play_environment_removes_teacher_observations_and_sensor():
  cfg = unitree_go1_srma_student_env_cfg(play=True)
  assert set(cfg.observations) == {"actor"}
  assert "srma_teacher_scan" not in {
    sensor.name for sensor in (cfg.scene.sensors or ())
  }
  assert "push_robot" not in cfg.events
  assert cfg.viewer.latent_bar_attr == "srma_latent_estimate"
  assert cfg.viewer.latent_bar_label == "SRMA student l_hat"


def test_student_play_runner_builds_and_saves(tmp_path):
  cfg = unitree_go1_srma_student_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  wrapped = RslRlVecEnvWrapper(env)
  try:
    obs = wrapped.get_observations()
    assert set(obs.keys()) == {"actor"}
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (1, 45)
    assert torch.isfinite(actor_obs).all()

    runner_cfg = load_rl_cfg("Mjlab-SRMA-Student-Rough-Unitree-Go1")
    runner = SRMAStudentRunner(
      wrapped,
      asdict(runner_cfg),
      log_dir=None,
      device=get_test_device(),
    )
    assert isinstance(runner.alg, SRMAStudentDagger)
    assert runner.alg.teacher is None
    actions = runner.get_inference_policy(device=get_test_device())(obs)
    assert actions.shape == (1, 12)

    runner.current_learning_iteration = 7
    runner.alg.num_updates = 123
    checkpoint_path = tmp_path / "model_7.pt"
    runner.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["iter"] == 7
    assert checkpoint["infos"]["srma"]["num_updates"] == 123
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=get_test_device(),
    )
  finally:
    wrapped.close()
