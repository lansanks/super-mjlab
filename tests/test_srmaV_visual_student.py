"""Tests for the depth-conditioned Black SRMAV student."""

from dataclasses import asdict

import pytest
import torch
from conftest import get_test_device
from tensordict import TensorDict

import mjlab.tasks.srmaV.config.black  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.srmaV.config.black.env_cfg import (
  BLACK_TEACHER_HEIGHT_SHAPE,
)
from mjlab.tasks.srmaV.config.black.student_env_cfg import (
  black_srmaV_student_env_cfg,
)
from mjlab.tasks.srmaV.rl.algorithm.visual_student_dagger import (
  SRMAVVisualStudentDagger,
)
from mjlab.tasks.srmaV.rl.config import (
  BLACK_SRMAV_TEACHER_CHECKPOINT,
  RslRlSRMAVVisualStudentRunnerCfg,
)
from mjlab.tasks.srmaV.rl.network.teacher_actor import SRMAVTeacherActorModel
from mjlab.tasks.srmaV.rl.network.visual_student_actor import (
  SRMAVVisualStudentActorModel,
)
from mjlab.tasks.srmaV.rl.visual_student_runner import (
  SRMAVVisualStudentRunner,
)
from mjlab.tasks.srmaV.rl.visual_student_storage import (
  SRMAVVisualStudentRolloutStorage,
)

NUM_ENVS = 2
OBS_DIM = 45
DEPTH_SHAPE = (18, 32, 2)
HEIGHT_LATENT_DIM = 16
PRIVILEGE_LATENT_DIM = 6
ACTION_DIM = 12
HIDDEN_DIM = 8

OBS_GROUPS = {
  "actor": ["actor"],
  "critic": ["critic"],
}


def make_obs() -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(NUM_ENVS, OBS_DIM),
      "critic": torch.randn(NUM_ENVS, 16),
      "student_depth": torch.rand(NUM_ENVS, *DEPTH_SHAPE),
      "teacher_height_map": torch.randn(
        NUM_ENVS,
        BLACK_TEACHER_HEIGHT_SHAPE[0] * BLACK_TEACHER_HEIGHT_SHAPE[1],
      ),
      "teacher_friction": torch.randn(NUM_ENVS, 3),
      "teacher_prev_push": torch.randn(NUM_ENVS, 6),
      "teacher_velocity_gt": torch.randn(NUM_ENVS, 3),
    },
    batch_size=[NUM_ENVS],
  )


def make_models() -> tuple[
  SRMAVVisualStudentActorModel,
  SRMAVTeacherActorModel,
]:
  torch.manual_seed(0)
  distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }
  student = SRMAVVisualStudentActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(16, 8),
    distribution_cfg=distribution_cfg,
    student_cfg={
      "gru_hidden_dim": HIDDEN_DIM,
      "gru_num_layers": 1,
      "height_latent_dim": HEIGHT_LATENT_DIM,
      "privilege_latent_dim": PRIVILEGE_LATENT_DIM,
      "velocity_dim": 3,
      "depth_feature_dim": 8,
      "depth_channels": (4, 8),
    },
  )
  teacher = SRMAVTeacherActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(16, 8),
    distribution_cfg=distribution_cfg,
    srmaV_cfg={
      "height_shape": BLACK_TEACHER_HEIGHT_SHAPE,
      "height_latent_dim": HEIGHT_LATENT_DIM,
      "privilege_latent_dim": PRIVILEGE_LATENT_DIM,
      "height_hidden_dim": 8,
      "privilege_hidden_dim": 8,
    },
  )
  student.copy_actor_from_teacher(teacher.state_dict())
  return student, teacher


def test_visual_student_task_protocol_is_registered() -> None:
  cfg = load_rl_cfg("Mjlab-SRMAV-Student-Rough-Black")
  assert isinstance(cfg, RslRlSRMAVVisualStudentRunnerCfg)
  assert cfg.num_steps_per_env == 24
  assert cfg.teacher_checkpoint == BLACK_SRMAV_TEACHER_CHECKPOINT
  assert cfg.algorithm.pretrain_iterations == 20
  assert cfg.algorithm.teacher_fraction_start == 1.0
  assert cfg.algorithm.teacher_fraction_min == 0.2
  assert cfg.algorithm.num_mini_batches == 8
  student_cfg = asdict(cfg.student)["student_cfg"]
  assert student_cfg["depth_group"] == "student_depth"
  assert student_cfg["height_latent_dim"] == 16
  assert student_cfg["privilege_latent_dim"] == 6

  env_cfg = black_srmaV_student_env_cfg()
  assert "student_depth" in env_cfg.observations
  assert env_cfg.camera_update_period == 2

  play_cfg = black_srmaV_student_env_cfg(play=True)
  assert set(play_cfg.observations) == {"actor", "student_depth"}


def test_visual_student_outputs_and_teacher_actor_copy() -> None:
  student, teacher = make_models()
  output = student.forward_deterministic(make_obs())

  assert output.velocity.shape == (NUM_ENVS, 3)
  assert output.height_latent.shape == (NUM_ENVS, HEIGHT_LATENT_DIM)
  assert output.privilege_latent.shape == (NUM_ENVS, PRIVILEGE_LATENT_DIM)
  assert output.latent.shape == (
    NUM_ENVS,
    HEIGHT_LATENT_DIM + PRIVILEGE_LATENT_DIM,
  )
  assert output.actor_mean.shape == (NUM_ENVS, ACTION_DIM)
  assert student.get_hidden_state().shape == (
    2,
    1,
    NUM_ENVS,
    HIDDEN_DIM,
  )
  assert student.mlp[0].in_features == 70
  torch.testing.assert_close(student.mlp[0].weight, teacher.mlp[0].weight)


def test_action_loss_does_not_reach_either_estimator_gru() -> None:
  student, _ = make_models()
  obs = make_obs()
  time_steps = 2
  observations = obs["actor"].unsqueeze(0).repeat(time_steps, 1, 1)
  depth = obs["student_depth"].half().unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
  initial_hidden = torch.zeros(2, 1, NUM_ENVS, HIDDEN_DIM)
  dones = torch.zeros(time_steps, NUM_ENVS, 1)

  output = student.forward_sequence(
    observations,
    depth,
    initial_hidden,
    dones,
  )
  output.actor_mean.square().mean().backward()
  assert student.visual_gru.weight_ih_l0.grad is None
  assert student.proprio_gru.weight_ih_l0.grad is None
  assert student.mlp[0].weight.grad is not None

  student.zero_grad(set_to_none=True)
  output = student.forward_sequence(
    observations,
    depth,
    initial_hidden,
    dones,
  )
  (
    output.velocity.square().mean()
    + output.height_latent.square().mean()
    + output.privilege_latent.square().mean()
  ).backward()
  assert student.visual_gru.weight_ih_l0.grad is not None
  assert student.proprio_gru.weight_ih_l0.grad is not None


def test_visual_student_hidden_resets_only_done_environments() -> None:
  student, _ = make_models()
  student.forward_deterministic(make_obs())
  before = student.get_hidden_state().clone()
  dones = torch.tensor([1, 0], dtype=torch.long)

  student.reset(dones)
  after = student.get_hidden_state()
  torch.testing.assert_close(after[:, :, 0], torch.zeros_like(after[:, :, 0]))
  torch.testing.assert_close(after[:, :, 1:], before[:, :, 1:])


def test_visual_student_onnx_wrapper_shapes() -> None:
  student, _ = make_models()
  wrapper = student.as_onnx(verbose=False)
  obs, depth, hidden = wrapper.get_dummy_inputs()  # type: ignore[attr-defined]
  actions, hidden_out = wrapper(obs, depth, hidden)

  assert actions.shape == (1, ACTION_DIM)
  assert hidden_out.shape == (2, 1, 1, HIDDEN_DIM)
  assert wrapper.input_names == ["obs", "depth", "h_in"]
  assert wrapper.output_names == ["actions", "h_out"]


def test_visual_dagger_records_split_targets_and_updates() -> None:
  student, teacher = make_models()
  storage = SRMAVVisualStudentRolloutStorage(
    obs=make_obs(),
    num_envs=NUM_ENVS,
    num_transitions_per_env=2,
    action_dim=ACTION_DIM,
    velocity_dim=3,
    height_latent_dim=HEIGHT_LATENT_DIM,
    privilege_latent_dim=PRIVILEGE_LATENT_DIM,
    hidden_shape=(2, 1, NUM_ENVS, HIDDEN_DIM),
  )
  alg = SRMAVVisualStudentDagger(
    student=student,
    teacher=teacher,
    storage=storage,
    pretrain_iterations=1,
    num_learning_epochs=1,
    num_mini_batches=1,
    max_iterations=10,
  )

  for _step in range(2):
    obs = make_obs()
    action = alg.act(obs)
    assert alg.transition is not None
    assert alg.transition.obs_depth.shape == (NUM_ENVS, *DEPTH_SHAPE)
    assert alg.transition.teacher_height_latent.shape == (
      NUM_ENVS,
      HEIGHT_LATENT_DIM,
    )
    assert alg.transition.teacher_privilege_latent.shape == (
      NUM_ENVS,
      PRIVILEGE_LATENT_DIM,
    )
    alg.process_env_step(
      obs,
      torch.zeros(NUM_ENVS),
      torch.zeros(NUM_ENVS, dtype=torch.long),
      {"log": {}},
    )
    assert action.shape == (NUM_ENVS, ACTION_DIM)

  losses = alg.update()
  assert losses["student_action"] == 0.0
  assert losses["student_height_latent"] >= 0.0
  assert losses["student_privilege_latent"] >= 0.0


def test_dagger_schedule_matches_srma_deciles() -> None:
  student, _ = make_models()
  obs = make_obs()
  storage = SRMAVVisualStudentRolloutStorage(
    obs=obs,
    num_envs=NUM_ENVS,
    num_transitions_per_env=2,
    action_dim=ACTION_DIM,
    velocity_dim=3,
    height_latent_dim=HEIGHT_LATENT_DIM,
    privilege_latent_dim=PRIVILEGE_LATENT_DIM,
    hidden_shape=(2, 1, NUM_ENVS, HIDDEN_DIM),
  )
  alg = SRMAVVisualStudentDagger(
    student=student,
    teacher=None,
    storage=storage,
    pretrain_iterations=1,
    max_iterations=500,
  )

  expected = {
    0: 1.0,
    49: 1.0,
    50: 0.9,
    100: 0.8,
    400: 0.2,
    499: 0.2,
  }
  for iteration, fraction in expected.items():
    assert alg.teacher_fraction_for_iteration(iteration) == pytest.approx(fraction)


def test_visual_student_play_runner_builds_and_saves(tmp_path) -> None:
  cfg = black_srmaV_student_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  wrapped = RslRlVecEnvWrapper(env)
  try:
    obs = wrapped.get_observations()
    assert set(obs.keys()) == {"actor", "student_depth"}
    depth = obs["student_depth"]
    assert isinstance(depth, torch.Tensor)
    assert depth.shape == (1, 90, 160, 2)

    runner_cfg = load_rl_cfg("Mjlab-SRMAV-Student-Rough-Black")
    runner = SRMAVVisualStudentRunner(
      wrapped,
      asdict(runner_cfg),
      log_dir=None,
      device=get_test_device(),
    )
    assert isinstance(runner.alg, SRMAVVisualStudentDagger)
    assert runner.alg.teacher is None
    actions = runner.get_inference_policy(device=get_test_device())(obs)
    assert actions.shape == (1, ACTION_DIM)

    runner.current_learning_iteration = 3
    runner.alg.num_updates = 7
    checkpoint_path = tmp_path / "model_3.pt"
    runner.save(str(checkpoint_path))
    checkpoint = torch.load(
      checkpoint_path,
      map_location="cpu",
      weights_only=False,
    )
    assert checkpoint["iter"] == 3
    assert checkpoint["infos"]["srmaV"]["num_updates"] == 7
  finally:
    wrapped.close()
