"""Tests for SRMA student DAgger scheduling and optimization."""

import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.srma.rl.algorithm.student_dagger import SRMAStudentDagger
from mjlab.tasks.srma.rl.network.student_actor import SRMAStudentActorModel
from mjlab.tasks.srma.rl.network.teacher_actor import SRMATeacherActorModel
from mjlab.tasks.srma.rl.storage import SRMAStudentRolloutStorage

NUM_ENVS = 4
OBS_DIM = 6
ACTION_DIM = 2
LATENT_DIM = 4
HIDDEN_DIM = 8
STEPS = 2

OBS_GROUPS = {
  "actor": ["actor"],
  "critic": ["critic"],
}


def make_obs() -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(NUM_ENVS, OBS_DIM),
      "critic": torch.randn(NUM_ENVS, 5),
      "teacher_height_map": torch.randn(NUM_ENVS, 9),
      "teacher_friction": torch.randn(NUM_ENVS, 2),
      "teacher_prev_push": torch.randn(NUM_ENVS, 2),
      "teacher_velocity_gt": torch.randn(NUM_ENVS, 3),
    },
    batch_size=[NUM_ENVS],
  )


def make_models() -> tuple[SRMAStudentActorModel, SRMATeacherActorModel]:
  torch.manual_seed(0)
  student = SRMAStudentActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(HIDDEN_DIM,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    student_cfg={
      "gru_hidden_dim": HIDDEN_DIM,
      "gru_num_layers": 1,
      "latent_dim": LATENT_DIM,
      "velocity_dim": 3,
    },
  )
  teacher = SRMATeacherActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(HIDDEN_DIM,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    srma_cfg={
      "height_shape": (3, 3),
      "latent_dim": LATENT_DIM,
      "height_channels": (4, 8),
      "height_hidden_dim": HIDDEN_DIM,
      "vector_hidden_dim": HIDDEN_DIM,
      "fusion_hidden_dim": HIDDEN_DIM,
    },
  )
  student.copy_actor_from_teacher(teacher.state_dict())
  return student, teacher


def make_algorithm(
  max_iterations: int = 10,
  pretrain_iterations: int = 1,
) -> SRMAStudentDagger:
  student, teacher = make_models()
  storage = SRMAStudentRolloutStorage(
    obs=make_obs(),
    num_envs=NUM_ENVS,
    num_transitions_per_env=STEPS,
    action_dim=ACTION_DIM,
    velocity_dim=3,
    latent_dim=LATENT_DIM,
    hidden_shape=(1, NUM_ENVS, HIDDEN_DIM),
  )
  return SRMAStudentDagger(
    student=student,
    teacher=teacher,
    storage=storage,
    pretrain_iterations=pretrain_iterations,
    num_learning_epochs=1,
    num_mini_batches=2,
    max_iterations=max_iterations,
  )


def fill_rollout(alg: SRMAStudentDagger) -> None:
  for step in range(STEPS):
    obs = make_obs()
    alg.act(obs)
    rewards = torch.ones(NUM_ENVS)
    dones = torch.zeros(NUM_ENVS, dtype=torch.long)
    if step == STEPS - 1:
      dones[-1] = 1
    alg.process_env_step(obs, rewards, dones, {"log": {}})


def test_schedule_uses_deciles_and_keeps_twenty_percent_teacher():
  alg = make_algorithm(max_iterations=500)
  expected = {
    0: 1.0,
    49: 1.0,
    50: 0.9,
    100: 0.8,
    200: 0.6,
    399: 0.3,
    400: 0.2,
    499: 0.2,
  }
  for iteration, fraction in expected.items():
    assert alg.teacher_fraction_for_iteration(iteration) == pytest.approx(fraction)
  assert all(
    alg.teacher_fraction_for_iteration(iteration) >= 0.2 for iteration in range(500)
  )


def test_pretrain_freezes_actor_and_dagger_unfreezes_it():
  alg = make_algorithm(pretrain_iterations=1)
  fill_rollout(alg)
  losses = alg.update()

  assert losses["student_action"] == 0.0
  assert all(not parameter.requires_grad for parameter in alg._actor_parameters())
  estimator_parameters = [
    *alg.student.estimator_encoder.parameters(),
    *alg.student.estimator_gru.parameters(),
    *alg.student.velocity_head.parameters(),
    *alg.student.latent_head.parameters(),
    *alg.student.latent_norm.parameters(),
  ]
  assert all(parameter.requires_grad for parameter in estimator_parameters)
  assert all(
    not parameter.requires_grad
    for parameter in alg.teacher.parameters()  # type: ignore[union-attr]
  )

  fill_rollout(alg)
  losses = alg.update()
  assert losses["student_action"] > 0.0
  assert all(parameter.requires_grad for parameter in alg._actor_parameters())


def test_teacher_behavior_uses_deterministic_action_mean():
  alg = make_algorithm()
  obs = make_obs()
  action = alg.act(obs)

  teacher = alg.teacher
  assert teacher is not None
  with torch.no_grad():
    teacher.get_latent(obs)
    expected = teacher.action_mean_with_latent(obs, teacher.last_external_output.latent)
  torch.testing.assert_close(action, expected)
  assert alg.transition is not None
  torch.testing.assert_close(alg.transition.behavior_action, expected)


def test_roles_persist_until_done_and_actual_fraction_is_logged():
  alg = make_algorithm()
  alg.behavior_is_teacher[:] = torch.tensor([[True], [False], [True], [False]])
  before = alg.behavior_is_teacher.clone()
  obs = make_obs()
  alg.act(obs)
  extras = {"log": {}}
  alg.process_env_step(
    obs,
    torch.ones(NUM_ENVS),
    torch.zeros(NUM_ENVS, dtype=torch.long),
    extras,
  )
  torch.testing.assert_close(alg.behavior_is_teacher, before)
  assert extras["log"]["Metrics/student_teacher_fraction"] == pytest.approx(0.5)


def test_checkpoint_restores_schedule_and_loss_scales():
  alg = make_algorithm(max_iterations=500)
  fill_rollout(alg)
  alg.update()
  alg.num_updates = 123
  alg.actual_dagger_round = 100
  alg._velocity_scale.fill_(0.7)
  alg._action_scale.fill_(0.4)
  saved = alg.save()

  alg.num_updates = 0
  alg.actual_dagger_round = 0
  alg._velocity_scale.fill_(1.0)
  alg._action_scale.fill_(1.0)
  alg.load(saved, load_cfg=None, strict=True)

  assert alg.num_updates == 123
  assert alg.actual_dagger_round == 100
  torch.testing.assert_close(
    alg._velocity_scale,
    torch.full_like(alg._velocity_scale, 0.7),
  )
  torch.testing.assert_close(
    alg._action_scale,
    torch.full_like(alg._action_scale, 0.4),
  )
