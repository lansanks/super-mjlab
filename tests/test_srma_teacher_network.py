"""Tests for the privileged SRMA teacher network and task registration."""

import torch
from tensordict import TensorDict

from mjlab.tasks.registry import list_tasks
from mjlab.tasks.srma.rl.algorithm.teacher_ppo import compute_teacher_metrics
from mjlab.tasks.srma.rl.network.teacher_actor import SRMATeacherActorModel

NUM_ENVS = 4
OBS_DIM = 6
CRITIC_DIM = 5
HEIGHT_SHAPE = (3, 3)
ACTION_DIM = 2

OBS_GROUPS = {
  "actor": ["actor"],
  "critic": ["critic"],
}


def make_obs() -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(NUM_ENVS, OBS_DIM),
      "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
      "teacher_height_map": torch.randn(NUM_ENVS, HEIGHT_SHAPE[0] * HEIGHT_SHAPE[1]),
      "teacher_friction": torch.randn(NUM_ENVS, 2),
      "teacher_prev_push": torch.randn(NUM_ENVS, 2),
      "teacher_velocity_gt": torch.randn(NUM_ENVS, 3),
    },
    batch_size=[NUM_ENVS],
  )


def make_actor(external_mode: str = "full") -> SRMATeacherActorModel:
  torch.manual_seed(0)
  return SRMATeacherActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(8,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    srma_cfg={
      "height_group": "teacher_height_map",
      "friction_group": "teacher_friction",
      "push_group": "teacher_prev_push",
      "velocity_group": "teacher_velocity_gt",
      "height_shape": HEIGHT_SHAPE,
      "latent_dim": 4,
      "external_mode": external_mode,
      "height_channels": (4, 8),
      "height_hidden_dim": 8,
      "vector_hidden_dim": 8,
      "fusion_hidden_dim": 8,
    },
  )


def test_teacher_actor_concatenates_privileged_latent():
  actor = make_actor()
  obs = make_obs()
  latent = actor.get_latent(obs)

  assert latent.shape == (NUM_ENVS, OBS_DIM + 3 + 4)
  assert actor.last_external_output.latent.shape == (NUM_ENVS, 4)
  assert actor.last_external_output.velocity.shape == (NUM_ENVS, 3)
  actions = actor(obs, stochastic_output=True)
  assert actions.shape == (NUM_ENVS, ACTION_DIM)


def test_policy_gradient_reaches_external_encoder():
  actor = make_actor()
  obs = make_obs()
  actions = actor(obs, stochastic_output=True)
  actor.get_output_log_prob(actions).mean().backward()

  external_grads = [
    parameter.grad
    for parameter in actor.external_encoder.parameters()
    if parameter.grad is not None
  ]
  assert external_grads
  assert any(torch.any(gradient != 0.0) for gradient in external_grads)


def test_none_external_mode_produces_zero_latent():
  actor = make_actor(external_mode="none")
  actor.get_latent(make_obs())
  torch.testing.assert_close(
    actor.last_external_output.latent,
    torch.zeros(NUM_ENVS, 4),
  )


def test_teacher_task_is_registered():
  import mjlab.tasks.srma.config.go1  # noqa: F401

  assert "Mjlab-SRMA-Teacher-Rough-Unitree-Go1" in list_tasks()


def test_teacher_metrics_are_finite_and_named():
  actor = make_actor()
  metrics = compute_teacher_metrics(actor, make_obs(), sample_size=4)
  assert set(metrics) == {
    "Metrics/srma_latent_effective_rank",
    "Metrics/srma_height_map_std",
    "Metrics/srma_friction_std",
    "Metrics/srma_push_fraction",
    "Metrics/srma_action_delta_zero",
    "Metrics/srma_action_delta_shuffle",
  }
  assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
