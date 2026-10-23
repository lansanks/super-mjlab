"""Tests for the split SRMAV privileged teacher."""

import torch
from tensordict import TensorDict

from mjlab.tasks.registry import list_tasks
from mjlab.tasks.srmaV.rl.algorithm.teacher_ppo import compute_teacher_metrics
from mjlab.tasks.srmaV.rl.network.teacher_actor import SRMAVTeacherActorModel

NUM_ENVS = 8
OBS_DIM = 45
HEIGHT_SHAPE = (17, 11)
HEIGHT_DIM = HEIGHT_SHAPE[0] * HEIGHT_SHAPE[1]
FRICTION_DIM = 3
PUSH_DIM = 6
ACTION_DIM = 12
HEIGHT_LATENT_DIM = 16
PRIVILEGE_LATENT_DIM = 6

OBS_GROUPS = {
  "actor": ["actor"],
  "critic": ["critic"],
}


def make_obs() -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(NUM_ENVS, OBS_DIM),
      "critic": torch.randn(NUM_ENVS, 32),
      "teacher_height_map": torch.randn(NUM_ENVS, HEIGHT_DIM),
      "teacher_friction": torch.randn(NUM_ENVS, FRICTION_DIM),
      "teacher_prev_push": torch.randn(NUM_ENVS, PUSH_DIM),
      "teacher_velocity_gt": torch.randn(NUM_ENVS, 3),
    },
    batch_size=[NUM_ENVS],
  )


def make_actor(external_mode: str = "full") -> SRMAVTeacherActorModel:
  torch.manual_seed(0)
  return SRMAVTeacherActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(32, 16),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    srmaV_cfg={
      "height_group": "teacher_height_map",
      "friction_group": "teacher_friction",
      "push_group": "teacher_prev_push",
      "velocity_group": "teacher_velocity_gt",
      "height_shape": HEIGHT_SHAPE,
      "height_latent_dim": HEIGHT_LATENT_DIM,
      "privilege_latent_dim": PRIVILEGE_LATENT_DIM,
      "external_mode": external_mode,
      "height_hidden_dim": 16,
      "privilege_hidden_dim": 16,
    },
  )


def test_teacher_uses_split_latents_and_seventy_dimensional_policy_input():
  actor = make_actor()
  latent = actor.get_latent(make_obs())
  output = actor.last_external_output

  assert latent.shape == (NUM_ENVS, OBS_DIM + 3 + 16 + 6)
  assert output.height_latent.shape == (NUM_ENVS, 16)
  assert output.privilege_latent.shape == (NUM_ENVS, 6)
  assert output.latent.shape == (NUM_ENVS, 22)
  assert actor.mlp[0].in_features == 70

  actions = actor.action_mean_with_latent(make_obs(), output.latent)
  assert actions.shape == (NUM_ENVS, ACTION_DIM)


def test_gradients_reach_both_privileged_encoders():
  actor = make_actor()
  actor.get_latent(make_obs())
  output = actor.last_external_output
  loss = output.height_latent.square().mean() + output.privilege_latent.square().mean()
  loss.backward()

  for encoder in (actor.height_encoder, actor.privilege_encoder):
    gradients = [
      parameter.grad for parameter in encoder.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert any(torch.any(gradient != 0.0) for gradient in gradients)


def test_external_modes_zero_only_the_disabled_branch():
  modes = {
    "none": (True, True),
    "height": (False, True),
    "priv": (True, False),
    "full": (False, False),
  }
  for mode, (height_is_zero, privilege_is_zero) in modes.items():
    actor = make_actor(mode)
    actor.get_latent(make_obs())
    output = actor.last_external_output
    assert torch.all(output.height_latent == 0) == height_is_zero
    assert torch.all(output.privilege_latent == 0) == privilege_is_zero


def test_teacher_metrics_cover_both_latent_branches():
  metrics = compute_teacher_metrics(make_actor(), make_obs(), sample_size=NUM_ENVS)
  assert set(metrics) == {
    "Metrics/srmaV_latent_effective_rank",
    "Metrics/srmaV_height_latent_effective_rank",
    "Metrics/srmaV_privilege_latent_effective_rank",
    "Metrics/srmaV_height_map_std",
    "Metrics/srmaV_friction_std",
    "Metrics/srmaV_push_fraction",
    "Metrics/srmaV_action_delta_height_zero",
    "Metrics/srmaV_action_delta_height_shuffle",
    "Metrics/srmaV_action_delta_privilege_zero",
    "Metrics/srmaV_action_delta_privilege_shuffle",
  }
  assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_black_srmaV_teacher_task_is_registered():
  import mjlab.tasks.srmaV.config.black  # noqa: F401

  assert "Mjlab-SRMAV-Teacher-Rough-Black" in list_tasks()
