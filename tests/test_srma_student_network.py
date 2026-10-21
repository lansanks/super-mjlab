"""Tests for the deployable SRMA student actor."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab.tasks.srma.rl.network.student_actor import SRMAStudentActorModel
from mjlab.tasks.srma.rl.network.teacher_actor import SRMATeacherActorModel

NUM_ENVS = 4
OBS_DIM = 6
ACTION_DIM = 2
LATENT_DIM = 4
HIDDEN_DIM = 8
HEIGHT_SHAPE = (3, 3)

OBS_GROUPS = {
  "actor": ["actor"],
  "critic": ["critic"],
}


def make_obs() -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(NUM_ENVS, OBS_DIM),
      "critic": torch.randn(NUM_ENVS, 5),
      "teacher_height_map": torch.randn(NUM_ENVS, HEIGHT_SHAPE[0] * HEIGHT_SHAPE[1]),
      "teacher_friction": torch.randn(NUM_ENVS, 2),
      "teacher_prev_push": torch.randn(NUM_ENVS, 2),
      "teacher_velocity_gt": torch.randn(NUM_ENVS, 3),
    },
    batch_size=[NUM_ENVS],
  )


def make_student() -> SRMAStudentActorModel:
  torch.manual_seed(0)
  return SRMAStudentActorModel(
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


def make_teacher() -> SRMATeacherActorModel:
  torch.manual_seed(1)
  return SRMATeacherActorModel(
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
      "height_shape": HEIGHT_SHAPE,
      "latent_dim": LATENT_DIM,
      "height_channels": (4, 8),
      "height_hidden_dim": HIDDEN_DIM,
      "vector_hidden_dim": HIDDEN_DIM,
      "fusion_hidden_dim": HIDDEN_DIM,
    },
  )


def test_student_outputs_and_hidden_shapes():
  student = make_student()
  output = student.forward_deterministic(make_obs())

  assert output.velocity.shape == (NUM_ENVS, 3)
  assert output.latent.shape == (NUM_ENVS, LATENT_DIM)
  assert output.actor_mean.shape == (NUM_ENVS, ACTION_DIM)
  assert student.get_hidden_state().shape == (1, NUM_ENVS, HIDDEN_DIM)
  assert output.latent.norm(dim=-1).shape == (NUM_ENVS,)


def test_bound_env_receives_latent_and_velocity_estimates():
  student = make_student()
  env = SimpleNamespace(num_envs=NUM_ENVS, device="cpu")
  student.bind_debug_env(env)
  assert env.srma_latent_estimate.shape == (NUM_ENVS, LATENT_DIM)
  assert env.srma_velocity_estimate.shape == (NUM_ENVS, 3)

  output = student.forward_deterministic(make_obs())
  torch.testing.assert_close(env.srma_latent_estimate, output.latent)
  torch.testing.assert_close(env.srma_velocity_estimate, output.velocity)


def test_hidden_resets_only_for_done_environments():
  student = make_student()
  student.update_estimator(make_obs())
  before = student.get_hidden_state().clone()
  dones = torch.tensor([1, 0, 0, 0], dtype=torch.long)

  student.reset(dones)
  after = student.get_hidden_state()
  torch.testing.assert_close(after[:, 0], torch.zeros_like(after[:, 0]))
  torch.testing.assert_close(after[:, 1:], before[:, 1:])


def test_sequence_loss_gradient_routing():
  student = make_student()
  time_steps = 5
  obs = make_obs()
  observations = obs["actor"].unsqueeze(0).repeat(time_steps, 1, 1)
  initial_hidden = torch.zeros(1, NUM_ENVS, HIDDEN_DIM)
  dones = torch.zeros(time_steps, NUM_ENVS, 1)
  output = student.forward_sequence(observations, initial_hidden, dones)

  output.velocity.square().mean().backward()
  estimator_weight = student.estimator_gru.weight_ih_l0
  assert isinstance(estimator_weight, torch.Tensor)
  estimator_gradient = estimator_weight.grad
  assert estimator_gradient is not None
  assert estimator_gradient.abs().sum().item() > 0.0

  student.zero_grad(set_to_none=True)
  output.actor_mean.square().mean().backward()
  assert student.estimator_gru.weight_ih_l0.grad is None
  assert student.mlp[0].weight.grad is not None


def test_sequence_done_resets_hidden_before_next_step():
  student = make_student()
  time_steps = 3
  observations = make_obs()["actor"].unsqueeze(0).repeat(time_steps, 1, 1)
  initial_hidden = torch.zeros(1, NUM_ENVS, HIDDEN_DIM)
  dones = torch.zeros(time_steps, NUM_ENVS, 1)
  dones[0, 0, 0] = 1

  without_reset = student.forward_sequence(
    observations,
    initial_hidden,
    torch.zeros_like(dones),
  )
  with_reset = student.forward_sequence(
    observations,
    initial_hidden,
    dones,
  )
  torch.testing.assert_close(
    without_reset.velocity[0],
    with_reset.velocity[0],
  )
  assert not torch.allclose(
    without_reset.velocity[1:, 0],
    with_reset.velocity[1:, 0],
  )


def test_teacher_actor_copy_excludes_external_encoder():
  student = make_student()
  teacher = make_teacher()
  copied = student.copy_actor_from_teacher(teacher.state_dict())

  assert "mlp.0.weight" in copied
  assert "external_encoder.fusion.2.weight" not in student.state_dict()
  torch.testing.assert_close(
    student.mlp[0].weight,
    teacher.mlp[0].weight,
  )
  torch.testing.assert_close(
    student.obs_normalizer._mean,
    teacher.obs_normalizer._mean,
  )


def test_onnx_wrapper_shapes():
  wrapper = make_student().as_onnx(verbose=False)
  obs, hidden = wrapper.get_dummy_inputs()  # type: ignore[attr-defined]
  actions, hidden_out = wrapper(obs, hidden)

  assert actions.shape == (1, ACTION_DIM)
  assert hidden_out.shape == (1, 1, HIDDEN_DIM)
  assert wrapper.input_names == ["obs", "h_in"]
  assert wrapper.output_names == ["actions", "h_out"]
