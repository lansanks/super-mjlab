"""Structural compatibility between the split teacher and existing student."""

import torch
from tensordict import TensorDict

from mjlab.tasks.srmaV.rl.network.student_actor import SRMAVStudentActorModel
from mjlab.tasks.srmaV.rl.network.teacher_actor import SRMAVTeacherActorModel

NUM_ENVS = 2
OBS_DIM = 45
HEIGHT_SHAPE = (17, 11)
ACTION_DIM = 12


def make_obs() -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(NUM_ENVS, OBS_DIM),
      "critic": torch.randn(NUM_ENVS, 16),
      "teacher_height_map": torch.randn(
        NUM_ENVS,
        HEIGHT_SHAPE[0] * HEIGHT_SHAPE[1],
      ),
      "teacher_friction": torch.randn(NUM_ENVS, 3),
      "teacher_prev_push": torch.randn(NUM_ENVS, 6),
      "teacher_velocity_gt": torch.randn(NUM_ENVS, 3),
    },
    batch_size=[NUM_ENVS],
  )


def test_existing_student_can_copy_the_split_teacher_actor() -> None:
  obs = make_obs()
  obs_groups = {"actor": ["actor"], "critic": ["critic"]}
  distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }
  teacher = SRMAVTeacherActorModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(16, 8),
    distribution_cfg=distribution_cfg,
    srmaV_cfg={
      "height_shape": HEIGHT_SHAPE,
      "height_latent_dim": 16,
      "privilege_latent_dim": 6,
      "height_hidden_dim": 8,
      "privilege_hidden_dim": 8,
    },
  )
  student = SRMAVStudentActorModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(16, 8),
    distribution_cfg=distribution_cfg,
    student_cfg={
      "gru_hidden_dim": 8,
      "gru_num_layers": 1,
      "latent_dim": 22,
      "velocity_dim": 3,
    },
  )

  copied = student.copy_actor_from_teacher(teacher.state_dict())
  assert "mlp.0.weight" in copied
  assert student.mlp[0].in_features == 70
  output = student.forward_deterministic(obs)
  assert output.latent.shape == (NUM_ENVS, 22)
