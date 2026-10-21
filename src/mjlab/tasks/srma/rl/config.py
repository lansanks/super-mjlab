"""RL configuration for SRMA student DAgger training."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg

DEFAULT_TEACHER_CHECKPOINT = (
  "logs/rsl_rl/go1_srma_teacher/2026-10-19_14-32-41/model_3999.pt"
)


@dataclass
class RslRlSRMAStudentAlgorithmCfg:
  """Student estimator-prewarm and DAgger settings."""

  class_name: str = "mjlab.tasks.srma.rl.algorithm.student_dagger:SRMAStudentDagger"
  pretrain_iterations: int = 20
  num_learning_epochs: int = 4
  num_mini_batches: int = 4
  learning_rate: float = 5.0e-4
  velocity_loss_coef: float = 1.0
  latent_loss_coef: float = 1.0
  action_loss_coef: float = 1.0
  max_grad_norm: float = 1.0
  teacher_fraction_start: float = 1.0
  teacher_fraction_min: float = 0.2


@dataclass
class RslRlSRMAStudentRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config for SRMA student training."""

  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="mjlab.tasks.srma.rl.network.student_actor:SRMAStudentActorModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      student_cfg={
        "obs_group": "actor",
        "velocity_target_group": "teacher_velocity_gt",
        "gru_hidden_dim": 64,
        "gru_num_layers": 1,
        "latent_dim": 16,
        "velocity_dim": 3,
      },
    )
  )
  teacher: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="mjlab.tasks.srma.rl.network.teacher_actor:SRMATeacherActorModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
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
        "height_shape": (11, 9),
        "latent_dim": 16,
        "external_mode": "full",
        "height_channels": (16, 32),
        "height_hidden_dim": 64,
        "vector_hidden_dim": 32,
        "fusion_hidden_dim": 64,
      },
    )
  )
  algorithm: RslRlSRMAStudentAlgorithmCfg = field(
    default_factory=RslRlSRMAStudentAlgorithmCfg
  )
  teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT


def unitree_go1_srma_student_dagger_runner_cfg(
  max_iterations: int = 500,
) -> RslRlSRMAStudentRunnerCfg:
  """Create the Unitree Go1 SRMA student training configuration."""
  return _srma_student_dagger_runner_cfg(
    experiment_name="go1_srma_student",
    teacher_checkpoint=DEFAULT_TEACHER_CHECKPOINT,
    max_iterations=max_iterations,
  )


def black_srma_student_dagger_runner_cfg(
  teacher_checkpoint: str = "",
  max_iterations: int = 500,
  height_shape: tuple[int, int] = (11, 9),
) -> RslRlSRMAStudentRunnerCfg:
  """Create the Black SRMA student training configuration.

  The teacher checkpoint must come from a Black SRMA teacher run. It is left
  empty by default so the task cannot silently load Go1 weights.
  """
  return _srma_student_dagger_runner_cfg(
    experiment_name="black_srma_student",
    teacher_checkpoint=teacher_checkpoint,
    max_iterations=max_iterations,
    height_shape=height_shape,
  )


def _srma_student_dagger_runner_cfg(
  experiment_name: str,
  teacher_checkpoint: str,
  max_iterations: int,
  height_shape: tuple[int, int] = (11, 9),
) -> RslRlSRMAStudentRunnerCfg:
  cfg = RslRlSRMAStudentRunnerCfg(
    obs_groups={"actor": ("actor",)},
    experiment_name=experiment_name,
    teacher_checkpoint=teacher_checkpoint,
    save_interval=50,
    num_steps_per_env=50,
    max_iterations=max_iterations,
  )
  if cfg.teacher.srma_cfg is None:
    raise ValueError("SRMA student teacher config must define srma_cfg.")
  cfg.teacher.srma_cfg["height_shape"] = height_shape
  return cfg
