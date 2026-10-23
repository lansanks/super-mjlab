"""RL configuration for SRMAV student DAgger training."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg

DEFAULT_TEACHER_CHECKPOINT = ""
"""No checkpoint predates the split 16-D height and 6-D privilege teacher."""


@dataclass
class RslRlSRMAVStudentAlgorithmCfg:
  """Student estimator-prewarm and DAgger settings."""

  class_name: str = "mjlab.tasks.srmaV.rl.algorithm.student_dagger:SRMAVStudentDagger"
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
class RslRlSRMAVStudentRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config for SRMAV student training."""

  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="mjlab.tasks.srmaV.rl.network.student_actor:SRMAVStudentActorModel",
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
        "latent_dim": 22,
        "velocity_dim": 3,
      },
    )
  )
  teacher: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="mjlab.tasks.srmaV.rl.network.teacher_actor:SRMAVTeacherActorModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
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
        "height_shape": (17, 11),
        "height_latent_dim": 16,
        "privilege_latent_dim": 6,
        "external_mode": "full",
        "height_hidden_dim": 32,
        "privilege_hidden_dim": 32,
      },
    )
  )
  algorithm: RslRlSRMAVStudentAlgorithmCfg = field(
    default_factory=RslRlSRMAVStudentAlgorithmCfg
  )
  teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT


def unitree_go1_srmaV_student_dagger_runner_cfg(
  max_iterations: int = 500,
) -> RslRlSRMAVStudentRunnerCfg:
  """Create the Unitree Go1 SRMAV student training configuration."""
  return _srmaV_student_dagger_runner_cfg(
    experiment_name="go1_srmaV_student",
    teacher_checkpoint=DEFAULT_TEACHER_CHECKPOINT,
    max_iterations=max_iterations,
  )


def black_srmaV_student_dagger_runner_cfg(
  teacher_checkpoint: str = "",
  max_iterations: int = 500,
  height_shape: tuple[int, int] = (17, 11),
) -> RslRlSRMAVStudentRunnerCfg:
  """Create the Black SRMAV student training configuration.

  The teacher checkpoint must come from a Black SRMAV teacher run. It is left
  empty by default so the task cannot silently load Go1 weights.
  """
  return _srmaV_student_dagger_runner_cfg(
    experiment_name="black_srmaV_student",
    teacher_checkpoint=teacher_checkpoint,
    max_iterations=max_iterations,
    height_shape=height_shape,
  )


def _srmaV_student_dagger_runner_cfg(
  experiment_name: str,
  teacher_checkpoint: str,
  max_iterations: int,
  height_shape: tuple[int, int] = (17, 11),
) -> RslRlSRMAVStudentRunnerCfg:
  cfg = RslRlSRMAVStudentRunnerCfg(
    obs_groups={"actor": ("actor",)},
    experiment_name=experiment_name,
    teacher_checkpoint=teacher_checkpoint,
    save_interval=50,
    num_steps_per_env=50,
    max_iterations=max_iterations,
  )
  if cfg.teacher.srmaV_cfg is None:
    raise ValueError("SRMAV student teacher config must define srmaV_cfg.")
  cfg.teacher.srmaV_cfg["height_shape"] = height_shape
  return cfg
