"""RL configuration for SRMAV student DAgger training."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg

BLACK_SRMAV_TEACHER_CHECKPOINT = (
  "logs/rsl_rl/black_srmaV_teacher/2026-10-23_19-53-47/model_5498.pt"
)


@dataclass
class RslRlSRMAVVisualStudentAlgorithmCfg:
  """Visual student DAgger settings with split latent supervision."""

  class_name: str = (
    "mjlab.tasks.srmaV.rl.algorithm.visual_student_dagger:SRMAVVisualStudentDagger"
  )
  pretrain_iterations: int = 20
  num_learning_epochs: int = 4
  # Depth images make the recurrent update substantially heavier than a
  # proprioceptive-only estimator. Use smaller environment batches by default.
  num_mini_batches: int = 8
  learning_rate: float = 5.0e-4
  velocity_loss_coef: float = 1.0
  latent_loss_coef: float = 1.0
  action_loss_coef: float = 1.0
  max_grad_norm: float = 1.0
  teacher_fraction_start: float = 1.0
  teacher_fraction_min: float = 0.2


@dataclass
class RslRlSRMAVVisualStudentRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config for the depth-conditioned SRMAV student."""

  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name=(
        "mjlab.tasks.srmaV.rl.network.visual_student_actor:SRMAVVisualStudentActorModel"
      ),
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
        "depth_group": "student_depth",
        "velocity_target_group": "teacher_velocity_gt",
        "gru_hidden_dim": 64,
        "gru_num_layers": 1,
        "height_latent_dim": 16,
        "privilege_latent_dim": 6,
        "velocity_dim": 3,
        "depth_feature_dim": 64,
        "depth_channels": (8, 16, 32),
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
  algorithm: RslRlSRMAVVisualStudentAlgorithmCfg = field(
    default_factory=RslRlSRMAVVisualStudentAlgorithmCfg
  )
  teacher_checkpoint: str = BLACK_SRMAV_TEACHER_CHECKPOINT


def black_srmaV_student_dagger_runner_cfg(
  teacher_checkpoint: str = BLACK_SRMAV_TEACHER_CHECKPOINT,
  max_iterations: int = 500,
  height_shape: tuple[int, int] = (17, 11),
) -> RslRlSRMAVVisualStudentRunnerCfg:
  """Create the Black SRMAV student training configuration.

  The default teacher checkpoint comes from the final Black SRMAV teacher run.
  The student consumes depth at 25 FPS and keeps the deployable actor interface.
  """
  return _srmaV_visual_student_dagger_runner_cfg(
    experiment_name="black_srmaV_student",
    teacher_checkpoint=teacher_checkpoint,
    max_iterations=max_iterations,
    height_shape=height_shape,
  )


def _srmaV_visual_student_dagger_runner_cfg(
  experiment_name: str,
  teacher_checkpoint: str,
  max_iterations: int,
  height_shape: tuple[int, int] = (17, 11),
) -> RslRlSRMAVVisualStudentRunnerCfg:
  cfg = RslRlSRMAVVisualStudentRunnerCfg(
    obs_groups={"actor": ("actor",)},
    experiment_name=experiment_name,
    teacher_checkpoint=teacher_checkpoint,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
  if cfg.teacher.srmaV_cfg is None:
    raise ValueError("SRMAV visual student teacher config must define srmaV_cfg.")
  cfg.teacher.srmaV_cfg["height_shape"] = height_shape
  return cfg
