"""RL configuration for the Black SRMA teacher task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from .env_cfg import BLACK_TEACHER_HEIGHT_SHAPE


def black_srma_teacher_ppo_runner_cfg(
  external_mode: str = "full",
) -> RslRlOnPolicyRunnerCfg:
  """Create the privileged Black teacher PPO configuration."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
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
        "height_shape": BLACK_TEACHER_HEIGHT_SHAPE,
        "latent_dim": 16,
        "external_mode": external_mode,
        "height_channels": (16, 32),
        "height_hidden_dim": 64,
        "vector_hidden_dim": 32,
        "fusion_hidden_dim": 64,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="mjlab.tasks.srma.rl.algorithm.teacher_ppo:SRMATeacherPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    experiment_name="black_srma_teacher",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=3000,
  )
