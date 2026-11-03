"""RL configuration for the Black SRMAM teacher task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from .env_cfg import BLACK_TEACHER_HEIGHT_SHAPE


def black_srmaM_teacher_ppo_runner_cfg(
  external_mode: str = "full",
  experiment_name: str = "black_srmaM_teacher",
) -> RslRlOnPolicyRunnerCfg:
  """Create the privileged Black teacher PPO configuration."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="mjlab.tasks.srmaM.rl.network.teacher_actor:SRMAMTeacherActorModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      srmaM_cfg={
        "height_group": "teacher_height_map",
        "friction_group": "teacher_friction",
        "push_group": "teacher_prev_push",
        "velocity_group": "teacher_velocity_gt",
        "height_shape": BLACK_TEACHER_HEIGHT_SHAPE,
        "height_latent_dim": 16,
        "privilege_latent_dim": 6,
        "external_mode": external_mode,
        "height_hidden_dim": 32,
        "privilege_hidden_dim": 32,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="mjlab.tasks.srmaM.rl.algorithm.teacher_ppo:SRMAMTeacherPPO",
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
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=3000,
  )
