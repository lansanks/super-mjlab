"""RL configuration for the Unitree Go1 SRMAV teacher task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go1_srmaV_teacher_ppo_runner_cfg(
  external_mode: str = "full",
) -> RslRlOnPolicyRunnerCfg:
  """Create the privileged teacher PPO configuration."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
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
      class_name="mjlab.tasks.srmaV.rl.algorithm.teacher_ppo:SRMAVTeacherPPO",
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
    experiment_name="go1_srmaV_teacher",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=3000,
  )
