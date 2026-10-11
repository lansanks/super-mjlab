"""RL configuration for the Unitree Go1 DreamWaQ task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go1_dreamwaq_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for the Unitree Go1 DreamWaQ task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="mjlab.tasks.dreamwaq.rl.network.actor:DreamwaqActorModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="mjlab.tasks.dreamwaq.rl.algorithm.dreamwaq_ppo:DreamwaqPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      # The paper uses a fixed Adam learning rate for all networks; the
      # adaptive schedule would also rescale CENet's learning rate.
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "cenet": ("cenet_input",),
    },
    experiment_name="go1_dreamwaq",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
