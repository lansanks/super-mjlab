"""RL configuration for the Unitree Go1 HIM task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go1_him_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for the Unitree Go1 HIM task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="mjlab.tasks.himloco.rl.network.actor:HimActorModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      # The released HIM implementation does not normalize observations.
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      him_cfg={
        # Paper Fig. 2: o_t^H = [o_t, ..., o_{t-H}] with H = 5, l in R^16.
        "obs_group": "him_input",
        "state_group": "him_state",
        "latent_dim": 16,
        "encoder_hidden_dims": (128, 64),
        "target_hidden_dims": (128, 64),
        "num_prototypes": 32,
        "temperature": 3.0,
        "history_order": "oldest_first",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="mjlab.tasks.himloco.rl.algorithm.him_ppo:HimPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      # The released implementation uses its adaptive schedule and passes the
      # current learning rate to the estimator's optimizer each mini-batch.
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      him_cfg={
        # Released code: loss = L_est + L_swap, estimator gradient clip 10,
        # Sinkhorn with eps = 0.05 and 3 normalization iterations.
        "est_coef": 1.0,
        "swap_coef": 1.0,
        "max_grad_norm": 10.0,
        "sinkhorn_eps": 0.05,
        "sinkhorn_iters": 3,
      },
    ),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "him": ("him_input",),
    },
    experiment_name="go1_himloco",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=600,
  )
