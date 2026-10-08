"""RL configuration for the DreamWaQ Go1 task."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

ACTOR_CLASS = "mjlab.tasks.dreamwaq.cenet:CENetActorModel"
ALGORITHM_CLASS = "mjlab.tasks.dreamwaq.dreamwaq_ppo:DreamWaQPPO"


@dataclass
class CENetModelCfg(RslRlModelCfg):
  """Model config with the extra CENet arguments."""

  cenet_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "obs_dim": 45,
      "history_length": 5,
      "latent_dim": 16,
      "encoder_hidden_dims": (128, 64),
      "decoder_hidden_dims": (64, 128),
    }
  )
  history_group: str = "actor"
  priv_group: str = "priv"


@dataclass
class DreamWaQPpoCfg(RslRlPpoAlgorithmCfg):
  """PPO config with the DreamWaQ estimator loss and AdaBoot."""

  beta: float = 1.0
  """Weight of the beta-VAE KL term."""
  adaboot_cfg: dict[str, Any] = field(
    default_factory=lambda: {"enabled": True, "smoothing": 0.0}
  )


def unitree_go1_dreamwaq_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for the DreamWaQ Go1 task."""
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    num_steps_per_env=24,
    max_iterations=1000,  # Paper: 1000 iterations with 4096 envs.
    save_interval=50,
    experiment_name="go1_dreamwaq",
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "priv": ("priv",),
    },
    actor=CENetModelCfg(
      class_name=ACTOR_CLASS,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
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
    algorithm=DreamWaQPpoCfg(
      class_name=ALGORITHM_CLASS,
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
      beta=1.0,
    ),
  )
