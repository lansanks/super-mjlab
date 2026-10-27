"""Runner configuration for G1 LAFAN1 SuperMimic tracking."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class TrackingSuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  """Hyper-parameters for :class:`TrackingSuperMimicRunner`.

  Defaults follow the reference SuperMimic reproduction. They require a
  high-memory GPU; the README documents explicit overrides for 8 GB cards.
  """

  horizon: int = 32
  goal_frames: int = 10
  obs_history: int = 10
  wm_window: int = 8
  wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
  policy_encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  policy_decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden: tuple[int, ...] = (256, 128)
  latent_dim: int = 32

  wm_batch: int = 2048
  wm_batches: int = 10
  policy_batch: int = 4096
  buffer_steps: int = 1000
  utd: float = 0.4

  learning_rate: float = 2e-4
  lr_decay_iters: int = 250_000
  gamma: float = 0.99
  lam: float = 0.95
  action_noise: float = 0.1
  critic_epochs: int = 5
  critic_minibatches: int = 4
  warmup_steps: int = 64
  wm_only_iters: int = 500
  log_interval: int = 50
  ckpt_interval: int = 2000
  amp: str = "bf16"
  no_critic: bool = False


def unitree_g1_tracking_supermimic_runner_cfg() -> TrackingSuperMimicRunnerCfg:
  return TrackingSuperMimicRunnerCfg(
    experiment_name="g1_lafan1_tracking_supermimic",
    max_iterations=320_000,
    save_interval=2000,
    logger="wandb",
    wandb_project="mjlab",
    num_steps_per_env=1,
  )
