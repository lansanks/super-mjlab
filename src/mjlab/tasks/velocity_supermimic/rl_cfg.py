"""Runner configuration for the Go1 velocity SuperMimic tasks."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  """Hyper-parameters for :class:`SuperMimicRunner`.

  The defaults are sized for a single consumer GPU.  The paper's Table A.5 uses
  a 2048-wide world model, 4096 imagined rollouts and a 4096-environment
  buffer; those values need an 80 GB GPU and can be raised here when available.
  """

  class_name: str = "SuperMimicRunner"

  # Imagination / world model.
  horizon: int = 32
  wm_window: int = 8
  obs_history: int = 10
  wm_hidden: tuple[int, ...] = (512, 512, 512)
  pi_hidden: tuple[int, ...] = (512, 512, 512)
  critic_hidden: tuple[int, ...] = (256, 128)

  # Batches.
  wm_batch: int = 256
  wm_batches: int = 4
  pi_batch: int = 256
  buffer_steps: int = 500
  utd: float = 0.4

  # Optimization.
  lr: float = 2e-4
  lr_decay_iters: int = 200_000
  gamma: float = 0.99
  lam: float = 0.95
  critic_epochs: int = 3
  critic_minibatches: int = 4
  action_noise: float = 0.1
  obs_noise: float = 0.05
  action_clip: float = 5.0

  # Schedule.
  warmup_steps: int = 32
  wm_only_iters: int = 100
  log_interval: int = 10
  ckpt_interval: int = 500
  imagine_pushes: bool = True
  wm_err_threshold: float = 1.5
  bc_coef: float = 0.1
  height_term_min: float = 0.0
  """Analytic low-base termination height used during imagination (0 disables)."""
  term_pos_weight: float = 1.0
  """Positive-class weight for the learned termination head BCE."""


def unitree_go1_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
  return SuperMimicRunnerCfg(
    experiment_name="go1_velocity_supermimic",
    max_iterations=5000,
    save_interval=500,
    logger="wandb",
    wandb_project="mjlab",
    num_steps_per_env=24,
  )


def unitree_go1_supermimic_nocrouch_runner_cfg() -> SuperMimicRunnerCfg:
  """Smaller action bound and more exploration for the NoCrouch variant."""
  return SuperMimicRunnerCfg(
    experiment_name="go1_velocity_supermimic_nocrouch",
    max_iterations=5000,
    save_interval=500,
    logger="wandb",
    wandb_project="mjlab",
    num_steps_per_env=24,
    action_clip=2.0,
    action_noise=0.3,
    height_term_min=0.15,
    term_pos_weight=10.0,
  )
