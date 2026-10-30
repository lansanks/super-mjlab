"""Runner configuration for Unitree Go2 cycle tracking."""

from mjlab.tasks.tracking_supermimic.rl_cfg import TrackingSuperMimicRunnerCfg


def unitree_go2_tracking_supermimic_runner_cfg() -> TrackingSuperMimicRunnerCfg:
  return TrackingSuperMimicRunnerCfg(
    experiment_name="go2_cycles_tracking_supermimic",
    max_iterations=320_000,
    save_interval=2000,
    logger="wandb",
    wandb_project="mjlab",
    num_steps_per_env=1,
  )


__all__ = ["unitree_go2_tracking_supermimic_runner_cfg"]
