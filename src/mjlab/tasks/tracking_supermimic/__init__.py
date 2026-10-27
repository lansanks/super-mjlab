"""G1 LAFAN1 tracking with the SuperMimic model-based training loop."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_tracking_supermimic_env_cfg
from .rl_cfg import unitree_g1_tracking_supermimic_runner_cfg
from .runner import TrackingSuperMimicRunner

register_mjlab_task(
  task_id="Mjlab-Tracking-SuperMimic-Unitree-G1",
  env_cfg=unitree_g1_tracking_supermimic_env_cfg(),
  play_env_cfg=unitree_g1_tracking_supermimic_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_supermimic_runner_cfg(),
  runner_cls=TrackingSuperMimicRunner,
)
