"""Unitree Go2 cycle tracking with the inherited SuperMimic loop."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking_supermimic_go2.env_cfgs import (
  unitree_go2_tracking_supermimic_env_cfg,
)
from mjlab.tasks.tracking_supermimic_go2.rl_cfg import (
  unitree_go2_tracking_supermimic_runner_cfg,
)
from mjlab.tasks.tracking_supermimic_go2.runner import (
  Go2TrackingSuperMimicRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-SuperMimic-Unitree-Go2",
  env_cfg=unitree_go2_tracking_supermimic_env_cfg(),
  play_env_cfg=unitree_go2_tracking_supermimic_env_cfg(play=True),
  rl_cfg=unitree_go2_tracking_supermimic_runner_cfg(),
  runner_cls=Go2TrackingSuperMimicRunner,
)
