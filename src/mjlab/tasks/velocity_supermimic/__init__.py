"""Go1 velocity tracking with the SuperMimic model-based training loop."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_go1_supermimic_env_cfg,
  unitree_go1_supermimic_flat_env_cfg,
  unitree_go1_supermimic_nocrouch_env_cfg,
)
from .rl_cfg import (
  unitree_go1_supermimic_nocrouch_runner_cfg,
  unitree_go1_supermimic_runner_cfg,
)
from .supermimic.runner import SuperMimicRunner

register_mjlab_task(
  task_id="Mjlab-Velocity-SuperMimic-Unitree-Go1",
  env_cfg=unitree_go1_supermimic_env_cfg(),
  play_env_cfg=unitree_go1_supermimic_env_cfg(play=True),
  rl_cfg=unitree_go1_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-SuperMimic-Flat-Unitree-Go1",
  env_cfg=unitree_go1_supermimic_flat_env_cfg(),
  play_env_cfg=unitree_go1_supermimic_flat_env_cfg(play=True),
  rl_cfg=unitree_go1_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-SuperMimic-Unitree-Go1-NoCrouch",
  env_cfg=unitree_go1_supermimic_nocrouch_env_cfg(),
  play_env_cfg=unitree_go1_supermimic_nocrouch_env_cfg(play=True),
  rl_cfg=unitree_go1_supermimic_nocrouch_runner_cfg(),
  runner_cls=SuperMimicRunner,
)
