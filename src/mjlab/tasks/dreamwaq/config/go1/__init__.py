"""Unitree Go1 DreamWaQ task registration."""

from mjlab.tasks.dreamwaq.rl import DreamwaqOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
  unitree_go1_dreamwaq_env_cfg,
  unitree_go1_dreamwaq_flat_env_cfg,
)
from .rl_cfg import unitree_go1_dreamwaq_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-DreamWaQ-Rough-Unitree-Go1",
  env_cfg=unitree_go1_dreamwaq_env_cfg(),
  play_env_cfg=unitree_go1_dreamwaq_env_cfg(play=True),
  rl_cfg=unitree_go1_dreamwaq_ppo_runner_cfg(),
  runner_cls=DreamwaqOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-DreamWaQ-Flat-Unitree-Go1",
  env_cfg=unitree_go1_dreamwaq_flat_env_cfg(),
  play_env_cfg=unitree_go1_dreamwaq_flat_env_cfg(play=True),
  rl_cfg=unitree_go1_dreamwaq_ppo_runner_cfg(),
  runner_cls=DreamwaqOnPolicyRunner,
)
