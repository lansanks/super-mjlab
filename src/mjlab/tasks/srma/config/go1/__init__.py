"""Unitree Go1 SRMA task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import unitree_go1_srma_env_cfg
from .rl_cfg import unitree_go1_srma_teacher_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-SRMA-Teacher-Rough-Unitree-Go1",
  env_cfg=unitree_go1_srma_env_cfg(),
  play_env_cfg=unitree_go1_srma_env_cfg(play=True),
  rl_cfg=unitree_go1_srma_teacher_ppo_runner_cfg(),
)
