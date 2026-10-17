"""Unitree Go1 HIM task registration."""

from mjlab.tasks.himloco.config.go1.env_cfg import unitree_go1_him_env_cfg
from mjlab.tasks.himloco.config.go1.rl_cfg import unitree_go1_him_ppo_runner_cfg
from mjlab.tasks.himloco.rl import HimOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-HIM-Rough-Unitree-Go1",
  env_cfg=unitree_go1_him_env_cfg(),
  play_env_cfg=unitree_go1_him_env_cfg(play=True),
  rl_cfg=unitree_go1_him_ppo_runner_cfg(),
  runner_cls=HimOnPolicyRunner,
)
