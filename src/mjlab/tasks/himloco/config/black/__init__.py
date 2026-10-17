"""Black HIM task registration."""

from mjlab.tasks.himloco.config.black.env_cfg import (
  black_him_env_cfg,
  black_him_flat_env_cfg,
)
from mjlab.tasks.himloco.config.black.rl_cfg import black_him_ppo_runner_cfg
from mjlab.tasks.himloco.rl import HimOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-HIM-Rough-Black",
  env_cfg=black_him_env_cfg(),
  play_env_cfg=black_him_env_cfg(play=True),
  rl_cfg=black_him_ppo_runner_cfg(),
  runner_cls=HimOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HIM-Flat-Black",
  env_cfg=black_him_flat_env_cfg(),
  play_env_cfg=black_him_flat_env_cfg(play=True),
  rl_cfg=black_him_ppo_runner_cfg(),
  runner_cls=HimOnPolicyRunner,
)
