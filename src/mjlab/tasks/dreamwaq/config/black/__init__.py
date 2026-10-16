"""Black DreamWaQ task registration."""

from mjlab.tasks.dreamwaq.rl import DreamwaqOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
  black_dreamwaq_env_cfg,
  black_dreamwaq_flat_env_cfg,
)
from .rl_cfg import black_dreamwaq_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-DreamWaQ-Rough-Black",
  env_cfg=black_dreamwaq_env_cfg(),
  play_env_cfg=black_dreamwaq_env_cfg(play=True),
  rl_cfg=black_dreamwaq_ppo_runner_cfg(),
  runner_cls=DreamwaqOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-DreamWaQ-Flat-Black",
  env_cfg=black_dreamwaq_flat_env_cfg(),
  play_env_cfg=black_dreamwaq_flat_env_cfg(play=True),
  rl_cfg=black_dreamwaq_ppo_runner_cfg(),
  runner_cls=DreamwaqOnPolicyRunner,
)
