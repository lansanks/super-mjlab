"""DreamWaQ reproduction (Unitree Go1, rough terrain).

Registers:

* ``Mjlab-DreamWaQ-Rough-Unitree-Go1``
"""

from mjlab.tasks.dreamwaq.env_cfg import (
  unitree_go1_dreamwaq_rough_env_cfg,
)
from mjlab.tasks.dreamwaq.rl_cfg import unitree_go1_dreamwaq_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

register_mjlab_task(
  task_id="Mjlab-DreamWaQ-Rough-Unitree-Go1",
  env_cfg=unitree_go1_dreamwaq_rough_env_cfg(),
  play_env_cfg=unitree_go1_dreamwaq_rough_env_cfg(play=True),
  rl_cfg=unitree_go1_dreamwaq_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
