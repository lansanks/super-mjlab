"""Unitree Go1 SRMA task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.srma.rl.config import (
  unitree_go1_srma_student_dagger_runner_cfg,
)
from mjlab.tasks.srma.rl.runner import SRMAStudentRunner

from .env_cfg import unitree_go1_srma_env_cfg
from .rl_cfg import unitree_go1_srma_teacher_ppo_runner_cfg
from .student_env_cfg import unitree_go1_srma_student_env_cfg

register_mjlab_task(
  task_id="Mjlab-SRMA-Teacher-Rough-Unitree-Go1",
  env_cfg=unitree_go1_srma_env_cfg(),
  play_env_cfg=unitree_go1_srma_env_cfg(play=True),
  rl_cfg=unitree_go1_srma_teacher_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-SRMA-Student-Rough-Unitree-Go1",
  env_cfg=unitree_go1_srma_student_env_cfg(),
  play_env_cfg=unitree_go1_srma_student_env_cfg(play=True),
  rl_cfg=unitree_go1_srma_student_dagger_runner_cfg(),
  runner_cls=SRMAStudentRunner,
)
