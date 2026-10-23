"""Unitree Go1 SRMAV task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.srmaV.rl.config import (
  unitree_go1_srmaV_student_dagger_runner_cfg,
)
from mjlab.tasks.srmaV.rl.runner import SRMAVStudentRunner

from .env_cfg import unitree_go1_srmaV_env_cfg
from .rl_cfg import unitree_go1_srmaV_teacher_ppo_runner_cfg
from .student_env_cfg import unitree_go1_srmaV_student_env_cfg

register_mjlab_task(
  task_id="Mjlab-SRMAV-Teacher-Rough-Unitree-Go1",
  env_cfg=unitree_go1_srmaV_env_cfg(),
  play_env_cfg=unitree_go1_srmaV_env_cfg(play=True),
  rl_cfg=unitree_go1_srmaV_teacher_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-SRMAV-Student-Rough-Unitree-Go1",
  env_cfg=unitree_go1_srmaV_student_env_cfg(),
  play_env_cfg=unitree_go1_srmaV_student_env_cfg(play=True),
  rl_cfg=unitree_go1_srmaV_student_dagger_runner_cfg(),
  runner_cls=SRMAVStudentRunner,
)
