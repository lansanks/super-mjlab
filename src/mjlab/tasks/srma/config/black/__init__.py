"""Black SRMA task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.srma.rl.config import (
  black_srma_student_dagger_runner_cfg,
)
from mjlab.tasks.srma.rl.runner import SRMAStudentRunner

from .env_cfg import BLACK_TEACHER_HEIGHT_SHAPE, black_srma_env_cfg
from .rl_cfg import black_srma_teacher_ppo_runner_cfg
from .student_env_cfg import black_srma_student_env_cfg

register_mjlab_task(
  task_id="Mjlab-SRMA-Teacher-Rough-Black",
  env_cfg=black_srma_env_cfg(),
  play_env_cfg=black_srma_env_cfg(play=True),
  rl_cfg=black_srma_teacher_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-SRMA-Student-Rough-Black",
  env_cfg=black_srma_student_env_cfg(),
  play_env_cfg=black_srma_student_env_cfg(play=True),
  rl_cfg=black_srma_student_dagger_runner_cfg(height_shape=BLACK_TEACHER_HEIGHT_SHAPE),
  runner_cls=SRMAStudentRunner,
)
