"""Wolf SRMA task registration.

Wolf 的 SRMA 流水线与 Black 对齐：先训练 teacher（rough 地形），再用
DAgger 蒸馏出本体感知的学生 estimator。机器人侧差异全部落在
``config/wolf/env_cfg.py`` 里。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.srma.rl.config import (
  wolf_srma_student_dagger_runner_cfg,
)
from mjlab.tasks.srma.rl.runner import SRMAStudentRunner

from .env_cfg import WOLF_TEACHER_HEIGHT_SHAPE, wolf_srma_env_cfg
from .rl_cfg import wolf_srma_teacher_ppo_runner_cfg
from .student_env_cfg import wolf_srma_student_env_cfg

# Wolf 还没有现成的 teacher 检查点；训练学生前用
# ``--agent.teacher-checkpoint <path>`` 指到一个训练好的 Wolf teacher。
WOLF_SRMA_TEACHER_CHECKPOINT = ""

register_mjlab_task(
  task_id="Mjlab-SRMA-Teacher-Rough-Wolf",
  env_cfg=wolf_srma_env_cfg(),
  play_env_cfg=wolf_srma_env_cfg(play=True),
  rl_cfg=wolf_srma_teacher_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-SRMA-Student-Rough-Wolf",
  env_cfg=wolf_srma_student_env_cfg(),
  play_env_cfg=wolf_srma_student_env_cfg(play=True),
  rl_cfg=wolf_srma_student_dagger_runner_cfg(
    teacher_checkpoint=WOLF_SRMA_TEACHER_CHECKPOINT,
    height_shape=WOLF_TEACHER_HEIGHT_SHAPE,
  ),
  runner_cls=SRMAStudentRunner,
)
