"""Black SRMAM task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.srmaM.rl.config import (
  black_srmaM_student_dagger_runner_cfg,
)
from mjlab.tasks.srmaM.rl.visual_student_runner import SRMAMVisualStudentRunner

from .env_cfg import BLACK_TEACHER_HEIGHT_SHAPE, black_srmaM_env_cfg
from .expert_env_cfg import (
  black_srmaM_loco_expert_env_cfg,
  black_srmaM_stairs_expert_env_cfg,
)
from .rl_cfg import black_srmaM_teacher_ppo_runner_cfg
from .student_env_cfg import black_srmaM_student_env_cfg

register_mjlab_task(
  task_id="Mjlab-SRMAM-Teacher-Rough-Black",
  env_cfg=black_srmaM_env_cfg(),
  play_env_cfg=black_srmaM_env_cfg(play=True),
  rl_cfg=black_srmaM_teacher_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-SRMAM-Expert-Loco-Black",
  env_cfg=black_srmaM_loco_expert_env_cfg(),
  play_env_cfg=black_srmaM_loco_expert_env_cfg(play=True),
  rl_cfg=black_srmaM_teacher_ppo_runner_cfg(experiment_name="black_srmaM_expert_loco"),
)

register_mjlab_task(
  task_id="Mjlab-SRMAM-Expert-Stairs-Black",
  env_cfg=black_srmaM_stairs_expert_env_cfg(),
  play_env_cfg=black_srmaM_stairs_expert_env_cfg(play=True),
  rl_cfg=black_srmaM_teacher_ppo_runner_cfg(
    experiment_name="black_srmaM_expert_stairs"
  ),
)

register_mjlab_task(
  task_id="Mjlab-SRMAM-Student-Rough-Black",
  env_cfg=black_srmaM_student_env_cfg(),
  play_env_cfg=black_srmaM_student_env_cfg(play=True),
  rl_cfg=black_srmaM_student_dagger_runner_cfg(height_shape=BLACK_TEACHER_HEIGHT_SHAPE),
  runner_cls=SRMAMVisualStudentRunner,
)
