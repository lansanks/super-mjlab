"""Wolf SRMA student environment configurations.

SRMA 的学生是本体感知（proprioceptive）estimator，没有深度相机；因此这里
只复用 Wolf 的 teacher 环境，play 时裁掉 teacher / critic 观测组与 teacher
专用的地形扫描。训练时保留全部 teacher 观测组，方便冻结 teacher 给每个
student 状态打标签。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.srma.config.wolf.env_cfg import wolf_srma_env_cfg


def wolf_srma_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Wolf SRMA student environment."""
  cfg = wolf_srma_env_cfg(play=play)
  if not play:
    return cfg

  for group in (
    "critic",
    "teacher_height_map",
    "teacher_friction",
    "teacher_prev_push",
    "teacher_velocity_gt",
  ):
    cfg.observations.pop(group, None)
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "srma_teacher_scan"
  )
  cfg.viewer.latent_bar_attr = "srma_latent_estimate"
  cfg.viewer.latent_bar_label = "SRMA student l_hat"
  cfg.viewer.latent_bar_range = (-3.0, 3.0)
  return cfg
