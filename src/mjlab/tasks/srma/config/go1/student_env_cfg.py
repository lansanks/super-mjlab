"""Unitree Go1 SRMA student environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.srma.config.go1.env_cfg import unitree_go1_srma_env_cfg


def unitree_go1_srma_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the SRMA student environment.

  Training keeps the teacher observation groups and sensors so the frozen
  teacher can label every student-visited state. Play keeps only the deployable
  actor observation group.
  """
  cfg = unitree_go1_srma_env_cfg(play=play)
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
