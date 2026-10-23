"""Unitree Go1 SRMAV student environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.srmaV.config.go1.env_cfg import unitree_go1_srmaV_env_cfg


def unitree_go1_srmaV_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the SRMAV student environment.

  Training keeps the teacher observation groups and sensors so the frozen
  teacher can label every student-visited state. Play keeps only the deployable
  actor observation group.
  """
  cfg = unitree_go1_srmaV_env_cfg(play=play)
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
  cfg.viewer.latent_bar_attr = "srmaV_latent_estimate"
  cfg.viewer.latent_bar_label = "SRMAV student l_hat"
  cfg.viewer.latent_bar_range = (-3.0, 3.0)
  return cfg
