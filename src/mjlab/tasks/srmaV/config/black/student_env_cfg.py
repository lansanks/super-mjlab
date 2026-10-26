"""Black SRMAV student environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.srmaV import mdp
from mjlab.tasks.srmaV.config.black.env_cfg import (
  black_depth_camera_cfg,
  black_srmaV_env_cfg,
)


def black_srmaV_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Black SRMAV student environment.

  Training keeps the teacher observation groups and sensors so the frozen
  teacher can label every student-visited state. Play keeps only the deployable
  actor observation group.
  """
  cfg = black_srmaV_env_cfg(play=play)
  cfg.camera_update_period = 2
  if not play:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
      black_depth_camera_cfg(data_types=("depth",)),
    )

  cfg.observations["student_depth"] = ObservationGroupCfg(
    terms={
      "depth": ObservationTermCfg(
        func=mdp.student_depth_image,
        params={
          "sensor_name": "black_depth_camera",
          "min_depth": 0.3,
          "max_depth": 4.0,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

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
