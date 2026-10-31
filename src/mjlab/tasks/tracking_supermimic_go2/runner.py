"""SuperMimic runner specialized for the 12-DoF Unitree Go2."""

from __future__ import annotations

from mjlab.tasks.tracking_supermimic.mdp import RewardCfg
from mjlab.tasks.tracking_supermimic.runner import (
  RobotMotionSpec,
  TrackingSuperMimicRunner,
)

from .constants import ANCHOR_BODY, GO2_JOINT_NAMES, TRACKED_BODIES
from .kinematics import Go2Kinematics

GO2_ROBOT_SPEC = RobotMotionSpec(
  task_id="Mjlab-Tracking-SuperMimic-Unitree-Go2",
  joint_names=GO2_JOINT_NAMES,
  tracked_bodies=TRACKED_BODIES,
  anchor_body=ANCHOR_BODY,
  kinematics_cls=Go2Kinematics,
  foot_geom_regex=r"^[FR][LR]_foot_collision$",
  sim_context_body="base",
  reward_cfg=RewardCfg(term_height=0.15),
)


class Go2TrackingSuperMimicRunner(TrackingSuperMimicRunner):
  """Inherit the SuperMimic loop with Go2-specific robot metadata."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, robot_spec=GO2_ROBOT_SPEC, **kwargs)


__all__ = ["Go2TrackingSuperMimicRunner", "GO2_ROBOT_SPEC"]
