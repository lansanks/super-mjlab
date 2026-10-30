"""Differentiable Unitree Go2 forward kinematics."""

from __future__ import annotations

import torch

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import GO2_XML
from mjlab.tasks.tracking_supermimic.kinematics import MjcfKinematics

from .constants import GO2_JOINT_NAMES, TRACKED_BODIES


class Go2Kinematics(MjcfKinematics):
  """Compute tracked Go2 body poses and velocities from a floating-base state."""

  def __init__(self, device: str | torch.device):
    super().__init__(
      GO2_XML,
      GO2_JOINT_NAMES,
      TRACKED_BODIES,
      device,
    )
