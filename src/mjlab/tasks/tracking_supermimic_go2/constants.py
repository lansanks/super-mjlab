"""Shared constants for Unitree Go2 cycle tracking."""

from __future__ import annotations

from pathlib import Path

CONTROL_DT = 0.02
NUM_JOINTS = 12

GO2_JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)

TRACKED_BODIES = (
  "base",
  "FL_hip",
  "FL_thigh",
  "FL_calf",
  "FR_hip",
  "FR_thigh",
  "FR_calf",
  "RL_hip",
  "RL_thigh",
  "RL_calf",
  "RR_hip",
  "RR_thigh",
  "RR_calf",
)

ANCHOR_BODY = "base"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MOTION_FILE = str(REPO_ROOT / "data" / "go2_cycles" / "motion.npz")
