from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.tracking_supermimic.constants import (
  G1_JOINT_NAMES,
  TRACKED_BODIES,
)
from mjlab.tasks.tracking_supermimic.kinematics import G1Kinematics
from mjlab.tasks.tracking_supermimic.motion import LafanMotionLibrary
from mjlab.tasks.tracking_supermimic.scripts.convert_lafan1 import (
  _build_model_metadata,
  convert_clip,
)


def _identity_quat(frames: int) -> np.ndarray:
  quat = np.zeros((frames, 4), dtype=np.float32)
  quat[:, 0] = 1.0
  return quat


def test_motion_goal_pads_after_single_clip_end(tmp_path: Path):
  frames = 6
  motion_file = tmp_path / "motion.npz"
  np.savez(
    motion_file,
    joint_pos=np.zeros((frames, len(G1_JOINT_NAMES)), dtype=np.float32),
    joint_vel=np.ones((frames, len(G1_JOINT_NAMES)), dtype=np.float32),
    body_pos_w=np.zeros((frames, len(TRACKED_BODIES), 3), dtype=np.float32),
    body_quat_w=np.broadcast_to(
      _identity_quat(frames)[:, None, :],
      (frames, len(TRACKED_BODIES), 4),
    ).copy(),
    body_lin_vel_w=np.zeros((frames, len(TRACKED_BODIES), 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((frames, len(TRACKED_BODIES), 3), dtype=np.float32),
    body_names=np.asarray(TRACKED_BODIES),
    names=np.asarray(["clip"]),
    clip_start=np.asarray([0], dtype=np.int64),
    clip_len=np.asarray([frames], dtype=np.int64),
    clip_id=np.zeros(frames, dtype=np.int64),
  )
  motion = LafanMotionLibrary(motion_file, "cpu")
  goal = motion.goal(torch.tensor([2]), count=5).reshape(1, 5, 73)
  assert goal.shape == (1, 5, 73)
  # Requested frames 3, 4, 5 are recorded. The remaining two hold the pose
  # at frame 5 but zero synthetic velocities.
  assert torch.count_nonzero(goal[0, 3:, 9:15]) == 0
  assert torch.count_nonzero(goal[0, 3:, 44:]) == 0
  torch.testing.assert_close(goal[0, 2, 0:9], goal[0, 3, 0:9])
  torch.testing.assert_close(goal[0, 2, 15:44], goal[0, 3, 15:44])


def test_kinematics_matches_mujoco_body_poses():
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  q = torch.zeros(1, len(G1_JOINT_NAMES))
  position = torch.tensor([[0.1, -0.2, 0.8]])
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  rotation = torch.eye(3).unsqueeze(0)
  linear_velocity = torch.zeros(1, 3)
  angular_velocity = torch.zeros(1, 3)
  kinematics = G1Kinematics("cpu")
  body_pos, body_rot, _, _ = kinematics.forward(
    position,
    rotation,
    linear_velocity,
    angular_velocity,
    q,
    torch.zeros_like(q),
  )

  data.qpos[:] = 0.0
  data.qpos[0:3] = position[0].numpy()
  data.qpos[3:7] = quat[0].numpy()
  mujoco.mj_kinematics(model, data)
  ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TRACKED_BODIES
  ]
  expected_pos = torch.from_numpy(data.xpos[ids].copy()).float()
  assert torch.allclose(body_pos[0], expected_pos, atol=1e-5)
  assert torch.isfinite(body_rot).all()


def test_lafan_conversion_produces_body_trajectories(tmp_path: Path):
  source = tmp_path / "dance.csv"
  frames = 12
  raw = np.zeros((frames, 7 + len(G1_JOINT_NAMES)), dtype=np.float32)
  raw[:, 0] = np.linspace(0.0, 0.2, frames)
  raw[:, 2] = 0.8
  raw[:, 3:7] = np.asarray([0.0, 0.0, 0.0, 1.0])
  np.savetxt(source, raw, delimiter=",")
  model, joint_qpos_addresses, body_ids = _build_model_metadata()
  converted = convert_clip(
    source,
    model,
    joint_qpos_addresses,
    body_ids,
    source_fps=30.0,
  )
  frame_count = converted["joint_pos"].shape[0]
  assert frame_count == 19
  assert converted["body_pos_w"].shape == (
    frame_count,
    len(TRACKED_BODIES),
    3,
  )
  assert converted["body_lin_vel_w"].shape == (
    frame_count,
    len(TRACKED_BODIES),
    3,
  )
  pelvis = TRACKED_BODIES.index("pelvis")
  pelvis_pos = converted["body_pos_w"][:, pelvis]
  assert np.allclose(pelvis_pos[:, 1:], [0.0, 0.8], atol=1e-4)
  assert pelvis_pos[0, 0] == 0.0
  assert 0.19 < pelvis_pos[-1, 0] <= 0.2
