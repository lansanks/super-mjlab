from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import GO2_XML
from mjlab.tasks.tracking_supermimic.constants import (
  G1_JOINT_NAMES,
  TRACKED_BODIES,
)
from mjlab.tasks.tracking_supermimic.kinematics import G1Kinematics
from mjlab.tasks.tracking_supermimic.mdp import (
  critic_extra_dim,
  goal_dim_per_frame,
  invariant_dim,
  obs_dim,
  xi_dim,
)
from mjlab.tasks.tracking_supermimic.motion import LafanMotionLibrary
from mjlab.tasks.tracking_supermimic.nets import Policy, WorldModel
from mjlab.tasks.tracking_supermimic.onnx import export_policy_to_onnx
from mjlab.tasks.tracking_supermimic.scripts.convert_lafan1 import (
  _build_model_metadata,
  convert_clip,
)
from mjlab.tasks.tracking_supermimic_go2.constants import (
  GO2_JOINT_NAMES,
)
from mjlab.tasks.tracking_supermimic_go2.constants import (
  TRACKED_BODIES as GO2_TRACKED_BODIES,
)
from mjlab.tasks.tracking_supermimic_go2.kinematics import Go2Kinematics
from mjlab.tasks.tracking_supermimic_go2.scripts.convert_go2_cycles import (
  _build_model_metadata as _build_go2_model_metadata,
)
from mjlab.tasks.tracking_supermimic_go2.scripts.convert_go2_cycles import (
  convert_clip as convert_go2_clip,
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


def test_go2_kinematics_matches_mujoco_body_poses():
  model = mujoco.MjModel.from_xml_path(str(GO2_XML))
  data = mujoco.MjData(model)
  q = torch.zeros(1, len(GO2_JOINT_NAMES))
  position = torch.tensor([[0.1, -0.2, 0.4]])
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  rotation = torch.eye(3).unsqueeze(0)
  zeros = torch.zeros(1, 3)
  kinematics = Go2Kinematics("cpu")
  body_pos, body_rot, _, _ = kinematics.forward(
    position,
    rotation,
    zeros,
    zeros,
    q,
    torch.zeros_like(q),
  )

  data.qpos[:] = 0.0
  data.qpos[0:3] = position[0].numpy()
  data.qpos[3:7] = quat[0].numpy()
  mujoco.mj_kinematics(model, data)
  ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in GO2_TRACKED_BODIES
  ]
  expected_pos = torch.from_numpy(data.xpos[ids].copy()).float()
  assert torch.allclose(body_pos[0], expected_pos, atol=1e-5)
  assert torch.isfinite(body_rot).all()


def test_supermimic_networks_use_dynamic_joint_dimensions():
  num_joints = 12
  num_bodies = 13
  context_dim = xi_dim(num_joints)
  world_model = WorldModel(
    hidden=(16,),
    num_joints=num_joints,
    invariant_input_dim=invariant_dim(num_joints),
    sim_context_dim=context_dim,
  )
  policy = Policy(
    obs_dim(num_joints),
    goal_dim_per_frame(num_joints) * 10,
    history=10,
    latent_dim=4,
    encoder_hidden=(16,),
    decoder_hidden=(16,),
    action_dim=num_joints,
  )
  assert world_model.input_dim == 65
  assert world_model.output_dim == 36
  assert critic_extra_dim(num_joints, num_bodies, context_dim) == 76
  action = policy(torch.zeros(2, 10, 48), torch.zeros(2, 390))
  assert action.shape == (2, num_joints)


def test_go2_cycle_converter_maps_joints_and_repeats(tmp_path: Path):
  dataset_dir = tmp_path / "go2_trot_test"
  dataset_dir.mkdir()
  source_q = np.tile(
    [-0.9, 1.8, -0.1, -0.9, 1.8, 0.1, -0.9, 1.8, -0.1, -0.9, 1.8, 0.1],
    (4, 1),
  )
  base = np.zeros((4, 6))
  base[:, 0] = [0.0, 0.002, 0.004, 0.006]
  base[:, 1] = 0.295
  np.savez(
    dataset_dir / "traj.npz",
    t=np.arange(4) * 0.01,
    base=base,
    q=source_q,
    qd=np.zeros_like(source_q),
    tau=np.zeros_like(source_q),
    contact=np.ones((4, 4), dtype=np.int64),
    grf=np.zeros((4, 4, 2)),
  )
  (dataset_dir / "meta.json").write_text(
    '{"id": "go2_trot_test", "category": "trot"}',
    encoding="utf-8",
  )
  metadata = _build_go2_model_metadata()
  converted, stats, warnings = convert_go2_clip(
    dataset_dir,
    *metadata,
    target_duration_s=0.1,
  )

  assert stats.cycles > 1
  assert not warnings
  assert converted["joint_pos"].shape[1] == 12
  assert converted["body_pos_w"].shape[1] == 13
  assert tuple(converted["body_pos_w"].shape[2:]) == (3,)
  assert np.allclose(
    converted["joint_pos"][0],
    [0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8],
    atol=1e-6,
  )


def test_supermimic_policy_onnx_matches_pytorch(tmp_path: Path):
  ort = pytest.importorskip("onnxruntime")
  torch.manual_seed(7)
  policy = Policy(
    obs_dim=48,
    goal_dim=78,
    history=3,
    latent_dim=4,
    encoder_hidden=(16,),
    decoder_hidden=(16,),
    action_dim=12,
  )
  observation = torch.randn(1, 3, 48)
  goal = torch.randn(1, 78)
  expected = torch.clamp(policy(observation, goal), -20, 20).detach()
  output_path = tmp_path / "policy.onnx"
  export_policy_to_onnx(
    policy,
    output_path,
    history=3,
    observation_dim=48,
    goal_dim=78,
    action_clip=20,
  )

  session = ort.InferenceSession(str(output_path))
  actual = session.run(
    None,
    {
      "observation_history": observation.numpy(),
      "goal": goal.numpy(),
    },
  )[0]
  assert actual.shape == (1, 12)
  np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4)
