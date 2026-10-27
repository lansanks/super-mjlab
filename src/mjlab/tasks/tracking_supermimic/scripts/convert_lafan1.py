"""Convert 30 Hz LAFAN1 G1 CSV clips into one 50 Hz mjlab motion NPZ.

The output contains future-compatible root/joint states plus world-space
trajectories for the 14 bodies tracked by SuperMimic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from ..constants import CONTROL_DT, G1_JOINT_NAMES, TRACKED_BODIES
from ..rotations import quat_slerp, quat_to_mat, so3_log


def _gradient(value: torch.Tensor, dt: float) -> torch.Tensor:
  output = torch.zeros_like(value)
  output[1:-1] = (value[2:] - value[:-2]) / (2 * dt)
  output[0] = (value[1] - value[0]) / dt
  output[-1] = (value[-1] - value[-2]) / dt
  return output


def _angular_velocity(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  output = torch.zeros(rotations.shape[:-2] + (3,), dtype=rotations.dtype)
  output[1:-1] = so3_log(rotations[2:] @ rotations[:-2].transpose(-1, -2)) / (2 * dt)
  output[0] = so3_log(rotations[1] @ rotations[0].transpose(-1, -2)) / dt
  output[-1] = so3_log(rotations[-1] @ rotations[-2].transpose(-1, -2)) / dt
  return output


def _resample(
  raw: torch.Tensor,
  source_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if raw.ndim != 2 or raw.shape[1] < 7 + len(G1_JOINT_NAMES):
    raise ValueError(
      "LAFAN1 CSV must contain root position (3), root quaternion xyzw "
      f"(4), and {len(G1_JOINT_NAMES)} joints."
    )
  source_duration = (raw.shape[0] - 1) / source_fps
  output_frames = int(np.floor(source_duration / CONTROL_DT + 1e-6)) + 1
  times = torch.arange(output_frames, dtype=torch.float32) * CONTROL_DT
  source_indices = times * source_fps
  index_0 = torch.clamp(source_indices.floor().long(), max=raw.shape[0] - 2)
  blend = torch.clamp(source_indices - index_0.float(), 0.0, 1.0)

  position = raw[:, :3]
  quat_xyzw = raw[:, 3:7]
  joints = raw[:, 7 : 7 + len(G1_JOINT_NAMES)]
  quat_wxyz = torch.cat((quat_xyzw[:, 3:4], quat_xyzw[:, :3]), -1)
  position = (
    position[index_0] * (1 - blend[:, None]) + position[index_0 + 1] * blend[:, None]
  )
  joints = joints[index_0] * (1 - blend[:, None]) + joints[index_0 + 1] * blend[:, None]
  quat_wxyz = quat_slerp(quat_wxyz[index_0], quat_wxyz[index_0 + 1], blend)
  return position, quat_wxyz, joints


def _build_model_metadata():
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  joint_to_column = {name: index for index, name in enumerate(G1_JOINT_NAMES)}
  joint_qpos_addresses = np.zeros(len(G1_JOINT_NAMES), dtype=np.int32)
  for joint_id in range(model.njnt):
    name = model.joint(joint_id).name
    if name in joint_to_column:
      joint_qpos_addresses[joint_to_column[name]] = int(model.jnt_qposadr[joint_id])
  body_ids = []
  for name in TRACKED_BODIES:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      raise RuntimeError(f"G1 MJCF is missing body {name!r}.")
    body_ids.append(body_id)
  return model, joint_qpos_addresses, np.asarray(body_ids)


def _sample_body_trajectories(
  model: mujoco.MjModel,
  joint_qpos_addresses: np.ndarray,
  body_ids: np.ndarray,
  position: torch.Tensor,
  quat_wxyz: torch.Tensor,
  joints: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
  data = mujoco.MjData(model)
  body_positions = np.empty((joints.shape[0], len(body_ids), 3), dtype=np.float32)
  body_quaternions = np.empty((joints.shape[0], len(body_ids), 4), dtype=np.float32)
  positions = position.numpy()
  quaternions = quat_wxyz.numpy()
  joint_positions = joints.numpy()
  for frame in tqdm(range(joints.shape[0]), desc="  FK", unit="frame", leave=False):
    data.qpos[:] = 0.0
    data.qpos[0:3] = positions[frame]
    data.qpos[3:7] = quaternions[frame]
    data.qpos[joint_qpos_addresses] = joint_positions[frame]
    mujoco.mj_kinematics(model, data)
    body_positions[frame] = data.xpos[body_ids]
    body_quaternions[frame] = data.xquat[body_ids]
  return body_positions, body_quaternions


def convert_clip(
  csv_path: Path,
  model: mujoco.MjModel,
  joint_qpos_addresses: np.ndarray,
  body_ids: np.ndarray,
  source_fps: float,
) -> dict[str, np.ndarray]:
  raw = np.atleast_2d(np.loadtxt(csv_path, delimiter=","))
  raw_tensor = torch.as_tensor(raw, dtype=torch.float32)
  if raw_tensor.shape[0] < 2:
    raise ValueError(f"{csv_path} needs at least two frames.")
  position, quat_wxyz, joints = _resample(raw_tensor, source_fps)
  joint_vel = _gradient(joints, CONTROL_DT)
  body_pos, body_quat = _sample_body_trajectories(
    model,
    joint_qpos_addresses,
    body_ids,
    position,
    quat_wxyz,
    joints,
  )
  body_pos_tensor = torch.from_numpy(body_pos)
  body_rot = quat_to_mat(torch.from_numpy(body_quat))
  body_vel = _gradient(body_pos_tensor, CONTROL_DT).numpy()
  body_ang_vel = _angular_velocity(body_rot, CONTROL_DT).numpy()
  pelvis_index = TRACKED_BODIES.index("pelvis")
  root_pos = body_pos[:, pelvis_index]
  root_quat = body_quat[:, pelvis_index]
  root_rot = quat_to_mat(torch.from_numpy(root_quat))
  root_vel = _gradient(torch.from_numpy(root_pos), CONTROL_DT).numpy()
  root_ang_vel = _angular_velocity(root_rot, CONTROL_DT).numpy()
  return {
    "joint_pos": joints.numpy(),
    "joint_vel": joint_vel.numpy(),
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": body_vel,
    "body_ang_vel_w": body_ang_vel,
    "root_pos_w": root_pos,
    "root_quat_w": root_quat,
    "root_lin_vel_w": root_vel,
    "root_ang_vel_w": root_ang_vel,
  }


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert LAFAN1 G1 CSV clips to a multi-clip mjlab NPZ."
  )
  parser.add_argument(
    "--input-dir",
    type=Path,
    required=True,
    help="Directory containing LAFAN1 G1 CSV files.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("data/lafan1_g1/motion.npz"),
  )
  parser.add_argument("--source-fps", type=float, default=30.0)
  parser.add_argument(
    "--limit-clips",
    type=int,
    default=None,
    help="Convert only the first N sorted clips for a smoke test.",
  )
  args = parser.parse_args()

  files = sorted(args.input_dir.glob("*.csv"))
  if args.limit_clips is not None:
    files = files[: args.limit_clips]
  if not files:
    raise SystemExit(f"No CSV files found in {args.input_dir}")

  model, joint_qpos_addresses, body_ids = _build_model_metadata()
  clips: list[dict[str, np.ndarray]] = []
  names: list[str] = []
  for csv_path in files:
    print(f"[convert] {csv_path.name}")
    clips.append(
      convert_clip(
        csv_path,
        model,
        joint_qpos_addresses,
        body_ids,
        args.source_fps,
      )
    )
    names.append(csv_path.stem)

  clip_lengths = np.asarray(
    [clip["joint_pos"].shape[0] for clip in clips], dtype=np.int64
  )
  clip_starts = np.concatenate(
    (np.zeros(1, dtype=np.int64), np.cumsum(clip_lengths[:-1]))
  )
  frame_clip = np.concatenate(
    [
      np.full(length, index, dtype=np.int64)
      for index, length in enumerate(clip_lengths)
    ]
  )
  keys = tuple(key for key in clips[0] if key != "clip_id")
  output = {key: np.concatenate([clip[key] for clip in clips], axis=0) for key in keys}
  output.update(
    {
      "fps": np.asarray([1.0 / CONTROL_DT], dtype=np.float32),
      "names": np.asarray(names),
      "body_names": np.asarray(TRACKED_BODIES),
      "clip_start": clip_starts,
      "clip_len": clip_lengths,
      "clip_id": frame_clip,
    }
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, **output)
  total_frames = int(clip_lengths.sum())
  print(
    f"[convert] wrote {args.output} with {len(clips)} clips and "
    f"{total_frames} frames ({total_frames * CONTROL_DT / 3600:.2f} h)"
  )


if __name__ == "__main__":
  main()
