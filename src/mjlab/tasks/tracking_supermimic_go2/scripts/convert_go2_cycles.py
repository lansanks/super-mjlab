"""Convert generated Go2 cycle trajectories into a SuperMimic motion library.

The source datasets in ``data/datasets`` are stored at 100 Hz and contain one
gait cycle each. Periodic clips are extended to a 5 s training horizon by
default so the SuperMimic policy's 32-step imagination window has valid starts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import GO2_XML
from mjlab.tasks.tracking_supermimic.rotations import quat_to_mat, so3_log

from ..constants import CONTROL_DT, TRACKED_BODIES

SOURCE_LEGS = ("FR", "FL", "RR", "RL")
TARGET_LEGS = ("FL", "FR", "RL", "RR")
PERIODIC_CATEGORIES = {"trot", "bound", "biped", "handstand"}
FOOT_RADIUS = 0.022
TORQUE_LIMITS = {
  "hip": 23.7,
  "thigh": 23.7,
  "calf": 45.43,
}


@dataclass(frozen=True)
class ClipStats:
  source_frames: int
  output_frames: int
  cycles: int
  dz: float
  max_torque_ratio: float
  joint_limit_violations: int


def _gradient(value: torch.Tensor, dt: float) -> torch.Tensor:
  output = torch.zeros_like(value)
  if value.shape[0] == 1:
    return output
  output[1:-1] = (value[2:] - value[:-2]) / (2 * dt)
  output[0] = (value[1] - value[0]) / dt
  output[-1] = (value[-1] - value[-2]) / dt
  return output


def _angular_velocity(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  output = torch.zeros(rotations.shape[:-2] + (3,), dtype=rotations.dtype)
  if rotations.shape[0] == 1:
    return output
  output[1:-1] = so3_log(rotations[2:] @ rotations[:-2].transpose(-1, -2)) / (2 * dt)
  output[0] = so3_log(rotations[1] @ rotations[0].transpose(-1, -2)) / dt
  output[-1] = so3_log(rotations[-1] @ rotations[-2].transpose(-1, -2)) / dt
  return output


def _map_source_q(source_q: np.ndarray) -> np.ndarray:
  """Map source ``[HFE, KFE, HAA]`` legs to Go2 entity joint order."""
  output = np.empty_like(source_q)
  for target_leg_index, leg in enumerate(TARGET_LEGS):
    source_leg_index = SOURCE_LEGS.index(leg)
    source_slice = slice(3 * source_leg_index, 3 * source_leg_index + 3)
    target_slice = slice(3 * target_leg_index, 3 * target_leg_index + 3)
    hfe, kfe, haa = source_q[:, source_slice].T
    output[:, target_slice] = np.column_stack((haa, -hfe, -kfe))
  return output


def _map_contact(contact: np.ndarray) -> np.ndarray:
  source_indices = [SOURCE_LEGS.index(leg) for leg in TARGET_LEGS]
  return contact[:, source_indices]


def _map_torque(torque: np.ndarray) -> np.ndarray:
  return _map_source_q(torque)


def _root_quaternion(pitch: np.ndarray) -> np.ndarray:
  pitch = np.unwrap(pitch)
  quat = np.zeros((pitch.shape[0], 4), dtype=np.float64)
  quat[:, 0] = np.cos(pitch / 2)
  quat[:, 2] = -np.sin(pitch / 2)
  return quat


def _resample_cycle(
  source_t: np.ndarray,
  source_base: np.ndarray,
  source_q: np.ndarray,
  source_contact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  duration = float(source_t[-1] - source_t[0])
  if duration <= 0:
    raise ValueError("Source trajectory duration must be positive.")
  frame_count = int(np.floor(duration / CONTROL_DT + 1e-8)) + 1
  output_t = np.arange(frame_count, dtype=np.float64) * CONTROL_DT

  base = np.empty((frame_count, source_base.shape[1]), dtype=np.float64)
  for column in range(source_base.shape[1]):
    base[:, column] = np.interp(
      output_t,
      source_t,
      source_base[:, column],
    )
  q = np.empty((frame_count, source_q.shape[1]), dtype=np.float64)
  for column in range(source_q.shape[1]):
    q[:, column] = np.interp(output_t, source_t, source_q[:, column])

  source_indices = np.clip(
    np.searchsorted(source_t, output_t, side="left"),
    0,
    len(source_t) - 1,
  )
  previous = np.maximum(source_indices - 1, 0)
  choose_previous = np.abs(output_t - source_t[previous]) <= np.abs(
    source_t[source_indices] - output_t
  )
  contact_indices = np.where(choose_previous, previous, source_indices)
  return base, q, source_contact[contact_indices], output_t


def _extend_cycle(
  base: np.ndarray,
  q: np.ndarray,
  contact: np.ndarray,
  target_duration_s: float,
  repeat: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
  if not repeat or base.shape[0] <= 1:
    return base, q, contact, 1

  target_frames = max(
    int(np.ceil(target_duration_s / CONTROL_DT)),
    base.shape[0],
  )
  cycle_count = max(1, int(np.ceil((target_frames - 1) / (base.shape[0] - 1))))
  stride_x = float(base[-1, 0] - base[0, 0])

  base_parts = [base.copy()]
  q_parts = [q.copy()]
  contact_parts = [contact.copy()]
  for cycle_index in range(1, cycle_count):
    next_base = base[1:].copy()
    next_base[:, 0] += cycle_index * stride_x
    base_parts.append(next_base)
    q_parts.append(q[1:].copy())
    contact_parts.append(contact[1:].copy())

  return (
    np.concatenate(base_parts, axis=0),
    np.concatenate(q_parts, axis=0),
    np.concatenate(contact_parts, axis=0),
    cycle_count,
  )


def _build_model_metadata():
  model = mujoco.MjModel.from_xml_path(str(GO2_XML))
  qpos_addresses = np.zeros(12, dtype=np.int32)
  joint_limit_low = np.zeros(12, dtype=np.float64)
  joint_limit_high = np.zeros(12, dtype=np.float64)
  for leg_index, leg in enumerate(TARGET_LEGS):
    for part_index, part in enumerate(("hip", "thigh", "calf")):
      column = 3 * leg_index + part_index
      joint_name = f"{leg}_{part}_joint"
      joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
      )
      if joint_id < 0:
        raise RuntimeError(f"Go2 model is missing joint {joint_name!r}.")
      qpos_addresses[column] = int(model.jnt_qposadr[joint_id])
      joint_limit_low[column] = model.jnt_range[joint_id, 0]
      joint_limit_high[column] = model.jnt_range[joint_id, 1]

  body_ids = np.zeros(len(TRACKED_BODIES), dtype=np.int32)
  for body_index, body_name in enumerate(TRACKED_BODIES):
    body_id = mujoco.mj_name2id(
      model,
      mujoco.mjtObj.mjOBJ_BODY,
      body_name,
    )
    if body_id < 0:
      raise RuntimeError(f"Go2 model is missing body {body_name!r}.")
    body_ids[body_index] = body_id
  foot_site_ids = np.asarray(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg) for leg in TARGET_LEGS],
    dtype=np.int32,
  )
  return (
    model,
    qpos_addresses,
    body_ids,
    foot_site_ids,
    joint_limit_low,
    joint_limit_high,
  )


def _set_qpos(
  data: mujoco.MjData,
  qpos_addresses: np.ndarray,
  base: np.ndarray,
  root_quat: np.ndarray,
  q: np.ndarray,
  frame: int,
  dz: float,
) -> None:
  data.qpos[:] = 0.0
  data.qpos[0:3] = (base[frame, 0], 0.0, base[frame, 1] + dz)
  data.qpos[3:7] = root_quat[frame]
  data.qpos[qpos_addresses] = q[frame]


def _ground_offset(
  model: mujoco.MjModel,
  qpos_addresses: np.ndarray,
  foot_site_ids: np.ndarray,
  base: np.ndarray,
  root_quat: np.ndarray,
  q: np.ndarray,
  contact: np.ndarray,
) -> float:
  data = mujoco.MjData(model)
  offsets: list[float] = []
  for frame in np.flatnonzero(contact.any(-1)):
    _set_qpos(data, qpos_addresses, base, root_quat, q, int(frame), dz=0.0)
    mujoco.mj_kinematics(model, data)
    for leg_index in np.flatnonzero(contact[frame]):
      offsets.append(FOOT_RADIUS - float(data.site_xpos[foot_site_ids[leg_index], 2]))
  if not offsets:
    raise ValueError("Trajectory has no contact frames for vertical calibration.")
  return float(np.median(offsets))


def _sample_body_trajectories(
  model: mujoco.MjModel,
  qpos_addresses: np.ndarray,
  body_ids: np.ndarray,
  base: np.ndarray,
  root_quat: np.ndarray,
  q: np.ndarray,
  dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  data = mujoco.MjData(model)
  body_pos = np.empty((len(q), len(body_ids), 3), dtype=np.float32)
  body_quat = np.empty((len(q), len(body_ids), 4), dtype=np.float32)
  root_pos = np.empty((len(q), 3), dtype=np.float32)
  for frame in tqdm(range(len(q)), desc="  FK", unit="frame", leave=False):
    _set_qpos(data, qpos_addresses, base, root_quat, q, frame, dz)
    mujoco.mj_kinematics(model, data)
    root_pos[frame] = data.xpos[body_ids[0]]
    body_pos[frame] = data.xpos[body_ids]
    body_quat[frame] = data.xquat[body_ids]
  return root_pos, body_pos, body_quat


def convert_clip(
  dataset_dir: Path,
  model: mujoco.MjModel,
  qpos_addresses: np.ndarray,
  body_ids: np.ndarray,
  foot_site_ids: np.ndarray,
  joint_limit_low: np.ndarray,
  joint_limit_high: np.ndarray,
  target_duration_s: float,
) -> tuple[dict[str, np.ndarray], ClipStats, list[str]]:
  meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
  source = np.load(dataset_dir / "traj.npz", allow_pickle=False)
  source_q = _map_source_q(np.asarray(source["q"], dtype=np.float64))
  source_contact = _map_contact(np.asarray(source["contact"], dtype=np.int8))
  source_torque = _map_torque(np.asarray(source["tau"], dtype=np.float64))

  limit_violations = int(
    ((source_q < joint_limit_low) | (source_q > joint_limit_high)).sum()
  )
  torque_limits = np.asarray(
    [TORQUE_LIMITS[part] for _ in TARGET_LEGS for part in ("hip", "thigh", "calf")]
  )
  max_torque_ratio = float(np.max(np.abs(source_torque) / torque_limits[None, :]))

  base, q, contact, _ = _resample_cycle(
    np.asarray(source["t"], dtype=np.float64),
    np.asarray(source["base"], dtype=np.float64),
    source_q,
    source_contact,
  )
  resampled_seam_error = float(np.abs(q[-1] - q[0]).max())
  category = str(meta.get("category", meta.get("gait", "")))
  base, q, contact, cycles = _extend_cycle(
    base,
    q,
    contact,
    target_duration_s=target_duration_s,
    repeat=category in PERIODIC_CATEGORIES,
  )

  root_quat = _root_quaternion(base[:, 2])
  dz = _ground_offset(
    model,
    qpos_addresses,
    foot_site_ids,
    base,
    root_quat,
    q,
    contact,
  )
  root_pos, body_pos, body_quat = _sample_body_trajectories(
    model,
    qpos_addresses,
    body_ids,
    base,
    root_quat,
    q,
    dz,
  )

  q_tensor = torch.from_numpy(q.astype(np.float32))
  root_pos_tensor = torch.from_numpy(root_pos)
  body_pos_tensor = torch.from_numpy(body_pos)
  body_rot = quat_to_mat(torch.from_numpy(body_quat))
  root_rot = quat_to_mat(torch.from_numpy(root_quat.astype(np.float32)))

  warnings: list[str] = []
  if limit_violations:
    warnings.append(f"{limit_violations} joint-limit violations")
  if max_torque_ratio > 1.01:
    warnings.append(f"torque ratio {max_torque_ratio:.2f}x")
  if category in PERIODIC_CATEGORIES:
    if resampled_seam_error > 0.05:
      warnings.append(f"period seam {resampled_seam_error:.3f} rad")

  output = {
    "joint_pos": q_tensor.numpy(),
    "joint_vel": _gradient(q_tensor, CONTROL_DT).numpy(),
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": _gradient(body_pos_tensor, CONTROL_DT).numpy(),
    "body_ang_vel_w": _angular_velocity(body_rot, CONTROL_DT).numpy(),
    "root_pos_w": root_pos,
    "root_quat_w": root_quat.astype(np.float32),
    "root_lin_vel_w": _gradient(root_pos_tensor, CONTROL_DT).numpy(),
    "root_ang_vel_w": _angular_velocity(root_rot, CONTROL_DT).numpy(),
  }
  stats = ClipStats(
    source_frames=int(len(source["t"])),
    output_frames=int(len(q)),
    cycles=cycles,
    dz=dz,
    max_torque_ratio=max_torque_ratio,
    joint_limit_violations=limit_violations,
  )
  return output, stats, warnings


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert generated Go2 cycles to a SuperMimic NPZ.",
  )
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("data/go2_cycles/motion.npz"),
  )
  parser.add_argument("--target-duration", type=float, default=5.0)
  parser.add_argument("--max-clips", type=int, default=None)
  parser.add_argument(
    "--drop-infeasible",
    action="store_true",
    help="Skip clips whose source torque exceeds the Go2 actuator limits.",
  )
  args = parser.parse_args()

  dataset_metas = sorted(args.input_dir.glob("*/meta.json"))
  if args.max_clips is not None:
    dataset_metas = dataset_metas[: args.max_clips]
  if not dataset_metas:
    raise SystemExit(f"No dataset meta.json files found in {args.input_dir}")

  (
    model,
    qpos_addresses,
    body_ids,
    foot_site_ids,
    joint_limit_low,
    joint_limit_high,
  ) = _build_model_metadata()
  clips: list[dict[str, np.ndarray]] = []
  names: list[str] = []
  categories: list[str] = []
  for meta_path in dataset_metas:
    dataset_dir = meta_path.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"[convert] {dataset_dir.name}")
    try:
      clip, stats, warnings = convert_clip(
        dataset_dir,
        model,
        qpos_addresses,
        body_ids,
        foot_site_ids,
        joint_limit_low,
        joint_limit_high,
        args.target_duration,
      )
    except (KeyError, ValueError) as exc:
      print(f"  skip: {exc}")
      continue
    if args.drop_infeasible and stats.max_torque_ratio > 1.01:
      print(f"  skip infeasible torque ratio {stats.max_torque_ratio:.2f}x")
      continue
    if stats.joint_limit_violations:
      raise ValueError(
        f"{dataset_dir.name} has {stats.joint_limit_violations} joint-limit violations"
      )
    clips.append(clip)
    names.append(str(meta.get("id", dataset_dir.name)))
    categories.append(str(meta.get("category", "")))
    warning_text = f"; {', '.join(warnings)}" if warnings else ""
    print(
      f"  {stats.source_frames} source frames -> "
      f"{stats.output_frames} frames, cycles={stats.cycles}, "
      f"dz={stats.dz:.4f} m{warning_text}"
    )

  if not clips:
    raise SystemExit("No clips were converted.")

  clip_lengths = np.asarray(
    [clip["joint_pos"].shape[0] for clip in clips],
    dtype=np.int64,
  )
  clip_starts = np.concatenate(
    (np.zeros(1, dtype=np.int64), np.cumsum(clip_lengths[:-1]))
  )
  frame_clip = np.concatenate(
    [
      np.full(length, clip_index, dtype=np.int64)
      for clip_index, length in enumerate(clip_lengths)
    ]
  )
  output = {
    key: np.concatenate([clip[key] for clip in clips], axis=0) for key in clips[0]
  }
  output.update(
    {
      "fps": np.asarray([1.0 / CONTROL_DT], dtype=np.float32),
      "names": np.asarray(names),
      "categories": np.asarray(categories),
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
    f"[convert] wrote {args.output}: {len(clips)} clips, "
    f"{total_frames} frames, {total_frames * CONTROL_DT / 60:.2f} min"
  )


if __name__ == "__main__":
  main()
