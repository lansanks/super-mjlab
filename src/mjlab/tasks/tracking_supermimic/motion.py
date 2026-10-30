"""Multi-clip LAFAN1 reference motion used by the SuperMimic tracking task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .rotations import mat_to_6d, quat_to_mat, yaw_mat


class LafanMotionLibrary:
  """Runtime reference library stored as one flattened multi-clip NPZ."""

  def __init__(
    self,
    motion_file: str | Path,
    device: str | torch.device,
    anchor_body_name: str = "pelvis",
  ):
    self.device = torch.device(device)
    path = Path(motion_file)
    if not path.is_file():
      raise FileNotFoundError(
        f"Motion file does not exist: {path}. Run "
        "`python -m mjlab.tasks.tracking_supermimic.scripts.convert_lafan1` "
        "to build it from the LAFAN1 CSV files."
      )
    with np.load(path) as data:
      required = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "clip_start",
        "clip_len",
      )
      missing = [key for key in required if key not in data]
      if missing:
        raise ValueError(f"Motion file {path} is missing fields: {missing}")

      self.body_names = tuple(str(name) for name in data["body_names"].tolist())
      self.q = self._tensor(data["joint_pos"])
      self.qd = self._tensor(data["joint_vel"])
      self.body_p = self._tensor(data["body_pos_w"])
      body_quat = self._tensor(data["body_quat_w"])
      self.body_R = quat_to_mat(body_quat)
      self.body_v = self._tensor(data["body_lin_vel_w"])
      self.body_w = self._tensor(data["body_ang_vel_w"])
      self.clip_start = self._tensor(data["clip_start"], dtype=torch.long)
      self.clip_len = self._tensor(data["clip_len"], dtype=torch.long)
      if "clip_id" in data:
        self.frame_clip = self._tensor(data["clip_id"], dtype=torch.long)
      else:
        self.frame_clip = torch.cat(
          [
            torch.full(
              (int(length),),
              clip_index,
              dtype=torch.long,
              device=self.device,
            )
            for clip_index, length in enumerate(self.clip_len.tolist())
          ]
        )
      self.names = (
        [str(name) for name in data["names"].tolist()]
        if "names" in data
        else [f"clip_{i}" for i in range(len(self.clip_len))]
      )

    if self.body_p.shape[1] != len(self.body_names):
      raise ValueError("body_pos_w and body_names have inconsistent dimensions.")
    if anchor_body_name not in self.body_names:
      raise ValueError(
        f"Tracking motion must contain the anchor body {anchor_body_name!r}."
      )

    self.num_frames = int(self.q.shape[0])
    if self.num_frames == 0:
      raise ValueError("Motion file contains no frames.")
    self.num_clips = int(self.clip_len.shape[0])
    self.clip_end = self.clip_start + self.clip_len - 1
    self.anchor_body_name = anchor_body_name
    self.anchor_body_index = self.body_names.index(anchor_body_name)
    self.root_p = self.body_p[:, self.anchor_body_index]
    self.root_R = self.body_R[:, self.anchor_body_index]
    self.root_v = self.body_v[:, self.anchor_body_index]
    self.root_w = self.body_w[:, self.anchor_body_index]

    root_rt = self.root_R.transpose(-1, -2)
    self.root_v_local = (root_rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
    self.root_w_local = (root_rt @ self.root_w.unsqueeze(-1)).squeeze(-1)

    bin_clip: list[int] = []
    bin_start: list[int] = []
    bin_end: list[int] = []
    for clip_index, length in enumerate(self.clip_len.tolist()):
      start = int(self.clip_start[clip_index])
      for frame in range(0, int(length), 50):
        bin_clip.append(clip_index)
        bin_start.append(start + frame)
        bin_end.append(min(start + frame + 50, start + int(length)))
    self.bin_clip = torch.tensor(bin_clip, dtype=torch.long, device=self.device)
    self.bin_start = torch.tensor(bin_start, dtype=torch.long, device=self.device)
    self.bin_end = torch.tensor(bin_end, dtype=torch.long, device=self.device)
    self.num_bins = len(bin_clip)
    self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=self.device)
    for bin_index in range(self.num_bins):
      start = int(self.bin_start[bin_index])
      end = int(self.bin_end[bin_index])
      self.frame_bin[start:end] = bin_index

  def _tensor(
    self,
    value: np.ndarray,
    dtype: torch.dtype = torch.float32,
  ) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=self.device)

  @property
  def time_step_total(self) -> int:
    """Total frames across all clips; the viewer scrubber spans this range."""
    return int(self.q.shape[0])

  def clip_for_frame(self, frames: torch.Tensor) -> torch.Tensor:
    return self.frame_clip[frames]

  def future_frames(self, frames: torch.Tensor, count: int) -> torch.Tensor:
    """Return ``frames + 1 ... frames + count`` clamped at each clip end."""
    ends = self.clip_end[self.clip_for_frame(frames)]
    future = frames.unsqueeze(-1) + torch.arange(1, count + 1, device=frames.device)
    return torch.minimum(future, ends.unsqueeze(-1))

  def goal(self, frames: torch.Tensor, count: int) -> torch.Tensor:
    """Build the heading-invariant future reference window used by the encoder."""
    future = self.future_frames(frames, count)
    requested = frames.unsqueeze(-1) + torch.arange(1, count + 1, device=frames.device)
    recorded = (
      requested <= self.clip_end[self.clip_for_frame(frames)].unsqueeze(-1)
    ).unsqueeze(-1)

    root_v = torch.where(recorded, self.root_v_local[future], 0.0)
    root_w = torch.where(recorded, self.root_w_local[future], 0.0)
    joint_v = torch.where(recorded, self.qd[future], 0.0)

    heading = yaw_mat(self.root_R[frames])
    heading_t = heading.transpose(-1, -2).unsqueeze(1)
    delta_pos = (
      heading_t @ (self.root_p[future] - self.root_p[frames].unsqueeze(1)).unsqueeze(-1)
    ).squeeze(-1)
    relative_rot = mat_to_6d(heading_t @ self.root_R[future])
    features = torch.cat(
      (
        delta_pos,
        relative_rot,
        root_v,
        root_w,
        self.q[future],
        joint_v * 0.1,
      ),
      -1,
    )
    return features.flatten(1)

  def ref_state(self, frames: torch.Tensor):
    return (
      self.root_p[frames],
      self.root_R[frames],
      self.root_v[frames],
      self.root_w[frames],
      self.q[frames],
      self.qd[frames],
    )

  def reference_bodies(self, frames: torch.Tensor):
    return (
      self.body_p[frames],
      self.body_R[frames],
      self.body_v[frames],
      self.body_w[frames],
    )
