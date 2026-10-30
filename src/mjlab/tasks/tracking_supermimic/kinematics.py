"""Differentiable forward kinematics built from mjlab MJCF assets."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from .constants import G1_JOINT_NAMES, TRACKED_BODIES
from .rotations import axis_angle_to_mat, quat_to_mat


class MjcfKinematics:
  """Compute tracked MJCF body poses and velocities from a floating-base state."""

  def __init__(
    self,
    xml_path: str | Path,
    joint_names: tuple[str, ...],
    tracked_bodies: tuple[str, ...],
    device: str | torch.device,
  ):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.device = torch.device(device)
    joint_to_index = {name: i for i, name in enumerate(joint_names)}

    bodies: list[str] = []
    parent: list[int] = []
    joint_index: list[int] = []
    axes: list[np.ndarray] = []
    offsets_pos: list[np.ndarray] = []
    offsets_quat: list[np.ndarray] = []
    mj_to_local: dict[int, int] = {}

    for body_id in range(1, model.nbody):
      if model.body_jntnum[body_id] == 0:
        continue
      joint_id = int(model.body_jntadr[body_id])
      name = model.body(body_id).name
      mj_to_local[body_id] = len(bodies)
      bodies.append(name)

      if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
        parent.append(-1)
        joint_index.append(-1)
      else:
        parent_body = int(model.body_parentid[body_id])
        if parent_body not in mj_to_local:
          raise RuntimeError(
            f"Tracked body {name!r} has a fixed parent chain that is not "
            "represented in the differentiable model."
          )
        parent.append(mj_to_local[parent_body])
        joint_name = model.joint(joint_id).name
        joint_index.append(joint_to_index[joint_name])

      axes.append(model.jnt_axis[joint_id].copy())
      offsets_pos.append(model.body_pos[body_id].copy())
      offsets_quat.append(model.body_quat[body_id].copy())

    missing = [name for name in tracked_bodies if name not in bodies]
    if missing:
      raise RuntimeError(f"Tracked bodies missing from MJCF model: {missing}")

    self.body_names = tuple(bodies)
    self.parent = tuple(parent)
    self.joint_index = tuple(joint_index)
    self.axes = torch.as_tensor(
      np.asarray(axes), dtype=torch.float32, device=self.device
    )
    self.offset_pos = torch.as_tensor(
      np.asarray(offsets_pos), dtype=torch.float32, device=self.device
    )
    self.offset_rot = quat_to_mat(
      torch.as_tensor(np.asarray(offsets_quat), dtype=torch.float32, device=self.device)
    )
    self.tracked_indices = tuple(bodies.index(name) for name in tracked_bodies)

    q_lo = np.zeros(len(joint_names), dtype=np.float32)
    q_hi = np.zeros(len(joint_names), dtype=np.float32)
    for joint_id in range(model.njnt):
      name = model.joint(joint_id).name
      if name in joint_to_index:
        index = joint_to_index[name]
        q_lo[index], q_hi[index] = model.jnt_range[joint_id]
    self.q_lo = torch.as_tensor(q_lo, device=self.device)
    self.q_hi = torch.as_tensor(q_hi, device=self.device)

  def forward(
    self,
    p: torch.Tensor,
    rot: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tracked body positions, rotations, linear and angular velocities."""
    positions: list[torch.Tensor | None] = [None] * len(self.body_names)
    rotations: list[torch.Tensor | None] = [None] * len(self.body_names)
    linear_velocities: list[torch.Tensor | None] = [None] * len(self.body_names)
    angular_velocities: list[torch.Tensor | None] = [None] * len(self.body_names)

    for body_index, parent_index in enumerate(self.parent):
      if parent_index < 0:
        positions[body_index] = p
        rotations[body_index] = rot
        linear_velocities[body_index] = lin_vel
        angular_velocities[body_index] = ang_vel
        continue

      parent_pos = positions[parent_index]
      parent_rot = rotations[parent_index]
      parent_lin_vel = linear_velocities[parent_index]
      parent_ang_vel = angular_velocities[parent_index]
      assert parent_pos is not None
      assert parent_rot is not None
      assert parent_lin_vel is not None
      assert parent_ang_vel is not None

      joint = self.joint_index[body_index]
      body_pos = parent_pos + (
        parent_rot @ self.offset_pos[body_index].unsqueeze(-1)
      ).squeeze(-1)
      body_rot_unactuated = parent_rot @ self.offset_rot[body_index]
      body_rot = body_rot_unactuated @ axis_angle_to_mat(
        self.axes[body_index], q[..., joint]
      )
      axis_world = (body_rot_unactuated @ self.axes[body_index].unsqueeze(-1)).squeeze(
        -1
      )
      body_ang_vel = parent_ang_vel + axis_world * qd[..., joint : joint + 1]
      body_lin_vel = parent_lin_vel + torch.cross(
        parent_ang_vel, body_pos - parent_pos, dim=-1
      )

      positions[body_index] = body_pos
      rotations[body_index] = body_rot
      linear_velocities[body_index] = body_lin_vel
      angular_velocities[body_index] = body_ang_vel

    indices = self.tracked_indices

    def selected(values: list[torch.Tensor | None]) -> list[torch.Tensor]:
      output: list[torch.Tensor] = []
      for index in indices:
        value = values[index]
        assert value is not None
        output.append(value)
      return output

    return (
      torch.stack(selected(positions), -2),
      torch.stack(selected(rotations), -3),
      torch.stack(selected(linear_velocities), -2),
      torch.stack(selected(angular_velocities), -2),
    )


class KinematicsFactory(Protocol):
  def __call__(self, device: str | torch.device) -> MjcfKinematics: ...


class G1Kinematics(MjcfKinematics):
  """Compute tracked G1 link poses and velocities from a floating-base state."""

  def __init__(self, device: str | torch.device):
    super().__init__(
      G1_XML,
      G1_JOINT_NAMES,
      TRACKED_BODIES,
      device,
    )
