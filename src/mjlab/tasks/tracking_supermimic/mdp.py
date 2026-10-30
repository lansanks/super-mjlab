"""Analytic observation, reward and termination heads for SuperMimic tracking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import CONTROL_DT
from .kinematics import MjcfKinematics
from .motion import LafanMotionLibrary
from .rotations import (
  mat_to_6d,
  mat_to_quat,
  quat_to_mat,
  rot_angle_sq,
  yaw_mat,
)


def obs_dim(num_joints: int) -> int:
  return num_joints + num_joints + 3 + 3 + num_joints + 6


def xi_dim(num_joints: int) -> int:
  return 1 + num_joints + 3


def invariant_dim(num_joints: int) -> int:
  return 1 + 6 + 3 + 3 + num_joints + num_joints


def goal_dim_per_frame(num_joints: int) -> int:
  return 3 + 6 + 3 + 3 + num_joints + num_joints


def critic_extra_dim(num_joints: int, num_bodies: int, context_dim: int) -> int:
  return num_joints + 3 * num_bodies + 3 + 6 + context_dim


def observation_noise(num_joints: int) -> torch.Tensor:
  return torch.cat(
    (
      torch.full((num_joints,), 0.01),
      torch.full((num_joints,), 0.5),
      torch.full((3,), 0.2),
      torch.full((3,), 0.05),
      torch.zeros(num_joints),
      torch.full((6,), 0.05),
    )
  )


def observation_scale(num_joints: int) -> torch.Tensor:
  return torch.cat(
    (
      torch.ones(num_joints),
      torch.full((num_joints,), 0.1),
      torch.full((3,), 0.25),
      torch.ones(3),
      torch.ones(num_joints),
      torch.ones(6),
    )
  )


# G1 defaults are kept as module constants for backward compatibility.
OBS_DIM = obs_dim(29)
XI_DIM = xi_dim(29)
INV_DIM = invariant_dim(29)
OBS_NOISE = observation_noise(29)
OBS_SCALE = observation_scale(29)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
  body_pos_std: float = 0.3
  body_ori_std: float = 0.4
  body_linvel_std: float = 1.0
  body_angvel_std: float = 3.14
  root_pos_std: float = 0.3
  root_ori_std: float = 0.4
  w_action_rate: float = -0.1
  w_joint_limit: float = -2.0
  soft_limit_factor: float = 0.9
  term_height: float = 0.25
  term_tilt: float = 0.8


def flatten_state(state):
  p, rot, lin_vel, ang_vel, q, qd = state
  return torch.cat((p, mat_to_quat(rot), lin_vel, ang_vel, q, qd), -1)


def unflatten_state(flat: torch.Tensor, num_joints: int = 29):
  state_end = 13 + 2 * num_joints
  if flat.shape[-1] != state_end:
    raise ValueError(
      f"Expected a {state_end}-D state for {num_joints} joints, got {flat.shape[-1]}."
    )
  return (
    flat[..., 0:3],
    quat_to_mat(flat[..., 3:7]),
    flat[..., 7:10],
    flat[..., 10:13],
    flat[..., 13 : 13 + num_joints],
    flat[..., 13 + num_joints : state_end],
  )


def invariant_features(state) -> torch.Tensor:
  """Height/tilt/body-frame state features; global xy and yaw are removed."""
  p, rot, lin_vel, ang_vel, q, qd = state
  heading = yaw_mat(rot)
  rot_t = rot.transpose(-1, -2)
  return torch.cat(
    (
      p[..., 2:3],
      mat_to_6d(heading.transpose(-1, -2) @ rot),
      (rot_t @ lin_vel.unsqueeze(-1)).squeeze(-1),
      (rot_t @ ang_vel.unsqueeze(-1)).squeeze(-1),
      q,
      qd,
    ),
    -1,
  )


class MDP:
  def __init__(
    self,
    kinematics: MjcfKinematics,
    motion: LafanMotionLibrary,
    default_joint_pos: torch.Tensor,
    cfg: RewardCfg | None = None,
  ):
    self.kinematics = kinematics
    self.motion = motion
    self.cfg = cfg or RewardCfg()
    self.default_joint_pos = default_joint_pos
    self.num_joints = int(default_joint_pos.shape[-1])
    self.num_bodies = len(kinematics.tracked_indices)
    self.xi_dim = xi_dim(self.num_joints)
    middle = (kinematics.q_lo + kinematics.q_hi) / 2
    half = (kinematics.q_hi - kinematics.q_lo) / 2
    self.soft_lo = middle - half * self.cfg.soft_limit_factor
    self.soft_hi = middle + half * self.cfg.soft_limit_factor
    self.obs_noise = observation_noise(self.num_joints).to(default_joint_pos.device)
    self.obs_scale = observation_scale(self.num_joints).to(default_joint_pos.device)

  def obs(
    self,
    state,
    previous_action: torch.Tensor,
    sim_context: torch.Tensor,
    reference_frame: torch.Tensor,
  ) -> torch.Tensor:
    _, rot, _, ang_vel, q, qd = state
    rot_t = rot.transpose(-1, -2)
    joint_offset = sim_context[..., 1 : 1 + self.num_joints]
    gravity_body = -rot[..., 2, :]
    reference_relative_rot = mat_to_6d(rot_t @ self.motion.root_R[reference_frame])
    return torch.cat(
      (
        q - self.default_joint_pos - joint_offset,
        qd,
        (rot_t @ ang_vel.unsqueeze(-1)).squeeze(-1),
        gravity_body,
        previous_action,
        reference_relative_rot,
      ),
      -1,
    )

  def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
    return observation + (2 * torch.rand_like(observation) - 1) * self.obs_noise

  def reward(
    self,
    state,
    reference_frame: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    terms: bool = False,
  ):
    cfg = self.cfg
    motion = self.motion
    p, rot, lin_vel, ang_vel, q, qd = state
    body_p, body_rot, body_v, body_w = self.kinematics.forward(
      p, rot, lin_vel, ang_vel, q, qd
    )
    reference_p, reference_rot = (
      motion.root_p[reference_frame],
      motion.root_R[reference_frame],
    )

    delta_rot = yaw_mat(rot @ reference_rot.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], reference_p[..., 2:3]), -1)
    reference_body_p = anchor.unsqueeze(-2) + (
      delta_rot.unsqueeze(-3)
      @ (motion.body_p[reference_frame] - reference_p.unsqueeze(-2)).unsqueeze(-1)
    ).squeeze(-1)
    reference_body_rot = delta_rot.unsqueeze(-3) @ motion.body_R[reference_frame]

    body_pos_error = ((reference_body_p - body_p) ** 2).sum(-1).mean(-1)
    body_rot_error = rot_angle_sq(reference_body_rot, body_rot).mean(-1)
    body_lin_error = ((motion.body_v[reference_frame] - body_v) ** 2).sum(-1).mean(-1)
    body_ang_error = ((motion.body_w[reference_frame] - body_w) ** 2).sum(-1).mean(-1)
    root_pos_error = ((reference_p - p) ** 2).sum(-1)
    root_rot_error = rot_angle_sq(reference_rot, rot)

    tracking = (
      torch.exp(-body_pos_error / cfg.body_pos_std**2)
      + torch.exp(-body_rot_error / cfg.body_ori_std**2)
      + torch.exp(-body_lin_error / cfg.body_linvel_std**2)
      + torch.exp(-body_ang_error / cfg.body_angvel_std**2)
      + 0.5 * torch.exp(-root_pos_error / cfg.root_pos_std**2)
      + 0.5 * torch.exp(-root_rot_error / cfg.root_ori_std**2)
    )
    action_rate = ((action - previous_action) ** 2).sum(-1)
    joint_limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
    reward = (
      tracking + cfg.w_action_rate * action_rate + cfg.w_joint_limit * joint_limit
    ) * CONTROL_DT

    if terms:
      return reward, {
        "track": tracking * CONTROL_DT,
        "e_body_pos": body_pos_error,
        "e_body_ori": body_rot_error,
        "e_root_pos": root_pos_error,
        "action_rate": action_rate,
        "joint_limit": joint_limit,
      }
    return reward

  def termination(self, state, reference_frame: torch.Tensor) -> torch.Tensor:
    p, rot = state[0], state[1]
    reference_p = self.motion.root_p[reference_frame]
    reference_rot = self.motion.root_R[reference_frame]
    height_error = (reference_p[..., 2] - p[..., 2]).abs()
    tilt_error = (reference_rot[..., 2, 2] - rot[..., 2, 2]).abs()
    return (height_error > self.cfg.term_height) | (tilt_error > self.cfg.term_tilt)

  def soft_termination(
    self,
    state,
    reference_frame: torch.Tensor,
    height_scale: float = 0.02,
    tilt_scale: float = 0.04,
  ) -> torch.Tensor:
    p, rot = state[0], state[1]
    reference_p = self.motion.root_p[reference_frame]
    reference_rot = self.motion.root_R[reference_frame]
    height_error = (reference_p[..., 2] - p[..., 2]).abs()
    tilt_error = (reference_rot[..., 2, 2] - rot[..., 2, 2]).abs()
    alive = (
      1 - torch.sigmoid((height_error - self.cfg.term_height) / height_scale)
    ) * (1 - torch.sigmoid((tilt_error - self.cfg.term_tilt) / tilt_scale))
    return 1 - alive

  def critic_features(
    self,
    state,
    reference_frame: torch.Tensor,
    sim_context: torch.Tensor,
  ) -> torch.Tensor:
    motion = self.motion
    p, rot, lin_vel, ang_vel, q, qd = state
    heading = yaw_mat(rot)
    heading_t = heading.transpose(-1, -2)
    reference_p = motion.root_p[reference_frame]
    reference_rot = motion.root_R[reference_frame]
    body_p = self.kinematics.forward(p, rot, lin_vel, ang_vel, q, qd)[0]
    delta_rot = yaw_mat(rot @ reference_rot.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], reference_p[..., 2:3]), -1)
    reference_body_p = anchor.unsqueeze(-2) + (
      delta_rot.unsqueeze(-3)
      @ (motion.body_p[reference_frame] - reference_p.unsqueeze(-2)).unsqueeze(-1)
    ).squeeze(-1)
    body_error = (
      (heading_t.unsqueeze(-3) @ (reference_body_p - body_p).unsqueeze(-1))
      .squeeze(-1)
      .flatten(-2)
    )
    root_error = (heading_t @ (reference_p - p).unsqueeze(-1)).squeeze(-1)
    relative_rot = mat_to_6d(rot.transpose(-1, -2) @ reference_rot)
    return torch.cat(
      (
        invariant_features(state),
        motion.q[reference_frame] - q,
        body_error,
        root_error,
        relative_rot,
        sim_context,
      ),
      -1,
    )


CRITIC_EXTRA = critic_extra_dim(29, 14, XI_DIM)


def normalize_xi(xi: torch.Tensor, num_joints: int = 29) -> torch.Tensor:
  return torch.cat(
    (
      (xi[..., :1] - 0.75) / 0.45,
      xi[..., 1 : 1 + num_joints] / 0.01,
      xi[..., 1 + num_joints : 4 + num_joints] / 0.05,
    ),
    -1,
  )
