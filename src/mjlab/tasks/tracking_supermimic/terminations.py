"""Termination terms for LAFAN1 SuperMimic tracking."""

from __future__ import annotations

import torch

from .rotations import quat_to_mat


def _command(env):
  return env.command_manager.get_term("motion")


def _next_reference_frame(env) -> torch.Tensor:
  command = _command(env)
  return torch.minimum(command.time_steps + 1, command.clip_end)


def _robot_state(env):
  robot = env.scene["robot"]
  pose = robot.data.root_link_pose_w
  vel = robot.data.root_link_vel_w
  return (
    pose[:, :3] - env.scene.env_origins,
    quat_to_mat(pose[:, 3:7]),
    vel[:, :3],
    vel[:, 3:6],
    robot.data.joint_pos,
    robot.data.joint_vel,
  )


def bad_anchor_pos(env, threshold: float = 0.25) -> torch.Tensor:
  command = _command(env)
  reference = command.motion.root_p[_next_reference_frame(env)]
  robot_pos = _robot_state(env)[0]
  return (reference[:, 2] - robot_pos[:, 2]).abs() > threshold


def bad_anchor_ori(env, threshold: float = 0.8) -> torch.Tensor:
  command = _command(env)
  reference = command.motion.root_R[_next_reference_frame(env)]
  robot_rot = _robot_state(env)[1]
  return (reference[:, 2, 2] - robot_rot[:, 2, 2]).abs() > threshold


def motion_clip_end(env) -> torch.Tensor:
  command = _command(env)
  return command.time_steps + 1 >= command.clip_end
