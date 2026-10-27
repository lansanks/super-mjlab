"""Multi-clip LAFAN1 command with SuperMimic future-goal support."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
)
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from .mdp import MAX_STEP_REWARD
from .motion import LafanMotionLibrary
from .rotations import mat_to_quat


class LafanMotionCommand(MotionCommand):
  """Tracking command that samples complete clips instead of one continuous file."""

  cfg: LafanMotionCommandCfg

  def __init__(self, cfg: LafanMotionCommandCfg, env):
    CommandTerm.__init__(self, cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_ids, found_names = self.robot.find_bodies(
      cfg.body_names, preserve_order=True
    )
    self.body_indexes = torch.tensor(
      self.body_ids, dtype=torch.long, device=self.device
    )
    if tuple(found_names) != tuple(cfg.body_names):
      raise ValueError(
        "Tracking body names do not match the robot body order: "
        f"expected {cfg.body_names}, found {found_names}"
      )
    self.motion = LafanMotionLibrary(cfg.motion_file, self.device)
    if tuple(self.motion.body_names) != tuple(cfg.body_names):
      raise ValueError(
        "Motion and robot bodies differ: "
        f"motion={self.motion.body_names}, robot={cfg.body_names}"
      )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.start_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.bin_return = torch.zeros(self.motion.num_bins, device=self.device)
    self.bin_weight_average = torch.full(
      (self.motion.num_bins,),
      1.0 / self.motion.num_bins,
      device=self.device,
    )
    self.return_max = MAX_STEP_REWARD / (1 - 0.99)
    self.metrics = {
      "error_anchor_pos": torch.zeros(self.num_envs, device=self.device),
      "error_anchor_rot": torch.zeros(self.num_envs, device=self.device),
      "error_body_pos": torch.zeros(self.num_envs, device=self.device),
      "error_body_rot": torch.zeros(self.num_envs, device=self.device),
    }
    self._pending_forward = False

  @property
  def command(self) -> torch.Tensor:
    return torch.cat((self.joint_pos, self.joint_vel), dim=-1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.q[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.qd[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.motion.body_p[self.time_steps] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return mat_to_quat(self.motion.body_R[self.time_steps])

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_v[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def clip_end(self) -> torch.Tensor:
    return self.motion.clip_end[self.clip_ids]

  @property
  def clip_names(self) -> list[str]:
    return self.motion.names

  @property
  def num_clips(self) -> int:
    return self.motion.num_clips

  def future_goal(self, count: int) -> torch.Tensor:
    return self.motion.goal(self.time_steps, count)

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot.data.root_link_pos_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_w - self.robot.data.body_link_pos_w[:, self.body_ids],
      dim=-1,
    ).mean(-1)

  def _sample_clip_and_frame(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    mode = self.cfg.sampling_mode
    if mode == "all":
      clips = env_ids % self.motion.num_clips
      frames = torch.zeros_like(clips)
    elif mode == "start":
      clips = torch.zeros_like(env_ids)
      frames = torch.zeros_like(env_ids)
    elif mode == "adaptive":
      soft = torch.softmax(-25.0 * self.bin_return / self.return_max, 0)
      self.bin_weight_average.mul_(0.999).add_(0.001 * soft)
      probability = (
        0.5 / self.motion.num_bins
        + 0.4 * soft
        + 0.1 * self.bin_weight_average / self.bin_weight_average.sum()
      )
      bins = torch.multinomial(probability, len(env_ids), replacement=True)
      start = self.motion.bin_start[bins]
      end = torch.minimum(start + 50, self.motion.bin_end[bins])
      frames = (
        start
        + (torch.rand(len(env_ids), device=self.device) * (end - start).float()).long()
      )
      clips = self.motion.bin_clip[bins]
      self.start_bin[env_ids] = bins
      return clips, frames
    else:
      clips = torch.randint(
        0, self.motion.num_clips, (len(env_ids),), device=self.device
      )
      lengths = self.motion.clip_len[clips]
      frames = (torch.rand(len(env_ids), device=self.device) * lengths.float()).long()
      frames = torch.minimum(frames, lengths - 1)
    return clips, frames

  def record_episode_return(
    self,
    env_ids: torch.Tensor,
    discounted_return: torch.Tensor,
    undiscounted_return: torch.Tensor,
    lengths: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
  ) -> None:
    """Update the reset sampler after completed episodes."""
    timeout_bootstrap = torch.where(
      terminated,
      torch.zeros_like(lengths),
      gamma**lengths * (undiscounted_return / lengths.clamp(min=1)) / (1 - gamma),
    )
    estimate = discounted_return + timeout_bootstrap
    bins = self.start_bin[env_ids]
    self.bin_return.index_put_((bins,), 0.8 * self.bin_return[bins] + 0.2 * estimate)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    clips, frames = self._sample_clip_and_frame(env_ids)
    self.clip_ids[env_ids] = clips
    self.time_steps[env_ids] = frames
    self.start_bin[env_ids] = self.motion.frame_bin[frames]

    root_pos = self.motion.root_p[frames].clone()
    root_quat = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.motion.root_v[frames].clone()
    root_ang_vel = self.motion.root_w[frames].clone()
    joint_pos = self.motion.q[frames].clone()
    joint_vel = self.motion.qd[frames].clone()

    pose_ranges = torch.tensor(
      [
        self.cfg.pose_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    pose_noise = sample_uniform(
      pose_ranges[:, 0],
      pose_ranges[:, 1],
      (len(env_ids), 6),
      self.device,
    )
    root_pos += pose_noise[:, :3]
    orientation_delta = quat_from_euler_xyz(
      pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]
    )
    root_quat = quat_mul(orientation_delta, root_quat)

    velocity_ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    velocity_noise = sample_uniform(
      velocity_ranges[:, 0],
      velocity_ranges[:, 1],
      (len(env_ids), 6),
      self.device,
    )
    root_lin_vel += velocity_noise[:, :3]
    root_ang_vel += velocity_noise[:, 3:]
    joint_pos += sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      joint_pos.shape,
      self.device,
    )

    root_pos += self._env.scene.env_origins[env_ids]
    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_quat,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, limits[:, :, 0], limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(
      torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel), dim=-1),
      env_ids=env_ids,
    )
    self.robot.reset(env_ids=env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      self.time_steps += 1
      self.time_steps.clamp_max_(self.clip_end)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self.time_steps[env_ids] = frame
    self.clip_ids[env_ids] = self.motion.frame_clip[frame]
    root_pos = self.motion.root_p[frame] + self._env.scene.env_origins[env_ids]
    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      mat_to_quat(self.motion.root_R[frame]),
      self.motion.root_v[frame],
      self.motion.root_w[frame],
      self.motion.q[frame],
      self.motion.qd[frame],
    )


@dataclass(kw_only=True)
class LafanMotionCommandCfg(MotionCommandCfg):
  def build(self, env) -> LafanMotionCommand:
    return LafanMotionCommand(self, env)
