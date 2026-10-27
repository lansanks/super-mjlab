"""Extraction of the SuperMimic state, context and goal from an mjlab env."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .rotations import quat_to_rot6d

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TermSlicer:
  """Map named terms of a concatenated observation group to flat slices."""

  def __init__(self, obs_manager, group: str):
    self.group = group
    self.names: list[str] = list(obs_manager.active_terms[group])
    dims = obs_manager.group_obs_term_dim[group]
    self.slices: dict[str, slice] = {}
    start = 0
    for name, dim in zip(self.names, dims, strict=True):
      size = int(np.prod(dim))
      self.slices[name] = slice(start, start + size)
      start += size
    self.total = start

  def size(self, name: str) -> int:
    s = self.slices.get(name)
    return 0 if s is None else s.stop - s.start

  def get(self, obs: torch.Tensor, name: str, dim: int | None = None) -> torch.Tensor:
    s = self.slices.get(name)
    if s is not None:
      return obs[..., s]
    if dim is None:
      return torch.empty(obs.shape[0], 0, device=obs.device)
    return torch.zeros(obs.shape[0], dim, device=obs.device)


class StateAssembler:
  """Assemble world-model state / context / goal from mjlab observations."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.device = torch.device(env.device)
    self.robot = env.scene["robot"]
    self.actor = TermSlicer(env.observation_manager, "actor")
    self.critic = TermSlicer(env.observation_manager, "critic")
    self.action_dim = env.action_manager.total_action_dim
    self.goal_dim = self.actor.size("command")
    self.obs_dim = self.actor.total - self.goal_dim
    self.xi_dim = self.critic.size("height_scan")
    self.push_dim = 6
    self.control_dt = env.step_dt
    self.num_joints = self.robot.data.joint_pos.shape[-1]
    if self.num_joints != 12:
      raise ValueError(
        f"SuperMimic Go1 state expects 12 joints, got {self.num_joints}."
      )
    self._setup_push_sampler()

  # ------------------------------------------------------------------ state
  def state(self, obs_critic: torch.Tensor) -> torch.Tensor:
    pose = self.robot.data.root_link_pose_w
    vel = self.robot.data.root_link_vel_w
    q = self.robot.data.joint_pos
    qd = self.robot.data.joint_vel
    rot6 = quat_to_rot6d(pose[..., 3:7])
    # Use the position relative to the environment origin so that terrain tiles
    # placed at different world offsets share one dynamics model.
    pos = pose[..., 0:3] - self.env.scene.env_origins
    contact = self.critic.get(obs_critic, "foot_contact", 4).float()
    air = self.critic.get(obs_critic, "foot_air_time", 4).float()
    foot_h = self.critic.get(obs_critic, "foot_height", 4).float()
    return torch.cat((pos, rot6, vel, q, qd, contact, air, foot_h), -1)

  def xi(self, obs_critic: torch.Tensor) -> torch.Tensor:
    if self.xi_dim == 0:
      return torch.empty(obs_critic.shape[0], 0, device=self.device)
    return self.critic.get(obs_critic, "height_scan", self.xi_dim).float()

  def push(self, obs_critic: torch.Tensor) -> torch.Tensor:
    return self.critic.get(obs_critic, "push_velocity", self.push_dim).float()

  def split_actor(self, obs_actor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the actor observation into (proprioception without command, command)."""
    goal = self.actor.get(obs_actor, "command", self.goal_dim)
    parts = [
      obs_actor[..., self.actor.slices[name]]
      for name in self.actor.names
      if name != "command"
    ]
    return torch.cat(parts, -1), goal

  # ------------------------------------------------------------------ push
  def _setup_push_sampler(self) -> None:
    self.push_low = torch.zeros(6, device=self.device)
    self.push_high = torch.zeros(6, device=self.device)
    self.push_prob = 0.0
    event = self.env.cfg.events.get("push_robot")
    if event is None or event.params is None:
      return
    ranges = event.params.get("velocity_range")
    if not ranges:
      return
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    low = [float(ranges[k][0]) for k in keys]
    high = [float(ranges[k][1]) for k in keys]
    self.push_low = torch.tensor(low, device=self.device)
    self.push_high = torch.tensor(high, device=self.device)
    interval = event.interval_range_s
    mean_interval = float(sum(interval) / 2) if interval is not None else 2.0
    self.push_prob = float(self.control_dt / max(mean_interval, 1e-6))

  def sample_push(self, batch: int) -> torch.Tensor:
    """Sample the exogenous velocity kicks used during imagination."""
    out = torch.zeros(batch, 6, device=self.device)
    if self.push_prob <= 0.0:
      return out
    hit = torch.rand(batch, device=self.device) < self.push_prob
    n = int(hit.sum())
    if n == 0:
      return out
    low = self.push_low.expand(n, 6)
    high = self.push_high.expand(n, 6)
    out[hit] = low + (high - low) * torch.rand(n, 6, device=self.device)
    return out
