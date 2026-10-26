"""Rollout storage for the vision-conditioned SRMAV student."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass
class SRMAVVisualStudentTransition:
  """One transition recorded during visual student training."""

  obs_actor: torch.Tensor
  obs_depth: torch.Tensor
  teacher_action: torch.Tensor
  teacher_velocity: torch.Tensor
  teacher_height_latent: torch.Tensor
  teacher_privilege_latent: torch.Tensor
  behavior_action: torch.Tensor
  is_teacher: torch.Tensor
  reward: torch.Tensor | None = None


class SRMAVVisualStudentRolloutStorage:
  """Store proprioception, depth, and split teacher labels for DAgger."""

  def __init__(
    self,
    obs: TensorDict,
    num_envs: int,
    num_transitions_per_env: int,
    action_dim: int,
    velocity_dim: int,
    height_latent_dim: int,
    privilege_latent_dim: int,
    hidden_shape: tuple[int, ...],
    obs_group: str = "actor",
    depth_group: str = "student_depth",
    device: str = "cpu",
    depth_dtype: torch.dtype = torch.float16,
  ) -> None:
    for group in (obs_group, depth_group):
      if group not in obs:
        raise ValueError(
          f"Storage observation group '{group}' is missing: {list(obs.keys())}."
        )
    actor_obs = obs[obs_group]
    depth_obs = obs[depth_group]
    if not isinstance(actor_obs, torch.Tensor):
      raise TypeError(f"Observation '{obs_group}' must be a tensor.")
    if not isinstance(depth_obs, torch.Tensor):
      raise TypeError(f"Observation '{depth_group}' must be a tensor.")
    if actor_obs.ndim != 2:
      raise ValueError(
        f"Expected actor observation shape [B, D], got {tuple(actor_obs.shape)}."
      )
    if depth_obs.ndim != 4:
      raise ValueError(
        f"Expected depth shape [B, H, W, C], got {tuple(depth_obs.shape)}."
      )
    if actor_obs.shape[0] != num_envs or depth_obs.shape[0] != num_envs:
      raise ValueError(f"Expected {num_envs} environments for all observations.")

    self.device = device
    self.num_envs = int(num_envs)
    self.num_transitions_per_env = int(num_transitions_per_env)
    self.obs_dim = int(actor_obs.shape[-1])
    self.depth_shape = tuple(depth_obs.shape[1:])
    self.action_dim = int(action_dim)
    self.velocity_dim = int(velocity_dim)
    self.height_latent_dim = int(height_latent_dim)
    self.privilege_latent_dim = int(privilege_latent_dim)
    self.hidden_shape = tuple(hidden_shape)
    if len(self.hidden_shape) != 4 or self.hidden_shape[2] != self.num_envs:
      raise ValueError(
        "hidden_shape must be [beliefs, layers, num_envs, hidden_dim], got "
        f"{self.hidden_shape}."
      )

    shape = (self.num_transitions_per_env, self.num_envs)
    self.obs_actor = torch.zeros(
      *shape, self.obs_dim, device=device, dtype=actor_obs.dtype
    )
    self.obs_depth = torch.zeros(
      *shape,
      *self.depth_shape,
      device=device,
      dtype=depth_dtype,
    )
    self.teacher_action = torch.zeros(
      *shape, self.action_dim, device=device, dtype=actor_obs.dtype
    )
    self.teacher_velocity = torch.zeros(
      *shape, self.velocity_dim, device=device, dtype=actor_obs.dtype
    )
    self.teacher_height_latent = torch.zeros(
      *shape,
      self.height_latent_dim,
      device=device,
      dtype=actor_obs.dtype,
    )
    self.teacher_privilege_latent = torch.zeros(
      *shape,
      self.privilege_latent_dim,
      device=device,
      dtype=actor_obs.dtype,
    )
    self.behavior_action = torch.zeros_like(self.teacher_action)
    self.is_teacher = torch.zeros(*shape, 1, device=device, dtype=torch.bool)
    self.dones = torch.zeros(*shape, 1, device=device, dtype=torch.bool)
    self.rewards = torch.zeros(*shape, 1, device=device, dtype=actor_obs.dtype)
    self.initial_hidden = torch.zeros(
      *self.hidden_shape, device=device, dtype=actor_obs.dtype
    )
    self.step = 0

  def set_initial_hidden(self, hidden: torch.Tensor) -> None:
    if hidden.shape != self.initial_hidden.shape:
      raise ValueError(
        f"Expected hidden shape {tuple(self.initial_hidden.shape)}, "
        f"got {tuple(hidden.shape)}."
      )
    self.initial_hidden.copy_(hidden)

  def add_transition(self, transition: SRMAVVisualStudentTransition) -> None:
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("SRMAV visual student rollout buffer overflow.")
    self.obs_actor[self.step].copy_(transition.obs_actor)
    self.obs_depth[self.step].copy_(transition.obs_depth)
    self.teacher_action[self.step].copy_(transition.teacher_action)
    self.teacher_velocity[self.step].copy_(transition.teacher_velocity)
    self.teacher_height_latent[self.step].copy_(transition.teacher_height_latent)
    self.teacher_privilege_latent[self.step].copy_(transition.teacher_privilege_latent)
    self.behavior_action[self.step].copy_(transition.behavior_action)
    self.is_teacher[self.step].copy_(transition.is_teacher.view(-1, 1))
    if transition.reward is not None:
      self.rewards[self.step].copy_(transition.reward.reshape(-1, 1))
    self.step += 1

  def add_done(self, dones: torch.Tensor) -> None:
    if self.step == 0:
      raise RuntimeError("Cannot add done flags before a transition.")
    self.dones[self.step - 1].copy_(dones.reshape(-1, 1))

  def clear(self) -> None:
    self.step = 0

  def env_mini_batches(self, num_mini_batches: int) -> Iterator[torch.Tensor]:
    """Yield shuffled environment indices without changing time order."""
    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be positive.")
    if num_mini_batches > self.num_envs:
      raise ValueError(
        f"num_mini_batches={num_mini_batches} exceeds num_envs={self.num_envs}."
      )
    indices = torch.randperm(self.num_envs, device=self.device)
    base = self.num_envs // num_mini_batches
    remainder = self.num_envs % num_mini_batches
    start = 0
    for batch_index in range(num_mini_batches):
      size = base + (1 if batch_index < remainder else 0)
      yield indices[start : start + size]
      start += size
