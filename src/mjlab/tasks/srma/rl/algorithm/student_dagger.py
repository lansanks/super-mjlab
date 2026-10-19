"""Frozen-teacher DAgger training for the deployable SRMA student."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_class, resolve_obs_groups
from tensordict import TensorDict

from mjlab.tasks.srma.rl.network.student_actor import (
  SRMAStudentActorModel,
  SRMAStudentSequenceOutput,
)
from mjlab.tasks.srma.rl.network.teacher_actor import SRMATeacherActorModel
from mjlab.tasks.srma.rl.storage import (
  SRMAStudentRolloutStorage,
  SRMAStudentTransition,
)

TEACHER_OBS_GROUPS = (
  "teacher_height_map",
  "teacher_friction",
  "teacher_prev_push",
  "teacher_velocity_gt",
)


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected observation '{key}' to be a tensor.")
  return value


class SRMAStudentDagger:
  """Train the student estimator and deterministic actor from a frozen teacher."""

  def __init__(
    self,
    student: SRMAStudentActorModel,
    teacher: SRMATeacherActorModel | None,
    storage: SRMAStudentRolloutStorage,
    pretrain_iterations: int = 20,
    num_learning_epochs: int = 4,
    num_mini_batches: int = 4,
    learning_rate: float = 5.0e-4,
    velocity_loss_coef: float = 1.0,
    latent_loss_coef: float = 1.0,
    action_loss_coef: float = 1.0,
    max_grad_norm: float = 1.0,
    teacher_fraction_start: float = 1.0,
    teacher_fraction_min: float = 0.2,
    max_iterations: int = 500,
    teacher_checkpoint: str = "",
    device: str = "cpu",
    multi_gpu_cfg: dict | None = None,
  ) -> None:
    self.student = student.to(device)
    self.teacher = teacher.to(device) if teacher is not None else None
    if not isinstance(self.student, SRMAStudentActorModel):
      raise TypeError(
        "SRMAStudentDagger requires SRMAStudentActorModel, got "
        f"{type(self.student).__name__}."
      )
    if self.teacher is not None and not isinstance(self.teacher, SRMATeacherActorModel):
      raise TypeError(
        "SRMAStudentDagger requires SRMATeacherActorModel, got "
        f"{type(self.teacher).__name__}."
      )
    if multi_gpu_cfg is not None:
      raise NotImplementedError(
        "SRMA student DAgger currently supports single-GPU training only."
      )
    if pretrain_iterations < 0:
      raise ValueError("pretrain_iterations must be non-negative.")
    if max_iterations <= pretrain_iterations:
      raise ValueError(
        "max_iterations must be greater than pretrain_iterations, got "
        f"{max_iterations} and {pretrain_iterations}."
      )
    if not 0.0 <= teacher_fraction_min <= teacher_fraction_start <= 1.0:
      raise ValueError(
        "teacher fractions must satisfy "
        "0 <= min <= start <= 1, got "
        f"{teacher_fraction_min} and {teacher_fraction_start}."
      )

    self.device = device
    self.storage = storage
    self.pretrain_iterations = int(pretrain_iterations)
    self.num_learning_epochs = int(num_learning_epochs)
    self.num_mini_batches = int(num_mini_batches)
    self.learning_rate = float(learning_rate)
    self.velocity_loss_coef = float(velocity_loss_coef)
    self.latent_loss_coef = float(latent_loss_coef)
    self.action_loss_coef = float(action_loss_coef)
    self.max_grad_norm = float(max_grad_norm)
    self.teacher_fraction_start = float(teacher_fraction_start)
    self.teacher_fraction_min = float(teacher_fraction_min)
    self.max_iterations = int(max_iterations)
    self.teacher_checkpoint = str(teacher_checkpoint)
    self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)
    self.num_updates = 0
    self.actual_dagger_round = 0
    self._teacher_fraction = self.teacher_fraction_for_iteration(0)
    self.behavior_is_teacher = torch.ones(
      self.storage.num_envs, 1, dtype=torch.bool, device=device
    )
    self.transition: SRMAStudentTransition | None = None
    self._velocity_scale = torch.ones(self.student.velocity_dim, device=device)
    self._action_scale = torch.ones(self.storage.action_dim, device=device)
    self._latest_metrics: dict[str, torch.Tensor] = {}

    if self.teacher is not None:
      self.teacher.eval()
      self.teacher.requires_grad_(False)

  @property
  def teacher_fraction(self) -> float:
    return self._teacher_fraction

  def teacher_fraction_for_iteration(self, iteration: int) -> float:
    """Return the configured stepwise teacher ratio for an absolute iteration."""
    if iteration < 0:
      raise ValueError("iteration must be non-negative.")
    decile = math.floor(10.0 * iteration / self.max_iterations)
    return max(
      self.teacher_fraction_min,
      self.teacher_fraction_start - 0.1 * decile,
    )

  def _sample_behavior_roles(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    fraction = self.teacher_fraction_for_iteration(self.num_updates)
    self.behavior_is_teacher[env_ids] = (
      torch.rand(env_ids.numel(), 1, device=self.device) < fraction
    )

  def _student_parameters(self) -> Iterator[nn.Parameter]:
    return self.student.parameters()

  def _actor_parameters(self) -> Iterator[nn.Parameter]:
    yield from self.student.mlp.parameters()
    if self.student.distribution is not None:
      yield from self.student.distribution.parameters()

  def _set_actor_trainable(self, enabled: bool) -> None:
    for parameter in self._actor_parameters():
      parameter.requires_grad_(enabled)

  def act(self, obs: TensorDict) -> torch.Tensor:
    if self.teacher is None:
      raise RuntimeError("Teacher observations are unavailable in play mode.")
    if self.storage.step == 0:
      self.storage.set_initial_hidden(self.student.get_hidden_state())

    with torch.no_grad():
      self.teacher.get_latent(obs)
      teacher_output = self.teacher.last_external_output
      teacher_action_mean = self.teacher.action_mean_with_latent(
        obs, teacher_output.latent
      )
      student_output = self.student.forward_deterministic(obs)
      action = torch.where(
        self.behavior_is_teacher,
        teacher_action_mean,
        student_output.actor_mean,
      )
      self._latest_metrics = {
        "Metrics/student_velocity_error": (
          (student_output.velocity - teacher_output.velocity)
          .norm(dim=-1)
          .mean()
          .detach()
        ),
        "Metrics/student_latent_mse": (
          (student_output.latent - teacher_output.latent).square().mean().detach()
        ),
        "Metrics/student_action_mse": (
          (student_output.actor_mean - teacher_action_mean).square().mean().detach()
        ),
      }
      self.transition = SRMAStudentTransition(
        obs_actor=_tensor(obs, self.student.obs_group).detach(),
        teacher_action=teacher_action_mean.detach(),
        teacher_velocity=teacher_output.velocity.detach(),
        teacher_latent=teacher_output.latent.detach(),
        behavior_action=action.detach(),
        is_teacher=self.behavior_is_teacher.clone(),
      )
    self._teacher_fraction = self.teacher_fraction_for_iteration(self.num_updates)
    return action

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, Any],
  ) -> None:
    del obs
    if self.transition is None:
      raise RuntimeError("process_env_step called before act.")
    self.transition.reward = rewards.detach()
    self.storage.add_transition(self.transition)
    self.storage.add_done(dones.detach())

    self.student.reset(dones)
    done_ids = dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    self._sample_behavior_roles(done_ids)
    self.transition = None

    metrics = extras.get("log")
    if metrics is None:
      metrics = {}
      extras["log"] = metrics
    if self.storage.step == 1 or self.storage.step % 10 == 0:
      metrics["Metrics/student_teacher_fraction"] = float(
        self.behavior_is_teacher.float().mean()
      )
      metrics.update({key: float(value) for key, value in self._latest_metrics.items()})

  def compute_returns(self, obs: TensorDict) -> None:
    del obs

  def _update_loss_scales(self) -> None:
    velocity = self.storage.teacher_velocity.reshape(-1, self.student.velocity_dim)
    action = self.storage.teacher_action.reshape(-1, self.storage.action_dim)
    velocity_std = velocity.std(dim=0, unbiased=False).clamp_min(0.1)
    action_std = action.std(dim=0, unbiased=False).clamp_min(0.05)
    self._velocity_scale = (0.9 * self._velocity_scale + 0.1 * velocity_std).clamp_min(
      0.1
    )
    self._action_scale = (0.9 * self._action_scale + 0.1 * action_std).clamp_min(0.05)

  def _sequence_losses(
    self,
    output: SRMAStudentSequenceOutput,
    indices: torch.Tensor,
    estimator_only: bool,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity_target = self.storage.teacher_velocity[:, indices]
    latent_target = self.storage.teacher_latent[:, indices]
    velocity_loss = (
      ((output.velocity - velocity_target) / self._velocity_scale).square().mean()
    )
    latent_loss = (output.latent - latent_target).square().mean()
    if estimator_only:
      action_loss = latent_loss.new_zeros(())
    else:
      action_target = self.storage.teacher_action[:, indices]
      action_loss = (
        ((output.actor_mean - action_target) / self._action_scale).square().mean()
      )
    return velocity_loss, latent_loss, action_loss

  def update(self) -> dict[str, float]:
    if self.storage.step != self.storage.num_transitions_per_env:
      raise RuntimeError(
        "SRMA student update requires a full rollout buffer, got "
        f"{self.storage.step} steps."
      )
    self._update_loss_scales()
    estimator_only = self.num_updates < self.pretrain_iterations
    self._set_actor_trainable(not estimator_only)
    self.student.train()
    if self.teacher is not None:
      self.teacher.eval()

    totals = {"velocity": 0.0, "latent": 0.0, "action": 0.0}
    updates = 0
    for _ in range(self.num_learning_epochs):
      for indices in self.storage.env_mini_batches(self.num_mini_batches):
        output = self.student.forward_sequence(
          self.storage.obs_actor[:, indices],
          self.storage.initial_hidden[:, indices],
          self.storage.dones[:, indices],
        )
        velocity_loss, latent_loss, action_loss = self._sequence_losses(
          output, indices, estimator_only
        )
        loss = (
          self.velocity_loss_coef * velocity_loss + self.latent_loss_coef * latent_loss
        )
        if not estimator_only:
          loss = loss + self.action_loss_coef * action_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self._student_parameters()), self.max_grad_norm)
        self.optimizer.step()
        totals["velocity"] += float(velocity_loss.detach())
        totals["latent"] += float(latent_loss.detach())
        totals["action"] += float(action_loss.detach())
        updates += 1

    if self.student.obs_normalization:
      normalizer = self.student.obs_normalizer
      if not isinstance(normalizer, EmpiricalNormalization):
        raise TypeError("Student observation normalizer has an unexpected type.")
      normalizer.update(self.storage.obs_actor.reshape(-1, self.student.obs_dim))
    self.storage.clear()
    self.num_updates += 1
    if not estimator_only:
      self.actual_dagger_round += 1
    scale = max(updates, 1)
    return {
      "student_velocity": totals["velocity"] / scale,
      "student_latent": totals["latent"] / scale,
      "student_action": totals["action"] / scale,
    }

  def train_mode(self) -> None:
    self.student.train()
    if self.teacher is not None:
      self.teacher.eval()

  def eval_mode(self) -> None:
    self.student.eval()
    if self.teacher is not None:
      self.teacher.eval()

  def get_policy(self) -> SRMAStudentActorModel:
    return self.student

  def save(self) -> dict[str, Any]:
    saved: dict[str, Any] = {
      "student_state_dict": self.student.state_dict(),
      "teacher_state_dict": (
        self.teacher.state_dict() if self.teacher is not None else None
      ),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "infos": {
        "srma": {
          "pretrain_iterations": self.pretrain_iterations,
          "schedule_max_iterations": self.max_iterations,
          "num_updates": self.num_updates,
          "actual_dagger_round": self.actual_dagger_round,
          "teacher_fraction": self.teacher_fraction,
          "teacher_checkpoint": self.teacher_checkpoint,
          "velocity_scale": self._velocity_scale.detach().cpu().clone(),
          "action_scale": self._action_scale.detach().cpu().clone(),
        }
      },
    }
    return saved

  def load(
    self,
    loaded_dict: dict[str, Any],
    load_cfg: dict | None,
    strict: bool,
  ) -> bool:
    if load_cfg is None:
      load_cfg = {
        "student": True,
        "teacher": True,
        "optimizer": True,
        "iteration": True,
      }
    load_student = bool(load_cfg.get("student") or load_cfg.get("actor"))
    load_teacher = bool(load_cfg.get("teacher"))
    load_optimizer = bool(load_cfg.get("optimizer"))

    if load_student:
      if "student_state_dict" not in loaded_dict:
        raise KeyError("Checkpoint does not contain student_state_dict.")
      self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
    if load_teacher and self.teacher is not None:
      teacher_state = loaded_dict.get("teacher_state_dict")
      if teacher_state is None:
        teacher_state = loaded_dict.get("actor_state_dict")
      if teacher_state is None:
        raise KeyError("Checkpoint does not contain teacher parameters.")
      self.teacher.load_state_dict(teacher_state, strict=strict)
    if load_optimizer:
      if "optimizer_state_dict" not in loaded_dict:
        raise KeyError("Checkpoint does not contain optimizer_state_dict.")
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    infos = loaded_dict.get("infos") or {}
    srma = infos.get("srma")
    if load_cfg.get("iteration", False) and isinstance(srma, dict):
      self.pretrain_iterations = int(
        srma.get("pretrain_iterations", self.pretrain_iterations)
      )
      self.max_iterations = int(
        srma.get("schedule_max_iterations", self.max_iterations)
      )
      self.num_updates = int(srma.get("num_updates", self.num_updates))
      self.actual_dagger_round = int(
        srma.get("actual_dagger_round", self.actual_dagger_round)
      )
      self._teacher_fraction = float(
        srma.get("teacher_fraction", self._teacher_fraction)
      )
      if "velocity_scale" in srma:
        self._velocity_scale.copy_(srma["velocity_scale"].to(self.device))
      if "action_scale" in srma:
        self._action_scale.copy_(srma["action_scale"].to(self.device))
      self.teacher_checkpoint = str(
        srma.get("teacher_checkpoint", self.teacher_checkpoint)
      )
      self._sample_behavior_roles(
        torch.arange(self.storage.num_envs, device=self.device)
      )
    return bool(load_cfg.get("iteration", False))

  @classmethod
  def construct_algorithm(
    cls,
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> "SRMAStudentDagger":
    if cfg.get("multi_gpu") is not None:
      raise NotImplementedError(
        "SRMA student DAgger currently supports single-GPU training only."
      )
    obs_groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor"])
    student_class, student_kwargs = resolve_class(cfg["student"])
    student = cast(
      SRMAStudentActorModel,
      student_class(
        obs,
        obs_groups,
        "actor",
        env.num_actions,
        **student_kwargs,
      ).to(device),
    )

    teacher_cfg = cfg.get("teacher")
    available = [group in obs for group in TEACHER_OBS_GROUPS]
    teacher: SRMATeacherActorModel | None = None
    if teacher_cfg is not None and all(available):
      teacher_class, teacher_kwargs = resolve_class(teacher_cfg)
      teacher = cast(
        SRMATeacherActorModel,
        teacher_class(
          obs,
          obs_groups,
          "actor",
          env.num_actions,
          **teacher_kwargs,
        ).to(device),
      )
      checkpoint_path = str(cfg["teacher_checkpoint"])
      if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
      teacher_state = checkpoint.get("actor_state_dict")
      if teacher_state is None:
        raise KeyError(f"Teacher checkpoint has no actor_state_dict: {checkpoint_path}")
      teacher.load_state_dict(teacher_state, strict=True)
      teacher.eval()
      teacher.requires_grad_(False)
      student.copy_actor_from_teacher(teacher_state)
    elif any(available):
      raise ValueError(
        "Only some SRMA teacher observation groups are available: "
        f"{dict(zip(TEACHER_OBS_GROUPS, available, strict=True))}"
      )

    hidden = student.get_hidden_state()
    storage = SRMAStudentRolloutStorage(
      obs=obs,
      num_envs=env.num_envs,
      num_transitions_per_env=int(cfg["num_steps_per_env"]),
      action_dim=env.num_actions,
      velocity_dim=student.velocity_dim,
      latent_dim=student.latent_dim,
      hidden_shape=(
        int(hidden.shape[0]),
        int(hidden.shape[1]),
        int(hidden.shape[2]),
      ),
      obs_group=student.obs_group,
      device=device,
    )
    algorithm_cfg = dict(cfg["algorithm"])
    algorithm_cfg.pop("class_name", None)
    return cls(
      student=student,
      teacher=teacher,
      storage=storage,
      max_iterations=int(cfg["max_iterations"]),
      teacher_checkpoint=str(cfg.get("teacher_checkpoint", "")),
      device=device,
      multi_gpu_cfg=cfg.get("multi_gpu"),
      **algorithm_cfg,
    )
