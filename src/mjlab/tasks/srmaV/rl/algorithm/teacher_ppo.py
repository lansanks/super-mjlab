"""PPO wrapper that publishes SRMAV teacher diagnostics."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab.tasks.srmaV.rl.network.teacher_actor import (
  SRMAVExternalOutput,
  SRMAVTeacherActorModel,
)


def _parameter_grad_norm(module: nn.Module) -> float:
  squared_norm = 0.0
  for parameter in module.parameters():
    if parameter.grad is not None:
      squared_norm += float(parameter.grad.detach().square().sum())
  return math.sqrt(squared_norm)


def _effective_rank(latent: torch.Tensor) -> float:
  centered = latent - latent.mean(dim=0, keepdim=True)
  covariance = centered.T @ centered / max(latent.shape[0] - 1, 1)
  eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
  return float(
    eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1.0e-12)
  )


@torch.no_grad()
def compute_teacher_metrics(
  actor: SRMAVTeacherActorModel,
  observations: TensorDict,
  sample_size: int = 4096,
) -> dict[str, float]:
  """Compute latent, external-input and action-sensitivity diagnostics."""
  batch_size = observations.batch_size[0]
  sample_count = min(int(sample_size), batch_size)
  if sample_count == 0:
    return {}
  stride = max(1, batch_size // sample_count)
  indices = torch.arange(
    0,
    batch_size,
    stride,
    device=observations.device,
  )[:sample_count]
  sample = cast(TensorDict, observations[indices])

  actor.get_latent(sample)
  output: SRMAVExternalOutput = actor.last_external_output
  height_latent = output.height_latent
  privilege_latent = output.privilege_latent
  baseline_actions = actor.action_mean_with_latents(
    sample,
    height_latent,
    privilege_latent,
  )
  reverse_ids = torch.arange(
    height_latent.shape[0] - 1,
    -1,
    -1,
    device=height_latent.device,
  )
  zero_height_actions = actor.action_mean_with_latents(
    sample,
    torch.zeros_like(height_latent),
    privilege_latent,
  )
  shuffled_height_actions = actor.action_mean_with_latents(
    sample,
    height_latent[reverse_ids],
    privilege_latent,
  )
  zero_privilege_actions = actor.action_mean_with_latents(
    sample,
    height_latent,
    torch.zeros_like(privilege_latent),
  )
  shuffled_privilege_actions = actor.action_mean_with_latents(
    sample,
    height_latent,
    privilege_latent[reverse_ids],
  )

  return {
    "Metrics/srmaV_latent_effective_rank": _effective_rank(output.latent),
    "Metrics/srmaV_height_latent_effective_rank": _effective_rank(height_latent),
    "Metrics/srmaV_privilege_latent_effective_rank": _effective_rank(privilege_latent),
    "Metrics/srmaV_height_map_std": float(
      output.height_map.std(dim=0, correction=0).mean()
    ),
    "Metrics/srmaV_friction_std": float(
      output.friction.std(dim=0, correction=0).mean()
    ),
    "Metrics/srmaV_push_fraction": float(
      (output.previous_push.abs().sum(dim=-1) > 1.0e-8).float().mean()
    ),
    "Metrics/srmaV_action_delta_height_zero": float(
      (baseline_actions - zero_height_actions).norm(dim=-1).mean()
    ),
    "Metrics/srmaV_action_delta_height_shuffle": float(
      (baseline_actions - shuffled_height_actions).norm(dim=-1).mean()
    ),
    "Metrics/srmaV_action_delta_privilege_zero": float(
      (baseline_actions - zero_privilege_actions).norm(dim=-1).mean()
    ),
    "Metrics/srmaV_action_delta_privilege_shuffle": float(
      (baseline_actions - shuffled_privilege_actions).norm(dim=-1).mean()
    ),
  }


class SRMAVTeacherPPO(PPO):
  """Standard PPO with teacher-specific metrics in ``extras["log"]``."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if not isinstance(self.actor, SRMAVTeacherActorModel):
      raise TypeError(
        "SRMAVTeacherPPO requires an SRMAVTeacherActorModel actor, got "
        f"{type(self.actor).__name__}."
      )
    self._pending_srmaV_metrics: dict[str, float] = {}

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, Any],
  ) -> None:
    metrics = extras.get("log")
    if metrics is not None and self._pending_srmaV_metrics:
      metrics.update(self._pending_srmaV_metrics)
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    observations = cast(
      TensorDict,
      self.storage.observations.flatten(0, 1),
    )
    losses = super().update()
    actor = cast(SRMAVTeacherActorModel, self.actor)
    metrics = compute_teacher_metrics(actor, observations)
    metrics["Metrics/srmaV_height_encoder_grad_norm"] = _parameter_grad_norm(
      actor.height_encoder
    )
    metrics["Metrics/srmaV_privilege_encoder_grad_norm"] = _parameter_grad_norm(
      actor.privilege_encoder
    )
    self._pending_srmaV_metrics = metrics
    return losses
