"""PPO wrapper that publishes SRMA teacher diagnostics."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab.tasks.srma.rl.network.teacher_actor import (
  SRMAExternalOutput,
  SRMATeacherActorModel,
)


def _external_grad_norm(actor: SRMATeacherActorModel) -> float:
  squared_norm = 0.0
  for parameter in actor.external_encoder.parameters():
    if parameter.grad is not None:
      squared_norm += float(parameter.grad.detach().square().sum())
  return math.sqrt(squared_norm)


@torch.no_grad()
def compute_teacher_metrics(
  actor: SRMATeacherActorModel,
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
  output: SRMAExternalOutput = actor.last_external_output
  latent = output.latent
  baseline_actions = actor(sample)
  zero_actions = actor.action_mean_with_latent(sample, torch.zeros_like(latent))
  reverse_ids = torch.arange(
    latent.shape[0] - 1,
    -1,
    -1,
    device=latent.device,
  )
  shuffled_actions = actor.action_mean_with_latent(
    sample,
    latent[reverse_ids],
  )

  centered = latent - latent.mean(dim=0, keepdim=True)
  covariance = centered.T @ centered / max(latent.shape[0] - 1, 1)
  eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
  effective_rank = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(
    1.0e-12
  )
  return {
    "Metrics/srma_latent_effective_rank": float(effective_rank),
    "Metrics/srma_height_map_std": float(
      output.height_map.std(dim=0, correction=0).mean()
    ),
    "Metrics/srma_friction_std": float(output.friction.std(dim=0, correction=0).mean()),
    "Metrics/srma_push_fraction": float(
      (output.previous_push.abs().sum(dim=-1) > 1.0e-8).float().mean()
    ),
    "Metrics/srma_action_delta_zero": float(
      (baseline_actions - zero_actions).norm(dim=-1).mean()
    ),
    "Metrics/srma_action_delta_shuffle": float(
      (baseline_actions - shuffled_actions).norm(dim=-1).mean()
    ),
  }


class SRMATeacherPPO(PPO):
  """Standard PPO with teacher-specific metrics in ``extras["log"]``."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if not isinstance(self.actor, SRMATeacherActorModel):
      raise TypeError(
        "SRMATeacherPPO requires an SRMATeacherActorModel actor, got "
        f"{type(self.actor).__name__}."
      )
    self._pending_srma_metrics: dict[str, float] = {}

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, Any],
  ) -> None:
    metrics = extras.get("log")
    if metrics is not None and self._pending_srma_metrics:
      metrics.update(self._pending_srma_metrics)
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    observations = cast(
      TensorDict,
      self.storage.observations.flatten(0, 1),
    )
    losses = super().update()
    actor = cast(SRMATeacherActorModel, self.actor)
    metrics = compute_teacher_metrics(actor, observations)
    metrics["Metrics/srma_external_grad_norm"] = _external_grad_norm(actor)
    self._pending_srma_metrics = metrics
    return losses
