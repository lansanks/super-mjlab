"""CENet estimation and reconstruction diagnostics (algorithm side).

Two families of numbers monitor DreamWaQ's CENet, both expressed as percentages
of the signal's own variation so they are comparable across terms and runs:

- the body-velocity estimate: ``Metrics/est_vel_mean_pct``;
- the decoder reconstruction, per observation term:
  ``Metrics/recon_<term>_mean_pct``.

The value is ``100 * rms_d((x_hat_d - x_d) / sigma_d)``: each dimension's error
divided by the running standard deviation of that dimension, combined in
quadrature over the term's dimensions. 100% therefore reads as "the error is as
large as the typical fluctuation of the signal". The reconstruction is decoded
from the posterior mean ``mu``, so the numbers are deterministic and free of VAE
sampling noise (the training loss uses a posterior sample instead).

Note: the scales are running statistics. They start at their unit initialization
and only reflect the data after the first rollout, so the very first logged
percentages are anchored to a unit scale rather than to the signal.

These metrics cannot be environment metric terms (the
``mjlab.tasks.dreamwaq.mdp`` style): they need the CENet forward pass and its
running statistics, which only the algorithm sees. ``DreamwaqPPO`` computes
them every control step and writes them into ``extras["log"]``, mjlab's channel
for ``Metrics/...`` scalars.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization

from mjlab.tasks.dreamwaq.rl.network.cenet import CENet

DEFAULT_RECON_TERMS: tuple[tuple[str, int], ...] = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 12),
  ("joint_vel", 12),
  ("actions", 12),
  ("command", 3),
)
"""Layout of the ``cenet_target_o`` group of the Go1 DreamWaQ task."""


class CenetDecoder(Protocol):
  """Structural type of the CENet parts the metrics need."""

  @property
  def obs_normalizer(self) -> nn.Module:
    """Running normalization applied to the reconstruction target."""
    ...

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    """Decode context vectors into normalized observations."""
    ...


class CenetMetrics:
  """Per-step CENet errors as percentages of each signal's scale."""

  def __init__(
    self,
    obs_dim: int,
    velocity_dim: int = CENet.velocity_dim,
    recon_terms: dict[str, int] | tuple[tuple[str, int], ...] | None = None,
    device: str = "cpu",
  ) -> None:
    """Initialize the metric helper.

    Args:
      obs_dim: Dimension of a single proprioceptive observation ``o_t``.
      velocity_dim: Dimension of the estimated body velocity.
      recon_terms: Ordered ``(name, dim)`` layout of the reconstructed
        observation. Defaults to :data:`DEFAULT_RECON_TERMS`.
      device: Device of the running velocity statistics.
    """
    if recon_terms is None:
      terms = DEFAULT_RECON_TERMS
    elif isinstance(recon_terms, dict):
      terms = tuple(recon_terms.items())
    else:
      terms = tuple(recon_terms)
    total = sum(int(dim) for _, dim in terms)
    if total != obs_dim:
      raise ValueError(
        f"Reconstruction term dimensions sum to {total}, expected the policy "
        f"observation dimension {obs_dim}."
      )
    self.terms: tuple[tuple[str, int], ...] = tuple(
      (str(name), int(dim)) for name, dim in terms
    )
    self._slices: dict[str, slice] = {}
    start = 0
    for name, dim in self.terms:
      self._slices[name] = slice(start, start + dim)
      start += dim
    self.velocity_dim = int(velocity_dim)
    self.velocity_normalizer = EmpiricalNormalization(self.velocity_dim).to(device)
    self._last_mu: torch.Tensor | None = None
    self._mu_valid: torch.Tensor | None = None

  def state_dict(self) -> dict[str, torch.Tensor]:
    """Return the running velocity statistics for checkpointing."""
    return self.velocity_normalizer.state_dict()

  def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
    """Restore the running velocity statistics from a checkpoint."""
    self.velocity_normalizer.load_state_dict(state_dict)

  def update_velocity_statistics(self, velocity: torch.Tensor) -> None:
    """Update the running per-axis scale from ground-truth velocities (B, 3)."""
    self.velocity_normalizer.update(velocity.reshape(-1, self.velocity_dim))

  def velocity_metrics(
    self, estimate: torch.Tensor, target: torch.Tensor
  ) -> dict[str, float]:
    """Velocity estimate errors as percentages of the per-axis scale."""
    scale = self.velocity_normalizer.std + self.velocity_normalizer.eps
    return self._mean_percent((estimate - target) / scale, "est_vel")

  def latent_metrics(self, mu: torch.Tensor, done: torch.Tensor) -> dict[str, float]:
    """Latent drift and spread: is ``z`` a slow context or a fast memory?

    - ``latent_drift``: mean over environments of ``||mu_t - mu_{t-1}||``, i.e.
      how fast the context moves from one control step to the next. Steps right
      after an episode reset are skipped (the context is discontinuous there).
    - ``latent_spread``: mean over environments of the L2 norm of ``mu``'s
      deviation from the batch mean, i.e. how different the context is between
      robots. Same units as the drift, so ``drift / spread`` is directly
      comparable: small means the latent is dominated by between-robot
      (context) differences, large means it is dominated by fast motion.

    A context variable that captures terrain/contact properties should drift
    slowly relative to its spread across environments; a latent that mainly
    memorizes the recent gait phase drifts fast instead. The drift/spread ratio
    therefore tells apart "bought context" from "bought memory" when tuning β.
    """
    spread = (mu - mu.mean(dim=0, keepdim=True)).norm(dim=-1).mean()
    metrics = {"Metrics/latent_spread": float(spread)}
    if self._last_mu is None or self._last_mu.shape != mu.shape:
      self._last_mu = mu.detach().clone()
      # Reference established: it is valid for every environment that did not
      # reset this step, so the next call can measure the drift.
      self._mu_valid = ~done.reshape(-1).bool()
      return metrics
    valid = self._mu_valid
    assert valid is not None
    if bool(valid.any()):
      drift = (mu - self._last_mu).norm(dim=-1)
      metrics["Metrics/latent_drift"] = float(drift[valid].mean())
    self._last_mu = mu.detach().clone()
    self._mu_valid = ~done.reshape(-1).bool()
    return metrics

  def reconstruction_metrics(
    self,
    cenet: CenetDecoder,
    mu: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
  ) -> dict[str, float]:
    """Next-observation reconstruction errors per term, in percent.

    Args:
      cenet: The estimator whose decoder is evaluated.
      mu: Posterior mean of the context vector, shape (B, latent_dim).
      target: Clean next observation ``o_{t+1}``, shape (B, obs_dim).
      valid: Boolean mask of environments whose next observation belongs to the
        same episode (terminal steps are followed by a reset frame).
    """
    with torch.no_grad():
      decoded = cenet.decode(mu)
      normalizer = cenet.obs_normalizer
      if isinstance(normalizer, EmpiricalNormalization):
        # Normalized errors are already in units of the running per-dim std.
        error = decoded - normalizer(target)
      else:
        # Fall back to the batch scale, matching EmpiricalNormalization's
        # population variance.
        scale = target.std(dim=0, correction=0).clamp_min(1e-6)
        error = (decoded - target) / scale
      metrics: dict[str, float] = {}
      for name, dim_slice in self._slices.items():
        metrics.update(self._mean_percent(error[:, dim_slice], f"recon_{name}", valid))
    return metrics

  @staticmethod
  def _mean_percent(
    error: torch.Tensor, prefix: str, valid: torch.Tensor | None = None
  ) -> dict[str, float]:
    """Mean over environments of the per-environment RMS percentage."""
    per_env = error.square().mean(dim=-1).sqrt() * 100.0
    if valid is not None:
      per_env = per_env[valid]
    if per_env.numel() == 0:
      return {}
    return {f"Metrics/{prefix}_mean_pct": float(per_env.mean())}
