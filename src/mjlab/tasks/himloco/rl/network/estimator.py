"""Hybrid Internal Model (HIM) estimator: two encoders plus prototypes.

HIM (paper Sections 3.2-3.3) replaces an explicit model of the environmental
disturbances with a learned *hybrid internal embedding* of the robot's
response::

    o_t^H    = [o_t, ..., o_{t-H}]        proprioceptive history (H = 5)
    v_hat, l = source(o_t^H)              explicit velocity + implicit response
    l_target = target(o_{t+1})            embedding of the successor state

The embedding is optimized against the robot's successor state instead of
regressing the high-dimensional observation::

    L_est  = MSE(v_hat, v_{t+1})          explicit velocity regression
    L_swap = SwAV(l, l_target; E, tau)    contrastive cluster assignment

``L_swap`` follows SwAV (Caron et al., 2020). Both embeddings are compared
against a set of L2-normalized prototypes ``E``: the predicted distributions
``p`` are softmax over the cosine similarities, and the target distributions
``q`` are the Sinkhorn-Knopp transport plan of the same scores, which prevents
the assignment from collapsing onto a few prototypes. The two branches swap
targets, i.e. ``l`` is pulled towards the cluster assignment of the successor
state and vice versa.

Note that the velocity head predicts the *successor* velocity ``v_{t+1}``
(the official implementation regresses ``next_critic_obs``), which is what
makes it a model of the system response rather than a state estimator.

The estimator owns no optimizer: it is trained by ``HimPPO`` with a dedicated
Adam so that its update is independent of the PPO gradient (the policy treats
``v_hat`` and ``l`` as inputs, see ``HimActorModel``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP


@dataclass
class HimEstimatorOutput:
  """Outputs of a HIM estimator forward pass."""

  velocity: torch.Tensor
  """Estimated successor base linear velocity ``v_hat_{t+1}``, (num_envs, 3)."""

  latent: torch.Tensor
  """Implicit response ``l_t``, L2-normalized, (num_envs, latent_dim)."""


class HIMEstimator(nn.Module):
  """Source/target encoders with a SwAV prototype head."""

  velocity_dim: int = 3
  """Dimension of the estimated base linear velocity."""

  def __init__(
    self,
    obs_dim: int = 45,
    history_length: int = 6,
    state_dim: int = 45,
    latent_dim: int = 16,
    encoder_hidden_dims: tuple[int, ...] = (128, 64),
    target_hidden_dims: tuple[int, ...] = (128, 64),
    activation: str = "elu",
    num_prototypes: int = 16,
    temperature: float = 3.0,
    history_order: Literal["oldest_first", "newest_first"] = "oldest_first",
  ) -> None:
    """Initialize the estimator.

    Args:
      obs_dim: Dimension of a single proprioceptive observation ``o_t``.
      history_length: Number of frames in ``o_t^H``. The paper uses ``H = 5``,
        i.e. the current frame plus five past frames.
      state_dim: Dimension of the successor state fed to the target encoder.
      latent_dim: Dimension of the implicit response ``l``.
      encoder_hidden_dims: Hidden dimensions of the source encoder (paper:
        ``128 x 64``); its head outputs ``velocity_dim + latent_dim``.
      target_hidden_dims: Hidden dimensions of the target encoder.
      activation: Hidden-layer activation (the paper uses ELU).
      num_prototypes: Number of prototypes ``E``. The paper defaults to 16
        (the released code ships 32).
      temperature: Softmax temperature of the predicted assignment ``p``. The
        Sinkhorn target ``q`` uses its own ``eps`` (see :meth:`losses`).
      history_order: Frame order of the incoming history. ``"oldest_first"``
        matches mjlab's observation history buffers; the official
        implementation feeds ``"newest_first"`` and the history is converted
        internally, because the encoder's first layer is tied to frame order.
    """
    super().__init__()
    self.obs_dim = int(obs_dim)
    self.history_length = int(history_length)
    self.state_dim = int(state_dim)
    self.latent_dim = int(latent_dim)
    self.num_prototypes = int(num_prototypes)
    self.temperature = float(temperature)
    self.history_order = history_order

    # Source encoder: o_t^H -> [v_hat (3), l (latent_dim)].
    self.encoder = MLP(
      input_dim=self.input_dim,
      output_dim=self.velocity_dim + self.latent_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )
    # Target encoder: successor state -> l_target (latent_dim).
    self.target = MLP(
      input_dim=self.state_dim,
      output_dim=self.latent_dim,
      hidden_dims=target_hidden_dims,
      activation=activation,
    )
    # Prototypes E (num_prototypes, latent_dim), normalized before every use.
    self.proto = nn.Embedding(self.num_prototypes, self.latent_dim)

  @property
  def input_dim(self) -> int:
    """Dimension of the flattened observation history."""
    return self.obs_dim * self.history_length

  def forward(self, obs_history: torch.Tensor) -> HimEstimatorOutput:
    """Encode an observation history into the hybrid internal embedding."""
    velocity, latent = self.encode_history(obs_history)
    return HimEstimatorOutput(velocity=velocity, latent=latent)

  def encode_history(
    self, obs_history: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(v_hat, l)`` for a batch of histories."""
    frames = self._prepare_history(obs_history).flatten(start_dim=-2)
    output = self.encoder(frames)
    velocity = output[..., : self.velocity_dim]
    latent = F.normalize(output[..., self.velocity_dim :], dim=-1, p=2.0)
    return velocity, latent

  def encode_target(self, successor_state: torch.Tensor) -> torch.Tensor:
    """Return the L2-normalized embedding of a successor state."""
    return F.normalize(self.target(successor_state), dim=-1, p=2.0)

  def normalize_prototypes(self) -> None:
    """Project the prototypes onto the unit sphere in place."""
    with torch.no_grad():
      self.proto.weight.copy_(F.normalize(self.proto.weight.data, dim=-1, p=2.0))

  def losses(
    self,
    obs_history: torch.Tensor,
    successor_state: torch.Tensor,
    *,
    eps: float = 0.05,
    iters: int = 3,
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``(L_est, L_swap, diagnostics)`` for a batch of transitions.

    Args:
      obs_history: History ``o_t^H`` of the transition, ordered according to
        ``history_order``.
      successor_state: The state ``o_{t+1}`` the response is matched against.
        Its last ``velocity_dim`` entries are the velocity regression target.
      eps: Sinkhorn temperature (the released code uses 0.05).
      iters: Number of Sinkhorn normalization iterations (the released code
        uses 3).
    """
    velocity, source = self.encode_history(obs_history)
    target = self.encode_target(successor_state)

    # Cosine similarities against the normalized prototypes. Official code
    # normalizes the embedding weights in place before every use.
    self.normalize_prototypes()
    score_source = source @ self.proto.weight.T
    score_target = target @ self.proto.weight.T

    with torch.no_grad():
      q_source = sinkhorn(score_source, eps=eps, iters=iters)
      q_target = sinkhorn(score_target, eps=eps, iters=iters)

    log_p_source = F.log_softmax(score_source / self.temperature, dim=-1)
    log_p_target = F.log_softmax(score_target / self.temperature, dim=-1)
    # Swapped prediction: q_source supervises the target branch and vice versa.
    swap_loss = -0.5 * (q_source * log_p_target + q_target * log_p_source).mean()

    velocity_target = successor_state[..., -self.velocity_dim :]
    est_loss = F.mse_loss(velocity, velocity_target)

    return est_loss, swap_loss, self._diagnostics(score_source, q_target)

  def _diagnostics(
    self, score_source: torch.Tensor, q_target: torch.Tensor
  ) -> dict[str, float]:
    """Prototype health metrics of one batch (cheap, no gradient)."""
    with torch.no_grad():
      # Mean assignment mass per prototype: a normalized entropy of 1 means the
      # batch spreads evenly over the prototypes, ~0 means collapse.
      assignment = F.softmax(score_source / self.temperature, dim=-1).mean(dim=0)
      entropy = -(assignment * (assignment + 1e-9).log()).sum()
      active = (assignment > 0.5 / self.num_prototypes).sum()
      accuracy = (score_source.argmax(dim=-1) == q_target.argmax(dim=-1)).float().mean()
    return {
      "proto_entropy": float(entropy / math.log(self.num_prototypes)),
      "proto_active": float(active),
      "swap_accuracy": float(accuracy),
    }

  def _prepare_history(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Validate the history tensor and convert it to (num_envs, H, obs_dim)."""
    if obs_history.ndim == 2:
      obs_history = obs_history.reshape(-1, self.history_length, self.obs_dim)
    # Skip the shape checks during ONNX tracing; they compare traced shapes.
    if not torch.onnx.is_in_onnx_export() and (
      obs_history.ndim != 3
      or obs_history.shape[-2:] != (self.history_length, self.obs_dim)
    ):
      raise ValueError(
        "Expected obs_history of shape "
        f"(num_envs, {self.history_length}, {self.obs_dim}) or "
        f"(num_envs, {self.input_dim}), got {tuple(obs_history.shape)}."
      )
    if self.history_order == "oldest_first":
      obs_history = obs_history.flip(1)
    return obs_history


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
  """Sinkhorn-Knopp transport plan of SwAV, ported from the released code.

  Rows sum to ``1 / num_prototypes`` and columns to ``1 / batch_size``, i.e.
  every prototype receives an equal share of the batch, which is what keeps
  the clustering assignment balanced. ``out`` holds cosine similarities in
  ``[-1, 1]`` (both the embeddings and the prototypes are L2-normalized), which
  keeps ``exp(out / eps)`` well inside float32 range.
  """
  q = torch.exp(out / eps).T
  prototypes, batch = q.shape[0], q.shape[1]
  q /= q.sum()
  for _ in range(iters):
    # Normalize each row: total weight per prototype must be 1 / prototypes.
    q /= torch.sum(q, dim=1, keepdim=True)
    q /= prototypes
    # Normalize each column: total weight per sample must be 1 / batch.
    q /= torch.sum(q, dim=0, keepdim=True)
    q /= batch
  return (q * batch).T
