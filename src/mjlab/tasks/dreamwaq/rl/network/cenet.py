"""Context-aided estimator network (CENet) from the DreamWaQ paper.

CENet (Section II-C, Fig. 2) is a single shared encoder with two heads. The
velocity head regresses the base linear velocity ``v_t``; the context head
parameterizes a diagonal Gaussian posterior ``q(z_t | o_t^H)`` over a
``latent_dim`` context vector. A decoder reconstructs the next proprioceptive
observation from a sample of that posterior, i.e. a beta-VAE::

    L_CE = MSE(v_hat_t, v_t)
         + MSE(o_hat_{t+1}, o_{t+1})
         + beta * KL(q(z_t | o_t^H) || N(0, I))

The encoder consumes ``o_t^H = [o_t, o_{t-1}, ..., o_{t-H}]`` with ``H = 5``
(six frames in total). Observations are expected to be normalized to zero mean
and unit variance, which is also what makes the standard normal prior meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from rsl_rl.modules import MLP, EmpiricalNormalization


@dataclass
class CENetOutput:
  """Outputs of a CENet forward pass."""

  velocity: torch.Tensor
  """Estimated base linear velocity ``v_hat_t``, shape (num_envs, 3)."""

  z: torch.Tensor
  """Context vector ``z_t``, shape (num_envs, latent_dim)."""

  mu: torch.Tensor
  """Posterior mean of ``z_t``, shape (num_envs, latent_dim)."""

  log_var: torch.Tensor
  """Posterior log-variance of ``z_t``, shape (num_envs, latent_dim)."""


class CENet(nn.Module):
  """Shared encoder with a velocity regression head and a beta-VAE branch."""

  velocity_dim: int = 3
  """Dimension of the estimated base linear velocity."""

  def __init__(
    self,
    obs_dim: int = 45,
    history_length: int = 6,
    latent_dim: int = 16,
    encoder_hidden_dims: tuple[int, ...] = (128, 64),
    decoder_hidden_dims: tuple[int, ...] = (64, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    history_order: Literal["oldest_first", "newest_first"] = "oldest_first",
  ) -> None:
    """Initialize CENet.

    Args:
      obs_dim: Dimension of a single proprioceptive observation ``o_t``.
      history_length: Number of frames in ``o_t^H``. The paper uses ``H = 5``,
        i.e. the current frame plus five past frames.
      latent_dim: Dimension of the context vector ``z_t``.
      encoder_hidden_dims: Hidden dimensions of the shared encoder trunk (the
        paper's ``128 x 64``).
      decoder_hidden_dims: Hidden dimensions of the decoder (``64 x 128``).
      activation: Hidden-layer activation (the paper uses ELU).
      obs_normalization: Whether to apply running normalization to the encoder
        input and to the decoder target (see :meth:`normalize_obs`).
      history_order: Frame order inside the flattened history. ``"oldest_first"``
        matches mjlab's observation history buffers, ``"newest_first"`` matches
        the paper's ``[o_t, ..., o_{t-H}]``. Frames are converted to the paper
        order internally.
    """
    super().__init__()
    self.obs_dim = obs_dim
    self.history_length = history_length
    self.latent_dim = latent_dim
    self.history_order = history_order

    # One set of per-dimension statistics is shared by every frame of the
    # history and by the reconstruction target of the decoder.
    self.obs_normalizer: nn.Module = (
      EmpiricalNormalization(obs_dim) if obs_normalization else nn.Identity()
    )

    # Shared encoder: o_t^H -> [v_hat (3), mu (latent_dim), log_var (latent_dim)].
    self.encoder = MLP(
      input_dim=obs_dim * history_length,
      output_dim=self.velocity_dim + 2 * latent_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    # Decoder: z_t -> next observation o_{t+1} in normalized space.
    self.decoder = MLP(
      input_dim=latent_dim,
      output_dim=obs_dim,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )

  @property
  def input_dim(self) -> int:
    """Dimension of the flattened observation history."""
    return self.obs_dim * self.history_length

  def forward(
    self, obs_history: torch.Tensor, sample: bool | None = None
  ) -> CENetOutput:
    """Encode an observation history and draw a context vector.

    Args:
      obs_history: Observation history of shape (num_envs, input_dim) or
        (num_envs, history_length, obs_dim), ordered according to
        ``history_order``.
      sample: Whether to draw ``z_t`` with the reparameterization trick.
        ``None`` samples in training mode and uses the posterior mean in
        evaluation mode.

    Returns:
      The velocity estimate, context vector, and posterior parameters.
    """
    if sample is None:
      sample = self.training
    velocity, mu, log_var = self.encode(obs_history)
    z = self.reparameterize(mu, log_var) if sample else mu
    return CENetOutput(velocity=velocity, z=z, mu=mu, log_var=log_var)

  def encode(
    self, obs_history: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(v_hat_t, mu_t, log_var_t)`` for a batch of histories."""
    latent = self.obs_normalizer(self._prepare_history(obs_history))
    output = self.encoder(latent.flatten(start_dim=-2))
    velocity = output[..., : self.velocity_dim]
    mu = output[..., self.velocity_dim : self.velocity_dim + self.latent_dim]
    log_var = output[..., self.velocity_dim + self.latent_dim :]
    return velocity, mu, log_var

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    """Decode context vectors into next observations (normalized space)."""
    return self.decoder(z)

  @staticmethod
  def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Sample ``z = mu + sigma * eps`` with ``eps ~ N(0, I)``."""
    std = torch.exp(0.5 * log_var)
    return mu + std * torch.randn_like(std)

  @staticmethod
  def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence to the N(0, I) prior, reduced to shape (num_envs,)."""
    return 0.5 * (mu.square() + log_var.exp() - 1.0 - log_var).sum(dim=-1)

  def reconstruction_error(
    self, z: torch.Tensor, next_obs: torch.Tensor
  ) -> torch.Tensor:
    """Per-sample MSE between the decoded and normalized next observation."""
    target = self.normalize_obs(next_obs)
    return (self.decode(z) - target).square().mean(dim=-1)

  def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Normalize raw observations with the running statistics."""
    return self.obs_normalizer(obs)

  def update_normalization(self, obs_history: torch.Tensor) -> None:
    """Update running statistics from a batch of observation histories."""
    if isinstance(self.obs_normalizer, EmpiricalNormalization):
      frames = self._prepare_history(obs_history).flatten(0, 1)
      self.obs_normalizer.update(frames)

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
