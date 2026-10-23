"""Privileged SRMAV teacher encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


class SRMAVHeightMLPEncoder(nn.Module):
  """Encode the current support-normalized height scan into an extero latent."""

  def __init__(
    self,
    input_dim: int = 187,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.input_dim = int(input_dim)
    self.latent_dim = int(latent_dim)
    self.encoder = nn.Sequential(
      nn.Linear(self.input_dim, int(hidden_dim)),
      resolve_nn_activation(activation),
      nn.Linear(int(hidden_dim), self.latent_dim),
      nn.LayerNorm(self.latent_dim),
    )

  def forward(self, height_scan: torch.Tensor) -> torch.Tensor:
    """Return the height latent for a batch of flattened scans."""
    if height_scan.shape[-1] != self.input_dim:
      raise ValueError(
        f"Expected a {self.input_dim}-D height scan, got {height_scan.shape[-1]}."
      )
    return self.encoder(height_scan)


class SRMAVPrivilegeMLPEncoder(nn.Module):
  """Encode friction and delayed push into a compact private latent."""

  def __init__(
    self,
    input_dim: int = 9,
    latent_dim: int = 6,
    hidden_dim: int = 32,
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.input_dim = int(input_dim)
    self.latent_dim = int(latent_dim)
    self.encoder = nn.Sequential(
      nn.Linear(self.input_dim, int(hidden_dim)),
      resolve_nn_activation(activation),
      nn.Linear(int(hidden_dim), self.latent_dim),
      nn.LayerNorm(self.latent_dim),
    )

  def forward(self, privilege: torch.Tensor) -> torch.Tensor:
    """Return the private latent for a batch of privileged factors."""
    if privilege.shape[-1] != self.input_dim:
      raise ValueError(
        f"Expected a {self.input_dim}-D privilege vector, got {privilege.shape[-1]}."
      )
    return self.encoder(privilege)
