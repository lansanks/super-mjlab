"""Privileged external-environment encoder for the SRMA teacher."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


class SRMAExternalEncoder(nn.Module):
  """Encode a local height map and vector factors into a control latent."""

  def __init__(
    self,
    height_shape: tuple[int, int] = (11, 9),
    vector_dim: int = 9,
    latent_dim: int = 16,
    height_channels: tuple[int, ...] = (16, 32),
    height_hidden_dim: int = 64,
    vector_hidden_dim: int = 32,
    fusion_hidden_dim: int = 64,
    activation: str = "elu",
  ) -> None:
    super().__init__()
    if len(height_channels) != 2:
      raise ValueError(
        "SRMAExternalEncoder expects exactly two height convolution channels, "
        f"got {height_channels}."
      )
    self.height_shape = tuple(height_shape)
    self.vector_dim = int(vector_dim)
    self.latent_dim = int(latent_dim)

    first_channels, second_channels = height_channels
    conv_height = (self.height_shape[0] + 1) // 2
    conv_width = (self.height_shape[1] + 1) // 2
    flattened_height_dim = second_channels * conv_height * conv_width

    self.height_encoder = nn.Sequential(
      nn.Conv2d(
        1,
        first_channels,
        kernel_size=3,
        padding=1,
      ),
      resolve_nn_activation(activation),
      nn.Conv2d(
        first_channels,
        second_channels,
        kernel_size=3,
        stride=2,
        padding=1,
      ),
      resolve_nn_activation(activation),
      nn.Flatten(),
      nn.Linear(flattened_height_dim, height_hidden_dim),
      resolve_nn_activation(activation),
    )
    self.vector_encoder = nn.Sequential(
      nn.Linear(self.vector_dim, vector_hidden_dim),
      resolve_nn_activation(activation),
      nn.Linear(vector_hidden_dim, vector_hidden_dim),
      resolve_nn_activation(activation),
    )
    self.fusion = nn.Sequential(
      nn.Linear(height_hidden_dim + vector_hidden_dim, fusion_hidden_dim),
      resolve_nn_activation(activation),
      nn.Linear(fusion_hidden_dim, self.latent_dim),
      nn.LayerNorm(self.latent_dim),
    )

  def forward(
    self,
    height_map: torch.Tensor,
    vector_factors: torch.Tensor,
  ) -> torch.Tensor:
    """Return the normalized 16-D teacher latent."""
    expected_height = self.height_shape[0] * self.height_shape[1]
    if height_map.shape[-1] != expected_height:
      raise ValueError(
        f"Expected a flattened height map with {expected_height} values, "
        f"got {height_map.shape[-1]}."
      )
    if vector_factors.shape[-1] != self.vector_dim:
      raise ValueError(
        f"Expected a {self.vector_dim}-D vector factor group, "
        f"got {vector_factors.shape[-1]}."
      )
    height = height_map.reshape(-1, 1, *self.height_shape)
    height_feature = self.height_encoder(height)
    vector_feature = self.vector_encoder(vector_factors)
    return self.fusion(torch.cat((height_feature, vector_feature), dim=-1))
