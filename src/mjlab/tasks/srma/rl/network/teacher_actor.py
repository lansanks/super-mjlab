"""Privileged SRMA teacher actor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from mjlab.tasks.srma.rl.network.external_encoder import SRMAExternalEncoder


@dataclass
class SRMAExternalOutput:
  """Privileged inputs and latent produced by one teacher forward pass."""

  velocity: torch.Tensor
  height_map: torch.Tensor
  friction: torch.Tensor
  previous_push: torch.Tensor
  latent: torch.Tensor


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected observation '{key}' to be a tensor.")
  return value


class SRMATeacherActorModel(MLPModel):
  """Policy fed by true velocity and a control-oriented external latent."""

  _MODES = ("none", "map", "map_friction", "full")

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    srma_cfg: dict | None = None,
  ) -> None:
    cfg = dict(srma_cfg or {})
    self._height_group = str(cfg.pop("height_group", "teacher_height_map"))
    self._friction_group = str(cfg.pop("friction_group", "teacher_friction"))
    self._push_group = str(cfg.pop("push_group", "teacher_prev_push"))
    self._velocity_group = str(cfg.pop("velocity_group", "teacher_velocity_gt"))
    self._height_shape = tuple(cfg.pop("height_shape", (11, 9)))
    self._latent_dim = int(cfg.pop("latent_dim", 16))
    self._external_mode = str(cfg.pop("external_mode", "full"))
    self._height_channels = tuple(cfg.pop("height_channels", (16, 32)))
    self._height_hidden_dim = int(cfg.pop("height_hidden_dim", 64))
    self._vector_hidden_dim = int(cfg.pop("vector_hidden_dim", 32))
    self._fusion_hidden_dim = int(cfg.pop("fusion_hidden_dim", 64))

    if self._external_mode not in self._MODES:
      raise ValueError(
        f"external_mode must be one of {self._MODES}, got '{self._external_mode}'."
      )
    for group in (
      self._height_group,
      self._friction_group,
      self._push_group,
      self._velocity_group,
    ):
      if group not in obs:
        raise ValueError(
          f"Teacher observation group '{group}' is missing: {list(obs.keys())}."
        )

    friction_dim = int(_tensor(obs, self._friction_group).shape[-1])
    push_dim = int(_tensor(obs, self._push_group).shape[-1])
    velocity_dim = int(_tensor(obs, self._velocity_group).shape[-1])
    if velocity_dim != 3:
      raise ValueError(f"Expected 3-D velocity target, got {velocity_dim}.")
    if self._latent_dim < 1:
      raise ValueError(f"latent_dim must be positive, got {self._latent_dim}.")

    self._srma_extra_dim = velocity_dim + self._latent_dim
    self._last_external_output: SRMAExternalOutput | None = None

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    self.external_encoder = SRMAExternalEncoder(
      height_shape=self._height_shape,
      vector_dim=friction_dim + push_dim,
      latent_dim=self._latent_dim,
      height_channels=self._height_channels,
      height_hidden_dim=self._height_hidden_dim,
      vector_hidden_dim=self._vector_hidden_dim,
      fusion_hidden_dim=self._fusion_hidden_dim,
      activation=activation,
    )
    if cfg:
      raise ValueError(f"Unknown SRMA teacher config keys: {sorted(cfg)}.")

  @property
  def last_external_output(self) -> SRMAExternalOutput:
    if self._last_external_output is None:
      raise RuntimeError("The SRMA teacher has not run yet.")
    return self._last_external_output

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self._srma_extra_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    proprio = super().get_latent(obs, masks, hidden_state)
    height_map = _tensor(obs, self._height_group)
    friction = _tensor(obs, self._friction_group)
    previous_push = _tensor(obs, self._push_group)
    velocity = _tensor(obs, self._velocity_group)

    if self._external_mode == "none":
      height_map = torch.zeros_like(height_map)
      friction = torch.zeros_like(friction)
      previous_push = torch.zeros_like(previous_push)
    elif self._external_mode == "map":
      friction = torch.zeros_like(friction)
      previous_push = torch.zeros_like(previous_push)
    elif self._external_mode == "map_friction":
      previous_push = torch.zeros_like(previous_push)

    latent = (
      torch.zeros(
        proprio.shape[0],
        self._latent_dim,
        device=proprio.device,
        dtype=proprio.dtype,
      )
      if self._external_mode == "none"
      else self.external_encoder(
        height_map,
        torch.cat((friction, previous_push), dim=-1),
      )
    )
    self._last_external_output = SRMAExternalOutput(
      velocity=velocity,
      height_map=height_map,
      friction=friction,
      previous_push=previous_push,
      latent=latent,
    )
    return torch.cat((proprio, velocity, latent), dim=-1)

  def action_mean_with_latent(
    self,
    obs: TensorDict,
    latent: torch.Tensor,
  ) -> torch.Tensor:
    """Return deterministic actions with a caller-provided teacher latent."""
    proprio = super().get_latent(obs)
    velocity = _tensor(obs, self._velocity_group)
    policy_input = torch.cat((proprio, velocity, latent), dim=-1)
    output = self.mlp(policy_input)
    if self.distribution is None:
      return output
    return self.distribution.deterministic_output(output)

  def as_onnx(self, verbose: bool) -> nn.Module:
    raise NotImplementedError(
      "The privileged SRMA teacher does not support deployment ONNX export."
    )

  def as_jit(self) -> nn.Module:
    raise NotImplementedError(
      "The privileged SRMA teacher does not support TorchScript export."
    )
