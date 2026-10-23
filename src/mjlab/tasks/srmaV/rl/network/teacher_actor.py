"""Privileged SRMAV teacher actor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from mjlab.tasks.srmaV.rl.network.external_encoder import (
  SRMAVHeightMLPEncoder,
  SRMAVPrivilegeMLPEncoder,
)


@dataclass
class SRMAVExternalOutput:
  """Privileged inputs and split latents from one teacher forward pass."""

  velocity: torch.Tensor
  height_map: torch.Tensor
  friction: torch.Tensor
  previous_push: torch.Tensor
  height_latent: torch.Tensor
  privilege_latent: torch.Tensor

  @property
  def latent(self) -> torch.Tensor:
    """Return the concatenated latent in actor input order."""
    return torch.cat((self.height_latent, self.privilege_latent), dim=-1)


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected observation '{key}' to be a tensor.")
  return value


class SRMAVTeacherActorModel(MLPModel):
  """Policy fed by true velocity and separate height/private latents."""

  _MODES = ("none", "height", "priv", "full")

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
    srmaV_cfg: dict | None = None,
  ) -> None:
    cfg = dict(srmaV_cfg or {})
    self._height_group = str(cfg.pop("height_group", "teacher_height_map"))
    self._friction_group = str(cfg.pop("friction_group", "teacher_friction"))
    self._push_group = str(cfg.pop("push_group", "teacher_prev_push"))
    self._velocity_group = str(cfg.pop("velocity_group", "teacher_velocity_gt"))
    self._height_shape = tuple(cfg.pop("height_shape", (17, 11)))
    self._height_latent_dim = int(cfg.pop("height_latent_dim", 16))
    self._privilege_latent_dim = int(cfg.pop("privilege_latent_dim", 6))
    self._external_mode = str(cfg.pop("external_mode", "full"))
    self._height_hidden_dim = int(cfg.pop("height_hidden_dim", 32))
    self._privilege_hidden_dim = int(cfg.pop("privilege_hidden_dim", 32))

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

    height_dim = int(_tensor(obs, self._height_group).shape[-1])
    friction_dim = int(_tensor(obs, self._friction_group).shape[-1])
    push_dim = int(_tensor(obs, self._push_group).shape[-1])
    velocity_dim = int(_tensor(obs, self._velocity_group).shape[-1])
    expected_height_dim = math.prod(self._height_shape)
    if height_dim != expected_height_dim:
      raise ValueError(
        f"Expected a {expected_height_dim}-D height scan from "
        f"'{self._height_group}', got {height_dim}."
      )
    if velocity_dim != 3:
      raise ValueError(f"Expected 3-D velocity target, got {velocity_dim}.")
    if self._height_latent_dim < 1 or self._privilege_latent_dim < 1:
      raise ValueError("Teacher latent dimensions must be positive.")

    self._privilege_dim = friction_dim + push_dim
    self._srmaV_extra_dim = (
      velocity_dim + self._height_latent_dim + self._privilege_latent_dim
    )
    self._last_external_output: SRMAVExternalOutput | None = None

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

    self.height_encoder = SRMAVHeightMLPEncoder(
      input_dim=height_dim,
      latent_dim=self._height_latent_dim,
      hidden_dim=self._height_hidden_dim,
      activation=activation,
    )
    self.privilege_encoder = SRMAVPrivilegeMLPEncoder(
      input_dim=self._privilege_dim,
      latent_dim=self._privilege_latent_dim,
      hidden_dim=self._privilege_hidden_dim,
      activation=activation,
    )
    if cfg:
      raise ValueError(f"Unknown SRMAV teacher config keys: {sorted(cfg)}.")

  @property
  def last_external_output(self) -> SRMAVExternalOutput:
    if self._last_external_output is None:
      raise RuntimeError("The SRMAV teacher has not run yet.")
    return self._last_external_output

  @property
  def height_latent_dim(self) -> int:
    return self._height_latent_dim

  @property
  def privilege_latent_dim(self) -> int:
    return self._privilege_latent_dim

  @property
  def total_latent_dim(self) -> int:
    return self._height_latent_dim + self._privilege_latent_dim

  @property
  def latent_dim(self) -> int:
    """Return the combined height and privilege latent dimension."""
    return self.total_latent_dim

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self._srmaV_extra_dim

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

    height_latent = (
      self.height_encoder(height_map)
      if self._external_mode in ("height", "full")
      else height_map.new_zeros(height_map.shape[0], self._height_latent_dim)
    )
    if self._external_mode in ("priv", "full"):
      privilege_latent = self.privilege_encoder(
        torch.cat((friction, previous_push), dim=-1)
      )
    else:
      privilege_latent = friction.new_zeros(
        friction.shape[0], self._privilege_latent_dim
      )

    self._last_external_output = SRMAVExternalOutput(
      velocity=velocity,
      height_map=height_map,
      friction=friction,
      previous_push=previous_push,
      height_latent=height_latent,
      privilege_latent=privilege_latent,
    )
    return torch.cat(
      (proprio, velocity, height_latent, privilege_latent),
      dim=-1,
    )

  def action_mean_with_latents(
    self,
    obs: TensorDict,
    height_latent: torch.Tensor,
    privilege_latent: torch.Tensor,
  ) -> torch.Tensor:
    """Return deterministic actions with caller-provided split latents."""
    if height_latent.shape[-1] != self._height_latent_dim:
      raise ValueError(
        f"Expected {self._height_latent_dim}-D height latent, "
        f"got {height_latent.shape[-1]}."
      )
    if privilege_latent.shape[-1] != self._privilege_latent_dim:
      raise ValueError(
        f"Expected {self._privilege_latent_dim}-D privilege latent, "
        f"got {privilege_latent.shape[-1]}."
      )
    proprio = super().get_latent(obs)
    velocity = _tensor(obs, self._velocity_group)
    policy_input = torch.cat(
      (proprio, velocity, height_latent, privilege_latent),
      dim=-1,
    )
    output = self.mlp(policy_input)
    if self.distribution is None:
      return output
    return self.distribution.deterministic_output(output)

  def action_mean_with_latent(
    self,
    obs: TensorDict,
    latent: torch.Tensor,
  ) -> torch.Tensor:
    """Return deterministic actions with the concatenated split latent."""
    if latent.shape[-1] != self.total_latent_dim:
      raise ValueError(
        f"Expected a {self.total_latent_dim}-D teacher latent, got {latent.shape[-1]}."
      )
    return self.action_mean_with_latents(
      obs,
      latent[..., : self._height_latent_dim],
      latent[..., self._height_latent_dim :],
    )

  def as_onnx(self, verbose: bool) -> nn.Module:
    raise NotImplementedError(
      "The privileged SRMAV teacher does not support deployment ONNX export."
    )

  def as_jit(self) -> nn.Module:
    raise NotImplementedError(
      "The privileged SRMAV teacher does not support TorchScript export."
    )
