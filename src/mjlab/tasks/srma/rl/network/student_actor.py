"""Deployable SRMA student actor with a stateful GRU estimator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


@dataclass
class SRMAStudentEstimatorOutput:
  """Velocity and external-latent estimates for one control step."""

  velocity: torch.Tensor
  latent: torch.Tensor


@dataclass
class SRMAStudentStepOutput:
  """One-step estimator and deterministic actor output."""

  velocity: torch.Tensor
  latent: torch.Tensor
  normalized_observation: torch.Tensor
  actor_mean: torch.Tensor


@dataclass
class SRMAStudentSequenceOutput:
  """Temporally ordered outputs used by the DAgger update."""

  velocity: torch.Tensor
  latent: torch.Tensor
  actor_mean: torch.Tensor
  final_hidden: torch.Tensor


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected observation '{key}' to be a tensor.")
  return value


class SRMAStudentActorModel(MLPModel):
  """GRU estimator followed by the teacher-compatible actor MLP."""

  is_recurrent = True

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
    student_cfg: dict | None = None,
  ) -> None:
    cfg = dict(student_cfg or {})
    obs_group = str(cfg.pop("obs_group", "actor"))
    self._velocity_target_group = str(
      cfg.pop("velocity_target_group", "teacher_velocity_gt")
    )
    self._gru_hidden_dim = int(cfg.pop("gru_hidden_dim", 64))
    self._gru_num_layers = int(cfg.pop("gru_num_layers", 1))
    self._latent_dim = int(cfg.pop("latent_dim", 16))
    self._velocity_dim = int(cfg.pop("velocity_dim", 3))
    if self._gru_hidden_dim < 1:
      raise ValueError("gru_hidden_dim must be positive.")
    if self._gru_num_layers < 1:
      raise ValueError("gru_num_layers must be positive.")
    if self._latent_dim < 1:
      raise ValueError("latent_dim must be positive.")
    if self._velocity_dim != 3:
      raise ValueError(f"Expected 3-D velocity estimate, got {self._velocity_dim}.")
    if obs_group not in obs:
      raise ValueError(
        f"Student observation group '{obs_group}' is missing: {list(obs.keys())}."
      )

    self._student_obs_group = obs_group
    self._last_estimator_output: SRMAStudentEstimatorOutput | None = None
    self._last_normalized_observation: torch.Tensor | None = None

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

    batch_size = int(obs.batch_size[0])
    self.register_buffer(
      "_rollout_hidden",
      torch.zeros(
        self._gru_num_layers,
        batch_size,
        self._gru_hidden_dim,
        dtype=_tensor(obs, obs_group).dtype,
      ),
      persistent=False,
    )
    self.estimator_encoder = nn.Sequential(
      nn.Linear(self.obs_dim, self._gru_hidden_dim),
      resolve_nn_activation(activation),
    )
    self.estimator_gru = nn.GRU(
      input_size=self._gru_hidden_dim,
      hidden_size=self._gru_hidden_dim,
      num_layers=self._gru_num_layers,
    )
    self.velocity_head = nn.Linear(self._gru_hidden_dim, self._velocity_dim)
    self.latent_head = nn.Linear(self._gru_hidden_dim, self._latent_dim)
    self.latent_norm = nn.LayerNorm(self._latent_dim)
    if cfg:
      raise ValueError(f"Unknown student config keys: {sorted(cfg)}.")

  @property
  def obs_group(self) -> str:
    return self._student_obs_group

  @property
  def velocity_target_group(self) -> str:
    return self._velocity_target_group

  @property
  def latent_dim(self) -> int:
    return self._latent_dim

  @property
  def velocity_dim(self) -> int:
    return self._velocity_dim

  @property
  def gru_hidden_dim(self) -> int:
    return self._gru_hidden_dim

  @property
  def last_estimator_output(self) -> SRMAStudentEstimatorOutput:
    if self._last_estimator_output is None:
      raise RuntimeError("The SRMA student estimator has not run yet.")
    return self._last_estimator_output

  def bind_debug_env(self, env: Any) -> None:
    """Publish the latest velocity estimate for the play viewer."""
    self._debug_env = env
    env.srma_velocity_estimate = torch.zeros(
      env.num_envs, self.velocity_dim, device=env.device
    )

  def _publish_velocity_estimate(self, velocity: torch.Tensor) -> None:
    env = getattr(self, "_debug_env", None)
    if env is not None:
      env.srma_velocity_estimate.copy_(velocity.detach())

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self._velocity_dim + self._latent_dim

  def _normalized_observation(self, obs: TensorDict) -> torch.Tensor:
    return super().get_latent(obs)

  def _estimator_from_normalized(
    self,
    normalized_observation: torch.Tensor,
    hidden_state: torch.Tensor,
  ) -> tuple[SRMAStudentEstimatorOutput, torch.Tensor]:
    encoded = self.estimator_encoder(normalized_observation).unsqueeze(0)
    recurrent, next_hidden = self.estimator_gru(encoded, hidden_state)
    features = recurrent.squeeze(0)
    output = SRMAStudentEstimatorOutput(
      velocity=self.velocity_head(features),
      latent=self.latent_norm(self.latent_head(features)),
    )
    return output, next_hidden

  def _actor_mean(
    self,
    normalized_observation: torch.Tensor,
    velocity: torch.Tensor,
    latent: torch.Tensor,
  ) -> torch.Tensor:
    actor_input = torch.cat(
      (normalized_observation, velocity.detach(), latent.detach()), dim=-1
    )
    output = self.mlp(actor_input)
    if self.distribution is None:
      return output
    self.distribution.update(output)
    return self.distribution.deterministic_output(output)

  def update_estimator(self, obs: TensorDict) -> SRMAStudentEstimatorOutput:
    """Advance the rollout hidden state and return the current estimates."""
    normalized = self._normalized_observation(obs)
    output, next_hidden = self._estimator_from_normalized(
      normalized, self._rollout_hidden
    )
    self._rollout_hidden.copy_(next_hidden)
    self._last_estimator_output = output
    self._last_normalized_observation = normalized
    self._publish_velocity_estimate(output.velocity)
    return output

  def forward_deterministic(self, obs: TensorDict) -> SRMAStudentStepOutput:
    """Run one rollout step and return the deployed deterministic action."""
    estimates = self.update_estimator(obs)
    if self._last_normalized_observation is None:
      raise RuntimeError("Student observation cache was not populated.")
    normalized = self._last_normalized_observation
    actor_mean = self._actor_mean(normalized, estimates.velocity, estimates.latent)
    return SRMAStudentStepOutput(
      velocity=estimates.velocity,
      latent=estimates.latent,
      normalized_observation=normalized,
      actor_mean=actor_mean,
    )

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks
    if hidden_state is not None:
      if not isinstance(hidden_state, torch.Tensor):
        raise TypeError("SRMA student hidden_state must be a tensor.")
      self.reset(hidden_state=hidden_state)
    estimates = self.update_estimator(obs)
    if self._last_normalized_observation is None:
      raise RuntimeError("Student observation cache was not populated.")
    return torch.cat(
      (
        self._last_normalized_observation,
        estimates.velocity.detach(),
        estimates.latent.detach(),
      ),
      dim=-1,
    )

  def get_hidden_state(self) -> torch.Tensor:
    return self._rollout_hidden.detach()

  def reset(
    self,
    dones: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> None:
    if hidden_state is not None:
      if not isinstance(hidden_state, torch.Tensor):
        raise TypeError("SRMA student hidden_state must be a tensor.")
      if hidden_state.shape != self._rollout_hidden.shape:
        raise ValueError(
          "Expected hidden state shape "
          f"{tuple(self._rollout_hidden.shape)}, got {tuple(hidden_state.shape)}."
        )
      self._rollout_hidden.copy_(hidden_state)
      return
    if dones is None:
      self._rollout_hidden.zero_()
      return
    done_mask = dones.reshape(-1).bool()
    if done_mask.numel() != self._rollout_hidden.shape[1]:
      raise ValueError(
        f"Expected {self._rollout_hidden.shape[1]} done flags, got {done_mask.numel()}."
      )
    self._rollout_hidden[:, done_mask, :] = 0.0

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    if dones is None:
      self._rollout_hidden = self._rollout_hidden.detach()
      return
    done_mask = dones.reshape(-1).bool()
    self._rollout_hidden[:, done_mask, :] = self._rollout_hidden[
      :, done_mask, :
    ].detach()

  def forward_sequence(
    self,
    observation_sequence: torch.Tensor,
    initial_hidden: torch.Tensor,
    dones: torch.Tensor,
  ) -> SRMAStudentSequenceOutput:
    """Replay a sequence without modifying the rollout hidden state."""
    if observation_sequence.ndim != 3:
      raise ValueError(
        "observation_sequence must have shape [time, batch, obs_dim], got "
        f"{tuple(observation_sequence.shape)}."
      )
    time_steps, batch_size, obs_dim = observation_sequence.shape
    if obs_dim != self.obs_dim:
      raise ValueError(f"Expected observation dimension {self.obs_dim}, got {obs_dim}.")
    expected_hidden = (
      self._gru_num_layers,
      batch_size,
      self._gru_hidden_dim,
    )
    if initial_hidden.shape != expected_hidden:
      raise ValueError(
        f"Expected initial_hidden shape {expected_hidden}, "
        f"got {tuple(initial_hidden.shape)}."
      )
    if dones.shape[:2] != (time_steps, batch_size):
      raise ValueError(
        f"Expected dones shape [{time_steps}, {batch_size}, ...], "
        f"got {tuple(dones.shape)}."
      )

    normalized = self.obs_normalizer(observation_sequence.reshape(-1, obs_dim))
    normalized = normalized.reshape(time_steps, batch_size, obs_dim)
    hidden = initial_hidden
    velocity_outputs: list[torch.Tensor] = []
    latent_outputs: list[torch.Tensor] = []
    for step in range(time_steps):
      if step > 0:
        reset_mask = dones[step - 1].reshape(batch_size).bool()
        hidden = torch.where(
          reset_mask.view(1, batch_size, 1),
          torch.zeros_like(hidden),
          hidden,
        )
      encoded = self.estimator_encoder(normalized[step]).unsqueeze(0)
      recurrent, hidden = self.estimator_gru(encoded, hidden)
      features = recurrent.squeeze(0)
      velocity_outputs.append(self.velocity_head(features))
      latent_outputs.append(self.latent_norm(self.latent_head(features)))

    velocity = torch.stack(velocity_outputs, dim=0)
    latent = torch.stack(latent_outputs, dim=0)
    actor_input = torch.cat((normalized, velocity.detach(), latent.detach()), dim=-1)
    actor_output = self.mlp(actor_input.reshape(time_steps * batch_size, -1))
    if self.distribution is not None:
      actor_output = self.distribution.deterministic_output(actor_output)
    return SRMAStudentSequenceOutput(
      velocity=velocity,
      latent=latent,
      actor_mean=actor_output.reshape(time_steps, batch_size, -1),
      final_hidden=hidden,
    )

  def copy_actor_from_teacher(
    self,
    teacher_state_dict: dict[str, torch.Tensor],
    strict_shapes: bool = True,
  ) -> list[str]:
    """Copy the teacher policy MLP, normalizer and distribution parameters."""
    prefixes = ("mlp.", "obs_normalizer.", "distribution.")
    current = self.state_dict()
    copied: dict[str, torch.Tensor] = {}
    shape_errors: list[str] = []
    for key, value in teacher_state_dict.items():
      if not key.startswith(prefixes):
        continue
      if key not in current:
        if strict_shapes:
          shape_errors.append(f"{key}: missing in student")
        continue
      if current[key].shape != value.shape:
        if strict_shapes:
          shape_errors.append(
            f"{key}: teacher {tuple(value.shape)} != "
            f"student {tuple(current[key].shape)}"
          )
        continue
      copied[key] = value
    if shape_errors:
      raise ValueError("Teacher/student shape mismatch:\n" + "\n".join(shape_errors))
    if strict_shapes:
      missing = [
        key for key in current if key.startswith(prefixes) and key not in copied
      ]
      if missing:
        raise ValueError(
          "Teacher checkpoint is missing student actor keys: "
          + ", ".join(sorted(missing))
        )
    if not copied:
      raise ValueError("No compatible teacher actor parameters were found.")
    self.load_state_dict(copied, strict=False)
    return sorted(copied)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxSRMAStudentActorModel(self, verbose)

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("SRMA student export is supported through ONNX only.")


class _OnnxSRMAStudentActorModel(nn.Module):
  """Deterministic ONNX wrapper with explicit recurrent input/output."""

  is_recurrent = True

  def __init__(self, model: SRMAStudentActorModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.estimator_encoder = copy.deepcopy(model.estimator_encoder)
    self.estimator_gru = copy.deepcopy(model.estimator_gru)
    self.velocity_head = copy.deepcopy(model.velocity_head)
    self.latent_head = copy.deepcopy(model.latent_head)
    self.latent_norm = copy.deepcopy(model.latent_norm)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.obs_dim = model.obs_dim
    self.hidden_size = model.gru_hidden_dim
    self.num_layers = model._gru_num_layers

  def forward(
    self, obs: torch.Tensor, h_in: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = self.obs_normalizer(obs)
    encoded = self.estimator_encoder(normalized).unsqueeze(0)
    recurrent, h_out = self.estimator_gru(encoded, h_in)
    features = recurrent.squeeze(0)
    velocity = self.velocity_head(features)
    latent = self.latent_norm(self.latent_head(features))
    actor_input = torch.cat((normalized, velocity, latent), dim=-1)
    actions = self.deterministic_output(self.mlp(actor_input))
    return actions, h_out

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.obs_dim),
      torch.zeros(self.num_layers, 1, self.hidden_size),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", "h_in"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "h_out"]
