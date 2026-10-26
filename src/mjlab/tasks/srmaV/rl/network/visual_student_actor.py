"""Deployable SRMAV student with separate visual and proprioceptive beliefs."""

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
class SRMAVVisualEstimatorOutput:
  """Velocity, terrain, and privileged estimates for one control step."""

  velocity: torch.Tensor
  height_latent: torch.Tensor
  privilege_latent: torch.Tensor

  @property
  def latent(self) -> torch.Tensor:
    """Return the concatenated teacher-compatible external latent."""
    return torch.cat((self.height_latent, self.privilege_latent), dim=-1)


@dataclass
class SRMAVVisualStudentStepOutput:
  """One-step estimates and the deterministic student action."""

  velocity: torch.Tensor
  height_latent: torch.Tensor
  privilege_latent: torch.Tensor
  normalized_observation: torch.Tensor
  actor_mean: torch.Tensor

  @property
  def latent(self) -> torch.Tensor:
    return torch.cat((self.height_latent, self.privilege_latent), dim=-1)


@dataclass
class SRMAVVisualStudentSequenceOutput:
  """Temporally ordered outputs used by the DAgger update."""

  velocity: torch.Tensor
  height_latent: torch.Tensor
  privilege_latent: torch.Tensor
  actor_mean: torch.Tensor
  final_hidden: torch.Tensor

  @property
  def latent(self) -> torch.Tensor:
    return torch.cat((self.height_latent, self.privilege_latent), dim=-1)


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected observation '{key}' to be a tensor.")
  return value


class SRMAVVisualStudentActorModel(MLPModel):
  """Two-GRU student with depth-conditioned terrain belief.

  The visual GRU consumes proprioception and a depth feature to estimate the
  16-D teacher height latent. A separate proprioceptive GRU estimates the 6-D
  teacher privilege latent and ground-truth base linear velocity. All estimates
  are detached before the teacher-compatible actor MLP.
  """

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
    self._depth_group = str(cfg.pop("depth_group", "student_depth"))
    self._velocity_target_group = str(
      cfg.pop("velocity_target_group", "teacher_velocity_gt")
    )
    self._gru_hidden_dim = int(cfg.pop("gru_hidden_dim", 64))
    self._gru_num_layers = int(cfg.pop("gru_num_layers", 1))
    self._height_latent_dim = int(cfg.pop("height_latent_dim", 16))
    self._privilege_latent_dim = int(cfg.pop("privilege_latent_dim", 6))
    self._velocity_dim = int(cfg.pop("velocity_dim", 3))
    self._depth_feature_dim = int(cfg.pop("depth_feature_dim", 64))
    self._depth_channels = tuple(cfg.pop("depth_channels", (8, 16, 32)))
    if self._gru_hidden_dim < 1:
      raise ValueError("gru_hidden_dim must be positive.")
    if self._gru_num_layers < 1:
      raise ValueError("gru_num_layers must be positive.")
    if self._height_latent_dim < 1 or self._privilege_latent_dim < 1:
      raise ValueError("Student latent dimensions must be positive.")
    if self._velocity_dim != 3:
      raise ValueError(f"Expected 3-D velocity estimate, got {self._velocity_dim}.")
    if self._depth_feature_dim < 1:
      raise ValueError("depth_feature_dim must be positive.")
    if not self._depth_channels:
      raise ValueError("depth_channels must contain at least one channel count.")
    for group in (obs_group, self._depth_group):
      if group not in obs:
        raise ValueError(
          f"Student observation group '{group}' is missing: {list(obs.keys())}."
        )

    depth = _tensor(obs, self._depth_group)
    if depth.ndim != 4:
      raise ValueError(
        f"Student depth must have shape [B, H, W, C], got {tuple(depth.shape)}."
      )
    self._depth_shape = tuple(depth.shape[1:])
    self._student_obs_group = obs_group
    self._last_estimator_output: SRMAVVisualEstimatorOutput | None = None
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
        2,
        self._gru_num_layers,
        batch_size,
        self._gru_hidden_dim,
        dtype=_tensor(obs, obs_group).dtype,
      ),
      persistent=False,
    )

    depth_layers: list[nn.Module] = []
    in_channels = self._depth_shape[-1]
    for index, out_channels in enumerate(self._depth_channels):
      kernel_size = 5 if index == 0 else 3
      depth_layers.extend(
        (
          nn.Conv2d(
            in_channels,
            int(out_channels),
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
          ),
          resolve_nn_activation(activation),
        )
      )
      in_channels = int(out_channels)
    depth_layers.extend(
      (
        nn.AdaptiveAvgPool2d((3, 5)),
        nn.Flatten(),
        nn.Linear(in_channels * 3 * 5, self._depth_feature_dim),
        resolve_nn_activation(activation),
      )
    )
    self.depth_encoder = nn.Sequential(*depth_layers)
    self.visual_encoder = nn.Sequential(
      nn.Linear(self.obs_dim + self._depth_feature_dim, self._gru_hidden_dim),
      resolve_nn_activation(activation),
    )
    self.proprio_encoder = nn.Sequential(
      nn.Linear(self.obs_dim, self._gru_hidden_dim),
      resolve_nn_activation(activation),
    )
    self.visual_gru = nn.GRU(
      input_size=self._gru_hidden_dim,
      hidden_size=self._gru_hidden_dim,
      num_layers=self._gru_num_layers,
    )
    self.proprio_gru = nn.GRU(
      input_size=self._gru_hidden_dim,
      hidden_size=self._gru_hidden_dim,
      num_layers=self._gru_num_layers,
    )
    self.height_head = nn.Linear(self._gru_hidden_dim, self._height_latent_dim)
    self.height_norm = nn.LayerNorm(self._height_latent_dim)
    self.privilege_head = nn.Linear(self._gru_hidden_dim, self._privilege_latent_dim)
    self.privilege_norm = nn.LayerNorm(self._privilege_latent_dim)
    self.velocity_head = nn.Linear(self._gru_hidden_dim, self._velocity_dim)
    if cfg:
      raise ValueError(f"Unknown student config keys: {sorted(cfg)}.")

  @property
  def obs_group(self) -> str:
    return self._student_obs_group

  @property
  def depth_group(self) -> str:
    return self._depth_group

  @property
  def depth_shape(self) -> tuple[int, ...]:
    return self._depth_shape

  @property
  def velocity_target_group(self) -> str:
    return self._velocity_target_group

  @property
  def height_latent_dim(self) -> int:
    return self._height_latent_dim

  @property
  def privilege_latent_dim(self) -> int:
    return self._privilege_latent_dim

  @property
  def latent_dim(self) -> int:
    return self._height_latent_dim + self._privilege_latent_dim

  @property
  def velocity_dim(self) -> int:
    return self._velocity_dim

  @property
  def gru_hidden_dim(self) -> int:
    return self._gru_hidden_dim

  @property
  def last_estimator_output(self) -> SRMAVVisualEstimatorOutput:
    if self._last_estimator_output is None:
      raise RuntimeError("The SRMAV visual student has not run yet.")
    return self._last_estimator_output

  def bind_debug_env(self, env: Any) -> None:
    """Publish estimates for play-time viewer overlays."""
    self._debug_env = env
    env.srmaV_velocity_estimate = torch.zeros(
      env.num_envs, self.velocity_dim, device=env.device
    )
    env.srmaV_latent_estimate = torch.zeros(
      env.num_envs, self.latent_dim, device=env.device
    )

  def _publish_estimates(self, output: SRMAVVisualEstimatorOutput) -> None:
    env = getattr(self, "_debug_env", None)
    if env is not None:
      env.srmaV_velocity_estimate.copy_(output.velocity.detach())
      env.srmaV_latent_estimate.copy_(output.latent.detach())

  def _get_latent_dim(self) -> int:
    return (
      self.obs_dim
      + self._velocity_dim
      + self._height_latent_dim
      + self._privilege_latent_dim
    )

  def _normalized_observation(self, obs: TensorDict) -> torch.Tensor:
    return super().get_latent(obs)

  def _depth_features(self, depth: torch.Tensor) -> torch.Tensor:
    if depth.shape[-1] != self._depth_shape[-1]:
      raise ValueError(
        f"Expected {self._depth_shape[-1]} depth channels, got {depth.shape[-1]}."
      )
    depth_chw = depth.permute(0, 3, 1, 2)
    features = self.depth_encoder(depth_chw)
    if not isinstance(features, torch.Tensor):
      raise TypeError("Depth encoder output must be a tensor.")
    return features

  def _actor_mean(
    self,
    normalized_observation: torch.Tensor,
    output: SRMAVVisualEstimatorOutput,
  ) -> torch.Tensor:
    actor_input = torch.cat(
      (
        normalized_observation,
        output.velocity.detach(),
        output.height_latent.detach(),
        output.privilege_latent.detach(),
      ),
      dim=-1,
    )
    action = self.mlp(actor_input)
    if self.distribution is None:
      return action
    self.distribution.update(action)
    return self.distribution.deterministic_output(action)

  def update_estimators(self, obs: TensorDict) -> SRMAVVisualEstimatorOutput:
    """Advance both rollout hidden states and return current estimates."""
    normalized = self._normalized_observation(obs)
    depth = _tensor(obs, self.depth_group)
    depth_features = self._depth_features(depth)
    visual_input = self.visual_encoder(torch.cat((normalized, depth_features), dim=-1))
    visual_recurrent, visual_hidden = self.visual_gru(
      visual_input.unsqueeze(0), self._rollout_hidden[0]
    )
    proprio_input = self.proprio_encoder(normalized)
    proprio_recurrent, proprio_hidden = self.proprio_gru(
      proprio_input.unsqueeze(0), self._rollout_hidden[1]
    )
    self._rollout_hidden[0].copy_(visual_hidden)
    self._rollout_hidden[1].copy_(proprio_hidden)

    visual_features = visual_recurrent.squeeze(0)
    proprio_features = proprio_recurrent.squeeze(0)
    output = SRMAVVisualEstimatorOutput(
      velocity=self.velocity_head(proprio_features),
      height_latent=self.height_norm(self.height_head(visual_features)),
      privilege_latent=self.privilege_norm(self.privilege_head(proprio_features)),
    )
    self._last_estimator_output = output
    self._last_normalized_observation = normalized
    self._publish_estimates(output)
    return output

  def forward_deterministic(self, obs: TensorDict) -> SRMAVVisualStudentStepOutput:
    """Run one rollout step and return the deployed deterministic action."""
    output = self.update_estimators(obs)
    if self._last_normalized_observation is None:
      raise RuntimeError("Student observation cache was not populated.")
    normalized = self._last_normalized_observation
    return SRMAVVisualStudentStepOutput(
      velocity=output.velocity,
      height_latent=output.height_latent,
      privilege_latent=output.privilege_latent,
      normalized_observation=normalized,
      actor_mean=self._actor_mean(normalized, output),
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
        raise TypeError("SRMAV visual student hidden_state must be a tensor.")
      self.reset(hidden_state=hidden_state)
    output = self.update_estimators(obs)
    if self._last_normalized_observation is None:
      raise RuntimeError("Student observation cache was not populated.")
    return torch.cat(
      (
        self._last_normalized_observation,
        output.velocity.detach(),
        output.height_latent.detach(),
        output.privilege_latent.detach(),
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
        raise TypeError("SRMAV visual student hidden_state must be a tensor.")
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
    if done_mask.numel() != self._rollout_hidden.shape[2]:
      raise ValueError(
        f"Expected {self._rollout_hidden.shape[2]} done flags, got {done_mask.numel()}."
      )
    self._rollout_hidden[:, :, done_mask, :] = 0.0

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    if dones is None:
      self._rollout_hidden = self._rollout_hidden.detach()
      return
    done_mask = dones.reshape(-1).bool()
    self._rollout_hidden[:, :, done_mask, :] = self._rollout_hidden[
      :, :, done_mask, :
    ].detach()

  def forward_sequence(
    self,
    observation_sequence: torch.Tensor,
    depth_sequence: torch.Tensor,
    initial_hidden: torch.Tensor,
    dones: torch.Tensor,
  ) -> SRMAVVisualStudentSequenceOutput:
    """Replay a rollout sequence without modifying rollout hidden state."""
    if observation_sequence.ndim != 3:
      raise ValueError(
        "observation_sequence must have shape [time, batch, obs_dim], got "
        f"{tuple(observation_sequence.shape)}."
      )
    time_steps, batch_size, obs_dim = observation_sequence.shape
    if obs_dim != self.obs_dim:
      raise ValueError(f"Expected observation dimension {self.obs_dim}, got {obs_dim}.")
    expected_depth = (time_steps, batch_size, *self._depth_shape)
    if depth_sequence.shape != expected_depth:
      raise ValueError(
        f"Expected depth sequence shape {expected_depth}, "
        f"got {tuple(depth_sequence.shape)}."
      )
    expected_hidden = (
      2,
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
    proprio_inputs = self.proprio_encoder(normalized)

    visual_hidden = initial_hidden[0].clone()
    proprio_hidden = initial_hidden[1].clone()
    velocity_outputs: list[torch.Tensor] = []
    height_outputs: list[torch.Tensor] = []
    privilege_outputs: list[torch.Tensor] = []
    for step in range(time_steps):
      if step > 0:
        reset_mask = dones[step - 1].reshape(batch_size).bool()
        visual_hidden = torch.where(
          reset_mask.view(1, batch_size, 1),
          torch.zeros_like(visual_hidden),
          visual_hidden,
        )
        proprio_hidden = torch.where(
          reset_mask.view(1, batch_size, 1),
          torch.zeros_like(proprio_hidden),
          proprio_hidden,
        )
      depth_features = self._depth_features(
        depth_sequence[step].to(dtype=normalized.dtype)
      )
      visual_input = self.visual_encoder(
        torch.cat((normalized[step], depth_features), dim=-1)
      )
      visual_recurrent, visual_hidden = self.visual_gru(
        visual_input.unsqueeze(0), visual_hidden
      )
      proprio_recurrent, proprio_hidden = self.proprio_gru(
        proprio_inputs[step].unsqueeze(0), proprio_hidden
      )
      visual_features = visual_recurrent.squeeze(0)
      proprio_features = proprio_recurrent.squeeze(0)
      velocity_outputs.append(self.velocity_head(proprio_features))
      height_outputs.append(self.height_norm(self.height_head(visual_features)))
      privilege_outputs.append(
        self.privilege_norm(self.privilege_head(proprio_features))
      )

    velocity = torch.stack(velocity_outputs, dim=0)
    height_latent = torch.stack(height_outputs, dim=0)
    privilege_latent = torch.stack(privilege_outputs, dim=0)
    actor_input = torch.cat(
      (
        normalized,
        velocity.detach(),
        height_latent.detach(),
        privilege_latent.detach(),
      ),
      dim=-1,
    )
    actor_output = self.mlp(actor_input.reshape(time_steps * batch_size, -1))
    if self.distribution is not None:
      actor_output = self.distribution.deterministic_output(actor_output)
    return SRMAVVisualStudentSequenceOutput(
      velocity=velocity,
      height_latent=height_latent,
      privilege_latent=privilege_latent,
      actor_mean=actor_output.reshape(time_steps, batch_size, -1),
      final_hidden=torch.stack((visual_hidden, proprio_hidden), dim=0),
    )

  def copy_actor_from_teacher(
    self,
    teacher_state_dict: dict[str, torch.Tensor],
    strict_shapes: bool = True,
  ) -> list[str]:
    """Copy the teacher policy MLP, normalizer, and distribution parameters."""
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
    return _OnnxSRMAVVisualStudentActorModel(self, verbose)

  def as_jit(self) -> nn.Module:
    raise NotImplementedError(
      "SRMAV visual student export is supported through ONNX only."
    )


class _OnnxSRMAVVisualStudentActorModel(nn.Module):
  """Deterministic ONNX wrapper with depth and two recurrent states."""

  is_recurrent = True

  def __init__(self, model: SRMAVVisualStudentActorModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.depth_encoder = copy.deepcopy(model.depth_encoder)
    self.visual_encoder = copy.deepcopy(model.visual_encoder)
    self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
    self.visual_gru = copy.deepcopy(model.visual_gru)
    self.proprio_gru = copy.deepcopy(model.proprio_gru)
    self.height_head = copy.deepcopy(model.height_head)
    self.height_norm = copy.deepcopy(model.height_norm)
    self.privilege_head = copy.deepcopy(model.privilege_head)
    self.privilege_norm = copy.deepcopy(model.privilege_norm)
    self.velocity_head = copy.deepcopy(model.velocity_head)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.obs_dim = model.obs_dim
    self.depth_shape = model.depth_shape
    self.depth_feature_dim = model._depth_feature_dim
    self.hidden_size = model.gru_hidden_dim
    self.num_layers = model._gru_num_layers

  def forward(
    self,
    obs: torch.Tensor,
    depth: torch.Tensor,
    h_in: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = self.obs_normalizer(obs)
    depth_features = self.depth_encoder(depth.permute(0, 3, 1, 2))
    visual_input = self.visual_encoder(torch.cat((normalized, depth_features), dim=-1))
    proprio_input = self.proprio_encoder(normalized)
    visual_recurrent, visual_hidden = self.visual_gru(
      visual_input.unsqueeze(0), h_in[0]
    )
    proprio_recurrent, proprio_hidden = self.proprio_gru(
      proprio_input.unsqueeze(0), h_in[1]
    )
    visual_features = visual_recurrent.squeeze(0)
    proprio_features = proprio_recurrent.squeeze(0)
    velocity = self.velocity_head(proprio_features)
    height_latent = self.height_norm(self.height_head(visual_features))
    privilege_latent = self.privilege_norm(self.privilege_head(proprio_features))
    actor_input = torch.cat(
      (normalized, velocity, height_latent, privilege_latent), dim=-1
    )
    actions = self.deterministic_output(self.mlp(actor_input))
    return actions, torch.stack((visual_hidden, proprio_hidden), dim=0)

  def get_dummy_inputs(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.obs_dim),
      torch.zeros(1, *self.depth_shape),
      torch.zeros(2, self.num_layers, 1, self.hidden_size),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", "depth", "h_in"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "h_out"]
