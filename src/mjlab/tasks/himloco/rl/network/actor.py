"""HIM actor: policy MLP fed with the hybrid internal embedding.

The policy is ``pi(a_t | o_t, v_hat, l_t)``: the current proprioceptive
observation together with the estimator's explicit velocity and implicit
response. This module plugs ``HIMEstimator`` into rsl_rl's ``MLPModel``::

    obs["actor"]    (o_t)
    HIMEstimator(o_t^H) -> (v_hat, l)
        -> [o_t, v_hat, l] -> MLP -> action distribution

The estimator lives inside the actor so that its parameters follow the actor's
state dict and export path, but the policy path is stop-gradient: the estimator
forward runs under ``torch.no_grad()`` and the estimator is trained by its own
optimizer inside ``HimPPO`` (the released implementation does the same, see
``HimEstimator.optimizer`` in the official code).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from mjlab.tasks.himloco.rl.network.estimator import HIMEstimator, HimEstimatorOutput


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  """Return an observation tensor, rejecting nested observation groups."""
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(
      f"Expected observation '{key}' to be a tensor, got {type(value).__name__}."
    )
  return value


class HimActorModel(MLPModel):
  """Policy MLP fed with ``[o_t, v_hat, l]`` estimated by HIM.

  The policy MLP input grows from the actor observation (``o_t``) to
  ``o_t + v_hat + l``, so ``MLPModel``'s action distribution, PPO handling and
  state dict logic are reused as is. The estimator consumes the observation
  history from a separate observation group and the successor state from the
  task's ``him_state`` group (dimension only; the value is read by the
  algorithm from the rollout storage).
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    him_cfg: dict | None = None,
  ) -> None:
    """Initialize the actor and its HIM estimator.

    Args:
      obs: Observations used to resolve the observation dimensions.
      obs_groups: Observation sets resolved by rsl_rl. The history group is
        taken from the ``"him"`` set, or from ``him_cfg["obs_group"]``.
      obs_set: Observation set feeding the policy MLP (``"actor"``).
      output_dim: Action dimension.
      hidden_dims: Hidden dimensions of the policy MLP (paper: 512/256/128).
      activation: Activation of the policy MLP.
      obs_normalization: Whether to normalize ``o_t`` before the policy MLP.
        The released HIM code does not normalize observations; this defaults to
        ``False`` to match it.
      distribution_cfg: Output distribution config (Gaussian for PPO).
      him_cfg: Optional estimator overrides: ``obs_group``, ``state_group``,
        ``latent_dim``, ``encoder_hidden_dims``, ``target_hidden_dims``,
        ``activation``, ``num_prototypes``, ``temperature``, ``history_order``.
    """
    him_cfg = dict(him_cfg or {})

    obs_group = him_cfg.pop("obs_group", None)
    if obs_group is None:
      configured = obs_groups.get("him", [])
      if len(configured) > 1:
        raise ValueError(
          "HIM consumes a single observation history, but the 'him' "
          f"observation set lists {configured}."
        )
      obs_group = configured[0] if configured else "him_input"
    if obs_group not in obs:
      raise ValueError(
        f"HIM observation group '{obs_group}' is missing from the environment "
        f"observations: {list(obs.keys())}."
      )
    state_group = str(him_cfg.pop("state_group", "him_state"))
    if state_group not in obs:
      raise ValueError(
        f"HIM successor-state group '{state_group}' is missing from the "
        f"environment observations: {list(obs.keys())}."
      )
    state_dim = int(tuple(obs[state_group].shape)[-1])
    latent_dim = int(him_cfg.pop("latent_dim", 16))

    # The history group is either flattened (num_envs, L * obs_dim) or keeps
    # the time axis (num_envs, L, obs_dim).
    history_shape = tuple(obs[obs_group].shape)
    if len(history_shape) == 3:
      frame_count: int | None = int(history_shape[-2])
      frame_dim = int(history_shape[-1])
    else:
      frame_count = None
      frame_dim = int(history_shape[-1])

    # ``_get_latent_dim`` runs inside ``MLPModel.__init__``, so the extra input
    # dimension must be known before calling it. Plain attributes can be set
    # before ``nn.Module.__init__``.
    self._him_obs_group = obs_group
    self._him_state_group = state_group
    self._him_extra_dim = HIMEstimator.velocity_dim + latent_dim
    self._last_him_output: HimEstimatorOutput | None = None

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

    # The history stacks frames of the same vector the policy observes.
    if frame_count is not None:
      if frame_dim != self.obs_dim:
        raise ValueError(
          f"HIM history frames have dimension {frame_dim}, expected "
          f"{self.obs_dim} (the policy observation dimension)."
        )
      history_length = frame_count
    else:
      if frame_dim % self.obs_dim != 0:
        raise ValueError(
          f"HIM history dimension {frame_dim} is not a multiple of the "
          f"policy observation dimension {self.obs_dim}."
        )
      history_length = frame_dim // self.obs_dim

    self.estimator = HIMEstimator(
      obs_dim=self.obs_dim,
      history_length=int(history_length),
      state_dim=state_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=tuple(him_cfg.pop("encoder_hidden_dims", (128, 64))),
      target_hidden_dims=tuple(him_cfg.pop("target_hidden_dims", (128, 64))),
      activation=him_cfg.pop("activation", activation),
      num_prototypes=int(him_cfg.pop("num_prototypes", 16)),
      temperature=float(him_cfg.pop("temperature", 3.0)),
      history_order=him_cfg.pop("history_order", "oldest_first"),
    )
    if him_cfg:
      raise ValueError(f"Unknown HIM config keys: {sorted(him_cfg)}.")

  @property
  def him_group(self) -> str:
    """Observation group holding the flattened HIM history input."""
    return self._him_obs_group

  @property
  def state_group(self) -> str:
    """Observation group holding the successor-state frame."""
    return self._him_state_group

  @property
  def last_him_output(self) -> HimEstimatorOutput | None:
    """Estimator outputs of the most recent policy forward (detached).

    ``None`` until the actor has run at least once.
    """
    return self._last_him_output

  def bind_debug_env(self, env: Any) -> None:
    """Publish the implicit response for the play-time latent panel."""
    self._debug_env = env
    env.him_latent_estimate = torch.zeros(
      env.num_envs, self.estimator.latent_dim, device=env.device
    )
    env.cfg.viewer.latent_bar_attr = "him_latent_estimate"
    env.cfg.viewer.latent_bar_label = "HIM latent l"
    env.cfg.viewer.latent_bar_range = (-1.0, 1.0)

  def _publish_latent_estimate(self, latent: torch.Tensor) -> None:
    env = getattr(self, "_debug_env", None)
    if env is not None:
      env.him_latent_estimate.copy_(latent.detach())

  def _get_latent_dim(self) -> int:
    """Dimension of the policy input: ``[o_t, v_hat, l]``."""
    return self.obs_dim + self._him_extra_dim

  def _him_history(self, obs: TensorDict) -> torch.Tensor:
    """Return the observation history tensor from the observation dict."""
    return _tensor(obs, self._him_obs_group)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Concatenate ``o_t`` with the estimated velocity and implicit response."""
    proprio = super().get_latent(obs, masks, hidden_state)
    # Stop-gradient: the policy consumes v_hat and l as inputs, so the PPO loss
    # must not update the estimator. HimPPO trains the estimator with its own
    # optimizer and its own forward pass.
    with torch.no_grad():
      output = self.estimator(self._him_history(obs))
    self._last_him_output = output
    self._publish_latent_estimate(output.latent)
    return torch.cat((proprio, output.velocity, output.latent), dim=-1)

  def as_onnx(self, verbose: bool) -> nn.Module:
    """Return the ONNX export wrapper, estimator included.

    The wrapper takes the flattened observation history and uses its newest
    frame as ``o_t``, so the policy must consume a single observation group.
    """
    if len(self.obs_groups) != 1:
      raise ValueError(
        "ONNX export assumes the policy consumes a single observation group "
        f"whose newest history frame doubles as o_t, got {self.obs_groups}."
      )
    return _OnnxHimActorModel(self, verbose)

  def as_jit(self) -> nn.Module:
    """TorchScript export is not supported for the HIM actor."""
    raise NotImplementedError(
      "TorchScript export is not supported for HimActorModel. Use as_onnx(), "
      "which exports the full estimator + policy stack."
    )


class _OnnxHimActorModel(nn.Module):
  """Deterministic inference wrapper used for ONNX export.

  The exported model takes the flattened observation history ``o_t^H`` (the
  newest frame doubles as ``o_t``) and returns the deterministic action.
  """

  is_recurrent: bool = False

  def __init__(self, model: HimActorModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.estimator = copy.deepcopy(model.estimator)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.estimator.input_dim
    self.obs_dim = model.estimator.obs_dim
    self.history_length = model.estimator.history_length
    self.newest_first = model.estimator.history_order == "newest_first"

  def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Run the deterministic policy from a flattened observation history."""
    velocity, latent = self.estimator.encode_history(obs_history)
    frames = obs_history.reshape(-1, self.history_length, self.obs_dim)
    proprio = frames[:, 0] if self.newest_first else frames[:, -1]
    return self.deterministic_output(
      self.mlp(torch.cat((self.obs_normalizer(proprio), velocity, latent), dim=-1))
    )

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    """Return representative dummy inputs for ONNX tracing."""
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    """Return ONNX input tensor names."""
    return ["obs_history"]

  @property
  def output_names(self) -> list[str]:
    """Return ONNX output tensor names."""
    return ["actions"]
