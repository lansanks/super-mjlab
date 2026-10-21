"""DreamWaQ actor: policy MLP fed with CENet estimates.

The paper's policy is ``pi_phi(a_t | o_t, v_t, z_t)``: the current
proprioceptive observation together with the body velocity and the context
vector estimated by CENet. This module plugs CENet into rsl_rl's ``MLPModel``::

    obs["actor"]    (o_t)
    CENet(history)  (v_hat_t, z_t)
        -> [o_t, v_hat_t, z_t] -> MLP -> action distribution

CENet is a submodule of the actor, so it shares the actor's parameters,
optimizer, state dict and export path. The policy treats ``v_hat_t`` and ``z_t``
as inputs, so the actor path stops the gradient (``detach``): PPO's policy
gradient never reaches the CENet encoder, which is optimized by the auxiliary
``L_CE`` loss alone (paper II-C: "CENet is optimized using a hybrid loss
function"). The policy consumes the posterior mean ``mu_t`` (deterministic
given the history), which keeps the PPO importance ratio exact; the VAE branch
of the auxiliary loss samples ``z_t`` from the posterior instead.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from mjlab.tasks.dreamwaq.rl.network.cenet import CENet, CENetOutput

BOOT_FLAG_KEY = "cenet_boot_flag"
"""Observation key holding AdaBoot's per-step mask (1 = use v_hat, 0 = use v_gt).

Written by ``DreamwaqPPO`` during rollouts and replayed from the rollout
storage during updates, so the policy input is identical in both passes.
"""


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  """Return an observation tensor, rejecting nested observation groups."""
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(
      f"Expected observation '{key}' to be a tensor, got {type(value).__name__}."
    )
  return value


class DreamwaqActorModel(MLPModel):
  """Policy MLP fed with ``[o_t, v_hat_t, z_t]`` estimated by CENet.

  The policy MLP input grows from the actor observation (``o_t``) to
  ``o_t + v_hat + z``, so ``MLPModel``'s action distribution, PPO handling and
  state dict logic are reused as is. CENet consumes the observation history
  ``o_t^H`` from a separate observation group.
  """

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
    cenet_cfg: dict | None = None,
  ) -> None:
    """Initialize the actor and its CENet.

    Args:
      obs: Observations used to resolve the observation dimensions.
      obs_groups: Observation sets resolved by rsl_rl. CENet's history group is
        taken from the ``"cenet"`` set, or from ``cenet_cfg["obs_group"]``.
        The group may be flattened ``(num_envs, L * obs_dim)`` or keep the time
        axis ``(num_envs, L, obs_dim)``; the frame count is inferred from it.
      obs_set: Observation set feeding the policy MLP (``"actor"``).
      output_dim: Action dimension.
      hidden_dims: Hidden dimensions of the policy MLP (paper: 512/256/128).
      activation: Activation of the policy MLP.
      obs_normalization: Whether to normalize ``o_t`` before the policy MLP.
      distribution_cfg: Output distribution config (Gaussian for PPO).
      cenet_cfg: Optional CENet overrides: ``obs_group``, ``history_length``,
        ``latent_dim``, ``encoder_hidden_dims``, ``decoder_hidden_dims``,
        ``activation``, ``obs_normalization``, ``history_order``.
    """
    cenet_cfg = dict(cenet_cfg or {})

    obs_group = cenet_cfg.pop("obs_group", None)
    if obs_group is None:
      configured = obs_groups.get("cenet", [])
      if len(configured) > 1:
        raise ValueError(
          "CENet consumes a single observation group, but the 'cenet' "
          f"observation set lists {configured}."
        )
      obs_group = configured[0] if configured else "cenet_input"
    if obs_group not in obs:
      raise ValueError(
        f"CENet observation group '{obs_group}' is missing from the "
        f"environment observations: {list(obs.keys())}."
      )

    latent_dim = int(cenet_cfg.pop("latent_dim", 16))

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
    self._cenet_obs_group = obs_group
    self._cenet_extra_dim = CENet.velocity_dim + latent_dim
    self._last_cenet_output: CENetOutput | None = None
    self._v_target_group = str(cenet_cfg.pop("velocity_target_group", "cenet_target_v"))

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
          f"CENet history frames have dimension {frame_dim}, expected "
          f"{self.obs_dim} (the policy observation dimension)."
        )
      history_length = frame_count
    else:
      if frame_dim % self.obs_dim != 0:
        raise ValueError(
          f"CENet history dimension {frame_dim} is not a multiple of the "
          f"policy observation dimension {self.obs_dim}."
        )
      history_length = frame_dim // self.obs_dim
    configured_length = cenet_cfg.pop("history_length", None)
    if configured_length is not None and int(configured_length) != history_length:
      raise ValueError(
        f"cenet_cfg history_length={configured_length} does not match the "
        f"observation group ({history_length} frames)."
      )

    self.cenet = CENet(
      obs_dim=self.obs_dim,
      history_length=int(history_length),
      latent_dim=latent_dim,
      encoder_hidden_dims=tuple(cenet_cfg.pop("encoder_hidden_dims", (128, 64))),
      decoder_hidden_dims=tuple(cenet_cfg.pop("decoder_hidden_dims", (64, 128))),
      activation=cenet_cfg.pop("activation", activation),
      obs_normalization=bool(cenet_cfg.pop("obs_normalization", True)),
      history_order=cenet_cfg.pop("history_order", "oldest_first"),
      log_var_range=tuple(cenet_cfg.pop("log_var_range", (-10.0, 10.0))),
    )
    if cenet_cfg:
      raise ValueError(f"Unknown CENet config keys: {sorted(cenet_cfg)}.")

  @property
  def cenet_group(self) -> str:
    """Observation group holding the flattened CENet history input."""
    return self._cenet_obs_group

  @property
  def velocity_target_group(self) -> str:
    """Observation group holding the ground-truth base linear velocity v_t."""
    return self._v_target_group

  def bind_debug_env(self, env: Any) -> None:
    """Expose the latest velocity estimate to a play environment.

    The viewer's debug visualizer (the task's velocity command term) reads
    ``env.cenet_velocity_estimate`` and draws it as an arrow next to the
    commanded and measured velocities. Only the play path binds an
    environment, so training never pays this cost.
    """
    self._debug_env = env
    env.cenet_velocity_estimate = torch.zeros(
      env.num_envs, self.cenet.velocity_dim, device=env.device
    )
    env.cenet_latent_estimate = torch.zeros(
      env.num_envs, self.cenet.latent_dim, device=env.device
    )
    env.cfg.viewer.latent_bar_attr = "cenet_latent_estimate"
    env.cfg.viewer.latent_bar_label = "DreamWaQ CENet mu"
    env.cfg.viewer.latent_bar_range = (-3.0, 3.0)

  def _publish_velocity_estimate(self, velocity: torch.Tensor) -> None:
    """Mirror the estimate onto the bound play environment (debug only)."""
    env = getattr(self, "_debug_env", None)
    if env is not None:
      env.cenet_velocity_estimate.copy_(velocity.detach())

  def _publish_latent_estimate(self, latent: torch.Tensor) -> None:
    env = getattr(self, "_debug_env", None)
    if env is not None:
      env.cenet_latent_estimate.copy_(latent.detach())

  @property
  def last_cenet_output(self) -> CENetOutput:
    """CENet outputs of the most recent forward pass.

    The auxiliary losses read this right after re-running the actor on a
    mini-batch; the cached tensors keep their autograd graph.
    """
    if self._last_cenet_output is None:
      raise RuntimeError("CENet has not run yet; forward the actor first.")
    return self._last_cenet_output

  def _get_latent_dim(self) -> int:
    """Dimension of the policy input: ``[o_t, v_hat_t, z_t]``."""
    return self.obs_dim + self._cenet_extra_dim

  def _cenet_history(self, obs: TensorDict) -> torch.Tensor:
    """Return the observation history tensor from the observation dict."""
    history = obs[self._cenet_obs_group]
    if not isinstance(history, torch.Tensor):
      raise TypeError(
        f"Expected observation '{self._cenet_obs_group}' to be a tensor, "
        f"got {type(history).__name__}."
      )
    return history

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Concatenate ``o_t`` with the CENet velocity and context estimates."""
    proprio = super().get_latent(obs, masks, hidden_state)
    # ``sample=False`` uses the posterior mean as the policy's context vector.
    # The VAE reconstruction loss samples ``z_t`` separately, so the policy
    # input stays deterministic and the PPO ratio has no resampling noise.
    cenet_output = self.cenet(self._cenet_history(obs), sample=False)
    self._last_cenet_output = cenet_output
    # Stop-gradient: the actor sees v_hat and z as inputs, so the PPO loss must
    # not update CENet (paper II-C defines CENet's objective as L_CE alone).
    # CENet keeps its training signal from the auxiliary losses, which read the
    # live tensors cached above.
    velocity = cenet_output.velocity.detach()
    self._publish_velocity_estimate(velocity)
    self._publish_latent_estimate(cenet_output.mu.detach())
    # AdaBoot (paper Eq. 8): swap the velocity estimate for the simulated
    # ground truth wherever the AdaBoot mask says so. The mask is absent unless
    # AdaBoot is enabled, which also covers inference where no ground truth
    # exists and the estimate is always used.
    if BOOT_FLAG_KEY in obs:
      mask = _tensor(obs, BOOT_FLAG_KEY)
      ground_truth = _tensor(obs, self._v_target_group)
      velocity = mask * velocity + (1.0 - mask) * ground_truth
    return torch.cat((proprio, velocity, cenet_output.z.detach()), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    """Update the actor and CENet observation normalization statistics."""
    super().update_normalization(obs)
    self.cenet.update_normalization(self._cenet_history(obs))

  def as_onnx(self, verbose: bool) -> nn.Module:
    """Return the ONNX export wrapper, CENet included.

    The wrapper takes the flattened observation history and uses its newest
    frame as ``o_t``, so the policy must consume a single observation group.
    """
    if len(self.obs_groups) != 1:
      raise ValueError(
        "ONNX export assumes the policy consumes a single observation group "
        f"whose newest history frame doubles as o_t, got {self.obs_groups}."
      )
    return _OnnxDreamwaqActorModel(self, verbose)

  def as_jit(self) -> nn.Module:
    """TorchScript export is not supported for the CENet actor."""
    raise NotImplementedError(
      "TorchScript export is not supported for DreamwaqActorModel. Use "
      "as_onnx(), which exports the full CENet + policy stack."
    )


class _OnnxDreamwaqActorModel(nn.Module):
  """Deterministic inference wrapper used for ONNX export.

  The exported model takes the flattened observation history ``o_t^H`` (the
  newest frame doubles as ``o_t``) and returns the deterministic action.
  """

  is_recurrent: bool = False

  def __init__(self, model: DreamwaqActorModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cenet = copy.deepcopy(model.cenet)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.cenet.input_dim
    self.obs_dim = model.cenet.obs_dim
    self.history_length = model.cenet.history_length
    self.newest_first = model.cenet.history_order == "newest_first"

  def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Run the deterministic policy from a flattened observation history."""
    velocity, mu, _ = self.cenet.encode(obs_history)
    frames = obs_history.reshape(-1, self.history_length, self.obs_dim)
    proprio = frames[:, 0] if self.newest_first else frames[:, -1]
    latent = torch.cat((self.obs_normalizer(proprio), velocity, mu), dim=-1)
    return self.deterministic_output(self.mlp(latent))

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
