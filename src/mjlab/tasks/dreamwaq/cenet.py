"""DreamWaQ context-aided estimator (CENet) as an RSL-RL actor model.

Reference: Nahrendra et al., "DreamWaQ: Learning Robust Quadrupedal Locomotion
With Implicit Terrain Imagination via Deep Reinforcement Learning", ICRA 2023.

CENet is a single shared encoder with two heads:

* a body-velocity head that regresses ``v_t`` from a history of proprioceptive
  observations ``o^H_t = [o_t, o_{t-1}, ..., o_{t-H}]`` (supervised by the
  privileged simulator velocity), and
* a beta-VAE head that infers a 16-dim context ``z_t`` and decodes the next
  proprioceptive frame ``o_{t+1}``.

The policy (actor) consumes ``[o_t, v_t, z_t]``. Because the actor, the
velocity estimator and the context estimator share one encoder, a single
policy-gradient step also shapes the estimator (the "interplay between policy
and value networks" described in the paper).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_class
from tensordict import TensorDict


@dataclass
class CENetCfg:
  """Architecture of the context-aided estimator network."""

  obs_dim: int = 45
  """Dimension of a single proprioceptive frame."""
  history_length: int = 5
  """Number of proprioceptive frames stacked into the encoder input."""
  latent_dim: int = 16
  """Dimension of the context vector ``z_t``."""
  encoder_hidden_dims: tuple[int, ...] = (128, 64)
  """Hidden layers of the shared encoder (paper: 225 -> 128 -> 64 -> 19)."""
  decoder_hidden_dims: tuple[int, ...] = (64, 128)
  """Hidden layers of the next-observation decoder."""
  beta: float = 1.0
  """Weight of the KL term in the beta-VAE loss."""


class CENet(nn.Module):
  """Shared encoder + velocity head + beta-VAE context head + decoder."""

  def __init__(self, cfg: CENetCfg) -> None:
    super().__init__()
    self.cfg = cfg

    in_dim = cfg.obs_dim * cfg.history_length
    encoder_dims = [in_dim, *cfg.encoder_hidden_dims]
    encoder: list[nn.Module] = []
    for d_in, d_out in zip(encoder_dims[:-1], encoder_dims[1:], strict=True):
      encoder += [nn.Linear(d_in, d_out), nn.ELU()]
    self.encoder = nn.Sequential(*encoder)

    hidden = cfg.encoder_hidden_dims[-1]
    self.velocity_head = nn.Linear(hidden, 3)
    self.latent_mean_head = nn.Linear(hidden, cfg.latent_dim)
    self.latent_logvar_head = nn.Linear(hidden, cfg.latent_dim)

    decoder_dims = [3 + cfg.latent_dim, *cfg.decoder_hidden_dims]
    decoder: list[nn.Module] = []
    for d_in, d_out in zip(decoder_dims[:-1], decoder_dims[1:], strict=True):
      decoder += [nn.Linear(d_in, d_out), nn.ELU()]
    decoder.append(nn.Linear(decoder_dims[-1], cfg.obs_dim))
    self.decoder = nn.Sequential(*decoder)

  def encode(
    self, obs_history: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(v_hat, z, mu, logvar)`` for a batch of observation histories."""
    features = self.encoder(obs_history)
    v_hat = self.velocity_head(features)
    mu = self.latent_mean_head(features)
    logvar = self.latent_logvar_head(features)
    # Reparameterization trick. The std is clamped for numerical stability.
    std = torch.exp(0.5 * logvar).clamp(min=1e-6)
    z = mu + std * torch.randn_like(std)
    return v_hat, z, mu, logvar

  def encode_deterministic(
    self, obs_history: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(v_hat, z)`` using the posterior mean (inference / export)."""
    features = self.encoder(obs_history)
    return self.velocity_head(features), self.latent_mean_head(features)

  def decode(self, v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Decode ``[v, z]`` into the next proprioceptive frame."""
    return self.decoder(torch.cat([v, z], dim=-1))


class CENetActorModel(nn.Module):
  """RSL-RL 5.x actor model whose latent is produced by :class:`CENet`.

  The environment must expose two extra observation groups:

  * ``history_group`` (default ``"history"``): ``(num_envs, H, obs_dim)`` stack of
    the proprioceptive frames, oldest first (``flatten_history_dim=False``).
  * ``priv_group`` (default ``"priv"``): the privileged base linear velocity,
    used as the regression target of the velocity head and (optionally) as the
    AdaBoot teacher signal.
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    cenet_cfg: dict[str, Any] | None = None,
    history_group: str = "history",
    priv_group: str = "priv",
  ) -> None:
    """Initialize the model.

    Args:
        obs: Observation dictionary from the environment.
        obs_groups: Mapping from observation set (e.g. ``"actor"``) to the list
            of observation groups it consumes.
        obs_set: Observation set used by this model (``"actor"``).
        output_dim: Action dimension.
        hidden_dims: Hidden dimensions of the policy MLP.
        activation: Activation function of the policy MLP.
        obs_normalization: Whether to normalize the policy input with running
            statistics. Off by default: DreamWaQ relies on fixed observation
            scales so that the standard-normal prior of the VAE is meaningful.
        distribution_cfg: Config of the action distribution (e.g. Gaussian).
        cenet_cfg: Extra keyword arguments for :class:`CENetCfg`.
        history_group: Name of the observation group holding the frame stack.
        priv_group: Name of the observation group holding the privileged
            base linear velocity.
    """
    super().__init__()
    del obs_normalization  # Kept for API compatibility with `RslRlModelCfg`.

    self.obs_groups = list(obs_groups[obs_set])
    self.history_group = history_group
    self.priv_group = priv_group
    self.cenet_cfg = CENetCfg(**(cenet_cfg or {}))
    self.cenet = CENet(self.cenet_cfg)

    # Validate the environment observation layout early: catching shape
    # mismatches here is much cheaper than debugging PPO later.
    hist_shape = tuple(obs[history_group].shape[1:])
    expected = (self.cenet_cfg.history_length, self.cenet_cfg.obs_dim)
    if hist_shape != expected:
      raise ValueError(
        f"Observation group '{history_group}' has shape {hist_shape} per env, "
        f"expected {expected}. Set `history_length` on the group and "
        "`flatten_history_dim=False`."
      )
    if tuple(obs[priv_group].shape[1:]) != (3,):
      raise ValueError(
        f"Observation group '{priv_group}' must be the 3-dim base linear "
        f"velocity, got shape {tuple(obs[priv_group].shape[1:])}."
      )

    # Action distribution.
    if distribution_cfg is not None:
      dist_class, dist_cfg = resolve_class(distribution_cfg)
      self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
      mlp_output_dim: int = self.distribution.input_dim
    else:
      self.distribution = None
      mlp_output_dim = output_dim

    actor_in_dim = self.cenet_cfg.obs_dim + 3 + self.cenet_cfg.latent_dim
    self.mlp = MLP(actor_in_dim, mlp_output_dim, hidden_dims, activation)
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

    # Set by the training algorithm when AdaptiveBootstrapping decides that the
    # policy should consume the privileged (simulator) velocity instead of the
    # estimate. Always False at deployment.
    self.use_privileged_velocity = False

  # Observation helpers.

  def history_frames(self, obs: TensorDict) -> torch.Tensor:
    """Return the frame stack as ``(batch, H, obs_dim)`` (oldest first)."""
    frames = cast(torch.Tensor, obs[self.history_group])
    if frames.dim() == 2:  # Flattened history: (batch, H * obs_dim).
      frames = frames.view(-1, self.cenet_cfg.history_length, self.cenet_cfg.obs_dim)
    return frames

  def encoder_input(self, obs: TensorDict) -> torch.Tensor:
    """Return ``o^H_t = [o_t, o_{t-1}, ..., o_{t-H}]`` flattened per sample."""
    # The buffer stores frames oldest-first; the paper lists the newest frame
    # first, hence the flip.
    return self.history_frames(obs).flip(1).reshape(obs.batch_size[0], -1)

  def current_frame(self, obs: TensorDict) -> torch.Tensor:
    """Return the newest proprioceptive frame ``o_t``."""
    return self.history_frames(obs)[:, -1, :]

  # Model API expected by RSL-RL's PPO.

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Concatenate ``[o_t, v_t, z_t]`` into the policy input."""
    del masks, hidden_state
    obs_history = self.encoder_input(obs)
    v_hat, z, _, _ = self.cenet.encode(obs_history)
    velocity = (
      cast(torch.Tensor, obs[self.priv_group])
      if self.use_privileged_velocity
      else v_hat
    )
    return torch.cat([self.current_frame(obs), velocity, z], dim=-1)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: torch.Tensor | None = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    latent = self.get_latent(obs, masks, hidden_state)
    mlp_output = self.mlp(latent)
    if self.distribution is None:
      return mlp_output
    if stochastic_output:
      self.distribution.update(mlp_output)
      return self.distribution.sample()
    return self.distribution.deterministic_output(mlp_output)

  def cenet_loss(
    self, obs: TensorDict, next_obs: torch.Tensor
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute ``L_CE = L_est + L_VAE`` for a mini-batch.

    Args:
        obs: Mini-batch observation dictionary. Must contain the history group
            and the privileged velocity group.
        next_obs: Next proprioceptive frame ``o_{t+1}`` per sample.

    Returns:
        The CENet loss and a dict of its individual terms (for logging).
    """
    obs_history = self.encoder_input(obs)
    v_hat, z, mu, logvar = self.cenet.encode(obs_history)
    decode = self.cenet.decode(v_hat, z)

    vel_target = cast(torch.Tensor, obs[self.priv_group]).detach()
    decode_target = next_obs.detach()

    est_loss = F.mse_loss(v_hat, vel_target)
    rec_loss = F.mse_loss(decode, decode_target)
    # KL(q(z|o^H) || N(0, I)) for a diagonal Gaussian posterior.
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl_loss = kl.mean()

    loss = est_loss + rec_loss + self.cenet_cfg.beta * kl_loss
    terms = {
      "cenet": loss.detach(),
      "cenet_est": est_loss.detach(),
      "cenet_rec": rec_loss.detach(),
      "cenet_kl": kl_loss.detach(),
    }
    return loss, terms

  # Recurrent-model API (no-ops).

  def reset(self, dones: torch.Tensor | None = None) -> None:
    del dones

  def get_hidden_state(self) -> None:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  # Distribution helpers, mirroring `rsl_rl.models.MLPModel`.

  @property
  def output_mean(self) -> torch.Tensor:
    return self.distribution.mean  # type: ignore[union-attr]

  @property
  def output_std(self) -> torch.Tensor:
    return self.distribution.std  # type: ignore[union-attr]

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.distribution.entropy  # type: ignore[union-attr]

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    return self.distribution.params  # type: ignore[union-attr]

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

  def get_kl_divergence(
    self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
  ) -> torch.Tensor:
    return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

  def update_normalization(self, obs: TensorDict) -> None:
    """No-op: DreamWaQ uses fixed observation scales instead of running stats."""
    del obs

  # Export.

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxCENetActor(self, verbose)


class _OnnxCENetActor(nn.Module):
  """Export wrapper: observation history in, deterministic actions out."""

  is_recurrent: bool = False

  def __init__(self, model: CENetActorModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.cfg = model.cenet_cfg
    self.cenet = copy.deepcopy(model.cenet)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Run inference from a flattened ``[o_t, ..., o_{t-H}]`` history."""
    frames = obs_history.view(-1, self.cfg.history_length, self.cfg.obs_dim).flip(1)
    obs_t = frames[:, -1, :]
    v_hat, z = self.cenet.encode_deterministic(obs_history)
    latent = torch.cat([obs_t, v_hat, z], dim=-1)
    return self.deterministic_output(self.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.cfg.history_length * self.cfg.obs_dim),)

  @property
  def input_size(self) -> int:
    return self.cfg.history_length * self.cfg.obs_dim

  @property
  def input_names(self) -> list[str]:
    return ["obs_history"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
