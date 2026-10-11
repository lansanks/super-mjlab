"""DreamWaQ PPO: PPO with the CENet auxiliary losses.

On top of rsl_rl's PPO this algorithm optimizes CENet's hybrid loss together
with the policy::

    L_est   = MSE(v_hat_t, v_t)             # body velocity, simulator truth
    L_recon = MSE(o_hat_{t+1}, o_{t+1})     # next observation, done steps masked
    L_kl    = KL(q(z_t | o_t^H) || N(0, I)) # diagonal Gaussian posterior

    loss = L_PPO + ce_coef * (L_est + L_recon + beta * L_kl)

The whole loss is optimized in a single backward pass with the shared optimizer.
CENet lives inside the actor (see ``DreamwaqActorModel``), so its parameters
already belong to ``actor.parameters()``.

Reconstruction targets: the environment stores the clean current observation in
``cenet_target_o``; right before the update this algorithm shifts it one step
along the time axis (``o_{t+1}`` = next stored frame, last row filled from the
observation returned after the final environment step) and injects it into the
rollout storage, so the shuffled mini-batches carry it automatically. Steps
that ended an episode are masked out, because their "next frame" belongs to a
new episode.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch.distributed import ReduceOp, all_reduce, broadcast

from mjlab.tasks.dreamwaq.rl.network.actor import DreamwaqActorModel


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  """Return an observation tensor, rejecting nested observation groups."""
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(
      f"Expected observation '{key}' to be a tensor, got {type(value).__name__}."
    )
  return value


class DreamwaqPPO(PPO):
  """PPO with DreamWaQ's CENet auxiliary losses."""

  recon_target_key = "cenet_recon_target"
  """Storage key holding ``o_{t+1}``; assembled in :meth:`compute_returns`."""

  recon_mask_key = "cenet_recon_mask"
  """Storage key marking steps with a valid reconstruction target (1 = valid)."""

  def __init__(self, *args, dreamwaq_cfg: dict | None = None, **kwargs) -> None:
    """Initialize the algorithm.

    Args:
      *args: Positional arguments forwarded to :class:`rsl_rl.algorithms.PPO`
        (actor, critic and rollout storage).
      dreamwaq_cfg: Optional overrides: ``beta`` (KL weight), ``ce_coef``
        (weight of ``L_CE`` relative to the PPO loss), ``obs_target_group`` and
        ``velocity_target_group`` (environment observation groups).
      **kwargs: Keyword arguments forwarded to :class:`rsl_rl.algorithms.PPO`.
    """
    super().__init__(*args, **kwargs)
    cfg = dict(dreamwaq_cfg or {})
    self.beta = float(cfg.pop("beta", 1.0))
    self.ce_coef = float(cfg.pop("ce_coef", 1.0))
    self.o_target_group = str(cfg.pop("obs_target_group", "cenet_target_o"))
    self.v_target_group = str(cfg.pop("velocity_target_group", "cenet_target_v"))
    if cfg:
      raise ValueError(f"Unknown dreamwaq_cfg keys: {sorted(cfg)}.")

  def compute_returns(self, obs: TensorDict) -> None:
    """Compute returns and assemble the reconstruction targets."""
    super().compute_returns(obs)
    st = self.storage
    last_obs = _tensor(obs, self.o_target_group)
    # Shift [o_t] one step along the time axis: [o_{t+1}], last row from the
    # observation returned after the final environment step. This must happen
    # before the mini-batch generator shuffles the (time, env) pairs.
    target = torch.cat(
      (_tensor(st.observations, self.o_target_group)[1:], last_obs.unsqueeze(0)),
      dim=0,
    )
    st.observations[self.recon_target_key] = target
    # Terminal steps are followed by a reset, so their "next frame" belongs to
    # a new episode and must not be reconstructed.
    st.observations[self.recon_mask_key] = 1.0 - st.dones.float()

  def update(self) -> dict[str, float]:
    """Run PPO epochs with the CENet auxiliary losses."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_est_loss = 0.0
    mean_recon_loss = 0.0
    mean_kl_loss = 0.0
    # RND loss
    mean_rnd_loss = 0.0 if self.rnd else None
    # Symmetry loss
    mean_symmetry_loss = 0.0 if self.symmetry else None

    # Get mini-batch generator
    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    # Iterate over mini-batches
    for batch in generator:
      initial_observations = batch.observations
      if initial_observations is None:
        raise RuntimeError("DreamwaqPPO requires an RL mini-batch with PPO fields.")
      original_batch_size = initial_observations.batch_size[0]
      rnd_loss: torch.Tensor | None = None
      symmetry_loss: torch.Tensor | None = None

      # Check if we should normalize advantages per mini-batch
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          if batch.advantages is None:
            raise RuntimeError("DreamwaqPPO requires an RL mini-batch with PPO fields.")
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      # Perform symmetric augmentation if enabled
      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      # Read the PPO fields after augmentation so the surrogate and value
      # losses see the augmented batch, matching the base PPO update.
      observations = batch.observations
      actions = batch.actions
      old_actions_log_prob = batch.old_actions_log_prob
      old_distribution_params = batch.old_distribution_params
      old_values = batch.values
      returns = batch.returns
      advantages = batch.advantages
      if (
        observations is None
        or actions is None
        or old_actions_log_prob is None
        or old_distribution_params is None
        or old_values is None
        or returns is None
        or advantages is None
      ):
        raise RuntimeError("DreamwaqPPO requires an RL mini-batch with PPO fields.")

      # Optionally use mixed precision for the forward pass and loss computation
      with torch.amp.autocast(
        device_type=torch.device(self.device).type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        # Recompute actions log prob and entropy for current batch of transitions
        # Note: We need to do this because we updated the policy with new
        # parameters
        self.actor(
          observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[0],
          stochastic_output=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(actions)
        values = self.critic(
          observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[1],
        )
        # Note: We only keep the following tensors for the original samples in
        # case of symmetry augmentation
        distribution_params = tuple(
          p[:original_batch_size] for p in self.actor.output_distribution_params
        )
        entropy = self.actor.output_entropy[:original_batch_size]

        # Compute KL divergence and adapt the learning rate
        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(
              old_distribution_params, distribution_params
            )
            kl_mean = torch.mean(kl)

            # Reduce the KL divergence across all GPUs
            if self.is_multi_gpu:
              all_reduce(kl_mean, op=ReduceOp.SUM)
              kl_mean /= self.gpu_world_size

            # Update the learning rate only on the main process
            if self.gpu_global_rank == 0:
              if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
              elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            # Update the learning rate for all GPUs
            if self.is_multi_gpu:
              lr_tensor = torch.tensor(self.learning_rate, device=self.device)
              broadcast(lr_tensor, src=0)
              self.learning_rate = lr_tensor.item()

            # Update the learning rate for all parameter groups
            for param_group in self.optimizer.param_groups:
              param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
          value_clipped = old_values + (values - old_values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - returns).pow(2)
          value_losses_clipped = (value_clipped - returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (returns - values).pow(2).mean()

        # CENet auxiliary loss (estimation + reconstruction + KL)
        est_loss, recon_loss, ce_kl_loss = self._cenet_losses(
          observations, original_batch_size
        )
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
          + self.ce_coef * (est_loss + recon_loss + self.beta * ce_kl_loss)
        )

        # RND loss
        if self.rnd:
          rnd_loss = self.rnd.compute_loss(
            cast(TensorDict, observations[:original_batch_size])
          )

        # Symmetry loss
        if self.symmetry:
          symmetry_loss = self.symmetry.compute_loss(
            self.actor, batch, original_batch_size
          )
          if self.symmetry.use_mirror_loss:
            loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      # Compute the gradients for PPO
      self.optimizer.zero_grad()
      loss.backward()
      # Compute the gradients for RND
      if self.rnd is not None and rnd_loss is not None:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()

      # Collect gradients from all GPUs
      if self.is_multi_gpu:
        self.reduce_parameters()

      # Apply the gradients for PPO
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      # Apply the gradients for RND
      if self.rnd:
        self.rnd.optimizer.step()

      # Store the losses
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_est_loss += est_loss.item()
      mean_recon_loss += recon_loss.item()
      mean_kl_loss += ce_kl_loss.item()
      # RND loss
      if mean_rnd_loss is not None and rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      # Symmetry loss
      if mean_symmetry_loss is not None and symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    # Update the normalizers
    obs = cast(TensorDict, self.storage.observations.flatten(0, 1))
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    # Divide the losses by the number of updates
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_est_loss /= num_updates
    mean_recon_loss /= num_updates
    mean_kl_loss /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    # Construct the loss dictionary
    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "cenet_est": mean_est_loss,
      "cenet_recon": mean_recon_loss,
      "cenet_kl": mean_kl_loss,
    }
    if mean_rnd_loss is not None:
      loss_dict["rnd"] = mean_rnd_loss
    if mean_symmetry_loss is not None:
      loss_dict["symmetry"] = mean_symmetry_loss

    # Clear the storage
    self.storage.clear()

    return loss_dict

  def _cenet_losses(
    self, observations: TensorDict, batch_size: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CENet's estimation, reconstruction and KL losses."""
    actor = self.actor
    if not isinstance(actor, DreamwaqActorModel):
      raise TypeError(
        f"DreamwaqPPO requires a DreamwaqActorModel actor, got {type(actor).__name__}."
      )
    cenet = actor.cenet
    output = actor.last_cenet_output
    velocity = output.velocity[:batch_size]
    mu = output.mu[:batch_size]
    log_var = output.log_var[:batch_size]

    # L_est: supervised body velocity estimate (simulator ground truth).
    v_gt = _tensor(observations, self.v_target_group)[:batch_size]
    est_loss = F.mse_loss(velocity, v_gt)

    # L_recon: reconstruct the next observation from a posterior sample. The
    # VAE samples z_t here, while the policy consumes the posterior mean.
    z = cenet.reparameterize(mu, log_var)
    target = cenet.normalize_obs(
      _tensor(observations, self.recon_target_key)[:batch_size]
    )
    squared_error = (cenet.decode(z) - target).pow(2).mean(dim=-1)
    mask = _tensor(observations, self.recon_mask_key)[:batch_size].squeeze(-1)
    recon_loss = (squared_error * mask).sum() / mask.sum().clamp_min(1.0)

    # L_kl: regularize the context posterior, not the velocity head.
    kl_loss = cenet.kl_divergence(mu, log_var).mean()
    return est_loss, recon_loss, kl_loss
