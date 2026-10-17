"""HIM PPO: PPO with the hybrid internal optimization (HIO) auxiliary losses.

On top of rsl_rl's PPO this algorithm trains the hybrid internal model::

    L_est  = MSE(v_hat, 2 * v_{t+1})        # successor body velocity, scaled
    L_swap = SwAV(l_t, l_target)            # contrastive cluster assignment
    L_HIO  = est_coef * L_est + swap_coef * L_swap

The estimator has its own Adam (the released implementation does the same) and
is updated once per mini-batch, *before* the PPO loss is back-propagated. The
policy path is stop-gradient, so the two updates do not share gradients and the
estimator needs no PPO gradient to be reduced across GPUs.

Successor targets: the environment stores the current state frame in
``him_state``. Right before the update this algorithm shifts it one step along
the time axis (``him_state_next[t] = him_state[t+1]``, the last row filled from
the observation returned after the final environment step) and injects it into
the rollout storage, so the shuffled mini-batches carry it automatically.
Transitions that ended an episode are dropped from the HIO loss: their
successor observation belongs to a new episode, and using it would train the
response model against a reset state.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab.tasks.himloco.rl.network.actor import HimActorModel
from mjlab.tasks.himloco.rl.network.estimator import HIMEstimator


def _tensor(obs: TensorDict, key: str) -> torch.Tensor:
  """Return an observation tensor, rejecting nested observation groups."""
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(
      f"Expected observation '{key}' to be a tensor, got {type(value).__name__}."
    )
  return value


class HimPPO(PPO):
  """PPO with HIM's estimator trained by a dedicated optimizer."""

  state_group = "him_state"
  """Observation group holding the current successor-state frame."""

  next_state_key = "him_state_next"
  """Storage key holding the successor state; assembled in compute_returns."""

  valid_key = "him_valid"
  """Storage key marking steps with a valid successor (1 = valid)."""

  def __init__(self, *args, him_cfg: dict | None = None, **kwargs) -> None:
    """Initialize the algorithm.

    Args:
      *args: Positional arguments forwarded to :class:`rsl_rl.algorithms.PPO`
        (actor, critic and rollout storage).
      him_cfg: Optional overrides: ``state_group`` (observation group holding
        the state frame), ``est_coef`` and ``swap_coef`` (loss weights),
        ``max_grad_norm`` (estimator gradient clipping, the released code uses
        10) and ``sinkhorn_eps`` / ``sinkhorn_iters`` (SwAV transport plan).
      **kwargs: Keyword arguments forwarded to :class:`rsl_rl.algorithms.PPO`.
    """
    super().__init__(*args, **kwargs)
    cfg = dict(him_cfg or {})
    actor = self.actor
    if not isinstance(actor, HimActorModel):
      raise TypeError(
        f"HimPPO requires a HimActorModel actor, got {type(actor).__name__}."
      )
    if self.is_multi_gpu:
      raise NotImplementedError(
        "HimPPO does not support multi-GPU training yet: the estimator has its "
        "own optimizer and its gradients are not reduced across GPUs."
      )

    self.state_group = str(
      cfg.pop("state_group", getattr(actor, "state_group", self.state_group))
    )
    self.est_coef = float(cfg.pop("est_coef", 1.0))
    self.swap_coef = float(cfg.pop("swap_coef", 1.0))
    self.estimator_max_grad_norm = float(cfg.pop("max_grad_norm", 10.0))
    self.sinkhorn_eps = float(cfg.pop("sinkhorn_eps", 0.05))
    self.sinkhorn_iters = int(cfg.pop("sinkhorn_iters", 3))
    if cfg:
      raise ValueError(f"Unknown him_cfg keys: {sorted(cfg)}.")

    # The estimator is trained by its own optimizer. It is a submodule of the
    # actor (so it follows the actor's state dict and export path), which means
    # the PPO optimizer was built over its parameters too; drop them there so
    # the main Adam only owns the policy MLP and the critic.
    estimator_ids = {id(p) for p in actor.estimator.parameters()}
    for group in self.optimizer.param_groups:
      group["params"] = [p for p in group["params"] if id(p) not in estimator_ids]
    self.estimator_optimizer = torch.optim.Adam(
      actor.estimator.parameters(), lr=self.learning_rate
    )

    self._pending_metrics: dict[str, float] = {}

  def compute_returns(self, obs: TensorDict) -> None:
    """Compute returns and assemble the successor-state targets."""
    super().compute_returns(obs)
    st = self.storage
    # Shift [s_t] one step along the time axis: [s_{t+1}], last row from the
    # observation returned after the final environment step. This must happen
    # before the mini-batch generator shuffles the (time, env) pairs.
    state = _tensor(st.observations, self.state_group)
    last_state = _tensor(obs, self.state_group)
    st.observations[self.next_state_key] = torch.cat(
      (state[1:], last_state.unsqueeze(0)), dim=0
    )
    # Terminal steps are followed by a reset, so their "next state" belongs to
    # a new episode and must not supervise the response model.
    st.observations[self.valid_key] = 1.0 - st.dones.float()

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
  ) -> None:
    """Publish HIM metrics, then record the transition."""
    metrics = extras.get("log")
    if metrics is not None:
      metrics.update(self._rollout_metrics(obs, dones))
      # Prototype diagnostics are computed during the update; they become
      # visible in the next rollout step (the logger reads extras["log"]).
      metrics.update(self._pending_metrics)
    super().process_env_step(obs, rewards, dones, extras)

  def _rollout_metrics(self, obs: TensorDict, dones: torch.Tensor) -> dict[str, float]:
    """Velocity estimation error of the policy step that was just taken.

    ``obs`` is the post-step observation ``o_{t+1}``, so its state frame holds
    the ground truth the previous estimate was regressed against. The estimate
    and the target are both scaled by ``obs_scales.lin_vel = 2``; the metric is
    reported in m/s to stay comparable to DreamWaQ's velocity error.
    """
    actor = self.actor
    output = actor.last_him_output if isinstance(actor, HimActorModel) else None
    if output is None:
      return {}
    estimate = output.velocity.detach()
    state = _tensor(obs, self.state_group)[..., -HIMEstimator.velocity_dim :].detach()
    error = (estimate - state).norm(dim=-1)[dones.reshape(-1) <= 0]
    if error.numel() == 0:
      return {}
    return {"Metrics/him_vel_error": float(error.mean()) / 2.0}

  def update(self) -> dict[str, float]:
    """Run PPO epochs with the HIO auxiliary loss on the estimator."""
    if self.rnd is not None or self.symmetry is not None:
      raise NotImplementedError(
        "HimPPO does not support the RND or symmetry extensions."
      )

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_est_loss = 0.0
    mean_swap_loss = 0.0
    mean_hio_loss = 0.0
    valid_updates = 0
    estimator_updates = 0
    skipped_ppo = 0
    skipped_estimator = 0
    proto_entropy: list[float] = []
    proto_active: list[float] = []
    swap_accuracy: list[float] = []

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )

    for batch in generator:
      observations = batch.observations
      if observations is None:
        raise RuntimeError("HimPPO requires an RL mini-batch with PPO fields.")
      actions = batch.actions
      old_actions_log_prob = batch.old_actions_log_prob
      old_distribution_params = batch.old_distribution_params
      old_values = batch.values
      returns = batch.returns
      advantages = batch.advantages
      if (
        actions is None
        or old_actions_log_prob is None
        or old_distribution_params is None
        or old_values is None
        or returns is None
        or advantages is None
      ):
        raise RuntimeError("HimPPO requires an RL mini-batch with PPO fields.")

      # Check if we should normalize advantages per mini-batch
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch.advantages = advantages

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
        distribution_params = tuple(self.actor.output_distribution_params)
        entropy = self.actor.output_entropy

        # Compute KL divergence and adapt the learning rate
        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(
              old_distribution_params, distribution_params
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
              param_group["lr"] = self.learning_rate

      # Hybrid internal optimization: the estimator forward runs in full
      # precision (SwAV's exp() is sensitive to low-precision scores) and its
      # own optimizer applies the update before the PPO update of this batch.
      estimator_stats = self._update_estimator(observations)
      if estimator_stats is None:
        skipped_estimator += 1
      else:
        est_loss, swap_loss, diagnostics = estimator_stats
        estimator_updates += 1
        mean_est_loss += est_loss
        mean_swap_loss += swap_loss
        mean_hio_loss += self.est_coef * est_loss + self.swap_coef * swap_loss
        proto_entropy.append(diagnostics["proto_entropy"])
        proto_active.append(diagnostics["proto_active"])
        swap_accuracy.append(diagnostics["swap_accuracy"])

      with torch.amp.autocast(
        device_type=torch.device(self.device).type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
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

        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
        )

      # Guard against a non-finite loss (e.g. a diverged estimator): skipping
      # the mini-batch keeps NaN out of the shared optimizer state.
      if not bool(torch.isfinite(loss)):
        self.optimizer.zero_grad()
        skipped_ppo += 1
        continue

      # Compute the gradients for PPO
      self.optimizer.zero_grad()
      loss.backward()

      # Apply the gradients for PPO
      actor_grad_norm = nn.utils.clip_grad_norm_(
        self.actor.parameters(), self.max_grad_norm
      )
      critic_grad_norm = nn.utils.clip_grad_norm_(
        self.critic.parameters(), self.max_grad_norm
      )
      if not (
        bool(torch.isfinite(actor_grad_norm)) and bool(torch.isfinite(critic_grad_norm))
      ):
        self.optimizer.zero_grad()
        skipped_ppo += 1
        continue
      self.optimizer.step()
      valid_updates += 1

      # Store the losses
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()

    # Update the normalizers (no-op when observation normalization is disabled)
    obs = cast(TensorDict, self.storage.observations.flatten(0, 1))
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)

    # Divide the losses by the number of updates
    num_updates = max(valid_updates, 1)
    num_estimator_updates = max(estimator_updates, 1)
    if skipped_ppo or skipped_estimator:
      total_batches = self.num_learning_epochs * self.num_mini_batches
      print(
        f"[WARN] HimPPO skipped {skipped_ppo}/{total_batches} non-finite PPO "
        f"mini-batches and {skipped_estimator} estimator mini-batches."
      )
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_est_loss /= num_estimator_updates
    mean_swap_loss /= num_estimator_updates
    mean_hio_loss /= num_estimator_updates

    self._pending_metrics = {}
    if proto_entropy:
      self._pending_metrics = {
        "Metrics/him_proto_entropy": sum(proto_entropy) / len(proto_entropy),
        "Metrics/him_proto_active": sum(proto_active) / len(proto_active),
        "Metrics/him_swap_accuracy": sum(swap_accuracy) / len(swap_accuracy),
      }

    # Clear the storage
    self.storage.clear()

    return {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "him_est": mean_est_loss,
      "him_swap": mean_swap_loss,
      "him_total": mean_hio_loss,
    }

  def _update_estimator(
    self, observations: TensorDict
  ) -> tuple[float, float, dict[str, float]] | None:
    """Run one HIO update of the estimator.

    Terminal transitions are dropped: their successor observation belongs to
    the next episode. Rows are removed (instead of zeroing the loss) so the
    Sinkhorn plan still balances the batch it is computed on.
    """
    actor = self.actor
    if not isinstance(actor, HimActorModel):
      raise TypeError(f"Expected a HimActorModel actor, got {type(actor).__name__}.")
    valid = _tensor(observations, self.valid_key).reshape(-1) > 0
    if int(valid.sum()) < 2:
      return None
    history = _tensor(observations, actor.him_group)[valid]
    successor = _tensor(observations, self.next_state_key)[valid]

    est_loss, swap_loss, diagnostics = actor.estimator.losses(
      history,
      successor,
      eps=self.sinkhorn_eps,
      iters=self.sinkhorn_iters,
    )
    loss = self.est_coef * est_loss + self.swap_coef * swap_loss
    self.estimator_optimizer.zero_grad()
    if not bool(torch.isfinite(loss)):
      return None
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(
      actor.estimator.parameters(), self.estimator_max_grad_norm
    )
    if not bool(torch.isfinite(grad_norm)):
      self.estimator_optimizer.zero_grad()
      return None
    # Keep the estimator's learning rate in sync with PPO's adaptive schedule
    # (the released implementation passes lr=self.learning_rate on every call).
    for param_group in self.estimator_optimizer.param_groups:
      param_group["lr"] = self.learning_rate
    self.estimator_optimizer.step()
    return float(est_loss.item()), float(swap_loss.item()), diagnostics

  def save(self) -> dict:
    """Save model state plus the estimator's optimizer state."""
    saved_dict = super().save()
    saved_dict["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """Restore model state and the estimator's optimizer state."""
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    estimator_state = loaded_dict.get("estimator_optimizer_state_dict")
    if estimator_state is None:
      print(
        "[WARN] Checkpoint has no estimator optimizer state; the HIO optimizer "
        "restarts from a fresh Adam."
      )
    else:
      self.estimator_optimizer.load_state_dict(estimator_state)
    return load_iteration
