"""PPO variant used by DreamWaQ.

Compared to :class:`rsl_rl.algorithms.PPO` this subclass

1. builds a CENet-based actor (see :mod:`mjlab.tasks.dreamwaq.cenet`),
2. adds the estimator loss ``L_CE = L_est + L_VAE`` to every mini-batch update,
3. implements AdaptiveBootstrapping (AdaBoot): the probability of feeding the
   *estimated* base velocity to the policy is ``1 - tanh(CV(R))``, where ``R``
   is the vector of per-environment rollout rewards. Early in training the
   coefficient of variation is large, so the policy is fed the privileged
   simulator velocity instead of the noisy estimate.

The update loop mirrors ``rsl_rl>=5.5``'s ``PPO.update`` (KL-adaptive learning
rate, clipped surrogate, clipped value loss). RND, symmetry augmentation and
multi-GPU gradient reduction are dropped for clarity; add them back only if you
need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_class, resolve_obs_groups
from tensordict import TensorDict

from mjlab.tasks.dreamwaq.cenet import CENetActorModel

NEXT_OBS_KEY = "dwq_next_obs"
"""Key of the next-frame reconstruction target inside the rollout storage."""


@dataclass
class AdaBootCfg:
  """AdaptiveBootstrapping configuration."""

  enabled: bool = True
  """If False, always feed the estimated velocity to the policy (the
  ``DreamWaQ w/o AdaBoot`` ablation)."""
  smoothing: float = 0.0
  """Exponential smoothing factor for ``p_boot`` across iterations (0 = off)."""
  min_prob: float = 0.0
  max_prob: float = 1.0
  """Clamp for ``p_boot``."""


class DreamWaQPPO(PPO):
  """PPO with a CENet actor and the DreamWaQ estimator loss."""

  actor: CENetActorModel

  def __init__(
    self,
    *args: Any,
    beta: float = 1.0,
    adaboot_cfg: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.actor.cenet_cfg.beta = beta
    self.adaboot_cfg = AdaBootCfg(**(adaboot_cfg or {}))
    self.pboot: float = 1.0
    self._rollout_reward_sum = torch.zeros(
      self.storage.num_envs, device=self.device, dtype=torch.float32
    )
    self.last_cenet_terms: dict[str, float] = {}

  # Rollout hooks.

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    # `obs` is already o_{t+1}: record its newest frame as the reconstruction
    # target of the transition that is about to be written to the storage.
    step = self.storage.step
    next_obs = self.actor.current_frame(obs).detach()
    self.storage.observations[NEXT_OBS_KEY][step].copy_(next_obs)
    self._rollout_reward_sum += rewards.detach()
    super().process_env_step(obs, rewards, dones, extras)

  # Learning.

  def _update_adaboot(self) -> None:
    """Decide whether the policy bootstraps from the estimator next rollout."""
    cfg = self.adaboot_cfg
    if not cfg.enabled:
      self.pboot = 1.0
      self.actor.use_privileged_velocity = False
      return

    rewards = self._rollout_reward_sum
    mean = rewards.mean()
    cv = rewards.std(unbiased=False) / (mean.abs() + 1e-6)
    pboot = float(torch.clamp(1.0 - torch.tanh(cv), cfg.min_prob, cfg.max_prob))
    if cfg.smoothing > 0.0:
      pboot = (1.0 - cfg.smoothing) * pboot + cfg.smoothing * self.pboot
    self.pboot = pboot
    # One coin flip per iteration decides whether *all* environments bootstrap
    # (feed the estimate) or fall back to the privileged velocity.
    bootstrap = bool(torch.rand(()).item() < pboot)
    self.actor.use_privileged_velocity = not bootstrap

  def update(self) -> dict[str, float]:
    """Run PPO epochs and return the mean losses."""
    self._update_adaboot()
    self._rollout_reward_sum.zero_()

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_cenet_loss = 0.0
    cenet_terms: dict[str, float] = {}

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      observations = batch.observations
      assert observations is not None
      old_distribution_params = batch.old_distribution_params
      assert old_distribution_params is not None
      old_values = batch.values
      returns = batch.returns
      advantages = batch.advantages
      assert old_values is not None and returns is not None and advantages is not None
      with torch.amp.autocast(
        device_type=torch.device(self.device).type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        # Re-evaluate the policy under the current parameters.
        self.actor(
          observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[0],
          stochastic_output=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
        values = self.critic(
          observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[1],
        )
        distribution_params = tuple(p for p in self.actor.output_distribution_params)
        entropy = self.actor.output_entropy

        # KL-adaptive learning rate (identical to `PPO.update`).
        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(
              old_distribution_params, distribution_params
            )
            kl_mean = torch.mean(kl)
            if self.gpu_global_rank == 0:
              if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
              elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for param_group in self.optimizer.param_groups:
              param_group["lr"] = self.learning_rate

        # Surrogate loss.
        ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss.
        if self.use_clipped_value_loss:
          value_clipped = old_values + (values - old_values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - returns).pow(2)
          value_losses_clipped = (value_clipped - returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (old_values - returns).pow(2).mean()

        # DreamWaQ estimator loss: velocity regression + beta-VAE.
        cenet_loss, terms = self.actor.cenet_loss(
          observations,
          observations[NEXT_OBS_KEY],  # type: ignore[arg-type]
        )

        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
          + cenet_loss
        )

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_cenet_loss += float(terms["cenet"])
      for key, value in terms.items():
        cenet_terms[key] = cenet_terms.get(key, 0.0) + float(value)

    obs = self.storage.observations.flatten(0, 1)
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "cenet": mean_cenet_loss / num_updates,
      **{k: v / num_updates for k, v in cenet_terms.items() if k != "cenet"},
      "pboot": self.pboot,
      "bootstrap": float(not self.actor.use_privileged_velocity),
    }
    self.last_cenet_terms = {k: v / num_updates for k, v in cenet_terms.items()}

    self.storage.clear()
    return loss_dict

  # Construction.

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict[str, Any], device: str
  ) -> "DreamWaQPPO":
    """Build the DreamWaQ actor, the critic, the storage and the algorithm."""
    # Drop None-valued optional model kwargs (the runner does this too, but the
    # algorithm must also work when constructed directly, e.g. in tests).
    for key in ("actor", "critic"):
      model_cfg = cfg.get(key)
      if not isinstance(model_cfg, dict):
        continue
      for opt in ("cnn_cfg", "distribution_cfg"):
        if model_cfg.get(opt) is None:
          model_cfg.pop(opt, None)
      if model_cfg.get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          model_cfg.pop(opt, None)

    alg_class, alg_cfg = resolve_class(cfg["algorithm"])
    actor_class, actor_cfg = resolve_class(cfg["actor"])
    critic_class, critic_cfg = resolve_class(cfg["critic"])

    for unsupported in ("rnd_cfg", "symmetry_cfg"):
      assert alg_cfg.get(unsupported) in (None, {}), (
        f"DreamWaQPPO does not implement '{unsupported}'."
      )
    alg_cfg.pop("rnd_cfg", None)
    alg_cfg.pop("symmetry_cfg", None)
    alg_cfg.pop("share_cnn_encoders", None)

    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

    actor: CENetActorModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
    ).to(device)
    print(f"Actor Model: {actor}")
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **critic_cfg
    ).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )
    # Buffer for the beta-VAE reconstruction target o_{t+1}. Extra keys added to
    # the storage observation dict are automatically sampled by the mini-batch
    # generator, so no custom storage class is needed.
    obs_dim = int(actor.cenet_cfg.obs_dim)
    storage.observations[NEXT_OBS_KEY] = torch.zeros(
      cfg["num_steps_per_env"], env.num_envs, obs_dim, device=device
    )

    alg: DreamWaQPPO = alg_class(
      actor,
      critic,
      storage,
      device=device,
      **alg_cfg,
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    alg.compile(cfg.get("torch_compile_mode"))
    return alg
