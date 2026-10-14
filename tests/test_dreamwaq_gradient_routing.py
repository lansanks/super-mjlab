"""Gradient routing between the DreamWaQ policy and its CENet.

Paper II-C defines CENet's objective as the hybrid loss ``L_CE`` alone, and the
policy treats ``v_hat_t`` and ``z_t`` as inputs. The actor path therefore stops
the gradient, so the PPO loss must not reach the CENet encoder, while the
auxiliary losses must still drive it.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mjlab.tasks.dreamwaq.rl.network.actor import DreamwaqActorModel

NUM_ENVS = 4
ACTION_DIM = 2
OBS_DIM = 6
HISTORY = 3
CRITIC_DIM = 5
OBS_GROUPS = {"actor": ["actor"], "critic": ["critic"], "cenet": ["cenet_input"]}


def make_obs() -> TensorDict:
  """Fake observations with the DreamWaQ groups (mirrors the AdaBoot tests)."""
  return TensorDict(
    {
      "actor": torch.randn(NUM_ENVS, OBS_DIM),
      "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
      "cenet_input": torch.randn(NUM_ENVS, HISTORY, OBS_DIM),
      "cenet_target_o": torch.randn(NUM_ENVS, OBS_DIM),
      "cenet_target_v": torch.randn(NUM_ENVS, 3),
    },
    batch_size=[NUM_ENVS],
  )


def tensor_of(obs: TensorDict, key: str) -> torch.Tensor:
  """Return an observation tensor (TensorDict indexing is a Tensor|TensorDict union)."""
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected observation '{key}' to be a tensor.")
  return value


def make_actor() -> DreamwaqActorModel:
  torch.manual_seed(0)
  return DreamwaqActorModel(
    obs=make_obs(),
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(8,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def test_policy_loss_does_not_reach_cenet():
  """The actor path stops the gradient: PPO leaves CENet parameters untouched."""
  actor = make_actor()
  obs = make_obs()

  actions = actor(obs, stochastic_output=True)
  actor.get_output_log_prob(actions).mean().backward()

  # The policy MLP is driven by the policy loss ...
  assert any(p.grad is not None for p in actor.mlp.parameters())
  # ... while CENet receives no gradient at all from it.
  assert all(p.grad is None for p in actor.cenet.parameters())


def test_auxiliary_losses_still_reach_cenet():
  """The cached CENet outputs stay live for L_est + L_recon + beta KL."""
  actor = make_actor()
  obs = make_obs()
  # The forward pass caches the graph-carrying outputs read by the losses.
  actor(obs, stochastic_output=True)
  output = actor.last_cenet_output

  est = F.mse_loss(output.velocity, tensor_of(obs, "cenet_target_v"))
  z = actor.cenet.reparameterize(output.mu, output.log_var)
  recon = F.mse_loss(
    actor.cenet.decode(z), actor.cenet.normalize_obs(tensor_of(obs, "cenet_target_o"))
  )
  kl = actor.cenet.kl_divergence(output.mu, output.log_var).mean()
  (est + recon + kl).backward()

  grads = [p.grad for p in actor.cenet.parameters()]
  assert any(g is not None and torch.any(g != 0) for g in grads)
