"""Tests for the HIM PPO plumbing: successor shift, masks and optimizers."""

from typing import cast

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.tasks.himloco.rl.algorithm.him_ppo import HimPPO
from mjlab.tasks.himloco.rl.network.actor import HimActorModel

NUM_ENVS = 4
NUM_STEPS = 3
OBS_DIM = 5
HISTORY = 3
STATE_DIM = 6
ACTION_DIM = 2
OBS_GROUPS = {
  "actor": ["actor"],
  "critic": ["critic"],
  "him": ["him_input"],
}


def tensor_of(obs: TensorDict, key: str) -> torch.Tensor:
  """Return an observation tensor (TensorDict indexing is a Tensor|TensorDict union)."""
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected observation '{key}' to be a tensor.")
  return value


def make_obs(step: int, num_envs: int = NUM_ENVS) -> TensorDict:
  """Observations whose state frames encode the time step they belong to."""
  obs = TensorDict(
    {
      "actor": torch.randn(num_envs, OBS_DIM),
      "critic": torch.randn(num_envs, OBS_DIM),
      "him_input": torch.randn(num_envs, HISTORY, OBS_DIM),
      "him_state": torch.full((num_envs, STATE_DIM), float(step)),
    },
    batch_size=[num_envs],
  )
  return obs


def make_algorithm() -> HimPPO:
  torch.manual_seed(0)
  obs = make_obs(0)
  actor = HimActorModel(
    obs=obs,
    obs_groups=OBS_GROUPS,
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(8,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    him_cfg={
      "obs_group": "him_input",
      "state_group": "him_state",
      "latent_dim": 4,
      "encoder_hidden_dims": (8,),
      "target_hidden_dims": (8,),
      "num_prototypes": 4,
    },
  )
  critic = MLPModel(
    obs=obs,
    obs_groups=OBS_GROUPS,
    obs_set="critic",
    output_dim=1,
    hidden_dims=(8,),
  )
  storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [ACTION_DIM], "cpu")
  return HimPPO(
    actor,
    critic,
    storage,
    device="cpu",
    num_learning_epochs=1,
    num_mini_batches=1,
    learning_rate=1e-3,
  )


def fill_storage(alg: HimPPO, terminated_at_last_step: bool = False) -> None:
  """Record ``NUM_STEPS`` transitions, using each observation as its own state."""
  for step in range(NUM_STEPS):
    obs = make_obs(step)
    with torch.no_grad():
      actions = alg.actor(obs, stochastic_output=True)
      log_prob = alg.actor.get_output_log_prob(actions)
      params = tuple(p.detach().clone() for p in alg.actor.output_distribution_params)
    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.actions = actions.detach()
    transition.values = torch.zeros(NUM_ENVS, 1)
    transition.actions_log_prob = log_prob.detach()
    transition.distribution_params = params
    transition.rewards = torch.ones(NUM_ENVS)
    transition.dones = torch.zeros(NUM_ENVS)
    alg.storage.add_transition(transition)
  if terminated_at_last_step:
    alg.storage.dones[-1, 0] = 1


def test_compute_returns_injects_shifted_state_and_validity():
  alg = make_algorithm()
  fill_storage(alg, terminated_at_last_step=True)
  final_obs = make_obs(NUM_STEPS)
  alg.compute_returns(final_obs)

  state = tensor_of(alg.storage.observations, "him_state")
  next_state = tensor_of(alg.storage.observations, "him_state_next")
  # next_state[t] == state[t + 1] and the last row comes from the final obs.
  assert torch.allclose(next_state[:-1], state[1:])
  assert torch.allclose(next_state[-1], tensor_of(final_obs, "him_state"))
  # Terminal transitions are flagged invalid.
  assert torch.allclose(
    tensor_of(alg.storage.observations, "him_valid"),
    1.0 - alg.storage.dones.float(),
  )
  assert tensor_of(alg.storage.observations, "him_valid")[-1, 0] == 0.0


def test_estimator_is_trained_by_its_own_optimizer():
  alg = make_algorithm()
  fill_storage(alg)
  alg.compute_returns(make_obs(NUM_STEPS))

  actor = alg.actor
  assert isinstance(actor, HimActorModel)
  # The estimator parameters are not in the PPO optimizer.
  ppo_params = {id(p) for group in alg.optimizer.param_groups for p in group["params"]}
  estimator_params = {id(p) for p in actor.estimator.parameters()}
  assert ppo_params.isdisjoint(estimator_params)
  assert set(alg.estimator_optimizer.param_groups[0]["params"]) == set(
    actor.estimator.parameters()
  )

  before = [p.detach().clone() for p in actor.estimator.parameters()]
  losses = alg.update()
  after = list(actor.estimator.parameters())

  # The HIO loss is reported and finite, and the estimator moved.
  for key in ("value", "surrogate", "entropy", "him_est", "him_swap", "him_total"):
    assert key in losses
    assert torch.isfinite(torch.tensor(losses[key]))
  assert any(
    not torch.allclose(old, new) for old, new in zip(before, after, strict=True)
  )
  # The estimator follows PPO's (adaptive) learning rate.
  assert alg.estimator_optimizer.param_groups[0]["lr"] == alg.learning_rate
  # Prototype diagnostics are published for the next rollout step.
  assert "Metrics/him_proto_entropy" in alg._pending_metrics
  assert "Metrics/him_swap_accuracy" in alg._pending_metrics


def test_estimator_gradients_are_cleared_before_ppo_clipping():
  alg = make_algorithm()
  fill_storage(alg)
  alg.compute_returns(make_obs(NUM_STEPS))

  actor = alg.actor
  assert isinstance(actor, HimActorModel)
  alg.update()

  # HIO gradients stay private to the estimator optimizer. The PPO clip only
  # sees actor and critic parameters, and no stale estimator gradient survives.
  assert all(parameter.grad is None for parameter in actor.estimator.parameters())
  policy_parameters = set(alg._policy_parameters())
  assert policy_parameters.isdisjoint(set(actor.estimator.parameters()))


def test_terminal_rows_are_dropped_from_the_hio_loss():
  alg = make_algorithm()
  fill_storage(alg)
  alg.compute_returns(make_obs(NUM_STEPS))
  # Mark every transition invalid except one sample: Sinkhorn needs at least two
  # samples, so the HIO update is skipped instead of running on a single row.
  alg.storage.observations["him_valid"][:] = 0.0
  alg.storage.observations["him_valid"][0, 0] = 1.0
  batch = cast(TensorDict, alg.storage.observations.flatten(0, 1))
  assert alg._update_estimator(batch) is None


def test_rollout_metric_is_reported():
  alg = make_algorithm()
  # The actor forward of step t publishes the estimate for step t + 1.
  alg.act(make_obs(0))
  extras: dict = {"log": {}}
  alg.process_env_step(
    make_obs(1), torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), extras
  )
  assert "Metrics/him_vel_error" in extras["log"]
  assert extras["log"]["Metrics/him_vel_error"] >= 0.0
