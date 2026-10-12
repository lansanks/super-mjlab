"""Tests for DreamWaQ's adaptive bootstrapping (AdaBoot)."""

import math
from typing import Any

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.tasks.dreamwaq.rl.algorithm.dreamwaq_ppo import DreamwaqPPO
from mjlab.tasks.dreamwaq.rl.network.actor import BOOT_FLAG_KEY, DreamwaqActorModel

NUM_ENVS = 4
NUM_STEPS = 3
ACTION_DIM = 2
OBS_DIM = 6
HISTORY = 3
CRITIC_DIM = 5
OBS_GROUPS = {"actor": ["actor"], "critic": ["critic"], "cenet": ["cenet_input"]}


def make_obs() -> TensorDict:
  """Build a fake observation dict with all DreamWaQ groups."""
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


def make_algorithm(
  adaboot: bool = True, granularity: str = "env"
) -> tuple[DreamwaqPPO, DreamwaqActorModel, RolloutStorage]:
  """Build a DreamWaQ algorithm around fake observations (no simulator)."""
  torch.manual_seed(0)
  obs = make_obs()
  actor = DreamwaqActorModel(
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
  )
  critic = MLPModel(
    obs=obs, obs_groups=OBS_GROUPS, obs_set="critic", output_dim=1, hidden_dims=(8,)
  )
  storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [ACTION_DIM], device="cpu")
  alg = DreamwaqPPO(
    actor,
    critic,
    storage,
    device="cpu",
    num_learning_epochs=1,
    num_mini_batches=1,
    learning_rate=1e-3,
    dreamwaq_cfg={
      "adaboot": adaboot,
      "boot_granularity": granularity,
      # The fake observations are one term of OBS_DIM dimensions.
      "recon_terms": {"obs": OBS_DIM},
    },
  )
  return alg, actor, storage


def fill_rollout(alg: DreamwaqPPO, storage: RolloutStorage) -> dict[str, Any]:
  """Run one fake rollout without episode endings; return the last extras."""
  obs = make_obs()
  extras: dict[str, Any] = {}
  for _ in range(NUM_STEPS):
    alg.act(obs)
    obs = make_obs()
    extras = {"log": {}}
    alg.process_env_step(
      obs,
      torch.randn(NUM_ENVS),
      torch.zeros(NUM_ENVS, dtype=torch.long),
      extras=extras,
    )
  alg.compute_returns(obs)
  return extras


def test_boot_probability_from_episode_returns():
  alg, _, _ = make_algorithm()
  alg._has_episode_return[:] = True

  alg._last_episode_return[:] = 5.0
  assert alg._boot_probability() == pytest.approx(1.0)

  alg._last_episode_return[:] = torch.tensor([5.0, 6.0, 4.0, 5.0])
  narrow = alg._boot_probability()
  alg._last_episode_return[:] = torch.tensor([5.0, 20.0, 0.5, 9.0])
  wide = alg._boot_probability()
  assert 0.0 < wide < narrow < 1.0

  # Fewer than two completed episodes leaves the CV undefined.
  alg._has_episode_return[:] = False
  alg._has_episode_return[0] = True
  assert alg._boot_probability() == 0.0

  # A non-positive mean must not produce a probability outside [0, 1].
  alg._has_episode_return[:] = True
  alg._last_episode_return[:] = torch.tensor([-5.0, 5.0, -5.0, 5.0])
  assert 0.0 <= alg._boot_probability() <= 1e-6


def test_boot_mask_is_drawn_and_replayed_into_storage():
  alg, _, storage = make_algorithm()
  alg.p_boot = 1.0
  fill_rollout(alg, storage)

  mask = storage.observations[BOOT_FLAG_KEY]
  assert mask.shape == (NUM_STEPS, NUM_ENVS, 1)
  assert torch.all(mask == 1.0)

  batch = next(storage.mini_batch_generator(1, 1))
  assert BOOT_FLAG_KEY in batch.observations
  assert batch.observations[BOOT_FLAG_KEY].shape == (NUM_STEPS * NUM_ENVS, 1)

  # A zero probability must produce an all-ground-truth rollout.
  storage.clear()
  alg.p_boot = 0.0
  fill_rollout(alg, storage)
  assert torch.all(storage.observations[BOOT_FLAG_KEY] == 0.0)


def test_iteration_granularity_draws_once_per_rollout():
  alg, _, storage = make_algorithm(granularity="iteration")
  alg.p_boot = 0.5
  for _ in range(4):
    storage.clear()
    fill_rollout(alg, storage)
    mask = storage.observations[BOOT_FLAG_KEY]
    # One draw for the whole rollout: every (step, env) shares the outcome.
    assert float(mask.min()) == float(mask.max())


def test_policy_input_mixes_estimate_and_ground_truth():
  alg, actor, _ = make_algorithm()
  obs = make_obs()

  estimate_latent = actor.get_latent(obs)
  # AdaBoot off for this dict: the estimate is used everywhere.
  assert actor.last_cenet_output is not None

  obs[BOOT_FLAG_KEY] = torch.ones(NUM_ENVS, 1)
  assert torch.allclose(actor.get_latent(obs), estimate_latent, atol=1e-6)

  obs[BOOT_FLAG_KEY] = torch.zeros(NUM_ENVS, 1)
  ground_truth_latent = actor.get_latent(obs)
  target_v = obs["cenet_target_v"]
  assert isinstance(target_v, torch.Tensor)
  expected = torch.cat(
    (
      estimate_latent[:, :OBS_DIM],
      target_v,
      estimate_latent[:, OBS_DIM + 3 :],
    ),
    dim=-1,
  )
  assert torch.allclose(ground_truth_latent, expected, atol=1e-6)

  obs[BOOT_FLAG_KEY] = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
  mixed_latent = actor.get_latent(obs)
  assert torch.allclose(mixed_latent[0], estimate_latent[0], atol=1e-6)
  assert torch.allclose(mixed_latent[1], expected[1], atol=1e-6)


def test_adaboot_disabled_keeps_legacy_behavior():
  alg, _, storage = make_algorithm(adaboot=False)
  obs = make_obs()
  alg.act(obs)
  assert BOOT_FLAG_KEY not in obs

  extras = fill_rollout(alg, storage)
  assert BOOT_FLAG_KEY not in storage.observations
  assert alg.p_boot == 0.0
  # The CENet velocity error is always worth logging; the AdaBoot metrics are not.
  assert "Metrics/velocity_est_error" in extras["log"]
  assert "Metrics/adaboot_pboot" not in extras["log"]
  assert "Metrics/adaboot_fraction" not in extras["log"]

  losses = alg.update()
  assert "adaboot_pboot" not in losses
  assert "cenet_total" in losses


def test_update_with_adaboot_produces_finite_losses():
  alg, _, storage = make_algorithm()
  # Pretend every environment completed one episode with a small spread, so
  # the next rollout bootstraps with a high probability.
  alg._has_episode_return[:] = True
  alg._last_episode_return[:] = torch.tensor([10.0, 10.5, 11.0, 9.5])
  fill_rollout(alg, storage)
  assert alg.p_boot > 0.9

  losses = alg.update()
  assert "cenet_total" in losses
  assert all(math.isfinite(value) for value in losses.values())
  assert storage.step == 0


def test_metrics_track_velocity_error_and_adaboot_probability():
  alg, _, storage = make_algorithm()
  alg._has_episode_return[:] = True
  alg._last_episode_return[:] = torch.tensor([10.0, 10.5, 11.0, 9.5])

  # First rollout uses the initial p_boot = 0 and updates it at the end.
  extras = fill_rollout(alg, storage)
  metrics = extras["log"]
  assert metrics["Metrics/velocity_est_error"] >= 0.0
  assert metrics["Metrics/est_vel_mean_pct"] >= 0.0
  # The fake harness declares a single observation term named "obs".
  assert metrics["Metrics/recon_obs_mean_pct"] >= 0.0
  assert metrics["Metrics/adaboot_pboot"] == 0.0
  assert metrics["Metrics/adaboot_fraction"] == 0.0
  assert alg.p_boot > 0.9

  # Second rollout reports the updated probability and mostly uses the estimate.
  storage.clear()
  extras = fill_rollout(alg, storage)
  metrics = extras["log"]
  assert metrics["Metrics/adaboot_pboot"] == pytest.approx(alg.p_boot)
  assert 0.0 <= metrics["Metrics/adaboot_fraction"] <= 1.0
  # Rollout-level summary: almost all draws use the estimate at p_boot ~ 0.94.
  assert alg._boot_fraction > 0.5


def test_checkpoint_round_trip_restores_adaboot_state():
  alg, _, _ = make_algorithm()
  alg.p_boot = 0.37
  alg._episode_return[:] = 1.5
  alg._last_episode_return[:] = torch.arange(float(NUM_ENVS))
  alg._has_episode_return[:] = True

  saved = alg.save()
  assert "adaboot_state_dict" in saved

  restored, _, _ = make_algorithm()
  restored.load(saved, None, True)
  assert restored.p_boot == pytest.approx(0.37)
  assert torch.equal(restored._episode_return, alg._episode_return)
  assert torch.equal(restored._last_episode_return, alg._last_episode_return)
  assert torch.equal(restored._has_episode_return, alg._has_episode_return)

  # Older checkpoints have no AdaBoot state: restart from zero, loudly.
  saved.pop("adaboot_state_dict")
  restored.p_boot = 0.5
  restored.load(saved, None, True)
  assert restored.p_boot == 0.0
  assert not restored._has_episode_return.any()


def test_non_finite_minibatch_is_skipped(monkeypatch, capsys):
  """A NaN loss must not reach the optimizer or the parameters."""
  alg, actor, storage = make_algorithm()
  fill_rollout(alg, storage)
  before = {name: param.detach().clone() for name, param in actor.named_parameters()}

  # Force a NaN value loss, the failure mode that poisons the action std.
  monkeypatch.setattr(
    alg.critic, "forward", lambda *args, **kwargs: torch.full((1, 1), math.nan)
  )
  losses = alg.update()

  assert all(math.isfinite(value) for value in losses.values()), losses
  for name, param in actor.named_parameters():
    assert torch.equal(param.detach(), before[name]), name
  assert "skipped 1/" in capsys.readouterr().out
  assert storage.step == 0


def test_load_tolerates_different_environment_count(capsys):
  """A 4096-env checkpoint must load into the 1-env play environment."""
  alg, _, _ = make_algorithm()
  alg.p_boot = 0.42
  alg._episode_return[:] = 1.5
  alg._last_episode_return[:] = torch.arange(float(NUM_ENVS))
  alg._has_episode_return[:] = True
  saved = alg.save()

  restored, _, _ = make_algorithm()
  # Simulate a smaller play environment: one environment instead of NUM_ENVS.
  restored._episode_return = torch.zeros(1)
  restored._last_episode_return = torch.zeros(1)
  restored._has_episode_return = torch.zeros(1, dtype=torch.bool)
  restored.load(saved, {"actor": True}, True)

  assert restored.p_boot == pytest.approx(0.42)
  assert restored._episode_return.shape == (1,)
  assert not restored._episode_return.any()
  assert not restored._has_episode_return.any()
  assert "different number of environments" in capsys.readouterr().out


def test_update_applies_finite_minibatches():
  alg, actor, storage = make_algorithm()
  fill_rollout(alg, storage)
  before = {name: param.detach().clone() for name, param in actor.named_parameters()}

  losses = alg.update()
  assert all(math.isfinite(value) for value in losses.values())
  changed = sum(
    1
    for name, param in actor.named_parameters()
    if not torch.equal(param.detach(), before[name])
  )
  assert changed > 0
  assert storage.step == 0
