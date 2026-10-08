"""Tests for the DreamWaQ reproduction task (no simulator required)."""

from dataclasses import asdict
from typing import cast

import pytest
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.tasks.dreamwaq.cenet import CENetActorModel
from mjlab.tasks.dreamwaq.dreamwaq_ppo import DreamWaQPPO
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

TASK_ID = "Mjlab-DreamWaQ-Rough-Unitree-Go1"


class _DummyEnv:
  """Minimal stand-in for `RslRlVecEnvWrapper` used by `construct_algorithm`."""

  def __init__(self, num_envs: int, num_actions: int) -> None:
    self.num_envs = num_envs
    self.num_actions = num_actions


def _make_obs(num_envs: int = 4) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(num_envs, 5, 45),
      "priv": torch.randn(num_envs, 3),
      "critic": torch.randn(num_envs, 259),
    },
    batch_size=[num_envs],
  )


def _model_cfg() -> dict:
  """Actor kwargs as the runner passes them (None-valued options removed)."""
  cfg = dict(asdict(load_rl_cfg(TASK_ID))["actor"])
  cfg.pop("class_name")
  if cfg.get("cnn_cfg") is None:
    cfg.pop("cnn_cfg", None)
  if cfg.get("rnn_type") is None:
    for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      cfg.pop(key, None)
  return cfg


def test_task_is_registered() -> None:
  assert TASK_ID in list_tasks()


def test_observation_layout() -> None:
  cfg = load_env_cfg(TASK_ID)
  actor = cfg.observations["actor"]
  assert actor.history_length == 5
  assert actor.flatten_history_dim is False
  assert actor.enable_corruption is True
  assert len(actor.terms) == 6  # omega, gravity, command, q, qd, a_{t-1}
  assert "base_lin_vel" not in actor.terms
  assert "height_scan" not in actor.terms
  assert cfg.observations["priv"].terms.keys() == {"base_lin_vel"}
  assert cfg.scale_rewards_by_dt is False
  assert load_env_cfg(TASK_ID, play=True).observations["actor"].enable_corruption is (
    False
  )


def test_reward_weights_match_paper() -> None:
  cfg = load_env_cfg(TASK_ID)
  expected = {
    "track_linear_velocity": 1.0,
    "track_angular_velocity": 0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -0.2,
    "dof_acc": -2.5e-7,
    "joint_power": -2e-5,
    "base_height": -1.0,
    "feet_clearance": -0.01,
    "action_rate": -0.01,
    "smoothness": -0.01,
    "power_distribution": -1e-5,
  }
  for name, weight in expected.items():
    assert cfg.rewards[name].weight == pytest.approx(weight)


def test_cenet_actor_forward_and_loss() -> None:
  obs = _make_obs()
  obs_groups = {"actor": ["actor"], "critic": ["critic"], "priv": ["priv"]}
  model = CENetActorModel(obs, obs_groups, "actor", 12, **_model_cfg())

  actions = model(obs, stochastic_output=True)
  assert actions.shape == (4, 12)
  assert model.output_std.shape[-1] == 12

  loss, terms = model.cenet_loss(obs, next_obs=torch.randn(4, 45))
  loss.backward()
  assert torch.isfinite(loss)
  assert set(terms) == {"cenet", "cenet_est", "cenet_rec", "cenet_kl"}
  assert any(p.grad is not None for p in model.cenet.parameters())


def test_dreamwaq_ppo_rollout_and_update() -> None:
  num_envs = 4
  obs = _make_obs(num_envs)
  env = _DummyEnv(num_envs=num_envs, num_actions=12)
  cfg = asdict(load_rl_cfg(TASK_ID))
  cfg["num_steps_per_env"] = 4
  cfg["multi_gpu"] = None
  cfg["algorithm"]["num_learning_epochs"] = 1
  cfg["algorithm"]["num_mini_batches"] = 2

  alg = DreamWaQPPO.construct_algorithm(obs, cast(VecEnv, env), cfg, device="cpu")

  for _ in range(cfg["num_steps_per_env"]):
    actions = alg.act(obs)
    assert actions.shape == (num_envs, 12)
    next_obs = _make_obs(num_envs)
    alg.process_env_step(
      next_obs,
      rewards=torch.rand(num_envs),
      dones=torch.zeros(num_envs),
      extras={"time_outs": torch.zeros(num_envs, dtype=torch.bool)},
    )
    obs = next_obs

  alg.compute_returns(obs)
  loss_dict = alg.update()
  for key in ("value", "surrogate", "cenet", "pboot", "bootstrap"):
    assert key in loss_dict
  assert 0.0 <= alg.pboot <= 1.0
  # The next-frame target must have been filled in by `process_env_step`.
  assert torch.isfinite(torch.tensor(loss_dict["cenet"]))
