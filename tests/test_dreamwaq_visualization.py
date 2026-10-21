"""Tests for the DreamWaQ predicted-velocity debug arrow."""

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from conftest import get_test_device
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.dreamwaq.config.go1.env_cfg import (
  unitree_go1_dreamwaq_flat_env_cfg,
)
from mjlab.tasks.dreamwaq.config.go1.rl_cfg import (
  unitree_go1_dreamwaq_ppo_runner_cfg,
)
from mjlab.tasks.dreamwaq.rl.network.actor import DreamwaqActorModel
from mjlab.tasks.dreamwaq.rl.runner import DreamwaqOnPolicyRunner
from mjlab.viewer.debug_visualizer import NullDebugVisualizer
from mjlab.viewer.viewer_config import ViewerConfig

PREDICTED_COLOR = (1.0, 0.2, 0.8, 0.8)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def play_env(device):
  cfg = unitree_go1_dreamwaq_flat_env_cfg(play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  yield env
  env.close()


class RecordingVisualizer(NullDebugVisualizer):
  """Collects the arrows the environment draws."""

  def __init__(self) -> None:
    super().__init__(env_idx=0, meansize=0.1)
    self.arrows: list[tuple[np.ndarray, np.ndarray, Any, str | None]] = []

  def add_arrow(
    self,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    del width
    self.arrows.append((np.asarray(start), np.asarray(end), tuple(color), label))


def make_obs(num_envs: int = 2, obs_dim: int = 6, history: int = 3) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(num_envs, obs_dim),
      "critic": torch.randn(num_envs, 5),
      "cenet_input": torch.randn(num_envs, history, obs_dim),
      "cenet_target_o": torch.randn(num_envs, obs_dim),
      "cenet_target_v": torch.randn(num_envs, 3),
    },
    batch_size=[num_envs],
  )


def make_actor(
  num_envs: int = 2, obs_dim: int = 6, history: int = 3
) -> DreamwaqActorModel:
  torch.manual_seed(0)
  return DreamwaqActorModel(
    obs=make_obs(num_envs, obs_dim, history),
    obs_groups={"actor": ["actor"], "critic": ["critic"], "cenet": ["cenet_input"]},
    obs_set="actor",
    output_dim=2,
    hidden_dims=(8,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def test_model_publishes_velocity_estimate_to_bound_env():
  actor = make_actor()
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    cfg=SimpleNamespace(viewer=ViewerConfig()),
  )
  actor.bind_debug_env(env)
  assert env.cenet_velocity_estimate.shape == (2, 3)
  assert not env.cenet_velocity_estimate.any()
  assert env.cenet_latent_estimate.shape == (2, 16)

  actor(make_obs())
  assert torch.allclose(env.cenet_velocity_estimate, actor.last_cenet_output.velocity)
  torch.testing.assert_close(
    env.cenet_latent_estimate,
    actor.last_cenet_output.mu.detach(),
  )
  assert env.cfg.viewer.latent_bar_attr == "cenet_latent_estimate"
  assert env.cfg.viewer.latent_bar_range == (-3.0, 3.0)


def test_command_term_draws_predicted_velocity_arrow(play_env):
  play_env.cenet_velocity_estimate = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
  recorder = RecordingVisualizer()
  # Same call the viewer makes for the command manager. The sensors' own debug
  # visuals are unrelated here and need a stepped simulation.
  play_env.command_manager.debug_vis(recorder)

  # Commanded/actual linear+angular arrows plus the predicted-velocity arrow.
  predicted = [arrow for arrow in recorder.arrows if arrow[2] == PREDICTED_COLOR]
  assert len(predicted) == 1
  start, end, _, label = predicted[0]
  assert label == "CENet v_hat"

  term = play_env.command_manager.get_term("twist")
  expected_length = term.cfg.viz.scale * 0.5  # |v_hat_xy| of environment 0
  assert float(np.linalg.norm(end - start)) == pytest.approx(expected_length, rel=1e-3)
  # The second environment has a zero estimate: nothing is drawn for it.
  assert len(recorder.arrows) == 5


def test_runner_binds_policy_to_play_env(play_env, device):
  venv = RslRlVecEnvWrapper(play_env, clip_actions=None)
  runner = DreamwaqOnPolicyRunner(
    venv,
    dataclasses.asdict(unitree_go1_dreamwaq_ppo_runner_cfg()),
    log_dir=None,
    device=device,
  )
  policy = runner.get_inference_policy(device)
  assert isinstance(policy, DreamwaqActorModel)
  assert policy._debug_env is play_env

  with torch.inference_mode():
    actions = policy(venv.get_observations())
  venv.step(actions)
  assert torch.allclose(
    play_env.cenet_velocity_estimate, policy.last_cenet_output.velocity
  )
  assert play_env.cenet_velocity_estimate.shape == (play_env.num_envs, 3)
