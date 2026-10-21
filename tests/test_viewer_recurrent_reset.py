"""Tests for viewer-side reset of stateful policies."""

from types import SimpleNamespace
from typing import Any

import torch

from mjlab.viewer.base import BaseViewer
from mjlab.viewer.viewer_config import ViewerConfig


class _Policy:
  is_recurrent = True

  def __init__(self) -> None:
    self.reset_calls: list[torch.Tensor | None] = []

  def __call__(self, obs) -> torch.Tensor:
    del obs
    return torch.zeros(2, 1)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.reset_calls.append(dones)


class _Env:
  num_envs = 2
  cfg: Any = SimpleNamespace(viewer=ViewerConfig())
  device = "cpu"

  def __init__(self) -> None:
    self.next_dones = torch.tensor([0, 1])

  @property
  def unwrapped(self):
    return self

  def get_observations(self) -> torch.Tensor:
    return torch.zeros(2, 3)

  def step(self, actions):
    del actions
    return (
      self.get_observations(),
      torch.zeros(2),
      self.next_dones,
      {},
    )

  def reset(self) -> None:
    pass

  def close(self) -> None:
    pass


class _Viewer(BaseViewer):
  def setup(self) -> None:
    pass

  def sync_env_to_viewer(self) -> None:
    pass

  def sync_viewer_to_env(self) -> None:
    pass

  def close(self) -> None:
    pass

  def is_running(self) -> bool:
    return True


def test_execute_step_resets_policy_only_for_done_environments():
  env = _Env()
  policy = _Policy()
  viewer = _Viewer(env, policy)

  assert viewer._execute_step()
  torch.testing.assert_close(policy.reset_calls[0], torch.tensor([0, 1]))

  env.next_dones = torch.tensor([0, 0])
  assert viewer._execute_step()
  assert len(policy.reset_calls) == 1
