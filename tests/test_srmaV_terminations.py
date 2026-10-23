"""Tests for SRMAV per-sub-terrain boundary termination."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.tasks.srmaV.mdp.terminations import out_of_sub_terrain_bounds


class _Scene(dict):
  def __init__(self, robot, terrain) -> None:
    super().__init__(robot=robot)
    self.terrain = terrain


def _make_env(positions: torch.Tensor, episode_steps: int = 3):
  generator = SimpleNamespace(size=(8.0, 8.0))
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_type="generator",
      terrain_generator=generator,
    ),
    terrain_origins=torch.zeros(2, 4, 3),
    terrain_levels=torch.tensor([0, 1]),
    terrain_types=torch.tensor([0, 3]),
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=positions),
  )
  return SimpleNamespace(
    num_envs=len(positions),
    device=positions.device,
    scene=_Scene(robot, terrain),
    episode_length_buf=torch.full((len(positions),), episode_steps),
  )


def test_sub_terrain_bounds_use_assigned_row_and_column() -> None:
  positions = torch.tensor(
    [
      [-4.0, -12.0, 0.3],
      [8.11, 12.0, 0.3],
    ]
  )
  terminated = out_of_sub_terrain_bounds(cast(Any, _make_env(positions)))
  assert terminated.tolist() == [False, True]


def test_sub_terrain_bounds_skip_fresh_reset_steps() -> None:
  positions = torch.tensor([[-4.0, 8.0, 0.3]])
  env = _make_env(positions, episode_steps=2)
  env.scene.terrain.terrain_levels = torch.tensor([0])
  env.scene.terrain.terrain_types = torch.tensor([0])
  env.scene.terrain.terrain_origins = torch.zeros(2, 1, 3)
  assert not bool(out_of_sub_terrain_bounds(cast(Any, env))[0])
