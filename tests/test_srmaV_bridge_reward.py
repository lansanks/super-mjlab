"""Tests for per-foot broken-bridge crossing rewards."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.srmaV.config.black.terrain_cfg import (
  BLACK_SRMAV_BRIDGE_TERRAIN,
  BLACK_SRMAV_TERRAINS_CFG,
)
from mjlab.tasks.srmaV.mdp.rewards import bridge_gap_crossing


class _Scene(dict):
  def __init__(self, robot, sensor, terrain) -> None:
    super().__init__(
      robot=robot,
      feet_ground_contact=sensor,
    )
    self.terrain = terrain


def _make_reward():
  bridge_column = list(BLACK_SRMAV_TERRAINS_CFG.sub_terrains).index(
    BLACK_SRMAV_BRIDGE_TERRAIN
  )
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=BLACK_SRMAV_TERRAINS_CFG),
    terrain_origins=torch.zeros(10, 4, 3),
    terrain_levels=torch.zeros(2, dtype=torch.long),
    terrain_types=torch.tensor([bridge_column, 0]),
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(site_pos_w=torch.zeros(2, 4, 3)),
    find_sites=lambda names, preserve_order: (
      [0, 1, 2, 3],
      list(names),
    ),
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.zeros(2, 4, dtype=torch.long),
      force=torch.zeros(2, 4, 3),
    )
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    step_dt=0.02,
    scene=_Scene(robot, sensor, terrain),
    command_manager=SimpleNamespace(
      get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ),
    extras={"log": {}},
  )
  cfg = RewardTermCfg(
    func=bridge_gap_crossing,
    weight=1.0,
    params={
      "contact_sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot",
        site_names=("FL", "FR", "RR", "RL"),
      ),
      "terrain_name": BLACK_SRMAV_BRIDGE_TERRAIN,
    },
  )
  reward = bridge_gap_crossing(cfg, cast(Any, env))
  return reward, env, terrain


def _call(reward, env) -> torch.Tensor:
  return reward(
    cast(Any, env),
    contact_sensor_name="feet_ground_contact",
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot"),
    terrain_name=BLACK_SRMAV_BRIDGE_TERRAIN,
  )


def test_each_foot_is_rewarded_once_and_four_feet_total_ten() -> None:
  reward, env, _ = _make_reward()
  robot = env.scene["robot"]
  sensor = reward._contact_sensor
  assert sensor.data.found is not None
  assert sensor.data.force is not None

  total_reward = 0.0
  for foot in range(4):
    robot.data.site_pos_w[0, foot, 0] = -37.8
    sensor.data.found[0, foot] = 1
    sensor.data.force[0, foot, 2] = 20.0
    total_reward += float(_call(reward, env)[0])

    # Repeated contact with the same foot must not award again.
    assert float(_call(reward, env)[0]) == 0.0

  assert total_reward * env.step_dt == 10.0
  assert int(reward._current_gap[0]) == 1
  assert not bool(reward._feet_rewarded[0].any())


def test_non_bridge_env_and_reset() -> None:
  reward, env, _ = _make_reward()
  robot = env.scene["robot"]
  sensor = reward._contact_sensor
  assert sensor.data.found is not None
  assert sensor.data.force is not None

  robot.data.site_pos_w[1, :, 0] = -37.8
  sensor.data.found[1] = 1
  sensor.data.force[1, :, 2] = 20.0
  assert float(_call(reward, env)[1]) == 0.0

  robot.data.site_pos_w[0, 0, 0] = -37.8
  sensor.data.found[0, 0] = 1
  sensor.data.force[0, 0, 2] = 20.0
  assert float(_call(reward, env)[0]) == 125.0
  assert bool(reward._feet_rewarded[0, 0])

  reward.reset(torch.tensor([0]))
  assert int(reward._current_gap[0]) == 0
  assert not bool(reward._feet_rewarded[0].any())
