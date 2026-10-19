"""Tests for SRMA-specific reward changes."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.srma import mdp

_STANCE_CLEARANCE = 0.0225


class StubHeightSensor(TerrainHeightSensor):
  def __init__(self, heights: torch.Tensor) -> None:
    cast(Any, self)._data = SimpleNamespace(heights=heights)

  @property
  def data(self):
    return self._data


def test_feet_clearance_only_penalizes_insufficient_height():
  heights = torch.tensor(
    [
      [0.02, 0.10],
      [0.20, 0.02],
    ]
  )
  foot_velocities = torch.tensor(
    [
      [[1.0, 0.0], [1.0, 0.0]],
      [[1.0, 0.0], [1.0, 0.0]],
    ]
  )
  asset = SimpleNamespace(
    data=SimpleNamespace(site_lin_vel_w=foot_velocities),
  )
  env = SimpleNamespace(
    scene={
      "robot": asset,
      "foot_height_scan": StubHeightSensor(heights),
    }
  )
  asset_cfg = SimpleNamespace(name="robot", site_ids=torch.tensor([0, 1]))

  cost = mdp.feet_clearance(
    cast(Any, env),
    target_height=0.1,
    height_sensor_name="foot_height_scan",
    asset_cfg=cast(Any, asset_cfg),
  )

  # Each row is one foot below target by 0.08, while the 0.20 m foot
  # contributes no penalty.
  torch.testing.assert_close(cost, torch.tensor([0.08, 0.08]))


def test_feet_swing_height_uses_fixed_insufficient_height_target():
  num_envs = 2
  num_feet = 2
  heights = torch.tensor(
    [
      [0.20, 0.02],
      [0.20, 0.20],
    ]
  )
  first_contact = torch.ones(num_envs, num_feet, dtype=torch.bool)
  contact_sensor = SimpleNamespace(
    data=SimpleNamespace(found=torch.zeros(num_envs, num_feet)),
    compute_first_contact=lambda dt: first_contact,
  )
  env = SimpleNamespace(
    scene={
      "feet_ground_contact": contact_sensor,
      "foot_height_scan": StubHeightSensor(heights),
    },
    command_manager=SimpleNamespace(
      get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ),
    extras={"log": {}},
  )
  term = object.__new__(mdp.feet_swing_height)
  term.peak_heights = torch.zeros(num_envs, num_feet)
  term.step_dt = 0.02

  cost = term(
    env,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
    target_height=0.1,
    command_name="twist",
    command_threshold=0.05,
  )

  # Environment 0's 0.02 m foot is 0.08 m below the fixed target and
  # contributes (0.08 / 0.10)^2. Environment 1 clears the target with both feet.
  torch.testing.assert_close(cost, torch.tensor([0.64, 0.0]))


class _StubStairAsset:
  def __init__(self, num_envs: int, site_names: tuple[str, ...]) -> None:
    self._site_names = list(site_names)
    self.data = SimpleNamespace(
      site_pos_w=torch.zeros(num_envs, len(site_names), 3),
    )
    self.data.site_pos_w[:, :, 2] = _STANCE_CLEARANCE

  def find_sites(
    self,
    site_names: tuple[str, ...] | None,
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    del preserve_order
    assert site_names is not None
    site_ids = [self._site_names.index(name) for name in site_names]
    return site_ids, list(site_names)


def _make_stair_progress_term(
  num_envs: int = 2,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
  site_names = ("FR", "FL", "RR", "RL")
  asset = _StubStairAsset(num_envs, site_names)
  height_sensor = StubHeightSensor(
    torch.full((num_envs, len(site_names)), _STANCE_CLEARANCE)
  )
  contact_data = SimpleNamespace(
    found=torch.ones(num_envs, len(site_names), dtype=torch.bool),
    force=torch.full((num_envs, len(site_names), 3), 50.0),
  )
  landing = torch.ones(num_envs, len(site_names), dtype=torch.bool)
  contact_sensor = SimpleNamespace(
    data=contact_data,
    compute_first_contact=lambda dt: landing,
  )
  scene = {
    "robot": asset,
    "foot_height_scan": height_sensor,
    "feet_ground_contact": contact_sensor,
  }
  command = torch.tensor(
    [[1.0, 0.0, 0.0]] * num_envs,
    dtype=torch.float,
  )
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    scene=scene,
    command_manager=SimpleNamespace(get_command=lambda name: command),
    extras={"log": {}},
  )
  params = {
    "contact_sensor_name": "feet_ground_contact",
    "height_sensor_name": "foot_height_scan",
    "command_name": "twist",
    "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    "max_step_height": 0.25,
    "min_height_gain": 0.02,
    "support_clearance_max": 0.05,
    "contact_force_threshold": 10.0,
    "command_threshold": 0.1,
    "first_step_multiplier": 3.0,
  }
  cfg = SimpleNamespace(params=params)
  term = mdp.stair_climb_progress(cfg=cast(Any, cfg), env=cast(Any, env))
  return term, env, asset, contact_sensor, params


def _set_ground_heights(asset: Any, heights: torch.Tensor) -> None:
  asset.data.site_pos_w[:, :, 2] = heights + _STANCE_CLEARANCE


def test_stair_climb_progress_rewards_each_foot_on_new_height():
  term, env, asset, contact_sensor, params = _make_stair_progress_term()
  term(env, **params)

  # 单脚第一次踩上高台立即奖励，不需要等待第二只脚。
  one_foot_high = torch.tensor(
    [
      [0.20, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
    ]
  )
  _set_ground_heights(asset, one_foot_high)
  torch.testing.assert_close(term(env, **params), torch.tensor([120.0, 0.0]))
  torch.testing.assert_close(term(env, **params), torch.zeros(2))

  # 另一只脚随后到达同一高度，仍然独立获得一次奖励。
  other_foot_high = torch.tensor([[0.0, 0.20, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
  _set_ground_heights(asset, other_foot_high)
  torch.testing.assert_close(term(env, **params), torch.tensor([120.0, 0.0]))

  # 两脚同时继续到达更高层时，两只脚分别得分。
  higher = torch.tensor([[0.40, 0.40, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
  _set_ground_heights(asset, higher)
  torch.testing.assert_close(term(env, **params), torch.tensor([80.0, 0.0]))
  assert contact_sensor.data.found.all()
  assert env.extras["log"] == {}


def test_stair_climb_progress_accumulates_small_gains_and_does_not_repeat():
  term, env, asset, _, params = _make_stair_progress_term(num_envs=1)
  term(env, **params)

  # 斜坡上不足 2 cm 的升高先累计，达到 2 cm 后一次结算。
  _set_ground_heights(asset, torch.tensor([[0.010, 0.0, 0.0, 0.0]]))
  torch.testing.assert_close(term(env, **params), torch.zeros(1))
  _set_ground_heights(asset, torch.tensor([[0.015, 0.0, 0.0, 0.0]]))
  torch.testing.assert_close(term(env, **params), torch.zeros(1))
  _set_ground_heights(asset, torch.tensor([[0.025, 0.0, 0.0, 0.0]]))
  torch.testing.assert_close(term(env, **params), torch.tensor([15.0]))

  # 停留、下降后回到原高度都不重复给分。
  torch.testing.assert_close(term(env, **params), torch.zeros(1))
  _set_ground_heights(asset, torch.zeros(1, 4))
  torch.testing.assert_close(term(env, **params), torch.zeros(1))
  _set_ground_heights(asset, torch.tensor([[0.025, 0.0, 0.0, 0.0]]))
  torch.testing.assert_close(term(env, **params), torch.zeros(1))


def test_stair_climb_progress_respects_command_and_reset():
  term, env, asset, _, params = _make_stair_progress_term(num_envs=2)
  term(env, **params)
  command = env.command_manager.get_command("twist")
  command[0] = 0.0
  _set_ground_heights(
    asset,
    torch.tensor(
      [
        [0.20, 0.0, 0.0, 0.0],
        [0.20, 0.0, 0.0, 0.0],
      ]
    ),
  )
  torch.testing.assert_close(term(env, **params), torch.tensor([0.0, 120.0]))

  # 零指令期间高度变化只同步基线，恢复指令后不能补领。
  command[0, 0] = 1.0
  torch.testing.assert_close(term(env, **params), torch.zeros(2))

  term.reset()
  _set_ground_heights(
    asset,
    torch.tensor(
      [
        [0.40, 0.0, 0.0, 0.0],
        [0.40, 0.0, 0.0, 0.0],
      ]
    ),
  )
  # reset 后第一次有效支撑只建立新基线。
  torch.testing.assert_close(term(env, **params), torch.zeros(2))
  _set_ground_heights(
    asset,
    torch.tensor(
      [
        [0.60, 0.0, 0.0, 0.0],
        [0.60, 0.0, 0.0, 0.0],
      ]
    ),
  )
  torch.testing.assert_close(term(env, **params), torch.tensor([120.0, 120.0]))


def _make_command_stall_term(
  num_envs: int = 1,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
  asset = SimpleNamespace(
    data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(num_envs, 3))
  )
  command = torch.tensor([[1.0, 0.0, 0.0]] * num_envs)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    scene={"robot": asset},
    command_manager=SimpleNamespace(get_command=lambda name: command),
    extras={"log": {}},
  )
  params = {
    "command_name": "twist",
    "asset_cfg": SceneEntityCfg("robot"),
    "command_threshold": 0.2,
    "stall_speed_threshold": 0.05,
    "recover_speed_threshold": 0.1,
    "command_off_threshold": 0.1,
    "grace_time": 0.5,
    "ramp_time": 1.0,
    "penalty_start": 0.1,
    "penalty_end": 0.5,
  }
  cfg = SimpleNamespace(params=params)
  term = mdp.command_stall_penalty(cfg=cast(Any, cfg), env=cast(Any, env))
  return term, env, asset, command, params


def test_command_stall_penalty_does_not_trigger_without_command_or_progress():
  term, env, asset, command, params = _make_command_stall_term(num_envs=2)
  command[0] = 0.0
  asset.data.root_link_lin_vel_b[:, 0] = 1.0

  for _ in range(60):
    torch.testing.assert_close(term(env, **params), torch.zeros(2))
  assert env.extras["log"] == {}


def test_command_stall_penalty_accumulates_from_grace_to_cap():
  term, env, _, _, params = _make_command_stall_term()

  rewards = []
  for _ in range(25):
    rewards.append(term(env, **params))
  torch.testing.assert_close(rewards[-1], torch.tensor([-5.0]))

  for _ in range(50):
    rewards.append(term(env, **params))
  cumulative = torch.stack(rewards).sum(dim=0) * env.step_dt
  torch.testing.assert_close(cumulative, torch.tensor([-0.5]))

  # 1.5 秒后持续保持 -0.5/s 的惩罚速率。
  for _ in range(10):
    torch.testing.assert_close(term(env, **params), torch.tensor([-0.5]))


def test_command_stall_penalty_resets_after_progress_without_refund():
  term, env, asset, _, params = _make_command_stall_term()

  rewards = [term(env, **params) for _ in range(75)]
  torch.testing.assert_close(
    torch.stack(rewards).sum(dim=0) * env.step_dt,
    torch.tensor([-0.5]),
  )

  asset.data.root_link_lin_vel_b[0, 0] = 1.0
  torch.testing.assert_close(term(env, **params), torch.zeros(1))

  # 恢复后重新卡住，从宽限期开始重新累计。
  asset.data.root_link_lin_vel_b[0, 0] = 0.0
  for _ in range(24):
    torch.testing.assert_close(term(env, **params), torch.zeros(1))
  torch.testing.assert_close(term(env, **params), torch.tensor([-5.0]))


def _make_command_stall_reset_term(
  num_envs: int = 1,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
  asset = SimpleNamespace(
    data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(num_envs, 3))
  )
  command = torch.tensor([[1.0, 0.0, 0.0]] * num_envs)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    scene={"robot": asset},
    command_manager=SimpleNamespace(get_command=lambda name: command),
  )
  params = {
    "command_name": "twist",
    "asset_cfg": SceneEntityCfg("robot"),
    "command_threshold": 0.2,
    "stall_speed_threshold": 0.05,
    "recover_speed_threshold": 0.1,
    "command_off_threshold": 0.1,
    "reset_time": 2.5,
  }
  cfg = SimpleNamespace(params=params)
  term = mdp.command_stall_reset(cfg=cast(Any, cfg), env=cast(Any, env))
  return term, env, asset, command, params


def test_command_stall_reset_triggers_at_2_5_seconds():
  term, env, _, _, params = _make_command_stall_reset_term()

  for _ in range(124):
    assert not term(env, **params).any()
  assert term(env, **params).all()


def test_command_stall_reset_clears_after_progress_and_reset():
  term, env, asset, _, params = _make_command_stall_reset_term()

  for _ in range(60):
    assert not term(env, **params).any()

  asset.data.root_link_lin_vel_b[0, 0] = 1.0
  assert not term(env, **params).any()

  asset.data.root_link_lin_vel_b[0, 0] = 0.0
  for _ in range(60):
    assert not term(env, **params).any()

  term.reset()
  for _ in range(124):
    assert not term(env, **params).any()
  assert term(env, **params).all()
