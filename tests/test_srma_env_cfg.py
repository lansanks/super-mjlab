"""Tests for the independent SRMA teacher environment."""

import math

import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.play import (
  _configure_fixed_stair_play,
  _normalize_fixed_stair_args,
)
from mjlab.tasks.srma import mdp
from mjlab.tasks.srma.config.go1.env_cfg import (
  unitree_go1_srma_env_cfg,
  unitree_go1_srma_flat_env_cfg,
)
from mjlab.tasks.srma.mdp.events import push_by_setting_velocity_with_record
from mjlab.terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.config import ROUGH_TERRAINS_CFG


def test_teacher_observation_groups_and_shapes():
  cfg = unitree_go1_srma_env_cfg(play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    expected_shapes = {
      "actor": (2, 45),
      "critic": (2, 265),
      "teacher_height_map": (2, 99),
      "teacher_friction": (2, 3),
      "teacher_prev_push": (2, 6),
      "teacher_velocity_gt": (2, 3),
    }
    for group, shape in expected_shapes.items():
      value = obs[group]
      assert isinstance(value, torch.Tensor)
      assert value.shape == shape, group

    assert "cenet_input" not in obs
    assert "cenet_target_o" not in obs
    assert "cenet_target_v" not in obs
    assert env.scene["srma_teacher_scan"].num_rays_per_frame == 99
    friction = obs["teacher_friction"]
    assert isinstance(friction, torch.Tensor)
    assert torch.all((-1.0 <= friction) & (friction <= 1.0))
  finally:
    env.close()


def test_flat_teacher_height_map_is_zero():
  cfg = unitree_go1_srma_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    height_map = obs["teacher_height_map"]
    assert isinstance(height_map, torch.Tensor)
    assert height_map.shape == (1, 99)
    torch.testing.assert_close(height_map, torch.zeros_like(height_map))
  finally:
    env.close()


def test_foot_friction_matches_shared_geom_values():
  cfg = unitree_go1_srma_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    robot = env.scene["robot"]
    geom_ids, _ = robot.find_geoms(
      (
        "FR_foot_collision",
        "FL_foot_collision",
        "RR_foot_collision",
        "RL_foot_collision",
      ),
      preserve_order=True,
    )
    global_geom_id = int(robot.indexing.geom_ids[geom_ids[0]])
    friction = env.sim.model.geom_friction[:, global_geom_id, :]
    log_friction = torch.log(friction.clamp_min(1.0e-8))
    low = torch.tensor(
      [math.log(0.3), math.log(1.0e-4), math.log(1.0e-5)],
      device=friction.device,
    )
    high = torch.tensor(
      [math.log(1.5), math.log(2.0e-2), math.log(5.0e-3)],
      device=friction.device,
    )
    expected = (2.0 * (log_friction - low) / (high - low) - 1.0).clamp(-1.0, 1.0)
    actual = obs["teacher_friction"]
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)
  finally:
    env.close()


def test_previous_push_is_strictly_one_step_delayed():
  cfg = unitree_go1_srma_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    event = env.event_manager.get_term_cfg("push_robot").func
    assert isinstance(event, push_by_setting_velocity_with_record)
    kick = torch.arange(6, device=env.device, dtype=torch.float)
    event.push_velocity[0] = kick
    event.last_push_step[0] = env.common_step_counter

    assert torch.count_nonzero(mdp.previous_push_velocity(env)) == 0
    env.common_step_counter += 1
    torch.testing.assert_close(mdp.previous_push_velocity(env)[0], kick)
    env.common_step_counter += 1
    assert torch.count_nonzero(mdp.previous_push_velocity(env)) == 0

    event.last_push_step[0] = env.common_step_counter - 1
    event.reset(torch.tensor([0], device=env.device))
    assert torch.count_nonzero(mdp.previous_push_velocity(env)) == 0
  finally:
    env.close()


def test_stair_height_cap_is_srma_only():
  cfg = unitree_go1_srma_env_cfg()
  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None

  stairs = generator.sub_terrains["pyramid_stairs"]
  inverse_stairs = generator.sub_terrains["pyramid_stairs_inv"]
  assert isinstance(stairs, BoxPyramidStairsTerrainCfg)
  assert isinstance(inverse_stairs, BoxInvertedPyramidStairsTerrainCfg)
  assert stairs.step_height_range == (0.0, 0.25)
  assert inverse_stairs.step_height_range == (0.0, 0.25)

  default_stairs = ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"]
  assert isinstance(default_stairs, BoxPyramidStairsTerrainCfg)
  assert default_stairs.step_height_range == (0.0, 0.2)


def test_play_mode_has_enough_contact_capacity():
  assert unitree_go1_srma_env_cfg().sim.nconmax == 35
  assert unitree_go1_srma_env_cfg(play=True).sim.nconmax == 128


def test_fixed_stair_play_uses_one_environment_per_height():
  cfg = unitree_go1_srma_env_cfg(play=True)
  _configure_fixed_stair_play(cfg, "0.15,0.20,0.30")

  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None
  assert generator.curriculum
  assert generator.num_rows == 1
  assert list(generator.sub_terrains) == [
    "stairs_0.15m",
    "stairs_0.20m",
    "stairs_0.30m",
  ]
  for name, height in zip(
    generator.sub_terrains,
    (0.15, 0.20, 0.30),
    strict=True,
  ):
    stair_cfg = generator.sub_terrains[name]
    assert isinstance(stair_cfg, BoxInvertedPyramidStairsTerrainCfg)
    assert stair_cfg.step_height_range == (height, height)
  assert terrain.max_init_terrain_level == 0
  assert cfg.scene.num_envs == 3
  assert "randomize_terrain" not in cfg.events


def test_fixed_stair_play_supports_one_height():
  cfg = unitree_go1_srma_env_cfg(play=True)
  _configure_fixed_stair_play(cfg, "0.20")

  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None
  assert list(generator.sub_terrains) == ["stairs_0.20m"]
  assert cfg.scene.num_envs == 1


def test_fixed_stair_cli_accepts_space_or_comma_separated_values():
  option = "--fixed-stair-heights"
  expected = [option, "0.15,0.20,0.30", "--viewer", "viser"]
  assert (
    _normalize_fixed_stair_args([option, "0.15", "0.20", "0.30", "--viewer", "viser"])
    == expected
  )
  assert (
    _normalize_fixed_stair_args([option, "0.15,0.20,0.30", "--viewer", "viser"])
    == expected
  )
  assert _normalize_fixed_stair_args([option, "0.20"]) == [option, "0.20"]
