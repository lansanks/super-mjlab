"""Tests for the Black SRMAV bridge terrain environment."""

import torch
from conftest import get_test_device

import mjlab.tasks.srmaV.config.black  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.play import _configure_play_terrain
from mjlab.tasks.srmaV.config.black.env_cfg import black_srmaV_env_cfg
from mjlab.tasks.srmaV.config.black.terrain_cfg import BLACK_SRMAV_BRIDGE_TERRAIN
from mjlab.tasks.srmaV.mdp.velocity_command import UniformVelocityCommand


def test_play_can_select_the_bridge_terrain() -> None:
  cfg = black_srmaV_env_cfg(play=True)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    _configure_play_terrain(env, BLACK_SRMAV_BRIDGE_TERRAIN)
    env.reset()
    terrain = env.scene.terrain
    assert terrain is not None
    cfg_terrain = cfg.scene.terrain
    assert cfg_terrain is not None
    generator = cfg_terrain.terrain_generator
    assert generator is not None
    bridge_column = list(generator.sub_terrains).index(BLACK_SRMAV_BRIDGE_TERRAIN)
    assert torch.all(terrain.terrain_types == bridge_column)
  finally:
    env.close()


def test_bridge_environments_and_commands() -> None:
  cfg = black_srmaV_env_cfg(play=True)
  cfg.scene.num_envs = 4
  assert "out_of_sub_terrain_bounds" not in cfg.terminations
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    cfg_terrain = cfg.scene.terrain
    assert cfg_terrain is not None
    terrain = env.scene.terrain
    assert terrain is not None
    generator = cfg_terrain.terrain_generator
    assert generator is not None
    terrain_names = list(generator.sub_terrains)
    bridge_column = terrain_names.index(BLACK_SRMAV_BRIDGE_TERRAIN)

    assert sorted(terrain.terrain_types.tolist()) == list(range(4))
    bridge_ids = (terrain.terrain_types == bridge_column).nonzero().flatten()
    assert bridge_ids.numel() == 1

    command = env.command_manager.get_term("twist")
    assert isinstance(command, UniformVelocityCommand)
    assert torch.all((command.command[bridge_ids, 0] >= 0.3))
    assert torch.all((command.command[bridge_ids, 0] <= 1.0))
    torch.testing.assert_close(
      command.command[bridge_ids, 1:],
      torch.zeros_like(command.command[bridge_ids, 1:]),
    )
    torch.testing.assert_close(
      command.heading_target[bridge_ids],
      torch.zeros_like(command.heading_target[bridge_ids]),
    )
    assert bool(command.is_heading_env[bridge_ids].all())

    robot = env.scene["robot"]
    torch.testing.assert_close(
      robot.data.root_link_pos_w[bridge_ids, :2],
      env.scene.env_origins[bridge_ids, :2],
    )
  finally:
    env.close()
