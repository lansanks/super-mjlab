"""Tests for the SRMAM normal-walking expert on SRMA's rough map."""

import pytest
from conftest import get_test_device  # noqa: F401

import mjlab.tasks.srmaM.config.black  # noqa: F401
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.srmaM.config.black.expert_env_cfg import (
  black_srmaM_loco_expert_env_cfg,
)
from mjlab.tasks.srmaM.config.black.terrain_cfg import (
  make_srmaM_loco_terrains_cfg,
)
from mjlab.tasks.srmaM.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)

EXPERT_TASK = "Mjlab-SRMAM-Expert-Loco-Black"

ROUGH_SUB_TERRAINS = [
  "flat",
  "pyramid_stairs",
  "pyramid_stairs_inv",
  "hf_pyramid_slope",
  "hf_pyramid_slope_inv",
  "random_rough",
  "wave_terrain",
]


def test_expert_uses_the_srma_rough_map() -> None:
  cfg = black_srmaM_loco_expert_env_cfg()
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  assert list(generator.sub_terrains) == ROUGH_SUB_TERRAINS
  assert generator.curriculum
  assert generator.num_rows == 10
  assert generator.size == (8.0, 8.0)

  stairs = generator.sub_terrains["pyramid_stairs"]
  inverse_stairs = generator.sub_terrains["pyramid_stairs_inv"]
  assert isinstance(stairs, BoxPyramidStairsTerrainCfg)
  assert isinstance(inverse_stairs, BoxInvertedPyramidStairsTerrainCfg)
  assert stairs.step_height_range == (0.0, 0.25)
  assert inverse_stairs.step_height_range == (0.0, 0.25)

  # The bridge columns must be gone, not merely unrewarded.
  assert "broken_bridge" not in generator.sub_terrains
  assert not any(name.startswith("broken_bridge") for name in generator.sub_terrains)


def test_loco_terrain_factory_does_not_mutate_the_shared_preset() -> None:
  from mjlab.terrains.config import ROUGH_TERRAINS_CFG

  assert not ROUGH_TERRAINS_CFG.curriculum
  stairs = ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"]
  assert isinstance(stairs, BoxPyramidStairsTerrainCfg)
  assert stairs.step_height_range == (0.0, 0.2)

  first = make_srmaM_loco_terrains_cfg()
  second = make_srmaM_loco_terrains_cfg()
  assert first is not second
  assert first.sub_terrains is not second.sub_terrains


def test_expert_drops_bridge_mdp_terms() -> None:
  cfg = black_srmaM_loco_expert_env_cfg()
  assert "bridge_gap_crossing" not in cfg.rewards
  assert "out_of_sub_terrain_bounds" not in cfg.terminations
  assert "out_of_terrain_bounds" in cfg.terminations

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  assert twist_cmd.terrain_override_name is None

  # The stock uniform reset replaces the bridge spawn.
  assert cfg.events["reset_base"].func.__name__ == "reset_root_state_uniform"


def test_expert_keeps_the_srmaM_teacher_interface() -> None:
  cfg = black_srmaM_loco_expert_env_cfg()
  for group in ("actor", "critic", "teacher_height_map", "teacher_friction"):
    assert group in cfg.observations
  assert "teacher_prev_push" in cfg.observations
  assert "teacher_velocity_gt" in cfg.observations
  # The expert is a teacher task, so it must not carry the depth group.
  assert "student_depth" not in cfg.observations


def test_expert_task_is_registered() -> None:
  env_cfg = load_env_cfg(EXPERT_TASK)
  play_cfg = load_env_cfg(EXPERT_TASK, play=True)
  assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
  assert isinstance(play_cfg, ManagerBasedRlEnvCfg)
  assert play_cfg.episode_length_s >= 1e9

  rl_cfg = load_rl_cfg(EXPERT_TASK)
  assert rl_cfg.experiment_name == "black_srmaM_expert_loco"
  assert rl_cfg.actor.srmaM_cfg is not None


@pytest.mark.parametrize("play", [False, True])
def test_expert_env_builds(play: bool) -> None:
  from mjlab.envs import ManagerBasedRlEnv

  cfg = black_srmaM_loco_expert_env_cfg(play=play)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    assert env.scene.terrain is not None
    assert env.scene.terrain.cfg.terrain_generator is not None
  finally:
    env.close()
