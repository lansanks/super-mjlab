"""Tests for the SRMAM stairs expert (15-30 cm, stairs only)."""

import math

import pytest
from conftest import get_test_device

import mjlab.tasks.srmaM.config.black  # noqa: F401
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.srmaM.config.black.expert_env_cfg import (
  SRMAM_STAIRS_THIGH_CONTACT_FORCE,
  black_srmaM_stairs_expert_env_cfg,
)
from mjlab.tasks.srmaM.config.black.terrain_cfg import (
  make_srmaM_stairs_terrains_cfg,
)
from mjlab.tasks.srmaM.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)

EXPERT_TASK = "Mjlab-SRMAM-Expert-Stairs-Black"


def test_stairs_expert_terrain_is_stairs_only() -> None:
  cfg = black_srmaM_stairs_expert_env_cfg()
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  assert list(generator.sub_terrains) == [
    "stairs_up_w42",
    "stairs_up_w36",
    "stairs_up_w30",
    "stairs_up_w26",
    "stairs_down_w42",
    "stairs_down_w36",
    "stairs_down_w30",
    "stairs_down_w26",
  ]
  assert generator.curriculum
  assert generator.num_rows == 10
  assert generator.size == (8.0, 8.0)

  widths = (0.42, 0.36, 0.30, 0.26)
  for width, tag in zip(widths, ("w42", "w36", "w30", "w26"), strict=True):
    for direction, cfg_cls in (
      ("up", BoxPyramidStairsTerrainCfg),
      ("down", BoxInvertedPyramidStairsTerrainCfg),
    ):
      sub = generator.sub_terrains[f"stairs_{direction}_{tag}"]
      assert isinstance(sub, cfg_cls)
      assert sub.step_height_range == (0.15, 0.30)
      # The run is fixed inside a column, so width is the band's difficulty.
      assert sub.step_width == width
      assert sub.step_width_range is None
      assert sub.step_run(0.0) == sub.step_run(1.0) == width

  # No flat, rough, slope, wave, or bridge columns.
  for name in ("flat", "random_rough", "wave_terrain", "broken_bridge"):
    assert name not in generator.sub_terrains


def test_stairs_expert_columns_get_steeper_as_the_run_narrows() -> None:
  generator = make_srmaM_stairs_terrains_cfg()
  rise = 0.30
  angles = []
  for name in ("stairs_up_w42", "stairs_up_w36", "stairs_up_w30", "stairs_up_w26"):
    sub = generator.sub_terrains[name]
    angles.append(math.degrees(math.atan2(rise, sub.step_width)))  # type: ignore[union-attr]
  assert angles == sorted(angles)
  assert angles[0] < 40.0
  assert angles[-1] > 45.0


def test_stairs_terrain_factory_is_independent() -> None:
  first = make_srmaM_stairs_terrains_cfg()
  second = make_srmaM_stairs_terrains_cfg()
  assert first is not second
  assert first.sub_terrains is not second.sub_terrains
  # Tuning one expert's range must not leak into the other.
  second.sub_terrains["stairs_up_w42"].step_height_range = (0.10, 0.20)  # type: ignore[union-attr]
  assert first.sub_terrains["stairs_up_w42"].step_height_range == (0.15, 0.30)  # type: ignore[union-attr]


def test_stairs_expert_drops_bridge_and_raises_credit_ceiling() -> None:
  cfg = black_srmaM_stairs_expert_env_cfg()

  assert "bridge_gap_crossing" not in cfg.rewards
  assert "out_of_sub_terrain_bounds" not in cfg.terminations
  assert "out_of_terrain_bounds" in cfg.terminations

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  assert twist_cmd.terrain_override_name is None
  assert cfg.events["reset_base"].func.__name__ == "reset_root_state_uniform"

  # A full 30 cm step must still earn one normalized unit.
  climb = cfg.rewards["stair_climb_progress"]
  assert climb.params["max_step_height"] == 0.30
  assert climb.weight == 0.15


def test_stairs_expert_relaxes_the_thigh_contact_gate() -> None:
  cfg = black_srmaM_stairs_expert_env_cfg()
  assert (
    cfg.terminations["illegal_contact"].params["force_threshold"]
    == SRMAM_STAIRS_THIGH_CONTACT_FORCE
  )
  assert SRMAM_STAIRS_THIGH_CONTACT_FORCE == 50.0
  # Same sensor, so only the threshold changes.
  assert (
    cfg.terminations["illegal_contact"].params["sensor_name"] == "thigh_ground_touch"
  )


def test_thigh_contact_relaxation_is_scoped_to_stairs() -> None:
  """Every other SRMAM task must keep the shared 10 N gate."""
  stairs = load_env_cfg(EXPERT_TASK)
  assert stairs.terminations["illegal_contact"].params["force_threshold"] == 50.0

  for task in (
    "Mjlab-SRMAM-Expert-Loco-Black",
    "Mjlab-SRMAM-Teacher-Rough-Black",
    "Mjlab-SRMAM-Student-Rough-Black",
  ):
    cfg = load_env_cfg(task)
    params = cfg.terminations["illegal_contact"].params
    assert params.get("force_threshold", 10.0) == 10.0, task
    # And the stairs task must not have leaked its 50 N into the defaults.
    assert (
      black_srmaM_stairs_expert_env_cfg()
      .terminations["illegal_contact"]
      .params["force_threshold"]
      == 50.0
    )


def test_stairs_expert_keeps_the_loco_reward_set() -> None:
  loco = load_env_cfg("Mjlab-SRMAM-Expert-Loco-Black")
  stairs = load_env_cfg(EXPERT_TASK)
  assert set(loco.rewards) == set(stairs.rewards)
  for name in loco.rewards:
    assert loco.rewards[name].weight == stairs.rewards[name].weight, name


def test_stairs_expert_task_is_registered() -> None:
  env_cfg = load_env_cfg(EXPERT_TASK)
  play_cfg = load_env_cfg(EXPERT_TASK, play=True)
  assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
  assert isinstance(play_cfg, ManagerBasedRlEnvCfg)
  assert play_cfg.episode_length_s >= 1e9

  rl_cfg = load_rl_cfg(EXPERT_TASK)
  assert rl_cfg.experiment_name == "black_srmaM_expert_stairs"
  assert rl_cfg.actor.srmaM_cfg is not None


@pytest.mark.parametrize("play", [False, True])
def test_stairs_expert_env_builds(play: bool) -> None:
  from mjlab.envs import ManagerBasedRlEnv

  cfg = black_srmaM_stairs_expert_env_cfg(play=play)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    assert env.scene.terrain is not None
    assert env.scene.terrain.cfg.terrain_generator is not None
  finally:
    env.close()
