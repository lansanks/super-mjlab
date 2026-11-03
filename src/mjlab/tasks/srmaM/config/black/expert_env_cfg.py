"""Terrain-and-bridge-free SRMAM expert environments.

Every SRMAM expert reuses the Black SRMAM robot, sensors, and teacher
observations; only the terrain and the bridge-specific MDP terms change. The
bridge column exists only in the SRMAV-style Black curriculum, so any other
terrain must strip its command override, spawn, reward, and patch boundary.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.srmaM.mdp import UniformVelocityCommandCfg
from mjlab.tasks.srmaM.srmaM_env_cfg import make_srmaM_env_cfg

from .env_cfg import black_srmaM_env_cfg
from .terrain_cfg import (
  make_srmaM_loco_terrains_cfg,
  make_srmaM_stairs_terrains_cfg,
)

SRMAM_STAIRS_THIGH_CONTACT_FORCE = 50.0
"""Thigh-contact termination threshold for the stairs expert, in newtons.

The shared Black config terminates at 10 N. Descending sweeps the front thigh
past the step nose, which trips that gate before the robot is actually falling,
so the policy learns that approaching an edge ends the episode. The stairs
expert alone relaxes it; every other SRMAM task keeps the 10 N default.
"""


def _strip_bridge_terms(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Remove the bridge-only MDP terms from a Black SRMAM config."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.terrain_override_name = None
  cfg.rewards.pop("bridge_gap_crossing", None)
  cfg.terminations.pop("out_of_sub_terrain_bounds", None)
  # Restore the stock uniform reset; the Black task replaces it with the
  # bridge spawn that faces the first gap.
  cfg.events["reset_base"] = deepcopy(make_srmaM_env_cfg().events["reset_base"])
  return cfg


def black_srmaM_loco_expert_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the walking expert on SRMA's rough map.

  Swapping in ``make_srmaM_loco_terrains_cfg`` gives the 10x7 SRMA curriculum
  (flat, stairs, slopes, rough, waves) instead of the bridge curriculum.
  """
  cfg = black_srmaM_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = make_srmaM_loco_terrains_cfg()

  return _strip_bridge_terms(cfg)


def black_srmaM_stairs_expert_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the stairs expert: ascending and descending stairs only.

  Rows interpolate the step height from 15 cm to 30 cm, so the curriculum starts
  at a height the walking expert already handles and ends 5 cm above the hardest
  step it saw during pretraining. Four run widths (42/36/30/26 cm) give four
  difficulty bands per direction, so the rise and the slope are separate axes.
  Rewards are unchanged except ``stair_climb_progress.max_step_height``, which
  follows the new 30 cm ceiling so a full step still earns one full unit.
  """
  cfg = black_srmaM_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = make_srmaM_stairs_terrains_cfg()
  cfg.rewards["stair_climb_progress"].params["max_step_height"] = 0.30
  # Stairs only: see SRMAM_STAIRS_THIGH_CONTACT_FORCE.
  cfg.terminations["illegal_contact"].params["force_threshold"] = (
    SRMAM_STAIRS_THIGH_CONTACT_FORCE
  )

  return _strip_bridge_terms(cfg)
