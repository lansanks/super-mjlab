"""Black HIM environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.dreamwaq.config.black.env_cfg import (
  black_dreamwaq_env_cfg,
  black_dreamwaq_flat_env_cfg,
)
from mjlab.tasks.himloco.himloco_env_cfg import add_himloco_observations


def black_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Black rough-terrain HIM configuration.

  The locomotion environment is exactly the DreamWaQ Black configuration
  (same rewards, terrain, domain randomization, commands and terminations);
  only the observation groups differ: DreamWaQ's CENet groups are replaced by
  HIM's ``him_input`` / ``him_state``.
  """
  return add_himloco_observations(black_dreamwaq_env_cfg(play=play), play=play)


def black_him_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Black flat-terrain HIM configuration."""
  return add_himloco_observations(black_dreamwaq_flat_env_cfg(play=play), play=play)
