"""Unitree Go1 HIM environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.dreamwaq.config.go1.env_cfg import unitree_go1_dreamwaq_env_cfg
from mjlab.tasks.himloco.himloco_env_cfg import add_himloco_observations


def unitree_go1_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go1 rough-terrain HIM configuration.

  The locomotion environment is exactly the DreamWaQ Go1 rough configuration
  (same rewards, terrain, domain randomization, commands and terminations);
  only the observation groups differ: DreamWaQ's CENet groups are replaced by
  HIM's ``him_input`` / ``him_state``.
  """
  cfg = unitree_go1_dreamwaq_env_cfg(play=play)
  return add_himloco_observations(cfg, play=play)
