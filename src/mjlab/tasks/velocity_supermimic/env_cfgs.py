"""Go1 velocity task variants for SuperMimic.

The only observation change relative to the stock mjlab Go1 velocity tasks is
that the actor is not allowed to see the base linear velocity or the terrain
height scan.  The critic (and the SuperMimic world model) keep the privileged
observation group, which is what makes the learned dynamics model useful.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.go1.env_cfgs import (
  unitree_go1_flat_env_cfg,
  unitree_go1_rough_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def _strip_actor_privileged_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove base linear velocity and terrain height from the actor group."""
  actor_terms = cfg.observations["actor"].terms
  actor_terms.pop("base_lin_vel", None)
  actor_terms.pop("height_scan", None)
  if "base_lin_vel" in actor_terms or "height_scan" in actor_terms:
    raise RuntimeError("Failed to remove privileged terms from the actor observation.")


def unitree_go1_supermimic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough-terrain Go1 velocity task with a proprioceptive-only actor."""
  cfg = unitree_go1_rough_env_cfg(play=play)
  _strip_actor_privileged_terms(cfg)
  # The SuperMimic collector needs the true terminal state, so training uses
  # manual resets.  Play mode keeps the standard auto-reset behaviour.
  cfg.auto_reset = play
  if play and cfg.sim.nconmax is not None:
    # Random play terrain can briefly exceed the training contact budget; one
    # environment makes the larger allocation cheap.
    cfg.sim.nconmax = max(cfg.sim.nconmax, 256)
  return cfg


def unitree_go1_supermimic_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat-terrain Go1 velocity task with a proprioceptive-only actor."""
  cfg = unitree_go1_flat_env_cfg(play=play)
  _strip_actor_privileged_terms(cfg)
  cfg.auto_reset = play
  return cfg


def _configure_forward_commands(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """Start with forward-only commands, then widen the command distribution."""
  twist = cfg.commands["twist"]
  if not isinstance(twist, UniformVelocityCommandCfg):
    raise TypeError(f"Expected a UniformVelocityCommandCfg, got {type(twist)}.")
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.2
  if play:
    # A clean straight-walk demo.  Keep the ranges non-zero (0.1) because the
    # Viser velocity GUI's "Max" slider requires a value of at least 0.1.
    twist.ranges.lin_vel_x = (0.5, 1.0)
    twist.ranges.lin_vel_y = (-0.1, 0.1)
    twist.ranges.ang_vel_z = (-0.1, 0.1)
    return

  twist.ranges.lin_vel_x = (0.3, 1.0)
  twist.ranges.lin_vel_y = (-0.2, 0.2)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if "command_vel" in cfg.curriculum:
    # ``step`` counts vectorized environment steps, as in the stock curriculum.
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.3, 1.0),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (0.0, 0.0),
      },
      {
        "step": 4_000,
        "lin_vel_x": (0.2, 1.2),
        "lin_vel_y": (-0.3, 0.3),
        "ang_vel_z": (-0.2, 0.2),
      },
      {
        "step": 8_000,
        "lin_vel_x": (0.0, 1.5),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-0.5, 0.5),
      },
    ]


def unitree_go1_supermimic_nocrouch_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Rough Go1 velocity variant that blocks the crouch/do-nothing optimum.

  Three changes relative to :func:`unitree_go1_supermimic_env_cfg`:

  * a low-base termination (root height below 0.15 m) so the policy cannot sit
    down and farm the angular-tracking reward for the whole episode;
  * a trunk-ground contact termination, which catches lying/crawling;
  * forward-only commands for the first part of training, then a gradual
    widening to lateral and yaw commands.
  """
  cfg = unitree_go1_supermimic_env_cfg(play=play)
  _configure_forward_commands(cfg, play)
  # Trunk contact is meaningful on both training and random play terrain.
  cfg.terminations["trunk_contact"] = TerminationTermCfg(
    func=velocity_mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch"},
  )
  if not play:
    # World-z based; play uses randomized terrain where this can false-trigger.
    cfg.terminations["low_base"] = TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.15},
    )
  return cfg
