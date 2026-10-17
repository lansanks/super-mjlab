"""Observation groups shared by the HIM locomotion tasks.

HIM adds two observation groups on top of the locomotion environment; the
released implementation derives both from the same buffers the policy and the
critic already use:

``him_input``
  The actor observation history ``o_t^H`` with ``H = 5`` (six frames including
  the current one), consumed by the source encoder.

``him_state``
  The state frame the response is matched against. The released code slices
  ``next_critic_obs[:, 3 : 45 + 3]``, i.e. the proprioceptive frame *without*
  the velocity command plus the base linear velocity. The velocity is scaled by
  ``obs_scales.lin_vel = 2`` in the released code, which is kept here (the
  policy receives the estimate in the same units, see the task document).

Neither group is routed to the actor or the critic: ``him_state`` carries
privileged information (the base linear velocity), so it must stay out of the
policy observation. Both groups are stored in the rollout storage, where the
algorithm uses them for the hybrid internal optimization.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

HISTORY_LENGTH = 6
"""Frames in ``him_input``: the current frame plus ``H = 5`` past frames."""

LIN_VEL_SCALE = 2.0
"""Scale of the base linear velocity, matching legged_gym's ``obs_scales``."""

DREAMWAQ_OBS_GROUPS = ("cenet_input", "cenet_target_o", "cenet_target_v")
"""Observation groups of the DreamWaQ task, dropped by the HIM tasks."""


def add_himloco_observations(
  cfg: ManagerBasedRlEnvCfg, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Replace DreamWaQ's CENet groups with HIM's observation groups.

  The rest of the configuration (rewards, terrain, domain randomization,
  commands, terminations) is left untouched, so the HIM task differs from the
  DreamWaQ / velocity tasks only in the observations the networks consume.
  """
  for group in DREAMWAQ_OBS_GROUPS:
    cfg.observations.pop(group, None)

  actor_terms = cfg.observations["actor"].terms
  cfg.observations["him_input"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
    history_length=HISTORY_LENGTH,
    flatten_history_dim=False,
  )
  cfg.observations["him_state"] = ObservationGroupCfg(
    terms={
      # Same terms (and noise) as the policy observation, minus the command.
      "base_ang_vel": actor_terms["base_ang_vel"],
      "projected_gravity": actor_terms["projected_gravity"],
      "joint_pos": actor_terms["joint_pos"],
      "joint_vel": actor_terms["joint_vel"],
      "actions": actor_terms["actions"],
      # Privileged simulator truth; the clip is applied before the scale, so
      # the value range is (-20, 20) after the 2x scaling.
      "base_lin_vel": ObservationTermCfg(
        func=envs_mdp.base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-10.0, 10.0),
        scale=LIN_VEL_SCALE,
      ),
    },
    concatenate_terms=True,
    enable_corruption=True,
  )

  if play:
    # Play mode runs the trained policy without observation noise.
    cfg.observations["him_input"].enable_corruption = False
    cfg.observations["him_state"].enable_corruption = False
  return cfg
