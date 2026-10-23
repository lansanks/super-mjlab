from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.srmaV.mdp.terrain_utils import (
  terrain_column_index,
  terrain_normal_from_sensors,
)
from mjlab.terrains import BoxBrokenBridgeTerrainCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize insufficient clearance height, weighted by foot velocity.

  Foot heights above the target are not penalized. This avoids discouraging the
  larger leg lifts needed for tall stairs and descending steps.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  target = torch.full_like(foot_height, target_height)
  shortfall = (target - foot_height).clamp_min(0.0)  # [B, F]
  cost = torch.sum(shortfall * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize insufficient swing height with a fixed target.

  Heights above the target are allowed so that tall steps can be cleared.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    shortfall = (target_height - self.peak_heights).clamp_min(0.0)
    relative_shortfall = shortfall / target_height
    cost = (
      torch.sum(
        torch.square(relative_shortfall) * first_contact.float(),
        dim=1,
      )
      * active
    )
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.peak_heights[env_ids] = 0.0


class stair_climb_progress:
  """按脚奖励首次到达的新高度。

  每只脚独立维护上次获得奖励的接触高度。有效承载接触高度增加至少
  ``min_height_gain`` 时，该脚获得一次与升高量成比例的奖励。台阶和斜坡都
  适用；同一只脚回到已奖励过的高度不会重复得分。
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._robot: Entity = env.scene[params["asset_cfg"].name]
    self._height_sensor: TerrainHeightSensor = env.scene[params["height_sensor_name"]]
    if not isinstance(self._height_sensor, TerrainHeightSensor):
      raise TypeError(
        "stair_climb_progress requires a TerrainHeightSensor, "
        f"got {type(self._height_sensor).__name__}."
      )
    self._contact_sensor: ContactSensor = env.scene[params["contact_sensor_name"]]

    site_ids, matched_sites = self._robot.find_sites(
      params["asset_cfg"].site_names,
      preserve_order=True,
    )
    if not site_ids:
      raise ValueError(
        f"stair_climb_progress requires foot sites, got {matched_sites}."
      )
    self._site_ids = torch.tensor(site_ids, dtype=torch.long, device=env.device)
    self._step_dt = float(env.step_dt)
    if self._step_dt <= 0.0:
      raise ValueError("stair_climb_progress requires a positive env.step_dt.")

    self._foot_anchor_height = torch.zeros(
      env.num_envs,
      self._site_ids.numel(),
      device=env.device,
    )
    self._foot_initialized = torch.zeros(
      env.num_envs,
      self._site_ids.numel(),
      dtype=torch.bool,
      device=env.device,
    )
    self._foot_has_rewarded = torch.zeros_like(self._foot_initialized)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    max_step_height: float = 0.25,
    min_height_gain: float = 0.02,
    support_clearance_max: float = 0.05,
    contact_force_threshold: float = 10.0,
    command_threshold: float = 0.1,
    first_step_multiplier: float = 3.0,
  ) -> torch.Tensor:
    del contact_sensor_name, height_sensor_name, asset_cfg
    if max_step_height <= 0.0:
      raise ValueError("max_step_height must be positive.")
    if min_height_gain <= 0.0:
      raise ValueError("min_height_gain must be positive.")
    if first_step_multiplier < 1.0:
      raise ValueError("first_step_multiplier must be at least one.")

    contact_data = self._contact_sensor.data
    found = contact_data.found
    assert found is not None
    found = found > 0
    if contact_data.force is not None:
      force_norm = torch.norm(contact_data.force, dim=-1)
      loaded = force_norm >= contact_force_threshold
    else:
      loaded = found

    clearance = self._height_sensor.data.heights
    if found.shape != clearance.shape:
      raise ValueError(
        "Contact and foot-height sensors must have matching shapes, got "
        f"{tuple(found.shape)} and {tuple(clearance.shape)}."
      )
    if found.shape[1] != self._site_ids.numel():
      raise ValueError(
        "stair_climb_progress foot site count does not match the contact sensor: "
        f"{self._site_ids.numel()} vs {found.shape[1]}."
      )

    foot_z = self._robot.data.site_pos_w[:, self._site_ids, 2]
    ground_z = foot_z - clearance
    valid = (
      found & loaded & torch.isfinite(ground_z) & (clearance <= support_clearance_max)
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_active = torch.norm(command[:, :2], dim=1) > command_threshold

    landing = valid & self._contact_sensor.compute_first_contact(dt=self._step_dt)
    first_contact = landing & ~self._foot_initialized
    candidate_gain = (ground_z - self._foot_anchor_height).clamp_min(0.0)
    reward_event = (
      landing
      & self._foot_initialized
      & command_active.unsqueeze(-1)
      & (candidate_gain >= min_height_gain)
    )
    first_reward_event = reward_event & ~self._foot_has_rewarded
    reward_multiplier = torch.where(
      first_reward_event,
      first_step_multiplier,
      torch.ones_like(candidate_gain),
    )
    event_gain = torch.where(
      reward_event,
      candidate_gain.clamp(max=max_step_height) * reward_multiplier,
      torch.zeros_like(candidate_gain),
    )

    # 指令为零时只同步基线，避免被动被推上高处后补领奖励。
    sync_anchor = landing & (~command_active.unsqueeze(-1) | first_contact)
    update_anchor = reward_event | sync_anchor
    self._foot_anchor_height = torch.where(
      first_contact,
      ground_z,
      torch.where(
        update_anchor,
        torch.maximum(self._foot_anchor_height, ground_z),
        self._foot_anchor_height,
      ),
    )
    self._foot_initialized |= landing
    self._foot_has_rewarded |= reward_event

    reward = (event_gain / max_step_height / self._step_dt).sum(dim=1)
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._foot_anchor_height[env_ids] = 0.0
    self._foot_initialized[env_ids] = False
    self._foot_has_rewarded[env_ids] = False


class command_stall_penalty:
  """按持续卡住时间积分惩罚沿指令方向无进展。

  不使用地形或接触信息。前向进度取实际速度在指令方向上的投影，宽限期后
  累计惩罚从 ``penalty_start`` 线性增加到 ``penalty_end``，之后继续以
  ``penalty_end`` 的每秒速率积分。
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._robot: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._step_dt = float(env.step_dt)
    if self._step_dt <= 0.0:
      raise ValueError("command_stall_penalty requires a positive env.step_dt.")
    self._stall_time = torch.zeros(env.num_envs, device=env.device)

  @staticmethod
  def _penalty_total(
    stall_time: torch.Tensor,
    grace_time: float,
    ramp_time: float,
    penalty_start: float,
    penalty_end: float,
  ) -> torch.Tensor:
    ramp_fraction = ((stall_time - grace_time) / ramp_time).clamp(min=0.0, max=1.0)
    ramp_total = penalty_start + (penalty_end - penalty_start) * ramp_fraction
    sustain_time = (stall_time - grace_time - ramp_time).clamp_min(0.0)
    total = ramp_total + penalty_end * sustain_time
    return torch.where(stall_time >= grace_time, -total, torch.zeros_like(total))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.2,
    stall_speed_threshold: float = 0.05,
    recover_speed_threshold: float = 0.1,
    command_off_threshold: float = 0.1,
    grace_time: float = 0.5,
    ramp_time: float = 1.0,
    penalty_start: float = 0.1,
    penalty_end: float = 0.5,
  ) -> torch.Tensor:
    del asset_cfg
    if command_threshold <= 0.0:
      raise ValueError("command_threshold must be positive.")
    if stall_speed_threshold < 0.0:
      raise ValueError("stall_speed_threshold must be non-negative.")
    if recover_speed_threshold < stall_speed_threshold:
      raise ValueError(
        "recover_speed_threshold must be at least stall_speed_threshold."
      )
    if command_off_threshold < 0.0:
      raise ValueError("command_off_threshold must be non-negative.")
    if grace_time < 0.0:
      raise ValueError("grace_time must be non-negative.")
    if ramp_time <= 0.0:
      raise ValueError("ramp_time must be positive.")
    if penalty_start < 0.0 or penalty_end < penalty_start:
      raise ValueError(
        "penalty thresholds must satisfy 0 <= penalty_start <= penalty_end."
      )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_xy = command[:, :2]
    command_norm = torch.norm(command_xy, dim=1)
    actual_xy = self._robot.data.root_link_lin_vel_b[:, :2]
    progress_speed = torch.sum(command_xy * actual_xy, dim=1) / command_norm.clamp_min(
      1.0e-6
    )

    stalled = (command_norm > command_threshold) & (
      progress_speed < stall_speed_threshold
    )
    recovered = (command_norm < command_off_threshold) | (
      progress_speed > recover_speed_threshold
    )

    previous_stall_time = self._stall_time
    self._stall_time = torch.where(
      stalled,
      self._stall_time + self._step_dt,
      torch.where(
        recovered,
        torch.zeros_like(self._stall_time),
        self._stall_time,
      ),
    )

    previous_total = self._penalty_total(
      previous_stall_time,
      grace_time,
      ramp_time,
      penalty_start,
      penalty_end,
    )
    current_total = self._penalty_total(
      self._stall_time,
      grace_time,
      ramp_time,
      penalty_start,
      penalty_end,
    )
    incremental_penalty = (current_total - previous_total).clamp(max=0.0)
    reward = torch.where(
      stalled,
      incremental_penalty / self._step_dt,
      torch.zeros_like(self._stall_time),
    )
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._stall_time[env_ids] = 0.0


class bridge_gap_crossing:
  """按脚奖励每只脚首次到达当前断桥缝隙远端。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._robot: Entity = env.scene[params["asset_cfg"].name]
    self._contact_sensor: ContactSensor = env.scene[params["contact_sensor_name"]]
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
      raise ValueError("bridge_gap_crossing requires generated terrain.")
    if terrain.terrain_origins is None:
      raise ValueError("bridge_gap_crossing requires terrain origins.")

    self._terrain = terrain
    self._bridge_column = terrain_column_index(env, params["terrain_name"])
    if self._bridge_column is None:
      raise ValueError("bridge_gap_crossing requires a named bridge terrain.")

    generator = terrain.cfg.terrain_generator
    bridge_cfg = generator.sub_terrains[params["terrain_name"]]
    if not isinstance(bridge_cfg, BoxBrokenBridgeTerrainCfg):
      raise TypeError(
        "bridge_gap_crossing requires BoxBrokenBridgeTerrainCfg, got "
        f"{type(bridge_cfg).__name__}."
      )
    self._bridge_cfg = bridge_cfg
    self._size_x = float(generator.size[0])
    self._num_rows = int(terrain.terrain_origins.shape[0])
    self._num_gaps = int(bridge_cfg.num_gaps)
    if self._num_gaps < 1:
      raise ValueError("bridge_gap_crossing requires at least one gap.")

    lower, upper = generator.difficulty_range
    gap_far_local = np.zeros((self._num_rows, self._num_gaps), dtype=np.float32)
    for row in range(self._num_rows):
      difficulty = lower + (upper - lower) * row / max(self._num_rows - 1, 1)
      gaps = bridge_cfg.gap_intervals(difficulty, patch_length=self._size_x)
      if len(gaps) != self._num_gaps:
        raise ValueError("Bridge gap count does not match its configuration.")
      gap_far_local[row] = [end for _, end in gaps]
    self._gap_far_local = torch.from_numpy(gap_far_local).to(env.device)

    site_ids, matched_sites = self._robot.find_sites(
      params["asset_cfg"].site_names,
      preserve_order=True,
    )
    if not site_ids:
      raise ValueError(f"bridge_gap_crossing requires foot sites, got {matched_sites}.")
    self._site_ids = torch.tensor(site_ids, dtype=torch.long, device=env.device)
    self._feet_rewarded = torch.zeros(
      env.num_envs,
      self._site_ids.numel(),
      dtype=torch.bool,
      device=env.device,
    )
    self._current_gap = torch.zeros(
      env.num_envs,
      dtype=torch.long,
      device=env.device,
    )
    self._step_dt = float(env.step_dt)
    if self._step_dt <= 0.0:
      raise ValueError("bridge_gap_crossing requires a positive env.step_dt.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    terrain_name: str,
    foot_bonus: float = 2.5,
    far_margin: float = 0.02,
    contact_force_threshold: float = 10.0,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    del contact_sensor_name, asset_cfg, terrain_name
    if foot_bonus < 0.0:
      raise ValueError("foot_bonus must be non-negative.")
    if far_margin < 0.0:
      raise ValueError("far_margin must be non-negative.")
    if contact_force_threshold < 0.0:
      raise ValueError("contact_force_threshold must be non-negative.")

    contact_data = self._contact_sensor.data
    found = contact_data.found
    assert found is not None
    loaded = found > 0
    if contact_data.force is not None:
      loaded &= torch.norm(contact_data.force, dim=-1) >= contact_force_threshold

    foot_x = self._robot.data.site_pos_w[:, self._site_ids, 0]
    if foot_x.shape != loaded.shape:
      raise ValueError(
        "Foot site and contact sensor counts must match, got "
        f"{tuple(foot_x.shape)} and {tuple(loaded.shape)}."
      )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_active = torch.norm(command[:, :2], dim=1) > command_threshold
    bridge_envs = self._terrain.terrain_types == self._bridge_column
    assert isinstance(bridge_envs, torch.Tensor)
    valid_gap = self._current_gap < self._num_gaps
    gap_index = self._current_gap.clamp_max(self._num_gaps - 1)
    levels = self._terrain.terrain_levels
    far_local = self._gap_far_local[levels, gap_index]
    patch_min_x = -0.5 * self._num_rows * self._size_x + levels.float() * self._size_x
    far_world_x = patch_min_x + far_local
    reached_far = foot_x > far_world_x.unsqueeze(-1) + far_margin
    valid_landing = (
      bridge_envs.unsqueeze(-1)
      & valid_gap.unsqueeze(-1)
      & command_active.unsqueeze(-1)
      & loaded
      & reached_far
    )
    new_reward = valid_landing & ~self._feet_rewarded
    self._feet_rewarded |= valid_landing
    reward = new_reward.sum(dim=1).float() * foot_bonus / self._step_dt

    all_feet_done = bridge_envs & valid_gap & self._feet_rewarded.all(dim=1)
    self._current_gap += all_feet_done.long()
    self._feet_rewarded[all_feet_done] = False

    env.extras["log"]["Metrics/bridge_gap_index"] = self._current_gap.float().mean()
    env.extras["log"]["Metrics/bridge_feet_rewarded"] = (
      self._feet_rewarded.float().sum(dim=1).mean()
    )
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._current_gap[env_ids] = 0
    self._feet_rewarded[env_ids] = False


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
