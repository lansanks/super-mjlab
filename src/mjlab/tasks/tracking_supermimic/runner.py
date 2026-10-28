"""SuperMimic training loop for mjlab G1 LAFAN1 tracking."""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any

import torch
import torch.nn as nn

import mjlab.tasks.tracking_supermimic.nets as networks

from .buffer import ReplayBuffer
from .constants import G1_JOINT_NAMES, NUM_JOINTS, TRACKED_BODIES
from .kinematics import G1Kinematics
from .mdp import (
  CRITIC_EXTRA,
  INV_DIM,
  MAX_STEP_REWARD,
  MDP,
  OBS_DIM,
  flatten_state,
  normalize_xi,
  unflatten_state,
)
from .motion_command import LafanMotionCommand
from .nets import Critic, Policy, WorldModel
from .rotations import quat_to_mat


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
  for parameter in module.parameters():
    parameter.requires_grad_(flag)


class TrackingSuperMimicRunner:
  """Collect simulator data, fit a world model, and train by imagined BPTT."""

  def __init__(
    self,
    env,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
    **kwargs,
  ):
    del registry_name, kwargs
    self.cfg = dict(train_cfg)
    self.device = torch.device(device)
    self.wrapper = env
    self.env = env.unwrapped if hasattr(env, "unwrapped") else env
    self.robot = self.env.scene["robot"]
    command = self.env.command_manager.get_term("motion")
    if not isinstance(command, LafanMotionCommand):
      raise TypeError(
        "TrackingSuperMimicRunner requires LafanMotionCommand, "
        f"got {type(command).__name__}."
      )
    self.command = command
    self.motion = command.motion
    self.log_dir = log_dir or os.getcwd()
    os.makedirs(self.log_dir, exist_ok=True)

    if tuple(self.robot.joint_names) != G1_JOINT_NAMES:
      joint_ids, names = self.robot.find_joints(G1_JOINT_NAMES, preserve_order=True)
      if tuple(names) != G1_JOINT_NAMES:
        raise ValueError(
          "Robot joint order does not contain the expected 29-DoF G1 joints."
        )
      self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
    else:
      self.joint_ids = torch.arange(NUM_JOINTS, dtype=torch.long, device=self.device)

    body_ids, body_names = self.robot.find_bodies(TRACKED_BODIES, preserve_order=True)
    if tuple(body_names) != TRACKED_BODIES:
      raise ValueError(
        f"Robot bodies differ from expected tracking bodies: {body_names}"
      )
    self.body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

    self.default_joint_pos = (
      self.robot.data.default_joint_pos[:, self.joint_ids][0].clone().float()
    )
    self.kinematics = G1Kinematics(self.device)
    self.mdp = MDP(
      self.kinematics,
      self.motion,
      self.default_joint_pos,
    )
    self._setup_sim_context()

    self.num_envs = int(self.env.num_envs)
    self.action_dim = int(self.env.action_manager.total_action_dim)
    self.history = int(self.cfg.get("obs_history", 10))
    self.horizon = int(self.cfg.get("horizon", 32))
    self.goal_frames = int(self.cfg.get("goal_frames", 10))
    self.goal_dim = 73 * self.goal_frames
    self.window = int(self.cfg.get("wm_window", 8))
    self.wm_batch = int(self.cfg.get("wm_batch", 512))
    self.wm_batches = int(self.cfg.get("wm_batches", 4))
    self.policy_batch = int(self.cfg.get("policy_batch", 512))
    self.buffer_steps = int(self.cfg.get("buffer_steps", 500))
    self.utd = float(self.cfg.get("utd", 0.4))
    self.learning_rate = float(self.cfg.get("learning_rate", 2e-4))
    self.lr_decay_iters = int(self.cfg.get("lr_decay_iters", 250_000))
    self.gamma = float(self.cfg.get("gamma", 0.99))
    self.lam = float(self.cfg.get("lam", 0.95))
    self.action_noise = float(self.cfg.get("action_noise", 0.1))
    self.action_clip = 20.0
    self.critic_epochs = int(self.cfg.get("critic_epochs", 5))
    self.critic_minibatches = int(self.cfg.get("critic_minibatches", 4))
    self.warmup_steps = int(self.cfg.get("warmup_steps", 64))
    self.wm_only_iters = int(self.cfg.get("wm_only_iters", 500))
    self.log_interval = int(self.cfg.get("log_interval", 50))
    self.ckpt_interval = int(self.cfg.get("ckpt_interval", 2000))
    self.no_critic = bool(self.cfg.get("no_critic", False))

    amp = str(self.cfg.get("amp", "bf16")).lower()
    networks.AMP_DTYPE = (
      torch.bfloat16 if amp == "bf16" and self.device.type == "cuda" else None
    )

    wm_hidden = tuple(self.cfg.get("wm_hidden", (512, 512, 512)))
    encoder_hidden = tuple(self.cfg.get("policy_encoder_hidden", (512, 256, 128)))
    decoder_hidden = tuple(self.cfg.get("policy_decoder_hidden", (512, 512, 512)))
    critic_hidden = tuple(self.cfg.get("critic_hidden", (256, 128)))
    latent_dim = int(self.cfg.get("latent_dim", 32))

    self.world_model = WorldModel(hidden=wm_hidden).to(self.device)
    self.policy = Policy(
      OBS_DIM,
      self.goal_dim,
      history=self.history,
      latent_dim=latent_dim,
      encoder_hidden=encoder_hidden,
      decoder_hidden=decoder_hidden,
    ).to(self.device)
    critic_input_dim = INV_DIM + CRITIC_EXTRA + self.goal_dim
    self.critic = Critic(critic_input_dim, hidden=critic_hidden).to(self.device)
    self.opt_wm = torch.optim.Adam(self.world_model.parameters(), lr=self.learning_rate)
    self.opt_policy = torch.optim.Adam(
      self.policy.parameters(),
      lr=self.learning_rate,
      betas=(0.9, 0.95),
    )
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
    self.buffer = ReplayBuffer(self.buffer_steps, self.num_envs, self.device)

    self.err_floor = torch.cat(
      (
        torch.full((3,), 3e-3),
        torch.full((3,), 5e-3),
        torch.full((3,), 2e-2),
        torch.full((3,), 5e-2),
        torch.full((NUM_JOINTS,), 3e-3),
        torch.full((NUM_JOINTS,), 5e-2),
      )
    ).to(self.device)

    self.current_obs, _ = self.env.reset()
    self.current_state = self.state()
    self.current_xi = self.sim_context()
    observation = self.mdp.obs(
      self.current_state,
      torch.zeros(self.num_envs, NUM_JOINTS, device=self.device),
      self.current_xi,
      self.command.time_steps,
    )
    self.obs_hist = observation.unsqueeze(1).repeat(1, self.history, 1)
    self.previous_action = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
    self.episode_step = self.env.episode_length_buf.clone()
    self.episode_return = torch.zeros(self.num_envs, device=self.device)
    self.episode_discounted_return = torch.zeros(self.num_envs, device=self.device)

    self.current_learning_iteration = 0
    self._sim_steps = 0
    self._completed_returns: list[float] = []
    self._completed_lengths: list[float] = []
    self._term_count = 0
    self._done_count = 0
    self._t_start = time.time()
    self._t_log = time.time()
    self._csv_file = None
    self._wandb = None
    self._inference_history: torch.Tensor | None = None

    self.mdp.obs_noise = self.mdp.obs_noise.to(self.device).float()
    self.mdp.obs_scale = self.mdp.obs_scale.to(self.device).float()

    self.compile_enabled = bool(self.cfg.get("compile", True))
    if self.compile_enabled and self.device.type == "cuda":
      self._compile_hot_paths()

  def _compile_hot_paths(self) -> None:
    """Fuse the per-iteration hot paths with ``torch.compile``.

    Iteration time here is dominated by kernel-launch overhead rather than FLOPs:
    every world-model update runs an 8-step autoregressive rollout and every
    policy update a 32-step imagination loop, and each step is a long chain of
    small elementwise ops (rotation helpers, splits, matmuls). The reference
    SuperMimic implementation reports 0.66 -> 3.1 it/s from compiling exactly
    these paths, so mirror that here. Compilation is lazy: the first iteration
    pays a one-off warmup of a few seconds per path.
    """
    mode = str(self.cfg.get("compile_mode", "default"))
    targets = (
      (self.world_model, "step"),
      (self.world_model, "state_error"),
      (self.policy, "forward"),
      (self.critic, "forward"),
      (self.mdp, "obs"),
      (self.mdp, "add_noise"),
      (self.mdp, "reward"),
      (self.mdp, "soft_termination"),
      (self.mdp, "critic_features"),
      (self.motion, "goal"),
    )
    names = []
    for owner, attribute in targets:
      compiled = torch.compile(getattr(owner, attribute), mode=mode)
      setattr(owner, attribute, compiled)
      names.append(f"{type(owner).__name__}.{attribute}")
    print(f"[SuperMimic] torch.compile enabled (mode={mode}): {', '.join(names)}")

  def _setup_sim_context(self) -> None:
    geom_ids, _ = self.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")
    self.foot_geom_ids = self.robot.indexing.geom_ids[
      torch.tensor(geom_ids, dtype=torch.long, device=self.device)
    ]
    torso_ids, _ = self.robot.find_bodies("torso_link")
    if len(torso_ids) != 1:
      raise ValueError("Expected exactly one torso_link body.")
    self.torso_global_body_id = int(self.robot.indexing.body_ids[torso_ids[0]])
    self.default_torso_com = self.env.sim.mj_model.body_ipos[
      self.torso_global_body_id
    ].copy()

  def state(self):
    pose = self.robot.data.root_link_pose_w.float()
    velocity = self.robot.data.root_link_vel_w.float()
    return (
      pose[:, :3] - self.env.scene.env_origins.float(),
      quat_to_mat(pose[:, 3:7]),
      velocity[:, :3],
      velocity[:, 3:6],
      self.robot.data.joint_pos[:, self.joint_ids].float(),
      self.robot.data.joint_vel[:, self.joint_ids].float(),
    )

  def sim_context(self) -> torch.Tensor:
    if len(self.foot_geom_ids) > 0:
      friction = (
        self.env.sim.model.geom_friction[:, self.foot_geom_ids, 0].mean(-1).float()
      )
    else:
      friction = torch.full((self.num_envs,), 0.75, device=self.device)
    bias = self.robot.data.encoder_bias[:, self.joint_ids].float()
    current_com = self.env.sim.model.body_ipos[:, self.torso_global_body_id].float()
    com_offset = current_com - torch.as_tensor(
      self.default_torso_com, device=self.device
    ).float().unsqueeze(0)
    return torch.cat((friction.unsqueeze(-1), bias, com_offset), -1)

  def _checkpoint_goal(self, frames: torch.Tensor) -> torch.Tensor:
    return self.motion.goal(frames, self.goal_frames).float()

  def _sample_valid_starts(self, batch: int):
    starts: list[dict[str, torch.Tensor]] = []
    histories: list[dict[str, torch.Tensor]] = []
    count = 0
    for _ in range(16):
      start, history = self.buffer.sample_starts(batch * 2, self.history)
      frames = start["f"]
      ends = self.motion.clip_end[self.motion.clip_for_frame(frames)]
      valid = frames + self.horizon <= ends
      valid_ids = valid.nonzero(as_tuple=False).squeeze(-1)[: batch - count]
      if len(valid_ids) == 0:
        continue
      starts.append({key: value[valid_ids] for key, value in start.items()})
      histories.append({key: value[valid_ids] for key, value in history.items()})
      count += len(valid_ids)
      if count >= batch:
        break
    if count < batch:
      raise RuntimeError(
        "Could not sample enough starts with a complete imagination "
        "horizon. Collect more transitions or use longer motion clips."
      )

    def concatenate(items):
      return {key: torch.cat([item[key] for item in items]) for key in items[0]}

    return concatenate(starts), concatenate(histories)

  @torch.no_grad()
  def collect(self, steps: int) -> None:
    for _ in range(steps):
      self._collect_step()

  @torch.no_grad()
  def _collect_step(self) -> None:
    state = self.current_state
    frame = self.command.time_steps
    sim_context = self.current_xi
    observation = self.mdp.add_noise(
      self.mdp.obs(
        state,
        self.previous_action,
        sim_context,
        frame,
      )
    )
    fresh = self.env.episode_length_buf == 0
    self.obs_hist = torch.where(
      fresh[:, None, None],
      observation.unsqueeze(1).expand_as(self.obs_hist),
      torch.cat((self.obs_hist[:, 1:], observation.unsqueeze(1)), 1),
    )
    goal = self._checkpoint_goal(frame)
    self.policy.obs_norm.update(observation)
    self.policy.goal_norm.update(goal)
    action = self.policy(self.obs_hist, goal)
    noisy_action = torch.clamp(
      action + self.action_noise * torch.randn_like(action),
      -self.action_clip,
      self.action_clip,
    )
    previous_action = self.previous_action
    episode_step = self.env.episode_length_buf.clone()

    _, _, terminated, timed_out, _ = self.env.step(noisy_action)
    next_state = self.state()
    next_frame = self.command.time_steps
    reward = self.mdp.reward(
      next_state,
      next_frame,
      noisy_action,
      previous_action,
    )
    done = terminated | timed_out

    self.buffer.add(
      {
        "s": self._flatten_state(state),
        "s1": self._flatten_state(next_state),
        "a": noisy_action,
        "a_prev": previous_action,
        "xi": sim_context,
        "r": reward,
        "f": frame,
        "ep_step": episode_step.int(),
        "done": done,
      }
    )
    self.world_model.input_norm.update(
      self.world_model.inputs(state, noisy_action, normalize_xi(sim_context))
    )
    self.world_model.output_norm.update(self.world_model.targets(state, next_state))

    self.episode_return += reward
    self.episode_discounted_return += self.gamma ** episode_step.float() * reward
    self._sim_steps += self.num_envs
    self._term_count += int(terminated.sum())
    self._done_count += int(done.sum())

    if done.any():
      finished = done.nonzero(as_tuple=False).squeeze(-1)
      lengths = self.env.episode_length_buf[finished].float()
      self.command.record_episode_return(
        finished,
        self.episode_discounted_return[finished],
        self.episode_return[finished],
        lengths,
        terminated[finished],
        self.gamma,
      )
      self._completed_returns.extend(
        (self.episode_return[finished] / (lengths * MAX_STEP_REWARD)).cpu().tolist()
      )
      self._completed_lengths.extend(lengths.cpu().tolist())

    observation_next = self.mdp.obs(
      next_state,
      noisy_action,
      sim_context,
      next_frame,
    )
    self.obs_hist = torch.cat((self.obs_hist[:, 1:], observation_next.unsqueeze(1)), 1)
    self.previous_action = noisy_action
    self.current_state = next_state
    self.current_xi = sim_context
    self.episode_step = self.env.episode_length_buf.clone()

    if done.any():
      reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
      self.env.reset(env_ids=reset_ids)
      reset_state = self.state()
      reset_context = self.sim_context()
      reset_observation = self.mdp.obs(
        reset_state,
        torch.zeros_like(noisy_action),
        reset_context,
        self.command.time_steps,
      )
      for source, target in zip(reset_state, self.current_state, strict=True):
        target[reset_ids] = source[reset_ids]
      self.current_xi[reset_ids] = reset_context[reset_ids]
      self.obs_hist[reset_ids] = reset_observation[reset_ids].unsqueeze(1)
      self.previous_action[reset_ids] = 0.0
      self.episode_return[reset_ids] = 0.0
      self.episode_discounted_return[reset_ids] = 0.0
      self.episode_step = self.env.episode_length_buf.clone()

  @staticmethod
  def _flatten_state(state) -> torch.Tensor:
    return flatten_state(state)

  def _world_model_update(self) -> tuple[float, float]:
    if self.buffer.size < self.window:
      return 0.0, 0.0
    windows = self.buffer.sample_windows(self.wm_batch, self.window)
    predicted_state = unflatten_state(windows["s"][:, 0])
    loss = torch.zeros((), device=self.device)
    last_error = torch.zeros((), device=self.device)
    for step in range(self.window):
      predicted_state = self.world_model.step(
        predicted_state,
        windows["a"][:, step],
        normalize_xi(windows["xi"][:, step]),
      )
      target_state = unflatten_state(windows["s1"][:, step])
      std = torch.maximum(self.world_model.output_norm.std(), self.err_floor)
      error = (self.world_model.state_error(predicted_state, target_state) / std) ** 2
      error = error.mean(-1)
      mask = windows["mask"][:, step]
      loss = loss + (error * mask).sum() / mask.sum().clamp(min=1)
      last_error = (error.detach() * windows["mask"][:, -1]).sum() / windows["mask"][
        :, -1
      ].sum().clamp(min=1)
    loss = loss / self.window
    self.opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
    self.opt_wm.step()
    return float(loss.detach()), float(last_error)

  def _policy_update(self):
    start, history = self._sample_valid_starts(self.policy_batch)
    state = unflatten_state(start["s"])
    sim_context = start["xi"]
    normalized_context = normalize_xi(sim_context)
    frame = start["f"]
    previous_action = start["a_prev"]
    history_observation = self.mdp.obs(
      unflatten_state(history["s"]),
      history["a_prev"],
      history["xi"],
      history["f"],
    )
    observation_history = self.mdp.add_noise(history_observation)
    goal = self.motion.goal(frame, self.goal_frames)

    _set_requires_grad(self.world_model, False)
    _set_requires_grad(self.critic, False)
    states = []
    actions = []
    previous_actions = []
    frames = []
    goals = []
    start_state = state
    start_frame = frame
    start_goal = goal

    for _ in range(self.horizon):
      previous_actions.append(previous_action)
      action = self.policy(observation_history, goal)
      next_state = self.world_model.step(state, action, normalized_context)
      next_frame = frame + 1
      next_observation = self.mdp.add_noise(
        self.mdp.obs(
          next_state,
          action,
          sim_context,
          next_frame,
        )
      )
      observation_history = torch.cat(
        (observation_history[:, 1:], next_observation.unsqueeze(1)),
        1,
      )
      state = next_state
      previous_action = action
      frame = next_frame
      goal = self.motion.goal(frame, self.goal_frames)
      states.append(state)
      actions.append(action)
      frames.append(frame)
      goals.append(goal)

    def flatten(values):
      if isinstance(values[0], tuple):
        return tuple(
          torch.stack([value[i] for value in values]).flatten(0, 1)
          for i in range(len(values[0]))
        )
      return torch.stack(values).flatten(0, 1)

    flat_states = flatten(states)
    flat_frames = flatten(frames)
    flat_actions = flatten(actions)
    flat_previous_actions = flatten(previous_actions)
    reward, done = (
      self.mdp.reward(
        flat_states,
        flat_frames,
        flat_actions,
        flat_previous_actions,
      ),
      self.mdp.soft_termination(flat_states, flat_frames),
    )
    reward = reward.view(self.horizon, self.policy_batch)
    done = done.view(self.horizon, self.policy_batch)

    alive = torch.cumprod(1 - done, 0)
    survival = torch.cat(
      (torch.ones(1, self.policy_batch, device=self.device), alive[:-1]),
      0,
    )
    discounts = self.gamma ** torch.arange(
      self.horizon, device=self.device, dtype=torch.float32
    ).unsqueeze(-1)
    objective = (discounts * survival * reward).sum(0)
    if not self.no_critic:
      objective = objective + (self.gamma**self.horizon) * alive[-1] * self.critic(
        torch.cat(
          (
            self.mdp.critic_features(state, frame, sim_context),
            goal,
          ),
          -1,
        )
      )
    policy_loss = -objective.mean()
    self.opt_policy.zero_grad(set_to_none=True)
    policy_loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
    nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
    if torch.isfinite(gradient_norm):
      self.opt_policy.step()
    _set_requires_grad(self.world_model, True)
    _set_requires_grad(self.critic, True)

    critic_loss = torch.zeros((), device=self.device)
    if not self.no_critic:
      with torch.no_grad():
        all_states = tuple(
          torch.cat(
            (
              start_state[index].unsqueeze(0),
              torch.stack([value[index].detach() for value in states]),
            ),
            0,
          ).flatten(0, 1)
          for index in range(6)
        )
        all_frames = torch.cat(
          (start_frame.unsqueeze(0), torch.stack(frames)), 0
        ).flatten()
        all_goals = torch.cat((start_goal.unsqueeze(0), torch.stack(goals)), 0).flatten(
          0, 1
        )
        all_context = (
          normalized_context.unsqueeze(0).expand(self.horizon + 1, -1, -1).flatten(0, 1)
        )
        critic_input = self._critic_input(
          all_states,
          all_frames,
          all_context,
          all_goals,
        ).view(self.horizon + 1, self.policy_batch, -1)
      self.critic.norm.update(critic_input.reshape(-1, critic_input.shape[-1]))
      with torch.no_grad():
        values = self.critic(critic_input.reshape(-1, critic_input.shape[-1])).view(
          self.horizon + 1, self.policy_batch
        )
        target = values[-1]
        targets = []
        detached_rewards = reward.detach()
        detached_done = done.detach()
        for step in range(self.horizon - 1, -1, -1):
          target = detached_rewards[step] + self.gamma * (1 - detached_done[step]) * (
            (1 - self.lam) * values[step + 1] + self.lam * target
          )
          targets.append(target)
        targets = torch.stack(targets[::-1])
      inputs = critic_input[:-1].reshape(-1, critic_input.shape[-1])
      outputs = targets.reshape(-1)
      n = inputs.shape[0]
      minibatch = max(n // self.critic_minibatches, 1)
      for _ in range(self.critic_epochs):
        permutation = torch.randperm(n, device=self.device)
        for index in range(self.critic_minibatches):
          ids = permutation[index * minibatch : (index + 1) * minibatch]
          if len(ids) == 0:
            continue
          critic_loss = ((self.critic(inputs[ids]) - outputs[ids]) ** 2).mean()
          self.opt_critic.zero_grad(set_to_none=True)
          critic_loss.backward()
          nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
          self.opt_critic.step()

    imagined_return = reward.detach().sum(0).mean() / 0.1
    return (
      float(policy_loss.detach()),
      float(critic_loss.detach()),
      float(imagined_return),
      float(gradient_norm.detach()),
      float(alive[-1].mean().detach()),
    )

  def _critic_input(self, state, frame, sim_context, goal) -> torch.Tensor:
    return torch.cat(
      (self.mdp.critic_features(state, frame, sim_context), goal),
      -1,
    )

  def _lr_at(self, iteration: int) -> float:
    if iteration >= self.lr_decay_iters:
      return self.learning_rate * 0.03
    progress = iteration / max(self.lr_decay_iters, 1)
    return self.learning_rate * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress)))

  def _init_logger(self) -> None:
    path = os.path.join(self.log_dir, "train_log.csv")
    is_new = not os.path.exists(path)
    self._csv_file = open(path, "a", newline="")
    if is_new:
      csv.writer(self._csv_file).writerow(
        (
          "iteration",
          "wall_h",
          "sim_steps",
          "wm_loss",
          "wm_err",
          "policy_loss",
          "critic_loss",
          "imagined_return",
          "grad_norm",
          "survival",
          "return_norm",
          "episode_len",
          "term_frac",
          "iterations_per_second",
          "learning_rate",
        )
      )
    if self.cfg.get("logger") == "wandb":
      try:
        import wandb

        self._wandb = wandb.init(
          project=self.cfg.get("wandb_project", "mjlab"),
          name=self.cfg.get("run_name") or os.path.basename(self.log_dir),
          dir=self.log_dir,
          config=self.cfg,
        )
      except Exception as exc:
        print(f"[SuperMimic] wandb unavailable ({exc}); using CSV only.")

  def _log(self, iteration: int, **metrics) -> None:
    wall_hours = (time.time() - self._t_start) / 3600
    iterations_per_second = self.log_interval / max(time.time() - self._t_log, 1e-6)
    self._t_log = time.time()
    mean_return = (
      sum(self._completed_returns) / len(self._completed_returns)
      if self._completed_returns
      else 0.0
    )
    mean_length = (
      sum(self._completed_lengths) / len(self._completed_lengths)
      if self._completed_lengths
      else 0.0
    )
    term_fraction = self._term_count / max(self._done_count, 1)
    row = {
      "iteration": iteration,
      "wall_h": wall_hours,
      "sim_steps": self._sim_steps,
      **metrics,
      "return_norm": mean_return,
      "episode_len": mean_length,
      "term_frac": term_fraction,
      "iterations_per_second": iterations_per_second,
    }
    if self._csv_file is not None:
      csv.writer(self._csv_file).writerow(
        [
          f"{value:.6g}" if isinstance(value, float) else value
          for value in row.values()
        ]
      )
      self._csv_file.flush()
    if self._wandb is not None:
      self._wandb.log(row, step=iteration)
    print(
      f"it {iteration} | {wall_hours:.3f} h | "
      f"wm {metrics.get('wm_loss', 0):.4f} "
      f"| pi {metrics.get('policy_loss', 0):.3f} "
      f"| imag_r {metrics.get('imagined_return', 0):.3f} "
      f"| sim_ret {mean_return:.3f} | {iterations_per_second:.2f} it/s"
    )
    self._completed_returns.clear()
    self._completed_lengths.clear()
    self._term_count = 0
    self._done_count = 0

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    del init_at_random_ep_len
    self._init_logger()
    self.collect(self.warmup_steps)
    wm_history: list[float] = []
    error_history: list[float] = []
    policy_history: list[float] = []
    critic_history: list[float] = []
    imagined_history: list[float] = []
    gradient_history: list[float] = []
    survival_history: list[float] = []
    step_accumulator = 0.0

    for iteration in range(self.current_learning_iteration, num_learning_iterations):
      learning_rate = self._lr_at(iteration)
      for optimizer in (self.opt_wm, self.opt_policy, self.opt_critic):
        for group in optimizer.param_groups:
          group["lr"] = learning_rate

      step_accumulator += 1.0 / max(self.utd, 1e-6)
      steps = int(step_accumulator)
      step_accumulator -= steps
      if steps > 0:
        self.collect(steps)

      for _ in range(self.wm_batches):
        wm_loss, wm_error = self._world_model_update()
        wm_history.append(wm_loss)
        error_history.append(wm_error)

      policy_loss = critic_loss = imagined_return = 0.0
      gradient_norm = survival = 0.0
      if iteration >= self.wm_only_iters:
        (
          policy_loss,
          critic_loss,
          imagined_return,
          gradient_norm,
          survival,
        ) = self._policy_update()
        policy_history.append(policy_loss)
        critic_history.append(critic_loss)
        imagined_history.append(imagined_return)
        gradient_history.append(gradient_norm)
        survival_history.append(survival)

      self.current_learning_iteration = iteration + 1
      if (iteration + 1) % self.log_interval == 0:

        def mean(values):
          return sum(values) / len(values) if values else 0.0

        self._log(
          iteration + 1,
          wm_loss=mean(wm_history),
          wm_err=mean(error_history),
          policy_loss=mean(policy_history),
          critic_loss=mean(critic_history),
          imagined_return=mean(imagined_history),
          grad_norm=mean(gradient_history),
          survival=mean(survival_history),
          learning_rate=learning_rate,
        )
        for values in (
          wm_history,
          error_history,
          policy_history,
          critic_history,
          imagined_history,
          gradient_history,
          survival_history,
        ):
          values.clear()

      if (iteration + 1) % self.ckpt_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{iteration + 1:06d}.pt"))

    self.save(os.path.join(self.log_dir, "last.pt"))
    if self._csv_file is not None:
      self._csv_file.close()
    if self._wandb is not None:
      self._wandb.finish()

  def save(self, path: str) -> None:
    payload = {
      "wm": self.world_model.state_dict(),
      "pi": self.policy.state_dict(),
      "critic": self.critic.state_dict(),
      "opt_wm": self.opt_wm.state_dict(),
      "opt_policy": self.opt_policy.state_dict(),
      "opt_critic": self.opt_critic.state_dict(),
      "iteration": self.current_learning_iteration,
      "config": self.cfg,
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    del load_cfg, strict
    checkpoint = torch.load(
      path,
      map_location=map_location or self.device,
      weights_only=False,
    )
    self.world_model.load_state_dict(checkpoint["wm"], strict=False)
    self.policy.load_state_dict(checkpoint["pi"], strict=False)
    self.critic.load_state_dict(checkpoint["critic"], strict=False)
    if "opt_wm" in checkpoint:
      self.opt_wm.load_state_dict(checkpoint["opt_wm"])
      self.opt_policy.load_state_dict(checkpoint["opt_policy"])
      self.opt_critic.load_state_dict(checkpoint["opt_critic"])
    self.current_learning_iteration = int(checkpoint.get("iteration", 0))
    return {}

  def get_inference_policy(self, device: str | None = None):
    del device
    self.policy.eval()
    self.world_model.eval()
    self.critic.eval()
    self._inference_history = None

    def policy(observation):
      del observation
      state = self.state()
      sim_context = self.sim_context()
      fresh = self.env.episode_length_buf == 0
      self.previous_action[fresh] = 0.0
      current = self.mdp.obs(
        state,
        self.previous_action,
        sim_context,
        self.command.time_steps,
      )
      if self._inference_history is None:
        self._inference_history = current.unsqueeze(1).repeat(1, self.history, 1)
      else:
        self._inference_history = torch.where(
          fresh[:, None, None],
          current.unsqueeze(1).expand_as(self._inference_history),
          torch.cat(
            (
              self._inference_history[:, 1:],
              current.unsqueeze(1),
            ),
            1,
          ),
        )
      goal = self.motion.goal(self.command.time_steps, self.goal_frames)
      with torch.no_grad():
        action = self.policy(self._inference_history, goal)
      action = torch.clamp(action, -self.action_clip, self.action_clip)
      self.previous_action = action.detach().clone()
      return action

    return policy

  def add_git_repo_to_log(self, *args, **kwargs) -> None:
    del args, kwargs
