"""SuperMimic training loop for an mjlab task.

This is a task-agnostic implementation of the paper's Algorithm A.1 on top of a
regular mjlab ``ManagerBasedRlEnv``:

1. collect simulator transitions into a replay buffer,
2. fit a differentiable world model on short windows, and
3. improve the policy by BPTT through imagined rollouts, with a critic that
   bootstraps the return beyond the imagination horizon.

For the Go1 velocity task the world model sees the privileged state assembled by
:class:`StateAssembler` (base pose/velocity, joints, foot contacts, terrain
scan), while the actor only sees the requested proprioceptive observation
(no base linear velocity and no terrain height).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .layout import STATE_DIM
from .nets import (
  Critic,
  ObservationHead,
  Policy,
  RewardHead,
  TerminationHead,
  WorldModel,
)
from .state import StateAssembler


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  return (x * mask).sum() / mask.sum().clamp(min=1.0)


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
  for p in module.parameters():
    p.requires_grad_(flag)


class SuperMimicRunner:
  """Standalone runner compatible with mjlab's ``train.py`` / ``play.py``."""

  def __init__(
    self,
    env,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
  ):
    self.cfg: dict[str, Any] = dict(train_cfg)
    self.device = torch.device(device)
    self.wrapper = env
    self.env = env.unwrapped if hasattr(env, "unwrapped") else env
    self.log_dir = log_dir or os.getcwd()
    os.makedirs(self.log_dir, exist_ok=True)

    self.assembler = StateAssembler(self.env)
    self.num_envs = int(self.env.num_envs)
    self.state_dim = STATE_DIM
    self.action_dim = self.assembler.action_dim
    self.obs_dim = self.assembler.obs_dim
    self.goal_dim = self.assembler.goal_dim
    self.xi_dim = self.assembler.xi_dim
    self.push_dim = self.assembler.push_dim

    self.hist = int(self.cfg.get("obs_history", 10))
    self.horizon = int(self.cfg.get("horizon", 32))
    self.window = int(self.cfg.get("wm_window", 8))
    self.wm_batch = int(self.cfg.get("wm_batch", 256))
    self.wm_batches = int(self.cfg.get("wm_batches", 4))
    self.pi_batch = int(self.cfg.get("pi_batch", 256))
    self.lr = float(self.cfg.get("lr", 2e-4))
    self.lr_decay_iters = int(self.cfg.get("lr_decay_iters", 200_000))
    self.gamma = float(self.cfg.get("gamma", 0.99))
    self.lam = float(self.cfg.get("lam", 0.95))
    self.action_noise = float(self.cfg.get("action_noise", 0.1))
    self.action_clip = float(self.cfg.get("action_clip", 5.0))
    self.obs_noise = float(self.cfg.get("obs_noise", 0.05))
    self.critic_epochs = int(self.cfg.get("critic_epochs", 3))
    self.critic_minibatches = int(self.cfg.get("critic_minibatches", 4))
    self.warmup_steps = int(self.cfg.get("warmup_steps", 32))
    self.wm_only_iters = int(self.cfg.get("wm_only_iters", 100))
    self.buffer_steps = int(self.cfg.get("buffer_steps", 500))
    self.utd = float(self.cfg.get("utd", 0.4))
    self.log_interval = int(self.cfg.get("log_interval", 10))
    self.ckpt_interval = int(self.cfg.get("ckpt_interval", 500))
    self.imagine_pushes = bool(self.cfg.get("imagine_pushes", True))
    self.wm_err_threshold = float(self.cfg.get("wm_err_threshold", 1.5))
    self.bc_coef = float(self.cfg.get("bc_coef", 0.1))
    self.height_term_min = float(self.cfg.get("height_term_min", 0.0))
    self.term_pos_weight = float(self.cfg.get("term_pos_weight", 1.0))

    wm_hidden = tuple(self.cfg.get("wm_hidden", (512, 512, 512)))
    pi_hidden = tuple(self.cfg.get("pi_hidden", (512, 512, 512)))
    critic_hidden = tuple(self.cfg.get("critic_hidden", (256, 128)))

    self.wm = WorldModel(
      self.action_dim, self.xi_dim, self.push_dim, hidden=wm_hidden
    ).to(self.device)
    self.pi = Policy(
      self.obs_dim,
      self.goal_dim,
      self.action_dim,
      hist=self.hist,
      hidden=pi_hidden,
    ).to(self.device)
    self.critic = Critic(
      self.state_dim + self.goal_dim + self.xi_dim + self.push_dim,
      hidden=critic_hidden,
    ).to(self.device)
    self.obs_head = ObservationHead(self.action_dim, self.goal_dim, self.obs_dim).to(
      self.device
    )
    self.rew_head = RewardHead(
      self.action_dim, self.goal_dim, self.xi_dim, self.push_dim
    ).to(self.device)
    self.term_head = TerminationHead(self.goal_dim).to(self.device)

    wm_params = (
      list(self.wm.parameters())
      + list(self.obs_head.parameters())
      + list(self.rew_head.parameters())
      + list(self.term_head.parameters())
    )
    self.opt_wm = torch.optim.Adam(wm_params, lr=self.lr)
    self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=self.lr, betas=(0.9, 0.95))
    self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    self.buffer = ReplayBuffer(
      self.buffer_steps,
      self.num_envs,
      self.action_dim,
      self.obs_dim,
      self.goal_dim,
      self.xi_dim,
      self.push_dim,
      self.device,
    )
    if self.device.type == "cuda":
      _, total = torch.cuda.mem_get_info(self.device)
      if total < 12e9 and self.num_envs > 1024:
        print(
          f"[SuperMimic] WARNING: {self.num_envs} environments on a "
          f"{total / 1e9:.1f} GB GPU is likely to run out of memory in "
          "sim.forward; use 512-1024 environments on an 8 GB card."
        )

    self.obs_hist = torch.zeros(
      self.num_envs, self.hist, self.obs_dim, device=self.device
    )
    self.a_prev = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._prev_push = torch.zeros(self.num_envs, self.push_dim, device=self.device)
    self.current_obs, _ = self.env.reset()
    self.current_state = self.assembler.state(self.current_obs["critic"])
    self.current_xi = self.assembler.xi(self.current_obs["critic"])
    o, _ = self.assembler.split_actor(self.current_obs["actor"])
    self.obs_hist[:] = o.unsqueeze(1)
    self.ep_step = self.env.episode_length_buf.clone()

    self.current_learning_iteration = 0
    self._sim_steps = 0
    self._rew_sum = 0.0
    self._rew_count = 0
    self._term_count = 0
    self._done_count = 0
    self._wandb = None
    self._csv_file = None
    self._t_start = time.time()
    self._t_log = time.time()

  # ------------------------------------------------------------------ logging
  def _init_logger(self) -> None:
    logger = self.cfg.get("logger", "wandb")
    if logger == "wandb":
      try:
        import wandb

        tags = [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t]
        name = self.cfg.get("run_name") or os.path.basename(self.log_dir)
        self._wandb = wandb.init(
          project=self.cfg.get("wandb_project", "mjlab"),
          name=name,
          dir=self.log_dir,
          config=self.cfg,
          tags=tags or None,
        )
      except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[SuperMimic] wandb unavailable ({exc}); using CSV logging.")
        self._wandb = None
    path = os.path.join(self.log_dir, "train_log.csv")
    new = not os.path.exists(path)
    self._csv_file = open(path, "a", newline="")
    if new:
      self._csv_file.write(
        "iteration,wall_h,sim_steps,wm_loss,wm_err,pi_loss,v_loss,imag_ret,"
        "pi_gradnorm,survival,sim_rew,term_frac,lr,it_per_s\n"
      )

  def _log(self, iteration: int, **metrics: float) -> None:
    wall_h = (time.time() - self._t_start) / 3600.0
    ips = self.log_interval / max(time.time() - self._t_log, 1e-6)
    self._t_log = time.time()
    row = {
      "iteration": iteration,
      "wall_h": wall_h,
      "sim_steps": self._sim_steps,
      **metrics,
      "it_per_s": ips,
    }
    if self._csv_file is not None:
      keys = list(row.keys())
      self._csv_file.write(
        ",".join(
          f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in keys
        )
        + "\n"
      )
      self._csv_file.flush()
    if self._wandb is not None:
      self._wandb.log(row, step=iteration)
    print(
      f"it {iteration} | {wall_h:.3f} h | wm {metrics.get('wm_loss', 0):.4f} "
      f"| pi {metrics.get('pi_loss', 0):.3f} v {metrics.get('v_loss', 0):.3f} "
      f"| imag_r {metrics.get('imag_ret', 0):.3f} "
      f"| sim_r {metrics.get('sim_rew', 0):.3f} "
      f"| term {metrics.get('term_frac', 0):.2f} | {ips:.2f} it/s"
    )
    self._rew_sum = 0.0
    self._rew_count = 0
    self._term_count = 0
    self._done_count = 0

  # ------------------------------------------------------------------ collect
  def _collect(self, steps: int) -> None:
    for _ in range(steps):
      self._step_env()

  @torch.no_grad()
  def _step_env(self) -> None:
    obs = self.current_obs
    o, goal = self.assembler.split_actor(obs["actor"])
    action = torch.clamp(
      self.pi(self.obs_hist, goal), -self.action_clip, self.action_clip
    )
    noisy_action = torch.clamp(
      action + self.action_noise * torch.randn_like(action),
      -self.action_clip,
      self.action_clip,
    )

    state = self.current_state
    xi = self.current_xi
    a_prev = self.a_prev
    ep_step = self.ep_step

    obs_next, rew, term, trunc, _ = self.env.step(noisy_action)
    state_next = self.assembler.state(obs_next["critic"])
    push_now = self.assembler.push(obs_next["critic"])
    # ``push_velocity`` holds the most recent kick until the next reset, so turn
    # it into a one-step exogenous input by detecting changes.
    changed = (push_now - self._prev_push).abs().sum(-1) > 1e-6
    push = torch.where(changed.unsqueeze(-1), push_now, torch.zeros_like(push_now))
    self._prev_push = push_now.clone()
    done = term | trunc

    self.buffer.add(
      {
        "s": state,
        "s1": state_next,
        "a": noisy_action,
        "a_prev": a_prev,
        "o": o,
        "g": goal,
        "xi": xi,
        "u": push,
        "r": rew,
        "d": term,
        "done": done,
        "ep_step": ep_step,
      }
    )
    self._sim_steps += self.num_envs
    self._rew_sum += float(rew.sum())
    self._rew_count += self.num_envs
    self._term_count += int(term.sum())
    self._done_count += int(done.sum())

    self._update_normalizers(
      state, noisy_action, xi, push, state_next, o, goal, rew, a_prev
    )

    o_next, goal_next = self.assembler.split_actor(obs_next["actor"])
    self.obs_hist = torch.cat((self.obs_hist[:, 1:], o_next.unsqueeze(1)), 1)
    self.current_obs = obs_next
    self.current_state = state_next
    self.current_xi = self.assembler.xi(obs_next["critic"])
    self.a_prev = noisy_action
    self.ep_step = self.env.episode_length_buf.clone()

    if bool(done.any()):
      reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
      reset_obs, _ = self.env.reset(env_ids=reset_ids)
      for key in self.current_obs:
        self.current_obs[key][reset_ids] = reset_obs[key][reset_ids]
      reset_state = self.assembler.state(reset_obs["critic"])
      self.current_state[reset_ids] = reset_state[reset_ids]
      self.current_xi[reset_ids] = self.assembler.xi(reset_obs["critic"])[reset_ids]
      reset_o, _ = self.assembler.split_actor(reset_obs["actor"])
      self.obs_hist[reset_ids] = reset_o[reset_ids].unsqueeze(1)
      self.a_prev[reset_ids] = 0.0
      self._prev_push[reset_ids] = 0.0
      self.ep_step[reset_ids] = self.env.episode_length_buf[reset_ids]

  def _update_normalizers(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    xi: torch.Tensor,
    push: torch.Tensor,
    state_next: torch.Tensor,
    obs: torch.Tensor,
    goal: torch.Tensor,
    reward: torch.Tensor,
    a_prev: torch.Tensor,
  ) -> None:
    with torch.no_grad():
      self.pi.obs_norm.update(obs)
      self.pi.goal_norm.update(goal)
      self.obs_head.in_norm.update(torch.cat((state, action, a_prev, goal), -1))
      self.obs_head.out_norm.update(obs)
      self.rew_head.in_norm.update(
        torch.cat((state, action, a_prev, goal, xi, push), -1)
      )
      self.rew_head.out_norm.update(reward.unsqueeze(-1))
      self.term_head.in_norm.update(torch.cat((state, goal), -1))
      self.wm.in_norm.update(self.wm.inputs(state, action, xi, push))
      targets = self.wm.targets(state, state_next)
      self.wm.dyn_norm.update(targets["dyn"])
      self.wm.air_norm.update(targets["air"])
      self.wm.fh_norm.update(targets["fh"])

  # ------------------------------------------------------------------ world model
  def _height_termination(self, state: torch.Tensor) -> torch.Tensor:
    """Differentiable low-base termination used during imagination.

    The learned termination head is trained on a heavily imbalanced buffer and
    can miss the rare low-base events, which lets the policy fall and believe it
    survives.  When ``height_term_min`` is set, combine the learned probability
    with an analytic height gate.
    """
    if self.height_term_min <= 0.0:
      return torch.zeros(state.shape[0], device=self.device)
    z = state[..., 2]
    return torch.sigmoid((self.height_term_min - z) / 0.01)

  def _wm_update(self) -> tuple[float, float]:
    if self.buffer.size < self.window:
      return 0.0, 0.0
    w = self.buffer.sample_windows(self.wm_batch, self.window)
    s = w["s"]
    s1 = w["s1"]
    mask = w["mask"]
    state = s[:, 0]
    loss = torch.zeros((), device=self.device)
    last_err = torch.zeros((), device=self.device)
    for j in range(self.window):
      state_next, aux = self.wm.step(
        state, w["a"][:, j], w["xi"][:, j], w["u"][:, j], return_aux=True
      )
      errs = self.wm.state_error(state_next, s1[:, j], aux)
      m = mask[:, j]
      loss = loss + _masked_mean(errs["dyn"], m)
      loss = loss + 0.2 * _masked_mean(errs["contact"], m)
      loss = loss + 0.1 * _masked_mean(errs["air"], m)
      loss = loss + 0.1 * _masked_mean(errs["fh"], m)

      r_hat = self.rew_head(
        state,
        w["a"][:, j],
        w["a_prev"][:, j],
        w["g"][:, j],
        w["xi"][:, j],
        w["u"][:, j],
      )
      r_target = self.rew_head.out_norm(w["r"][:, j].unsqueeze(-1)).squeeze(-1)
      loss = loss + _masked_mean((r_hat - r_target).pow(2), m)

      o_hat = self.obs_head(state, w["a"][:, j], w["a_prev"][:, j], w["g"][:, j])
      o_target = self.obs_head.out_norm(w["o"][:, j])
      loss = loss + _masked_mean((o_hat - o_target).pow(2).mean(-1), m)

      d_logit = self.term_head(state, w["g"][:, j])
      d_loss = F.binary_cross_entropy_with_logits(
        d_logit,
        w["d"][:, j].float(),
        reduction="none",
        pos_weight=torch.tensor(self.term_pos_weight, device=self.device),
      )
      loss = loss + 0.5 * _masked_mean(d_loss, m)

      last_err = _masked_mean(errs["dyn"], m).detach()
      state = state_next

    self.opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
      list(self.wm.parameters())
      + list(self.obs_head.parameters())
      + list(self.rew_head.parameters())
      + list(self.term_head.parameters()),
      1.0,
    )
    self.opt_wm.step()
    return float(loss.detach() / (self.window * 3)), float(last_err)

  # ------------------------------------------------------------------ policy
  def _policy_update(self) -> tuple[float, float, float, float, float]:
    st = self.buffer.sample_starts(self.pi_batch, self.hist)
    state = st["s"]
    obs_hist = st["o_hist"]
    a_prev = st["a_prev"]
    goal = st["g"]
    xi = st["xi"]
    a_bc = st["a"]

    _set_requires_grad(self.wm, False)
    _set_requires_grad(self.obs_head, False)
    _set_requires_grad(self.rew_head, False)
    _set_requires_grad(self.term_head, False)
    _set_requires_grad(self.critic, False)

    states = []
    actions = []
    rewards = []
    dones = []
    pushes = []
    bc_loss = torch.zeros((), device=self.device)
    for t in range(self.horizon):
      action = torch.clamp(self.pi(obs_hist, goal), -self.action_clip, self.action_clip)
      if t == 0 and self.bc_coef > 0.0:
        bc_loss = (action - a_bc).pow(2).mean()
      push = (
        self.assembler.sample_push(self.pi_batch)
        if self.imagine_pushes
        else torch.zeros(self.pi_batch, self.push_dim, device=self.device)
      )
      state_next, _ = self.wm.step(state, action, xi, push, return_aux=True)
      obs_hat = self.obs_head.decode(self.obs_head(state, action, a_prev, goal))
      obs_hat = obs_hat + self.obs_noise * torch.randn_like(obs_hat)
      obs_hist = torch.cat((obs_hist[:, 1:], obs_hat.unsqueeze(1)), 1)
      reward = self.rew_head.decode(
        self.rew_head(state, action, a_prev, goal, xi, push)
      )
      done = torch.sigmoid(self.term_head(state, goal))
      done = 1.0 - (1.0 - done) * (1.0 - self._height_termination(state))
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      dones.append(done)
      pushes.append(push)
      state = state_next
      a_prev = action

    push_h = (
      self.assembler.sample_push(self.pi_batch)
      if self.imagine_pushes
      else torch.zeros(self.pi_batch, self.push_dim, device=self.device)
    )
    value_h = self.critic(torch.cat((state, goal, xi, push_h), -1))

    r = torch.stack(rewards, 0)
    d = torch.stack(dones, 0)
    alive = torch.cumprod(1.0 - d, 0)
    c = torch.cat((torch.ones(1, self.pi_batch, device=self.device), alive[:-1]), 0)
    discount = self.gamma ** torch.arange(
      self.horizon, device=self.device, dtype=torch.float32
    ).unsqueeze(-1)
    J = (discount * c * r).sum(0)
    J = J + (self.gamma**self.horizon) * alive[-1] * value_h
    pi_loss = -J.mean() + self.bc_coef * bc_loss

    self.opt_pi.zero_grad(set_to_none=True)
    pi_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
    if torch.isfinite(grad_norm):
      self.opt_pi.step()

    _set_requires_grad(self.wm, True)
    _set_requires_grad(self.obs_head, True)
    _set_requires_grad(self.rew_head, True)
    _set_requires_grad(self.term_head, True)
    _set_requires_grad(self.critic, True)

    # Critic targets from the imagined transitions (detached).
    with torch.no_grad():
      state_c = torch.stack([s.detach() for s in states] + [state.detach()], 0)
      push_c = torch.stack(pushes + [push_h], 0)
      g_c = goal.unsqueeze(0).expand(self.horizon + 1, -1, -1)
      xi_c = xi.unsqueeze(0).expand(self.horizon + 1, -1, -1)
      X = torch.cat((state_c, g_c, xi_c, push_c), -1).reshape(
        (self.horizon + 1) * self.pi_batch, -1
      )
      self.critic.norm.update(X)
      V = self.critic(X).view(self.horizon + 1, self.pi_batch)
      target = V[-1]
      targets = []
      for t in range(self.horizon - 1, -1, -1):
        target = r[t].detach() + self.gamma * (1.0 - d[t].detach()) * (
          (1.0 - self.lam) * V[t + 1] + self.lam * target
        )
        targets.append(target)
      targets = torch.stack(targets[::-1])

    Xf = X[: self.horizon * self.pi_batch]
    Yf = targets.reshape(-1)
    v_loss = torch.zeros((), device=self.device)
    n = Xf.shape[0]
    mb = max(n // max(self.critic_minibatches, 1), 1)
    for _ in range(self.critic_epochs):
      perm = torch.randperm(n, device=self.device)
      for i in range(self.critic_minibatches):
        idx = perm[i * mb : (i + 1) * mb]
        if idx.numel() == 0:
          continue
        v_loss = (self.critic(Xf[idx]) - Yf[idx]).pow(2).mean()
        self.opt_v.zero_grad(set_to_none=True)
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_v.step()

    return (
      float(pi_loss.detach()),
      float(v_loss.detach()),
      float(J.mean().detach()),
      float(grad_norm),
      float(alive[-1].mean().detach()),
    )

  # ------------------------------------------------------------------ training
  def _lr_at(self, iteration: int) -> float:
    if iteration >= self.lr_decay_iters:
      return self.lr * 0.03
    progress = iteration / max(self.lr_decay_iters, 1)
    return self.lr * (0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * progress)))

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    del init_at_random_ep_len
    self._init_logger()
    self._collect(self.warmup_steps)

    wm_hist: list[float] = []
    wm_err_hist: list[float] = []
    pi_hist: list[float] = []
    v_hist: list[float] = []
    imag_hist: list[float] = []
    gn_hist: list[float] = []
    surv_hist: list[float] = []
    step_acc = 0.0

    for it in range(self.current_learning_iteration, num_learning_iterations):
      lr = self._lr_at(it)
      for opt in (self.opt_wm, self.opt_pi, self.opt_v):
        for group in opt.param_groups:
          group["lr"] = lr

      step_acc += 1.0 / max(self.utd, 1e-6)
      n_steps = int(step_acc)
      step_acc -= n_steps
      if n_steps > 0:
        self._collect(n_steps)

      wm_loss, wm_err = 0.0, 0.0
      for _ in range(self.wm_batches):
        wm_loss, wm_err = self._wm_update()
      wm_hist.append(wm_loss)
      wm_err_hist.append(wm_err)

      pi_loss = v_loss = imag_ret = grad_norm = survival = 0.0
      if it >= self.wm_only_iters and wm_err <= self.wm_err_threshold:
        pi_loss, v_loss, imag_ret, grad_norm, survival = self._policy_update()
        pi_hist.append(pi_loss)
        v_hist.append(v_loss)
        imag_hist.append(imag_ret)
        gn_hist.append(grad_norm)
        surv_hist.append(survival)

      self.current_learning_iteration = it + 1
      if (it + 1) % self.log_interval == 0:
        mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
        term_frac = self._term_count / max(self._done_count, 1)
        sim_rew = self._rew_sum / max(self._rew_count, 1)
        self._log(
          it + 1,
          wm_loss=mean(wm_hist),
          wm_err=mean(wm_err_hist),
          pi_loss=mean(pi_hist),
          v_loss=mean(v_hist),
          imag_ret=mean(imag_hist),
          pi_gradnorm=mean(gn_hist),
          survival=mean(surv_hist),
          sim_rew=sim_rew,
          term_frac=term_frac,
          lr=lr,
        )
        wm_hist.clear()
        wm_err_hist.clear()
        pi_hist.clear()
        v_hist.clear()
        imag_hist.clear()
        gn_hist.clear()
        surv_hist.clear()

      if (it + 1) % self.ckpt_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))

    self.save(os.path.join(self.log_dir, "last.pt"))
    if self._wandb is not None:
      self._wandb.finish()
    if self._csv_file is not None:
      self._csv_file.close()

  # ------------------------------------------------------------------ checkpoint
  def save(self, path: str) -> None:
    payload = {
      "wm": self.wm.state_dict(),
      "pi": self.pi.state_dict(),
      "critic": self.critic.state_dict(),
      "obs_head": self.obs_head.state_dict(),
      "rew_head": self.rew_head.state_dict(),
      "term_head": self.term_head.state_dict(),
      "opt_wm": self.opt_wm.state_dict(),
      "opt_pi": self.opt_pi.state_dict(),
      "opt_v": self.opt_v.state_dict(),
      "iteration": self.current_learning_iteration,
      "cfg": self.cfg,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    del load_cfg, strict
    ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
    self.wm.load_state_dict(ck["wm"], strict=False)
    self.pi.load_state_dict(ck["pi"], strict=False)
    self.critic.load_state_dict(ck["critic"], strict=False)
    self.obs_head.load_state_dict(ck["obs_head"], strict=False)
    self.rew_head.load_state_dict(ck["rew_head"], strict=False)
    self.term_head.load_state_dict(ck["term_head"], strict=False)
    if "opt_wm" in ck:
      self.opt_wm.load_state_dict(ck["opt_wm"])
      self.opt_pi.load_state_dict(ck["opt_pi"])
      self.opt_v.load_state_dict(ck["opt_v"])
    self.current_learning_iteration = int(ck.get("iteration", 0))
    return {}

  # ------------------------------------------------------------------ inference
  def get_inference_policy(self, device: str | None = None):
    device = device or str(self.device)
    self.pi.eval()
    for module in (self.wm, self.obs_head, self.rew_head, self.term_head, self.critic):
      module.eval()
    hist = torch.zeros(1, self.hist, self.obs_dim, device=device)
    initialized = False

    def policy(obs):
      nonlocal hist, initialized
      # ``play.py`` passes a TensorDict, which is a Mapping but not a ``dict``.
      if hasattr(obs, "keys") and "actor" in obs.keys():
        actor = obs["actor"]
      else:
        actor = obs
      o, goal = self.assembler.split_actor(actor)
      if not initialized or hist.shape[0] != o.shape[0]:
        hist = o.unsqueeze(1).expand(-1, self.hist, -1).contiguous()
        initialized = True
      else:
        hist = torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
      with torch.no_grad():
        action = self.pi(hist, goal)
      return torch.clamp(action, -self.action_clip, self.action_clip)

    return policy

  def add_git_repo_to_log(self, *args, **kwargs) -> None:
    del args, kwargs
