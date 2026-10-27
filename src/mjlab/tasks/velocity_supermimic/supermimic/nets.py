"""Networks for the Go1 velocity SuperMimic variant.

The structure follows the reference reproduction (``supermimic_vibed``) and the
paper's Table A.4, with two practical changes for the velocity task:

* the world model predicts base pose/velocity/joint increments plus auxiliary
  contact/air-time/foot-height quantities, and
* reward, observation and termination heads are *learned* instead of analytic,
  because the mjlab velocity reward depends on contact sensors and other
  quantities that are not a closed-form function of the reduced state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import (
  AIR,
  AIR_DIM,
  ANG,
  CONTACT,
  CONTACT_DIM,
  DYN_DIM,
  FH,
  FH_DIM,
  LIN,
  POS,
  QD,
  ROT6,
  STATE_DIM,
  Q,
)
from .rotations import normalize_rot6d


def mlp(
  in_dim: int,
  hidden: tuple[int, ...],
  out_dim: int,
  act: type[nn.Module] = nn.ELU,
  layer_norm: bool = False,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  dim = in_dim
  for h in hidden:
    layers.append(nn.Linear(dim, h))
    if layer_norm:
      layers.append(nn.LayerNorm(h))
    layers.append(act())
    dim = h
  layers.append(nn.Linear(dim, out_dim))
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  """Running mean/std normalizer, updated from data without gradients."""

  def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))
    self.max_count = max_count
    self.eps = eps

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1]).float()
    if self.count >= self.max_count:
      return
    n = x.shape[0]
    mean = x.mean(0)
    var = x.var(0, unbiased=False)
    total = self.count + n
    delta = mean - self.mean
    self.mean += delta * n / total
    self.var = (
      self.var * self.count + var * n + delta**2 * self.count * n / total
    ) / total
    self.count = total

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std()

  def std(self) -> torch.Tensor:
    return torch.sqrt(self.var + self.eps**2)


class WorldModel(nn.Module):
  """Deterministic dynamics ``s_{t+1} = f(s_t, a_t, xi_t, u_t)``.

  ``u_t`` is the external velocity kick applied during the transition (zero when
  no push occurred).  Keeping it as an explicit input lets the otherwise
  deterministic model represent the state discontinuity caused by mjlab's
  interval push events.
  """

  def __init__(
    self,
    action_dim: int,
    xi_dim: int,
    push_dim: int = 6,
    hidden: tuple[int, ...] = (512, 512, 512),
  ):
    super().__init__()
    self.action_dim = action_dim
    self.xi_dim = xi_dim
    self.push_dim = push_dim
    self.in_dim = STATE_DIM + action_dim + xi_dim + push_dim
    self.out_dim = DYN_DIM + CONTACT_DIM + AIR_DIM + FH_DIM
    self.net = mlp(self.in_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
    last = self.net[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    self.in_norm = RunningNorm(self.in_dim)
    self.dyn_norm = RunningNorm(DYN_DIM)
    self.air_norm = RunningNorm(AIR_DIM)
    self.fh_norm = RunningNorm(FH_DIM)

  def inputs(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    xi: torch.Tensor,
    u: torch.Tensor,
  ) -> torch.Tensor:
    return torch.cat((s, a, xi, u), -1)

  def step(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    xi: torch.Tensor,
    u: torch.Tensor,
    return_aux: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y = self.net(self.in_norm(self.inputs(s, a, xi, u)))
    dyn = y[..., :DYN_DIM] * self.dyn_norm.std() + self.dyn_norm.mean
    contact_logits = y[..., DYN_DIM : DYN_DIM + CONTACT_DIM]
    air_delta = y[..., DYN_DIM + CONTACT_DIM : DYN_DIM + CONTACT_DIM + AIR_DIM]
    air_delta = air_delta * self.air_norm.std() + self.air_norm.mean
    fh_delta = y[..., DYN_DIM + CONTACT_DIM + AIR_DIM :]
    fh_delta = fh_delta * self.fh_norm.std() + self.fh_norm.mean

    pos = s[..., POS] + dyn[..., 0:3]
    rot6 = normalize_rot6d(dyn[..., 3:9])
    lin = s[..., LIN] + dyn[..., 9:12]
    ang = s[..., ANG] + dyn[..., 12:15]
    q = s[..., Q] + dyn[..., 15:27]
    qd = s[..., QD] + dyn[..., 27:39]
    contact = torch.sigmoid(contact_logits)
    air = F.relu(s[..., AIR] + air_delta)
    fh = s[..., FH] + fh_delta
    s_next = torch.cat((pos, rot6, lin, ang, q, qd, contact, air, fh), -1)
    if not return_aux:
      return s_next
    return s_next, {
      "contact_logits": contact_logits,
      "air_delta": air_delta,
      "fh_delta": fh_delta,
    }

  @staticmethod
  def targets(s: torch.Tensor, s1: torch.Tensor) -> dict[str, torch.Tensor]:
    """Regression targets for dynamics and auxiliary heads."""
    dyn = torch.cat(
      (
        s1[..., POS] - s[..., POS],
        normalize_rot6d(s1[..., ROT6]),
        s1[..., LIN] - s[..., LIN],
        s1[..., ANG] - s[..., ANG],
        s1[..., Q] - s[..., Q],
        s1[..., QD] - s[..., QD],
      ),
      -1,
    )
    return {
      "dyn": dyn,
      "contact": s1[..., CONTACT],
      "air": s1[..., AIR] - s[..., AIR],
      "fh": s1[..., FH] - s[..., FH],
    }

  def state_error(
    self,
    s_hat: torch.Tensor,
    s_target: torch.Tensor,
    aux: dict[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    """Per-sample errors, already normalized by the running output scale."""
    dyn_std = self.dyn_norm.std()
    dyn_err = torch.cat(
      (
        (s_hat[..., POS] - s_target[..., POS]) / dyn_std[0:3],
        (s_hat[..., ROT6] - normalize_rot6d(s_target[..., ROT6])) / dyn_std[3:9],
        (s_hat[..., LIN] - s_target[..., LIN]) / dyn_std[9:12],
        (s_hat[..., ANG] - s_target[..., ANG]) / dyn_std[12:15],
        (s_hat[..., Q] - s_target[..., Q]) / dyn_std[15:27],
        (s_hat[..., QD] - s_target[..., QD]) / dyn_std[27:39],
      ),
      -1,
    )
    contact_err = F.binary_cross_entropy_with_logits(
      aux["contact_logits"], s_target[..., CONTACT], reduction="none"
    )
    air_err = (s_hat[..., AIR] - s_target[..., AIR]) / self.air_norm.std()
    fh_err = (s_hat[..., FH] - s_target[..., FH]) / self.fh_norm.std()
    return {
      "dyn": dyn_err.pow(2).mean(-1),
      "contact": contact_err.mean(-1),
      "air": air_err.pow(2).mean(-1),
      "fh": fh_err.pow(2).mean(-1),
    }


class Policy(nn.Module):
  """Deterministic goal-conditioned policy: ``a = dec(enc(g), o_{t-9:t})``."""

  def __init__(
    self,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    hist: int = 10,
    z_dim: int = 32,
    hidden: tuple[int, ...] = (512, 512, 512),
  ):
    super().__init__()
    self.hist = hist
    self.obs_dim = obs_dim
    self.goal_dim = goal_dim
    self.action_dim = action_dim
    self.enc = mlp(goal_dim, (512, 256, 128), z_dim)
    self.dec = mlp(z_dim + obs_dim * hist, hidden, action_dim)
    last = self.dec[-1]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
      last.weight.mul_(0.01)
      last.bias.zero_()
    self.obs_norm = RunningNorm(obs_dim)
    self.goal_norm = RunningNorm(goal_dim)

  def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    z = torch.tanh(self.enc(self.goal_norm(goal)))
    obs = self.obs_norm(obs_hist).flatten(-2)
    return self.dec(torch.cat((z, obs), -1))


class Critic(nn.Module):
  def __init__(
    self,
    in_dim: int,
    hidden: tuple[int, ...] = (256, 128),
  ):
    super().__init__()
    self.net = mlp(in_dim, hidden, 1)
    self.norm = RunningNorm(in_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(x)).squeeze(-1)


class ObservationHead(nn.Module):
  """Learned noise-free proprioceptive observation ``o_t = h(s_t, a_t, a_{t-1}, g_t)``."""

  def __init__(
    self,
    action_dim: int,
    goal_dim: int,
    obs_dim: int,
    hidden: tuple[int, ...] = (512, 512),
  ):
    super().__init__()
    self.in_dim = STATE_DIM + 2 * action_dim + goal_dim
    self.net = mlp(self.in_dim, hidden, obs_dim, act=nn.SiLU, layer_norm=True)
    self.in_norm = RunningNorm(self.in_dim)
    self.out_norm = RunningNorm(obs_dim)

  def forward(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    goal: torch.Tensor,
  ) -> torch.Tensor:
    x = torch.cat((s, a, a_prev, goal), -1)
    return self.net(self.in_norm(x))

  def decode(self, y: torch.Tensor) -> torch.Tensor:
    return self.out_norm.mean + y * self.out_norm.std()


class RewardHead(nn.Module):
  """Learned reward ``r_t = r(s_t, a_t, a_{t-1}, g_t, xi_t, u_t)``."""

  def __init__(
    self,
    action_dim: int,
    goal_dim: int,
    xi_dim: int,
    push_dim: int = 6,
    hidden: tuple[int, ...] = (512, 512),
  ):
    super().__init__()
    self.in_dim = STATE_DIM + 2 * action_dim + goal_dim + xi_dim + push_dim
    self.net = mlp(self.in_dim, hidden, 1, act=nn.SiLU, layer_norm=True)
    self.in_norm = RunningNorm(self.in_dim)
    self.out_norm = RunningNorm(1)

  def forward(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    goal: torch.Tensor,
    xi: torch.Tensor,
    u: torch.Tensor,
  ) -> torch.Tensor:
    x = torch.cat((s, a, a_prev, goal, xi, u), -1)
    return self.net(self.in_norm(x)).squeeze(-1)

  def decode(self, y: torch.Tensor) -> torch.Tensor:
    return self.out_norm.mean[0] + y * self.out_norm.std()[0]


class TerminationHead(nn.Module):
  """Learned termination logit ``d_t = d(s_t, g_t)`` (hard terminations only)."""

  def __init__(
    self,
    goal_dim: int,
    hidden: tuple[int, ...] = (256, 128),
  ):
    super().__init__()
    self.in_dim = STATE_DIM + goal_dim
    self.net = mlp(self.in_dim, hidden, 1)
    self.in_norm = RunningNorm(self.in_dim)

  def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return self.net(self.in_norm(torch.cat((s, goal), -1))).squeeze(-1)
