"""World model, policy and critic for G1 LAFAN1 tracking."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import NUM_JOINTS
from .mdp import INV_DIM, XI_DIM, invariant_features
from .rotations import so3_exp, so3_log, yaw_mat

WM_OUT = 3 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS


def mlp(
  input_dim: int,
  hidden: tuple[int, ...],
  output_dim: int,
  activation: type[nn.Module] = nn.ELU,
  layer_norm: bool = False,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  dim = input_dim
  for hidden_dim in hidden:
    layers.append(nn.Linear(dim, hidden_dim))
    if layer_norm:
      layers.append(nn.LayerNorm(hidden_dim))
    layers.append(activation())
    dim = hidden_dim
  layers.append(nn.Linear(dim, output_dim))
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))
    self.max_count = max_count
    self.eps = eps

  @torch.no_grad()
  def update(self, value: torch.Tensor) -> None:
    value = value.reshape(-1, value.shape[-1]).float()
    if self.count >= self.max_count:
      return
    count = value.shape[0]
    mean = value.mean(0)
    var = value.var(0, unbiased=False)
    total = self.count + count
    delta = mean - self.mean
    self.mean += delta * count / total
    self.var = (
      self.var * self.count + var * count + delta**2 * self.count * count / total
    ) / total
    self.count = total

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return (value - self.mean) / self.std()

  def std(self) -> torch.Tensor:
    return torch.sqrt(self.var + self.eps**2)


AMP_DTYPE: torch.dtype | None = None


def run_mlp(network: nn.Module, value: torch.Tensor) -> torch.Tensor:
  if AMP_DTYPE is None:
    return network(value)
  with torch.autocast(
    device_type=value.device.type,
    dtype=AMP_DTYPE,
    enabled=value.device.type == "cuda",
  ):
    return network(value).float()


class WorldModel(nn.Module):
  """Predict heading-invariant frame-local state increments."""

  def __init__(self, hidden: tuple[int, ...] = (2048, 2048, 2048)):
    super().__init__()
    self.input_dim = INV_DIM + NUM_JOINTS + XI_DIM
    self.net = mlp(
      self.input_dim,
      hidden,
      WM_OUT,
      activation=nn.SiLU,
      layer_norm=True,
    )
    last = self.net[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    self.input_norm = RunningNorm(self.input_dim)
    self.output_norm = RunningNorm(WM_OUT, eps=1e-4)

  def inputs(
    self,
    state,
    action: torch.Tensor,
    sim_context: torch.Tensor,
  ) -> torch.Tensor:
    return torch.cat((invariant_features(state), action, sim_context), -1)

  @staticmethod
  def targets(state, next_state) -> torch.Tensor:
    p, rot, lin_vel, ang_vel, q, qd = state
    p1, rot1, lin_vel1, ang_vel1, q1, qd1 = next_state
    heading = yaw_mat(rot)
    rot_t = rot.transpose(-1, -2)
    return torch.cat(
      (
        (heading.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
        so3_log(rot_t @ rot1),
        (rot_t @ (lin_vel1 - lin_vel).unsqueeze(-1)).squeeze(-1),
        (rot_t @ (ang_vel1 - ang_vel).unsqueeze(-1)).squeeze(-1),
        q1 - q,
        qd1 - qd,
      ),
      -1,
    )

  def step(
    self,
    state,
    action: torch.Tensor,
    sim_context: torch.Tensor,
  ):
    p, rot, lin_vel, ang_vel, q, qd = state
    normalized = run_mlp(
      self.net,
      self.input_norm(self.inputs(state, action, sim_context)),
    )
    delta = normalized * self.output_norm.std() + self.output_norm.mean
    dp, dtheta, dlin, dang, dq, dqd = delta.split(
      (3, 3, 3, 3, NUM_JOINTS, NUM_JOINTS), -1
    )
    heading = yaw_mat(rot)
    return (
      p + (heading @ dp.unsqueeze(-1)).squeeze(-1),
      rot @ so3_exp(dtheta),
      lin_vel + (rot @ dlin.unsqueeze(-1)).squeeze(-1),
      ang_vel + (rot @ dang.unsqueeze(-1)).squeeze(-1),
      q + dq,
      qd + dqd,
    )

  @staticmethod
  def state_error(predicted, target) -> torch.Tensor:
    p, rot, lin_vel, ang_vel, q, qd = target
    p_hat, rot_hat, lin_hat, ang_hat, q_hat, qd_hat = predicted
    heading = yaw_mat(rot)
    rot_t = rot.transpose(-1, -2)
    return torch.cat(
      (
        (heading.transpose(-1, -2) @ (p_hat - p).unsqueeze(-1)).squeeze(-1),
        so3_log(rot_t @ rot_hat),
        (rot_t @ (lin_hat - lin_vel).unsqueeze(-1)).squeeze(-1),
        (rot_t @ (ang_hat - ang_vel).unsqueeze(-1)).squeeze(-1),
        q_hat - q,
        qd_hat - qd,
      ),
      -1,
    )


class Policy(nn.Module):
  def __init__(
    self,
    obs_dim: int,
    goal_dim: int,
    history: int = 10,
    latent_dim: int = 32,
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256),
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256),
  ):
    super().__init__()
    self.history = history
    self.obs_dim = obs_dim
    self.goal_dim = goal_dim
    self.encoder = mlp(goal_dim, encoder_hidden, latent_dim)
    self.decoder = mlp(
      latent_dim + obs_dim * history,
      decoder_hidden,
      NUM_JOINTS,
    )
    last = self.decoder[-1]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
      last.weight.mul_(0.01)
      last.bias.zero_()
    self.obs_norm = RunningNorm(obs_dim)
    self.goal_norm = RunningNorm(goal_dim)

  def forward(
    self, observation_history: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    latent = torch.tanh(self.encoder(self.goal_norm(goal)))
    observation = self.obs_norm(observation_history).flatten(-2)
    return self.decoder(torch.cat((latent, observation), -1))


class Critic(nn.Module):
  def __init__(
    self,
    input_dim: int,
    hidden: tuple[int, ...] = (256, 128),
  ):
    super().__init__()
    self.net = mlp(input_dim, hidden, 1)
    self.norm = RunningNorm(input_dim)

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(value)).squeeze(-1)
