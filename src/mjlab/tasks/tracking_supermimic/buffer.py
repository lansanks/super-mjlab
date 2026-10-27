"""GPU replay buffer laid out as ``(time, environment)``."""

from __future__ import annotations

import torch

from .constants import NUM_JOINTS, STATE_DIM
from .mdp import XI_DIM

FIELDS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
  "s": ((STATE_DIM,), torch.float32),
  "s1": ((STATE_DIM,), torch.float32),
  "a": ((NUM_JOINTS,), torch.float32),
  "a_prev": ((NUM_JOINTS,), torch.float32),
  "xi": ((XI_DIM,), torch.float32),
  "r": ((), torch.float32),
  "f": ((), torch.long),
  "ep_step": ((), torch.int32),
  "done": ((), torch.bool),
}


class ReplayBuffer:
  def __init__(
    self,
    steps: int,
    num_envs: int,
    device: str | torch.device,
  ):
    self.steps = steps
    self.num_envs = num_envs
    self.device = device
    self.data = {
      key: torch.zeros(
        (steps, num_envs, *shape),
        dtype=dtype,
        device=device,
      )
      for key, (shape, dtype) in FIELDS.items()
    }
    self.ptr = 0
    self.size = 0

  def add(self, transition: dict[str, torch.Tensor]) -> None:
    for key in FIELDS:
      value = transition[key]
      if value.requires_grad:
        value = value.detach()
      self.data[key][self.ptr] = value
    self.ptr = (self.ptr + 1) % self.steps
    self.size = min(self.size + 1, self.steps)

  def _row(self, age: torch.Tensor) -> torch.Tensor:
    return (self.ptr - self.size + age) % self.steps

  def sample_windows(self, batch: int, window: int) -> dict[str, torch.Tensor]:
    start = torch.randint(0, self.size - window + 1, (batch,), device=self.device)
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    ages = start.unsqueeze(-1) + torch.arange(window, device=self.device)
    rows = self._row(ages)
    envs = env.unsqueeze(-1).expand_as(rows)
    keys = ("s", "s1", "a", "xi", "done")
    out = {key: self.data[key][rows, envs] for key in keys}
    done = out["done"].float()
    out["mask"] = torch.cumprod(
      torch.cat(
        (torch.ones(batch, 1, device=self.device), 1 - done[:, :-1]),
        -1,
      ),
      -1,
    )
    return out

  def sample_starts(
    self, batch: int, history: int
  ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    age = torch.randint(0, self.size, (batch,), device=self.device)
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    rows = self._row(age)
    episode_step = self.data["ep_step"][rows, env].long()
    back = torch.arange(history - 1, -1, -1, device=self.device)
    back = torch.minimum(back.unsqueeze(0), episode_step.unsqueeze(-1))
    back = torch.minimum(back, age.unsqueeze(-1))
    history_rows = self._row(age.unsqueeze(-1) - back)
    history_envs = env.unsqueeze(-1).expand_as(history_rows)
    start = {key: self.data[key][rows, env] for key in ("s", "a_prev", "xi", "f")}
    history_data = {
      key: self.data[key][history_rows, history_envs]
      for key in ("s", "a_prev", "xi", "f")
    }
    return start, history_data
