"""Fixed-size replay buffer for the Go1 velocity SuperMimic trainer.

The buffer stores vectorized transitions in ``(time, env)`` layout.  Windows and
start states are sampled by age, so a ring-buffer overwrite never changes the
meaning of an already-sampled index.
"""

from __future__ import annotations

import torch

from .layout import STATE_DIM


class ReplayBuffer:
  def __init__(
    self,
    steps: int,
    num_envs: int,
    action_dim: int,
    obs_dim: int,
    goal_dim: int,
    xi_dim: int,
    push_dim: int,
    device: str | torch.device,
  ):
    self.steps = steps
    self.num_envs = num_envs
    self.device = device
    self.ptr = 0
    self.size = 0
    self.data: dict[str, torch.Tensor] = {
      "s": torch.zeros(steps, num_envs, STATE_DIM, device=device),
      "s1": torch.zeros(steps, num_envs, STATE_DIM, device=device),
      "a": torch.zeros(steps, num_envs, action_dim, device=device),
      "a_prev": torch.zeros(steps, num_envs, action_dim, device=device),
      "o": torch.zeros(steps, num_envs, obs_dim, device=device),
      "g": torch.zeros(steps, num_envs, goal_dim, device=device),
      "xi": torch.zeros(steps, num_envs, xi_dim, device=device),
      "u": torch.zeros(steps, num_envs, push_dim, device=device),
      "r": torch.zeros(steps, num_envs, device=device),
      "d": torch.zeros(steps, num_envs, dtype=torch.bool, device=device),
      "done": torch.zeros(steps, num_envs, dtype=torch.bool, device=device),
      "ep_step": torch.zeros(steps, num_envs, dtype=torch.long, device=device),
    }

  def add(self, transition: dict[str, torch.Tensor]) -> None:
    for key, value in transition.items():
      if value.requires_grad:
        value = value.detach()
      self.data[key][self.ptr] = value
    self.ptr = (self.ptr + 1) % self.steps
    self.size = min(self.size + 1, self.steps)

  def _row(self, age: torch.Tensor) -> torch.Tensor:
    """Map age-ordered index (0 = oldest) to a storage row."""
    return (self.ptr - self.size + age) % self.steps

  def sample_windows(self, batch: int, window: int) -> dict[str, torch.Tensor]:
    """Sample ``window``-step windows with a validity mask (B, window)."""
    t0 = torch.randint(0, self.size - window + 1, (batch,), device=self.device)
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    ages = t0.unsqueeze(-1) + torch.arange(window, device=self.device)
    rows = self._row(ages)
    envs = env.unsqueeze(-1).expand_as(rows)
    keys = ("s", "s1", "a", "a_prev", "o", "g", "xi", "u", "r", "d", "done")
    out = {key: self.data[key][rows, envs] for key in keys}
    done = out["done"].float()
    mask = torch.cumprod(
      torch.cat((torch.ones(batch, 1, device=self.device), 1.0 - done[:, :-1]), -1),
      -1,
    )
    out["mask"] = mask
    return out

  def sample_starts(self, batch: int, hist: int) -> dict[str, torch.Tensor]:
    """Sample start states plus the preceding observation history."""
    age = torch.randint(0, self.size, (batch,), device=self.device)
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    row = self._row(age)
    ep_step = self.data["ep_step"][row, env]
    back = torch.arange(hist - 1, -1, -1, device=self.device)
    back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
    back = torch.minimum(back, age.unsqueeze(-1))
    hist_rows = self._row(age.unsqueeze(-1) - back)
    hist_envs = env.unsqueeze(-1).expand_as(hist_rows)
    out = {
      key: self.data[key][row, env]
      for key in ("s", "a", "a_prev", "xi", "g", "ep_step")
    }
    out["o_hist"] = self.data["o"][hist_rows, hist_envs]
    return out
