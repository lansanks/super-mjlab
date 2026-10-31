"""ONNX export utilities for SuperMimic policies."""

from __future__ import annotations

import os

import torch
from torch import nn

from .nets import Policy


class SuperMimicPolicyOnnxModel(nn.Module):
  """Expose the two-input SuperMimic policy as a deployment ONNX graph."""

  def __init__(self, policy: Policy, action_clip: float):
    super().__init__()
    self.policy = policy
    self.action_clip = action_clip

  def forward(
    self,
    observation_history: torch.Tensor,
    goal: torch.Tensor,
  ) -> torch.Tensor:
    action = self.policy(observation_history, goal)
    return torch.clamp(action, -self.action_clip, self.action_clip)


def export_policy_to_onnx(
  policy: Policy,
  output_path: str | os.PathLike[str],
  *,
  history: int,
  observation_dim: int,
  goal_dim: int,
  action_clip: float,
  verbose: bool = False,
) -> None:
  """Export a SuperMimic policy with fixed deployment input shapes."""
  policy = policy.cpu().eval()
  model = SuperMimicPolicyOnnxModel(policy, action_clip).eval()
  output_path = os.fspath(output_path)
  os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
  torch.onnx.export(
    model,
    (
      torch.zeros(1, history, observation_dim),
      torch.zeros(1, goal_dim),
    ),
    output_path,
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=["observation_history", "goal"],
    output_names=["actions"],
    dynamic_axes=None,
    dynamo=False,
  )
