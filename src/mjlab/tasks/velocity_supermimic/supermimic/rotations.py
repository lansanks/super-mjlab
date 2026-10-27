"""Small rotation helpers used by the SuperMimic world model.

The state stores the base orientation as a 6-D rotation representation (the first
two columns of the rotation matrix).  This keeps the regression target continuous
and avoids quaternion sign flips, while still allowing the predicted state to be
converted back to a valid rotation.
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
  """Convert wxyz quaternions to the 6-D rotation representation."""
  mat = matrix_from_quat(quat)
  return mat[..., :, :2].reshape(*mat.shape[:-2], 6)


def normalize_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
  """Gram-Schmidt orthonormalisation of a 6-D rotation representation."""
  a1, a2 = rot6d[..., :3], rot6d[..., 3:]
  b1 = torch.nn.functional.normalize(a1, dim=-1)
  b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
  b2 = torch.nn.functional.normalize(b2, dim=-1)
  return torch.cat((b1, b2), -1)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
  """Convert a 6-D rotation representation to a rotation matrix."""
  a1, a2 = rot6d[..., :3], rot6d[..., 3:]
  b1 = torch.nn.functional.normalize(a1, dim=-1)
  b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
  b2 = torch.nn.functional.normalize(b2, dim=-1)
  b3 = torch.cross(b1, b2, dim=-1)
  return torch.stack((b1, b2, b3), dim=-1)


def rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
  """Convert a 6-D rotation representation to a wxyz quaternion."""
  return quat_from_matrix(rot6d_to_matrix(rot6d))
