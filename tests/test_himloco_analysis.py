"""Regression tests for the HIM latent analysis metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

PHASE_CHECK_PATH = (
  Path(__file__).parents[1] / "src/mjlab/tasks/himloco/doc/实验/实验数据/phase_check.py"
)


def load_phase_check() -> Any:
  spec = importlib.util.spec_from_file_location("phase_check", PHASE_CHECK_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


phase_check = load_phase_check()


def test_phase_r2_does_not_count_constant_offset():
  rng = np.random.default_rng(0)
  phase = np.linspace(0.0, 8.0 * np.pi, 256)
  latent = 100.0 + rng.normal(size=(len(phase), 4))

  r2, residual = phase_check.phase_r2(latent, phase)

  assert residual.shape == latent.shape
  assert abs(r2) < 0.1


def test_pca_effective_dimension_uses_variances():
  rng = np.random.default_rng(0)
  primary = np.linspace(-1.0, 1.0, 512)
  latent = np.column_stack([primary, 0.01 * rng.normal(size=len(primary))])

  ratios, participation = phase_check.pca_variance_spectrum(latent)

  assert ratios[0] > 0.999
  assert 1.0 <= participation < 1.01
