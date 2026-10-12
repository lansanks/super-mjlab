"""Tests for DreamWaQ's CENet error metrics."""

import math

import pytest
import torch
import torch.nn as nn

from mjlab.tasks.dreamwaq.rl.metrics import CenetMetrics


class StubCenet:
  """Minimal decoder stand-in: returns a target plus an injected error."""

  def __init__(self, decode: torch.Tensor) -> None:
    self._decode = decode
    self.obs_normalizer = nn.Identity()

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    del z
    return self._decode


def test_recon_terms_must_match_observation_dimension():
  with pytest.raises(ValueError, match="sum to 5"):
    CenetMetrics(obs_dim=4, recon_terms={"a": 3, "b": 2})
  # The default layout matches the Go1 DreamWaQ observation.
  CenetMetrics(obs_dim=45)


def test_velocity_percentages_are_relative_to_the_running_scale():
  metrics = CenetMetrics(obs_dim=1, velocity_dim=1, recon_terms={"x": 1})
  target = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
  metrics.update_velocity_statistics(target)

  # Population std of the target is 1.0 and EmpiricalNormalization adds 1e-2.
  scale = 1.01
  estimate = target + scale
  values = metrics.velocity_metrics(estimate, target)
  assert values["Metrics/est_vel_mean_pct"] == pytest.approx(100.0, rel=1e-4)

  doubled = metrics.velocity_metrics(target + 2 * scale, target)
  assert doubled["Metrics/est_vel_mean_pct"] == pytest.approx(200.0, rel=1e-4)

  zero = metrics.velocity_metrics(target, target)
  assert zero["Metrics/est_vel_mean_pct"] == 0.0


def test_reconstruction_percentages_and_term_slices():
  metrics = CenetMetrics(obs_dim=2, velocity_dim=1, recon_terms={"x": 2})
  target = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
  # First environment reconstructs perfectly, the second is off by 1.0 in a
  # dimension whose scale (batch std) is 1.0.
  decoded = torch.tensor([[-1.0, 0.0], [2.0, 0.0]])
  values = metrics.reconstruction_metrics(
    StubCenet(decoded),
    mu=torch.zeros(2, 1),
    target=target,
    valid=torch.ones(2, dtype=torch.bool),
  )
  # Environment 1 reconstruction error is one standard deviation in one of two
  # dimensions: rms = sqrt(1/2) -> 70.7%; the mean also includes environment 0.
  assert values["Metrics/recon_x_mean_pct"] == pytest.approx(
    50.0 * math.sqrt(0.5), rel=1e-4
  )


def test_terminal_steps_are_excluded_from_reconstruction_metrics():
  metrics = CenetMetrics(obs_dim=2, velocity_dim=1, recon_terms={"x": 2})
  target = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
  decoded = torch.tensor([[-1.0, 0.0], [10.0, 0.0]])
  values = metrics.reconstruction_metrics(
    StubCenet(decoded),
    mu=torch.zeros(2, 1),
    target=target,
    valid=torch.tensor([True, False]),  # environment 1 just terminated
  )
  assert values["Metrics/recon_x_mean_pct"] == 0.0


def test_latent_metrics_track_drift_spread_and_reset():
  metrics = CenetMetrics(obs_dim=2, velocity_dim=1, recon_terms={"x": 2})
  done = torch.zeros(2, dtype=torch.long)

  # First call only establishes the reference frame.
  values = metrics.latent_metrics(torch.zeros(2, 2), done)
  assert values["Metrics/latent_spread"] == 0.0
  assert "Metrics/latent_drift" not in values

  # Environment 0 moves by 1.0, environment 1 stays: drift = (1 + 0) / 2.
  mu = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
  values = metrics.latent_metrics(mu, done)
  assert values["Metrics/latent_drift"] == pytest.approx(0.5)
  # Spread: both environments deviate from the batch mean by L2 norm 0.5.
  assert values["Metrics/latent_spread"] == pytest.approx(0.5)

  # Environment 0 terminates: its next step is a new episode and is skipped.
  values = metrics.latent_metrics(mu, torch.tensor([1, 0]))
  assert "Metrics/latent_drift" not in values or values["Metrics/latent_drift"] == 0.0
  moved = torch.tensor([[50.0, 0.0], [1.0, 0.0]])
  values = metrics.latent_metrics(moved, done)
  # Only environment 1 counts: ||(1,0) - (0,0)|| = 1.
  assert values["Metrics/latent_drift"] == pytest.approx(1.0)
