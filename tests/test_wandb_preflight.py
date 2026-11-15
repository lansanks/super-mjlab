"""Tests for the W&B connectivity preflight."""

import email.message
import os
import urllib.error

import pytest

from mjlab.utils.wandb import ensure_wandb_reachable


class _FakeResponse:
  def __init__(self, status: int) -> None:
    self.status = status

  def __enter__(self):
    return self

  def __exit__(self, *args):
    return False


@pytest.fixture
def wandb_mode(monkeypatch):
  """Reset WANDB_MODE to "unset" and restore it afterwards."""
  monkeypatch.setenv("WANDB_MODE", "")
  monkeypatch.delenv("WANDB_BASE_URL", raising=False)
  return monkeypatch


def test_reachable_api_keeps_online_logging(wandb_mode):
  wandb_mode.setattr(
    "urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(200)
  )

  assert ensure_wandb_reachable() is True
  assert "offline" not in os.environ["WANDB_MODE"]


def test_blocked_api_falls_back_to_offline(wandb_mode):
  def _forbidden(*args, **kwargs):
    raise urllib.error.HTTPError(
      "https://api.wandb.ai/graphql", 403, "Forbidden", email.message.Message(), None
    )

  wandb_mode.setattr("urllib.request.urlopen", _forbidden)

  assert ensure_wandb_reachable() is False
  assert os.environ["WANDB_MODE"] == "offline"


def test_network_error_falls_back_to_offline(wandb_mode):
  def _unreachable(*args, **kwargs):
    raise OSError("Network is unreachable")

  wandb_mode.setattr("urllib.request.urlopen", _unreachable)

  assert ensure_wandb_reachable() is False
  assert os.environ["WANDB_MODE"] == "offline"


def test_explicit_offline_mode_is_not_overridden(wandb_mode):
  def _never_called(*args, **kwargs):
    raise AssertionError("the API must not be probed in offline mode")

  wandb_mode.setattr("urllib.request.urlopen", _never_called)
  wandb_mode.setenv("WANDB_MODE", "disabled")

  assert ensure_wandb_reachable() is False
  assert os.environ["WANDB_MODE"] == "disabled"
