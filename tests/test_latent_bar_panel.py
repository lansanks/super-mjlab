"""Tests for the Viser latent heatmap and bar panel."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mjlab.viewer.viewer_config import ViewerConfig
from mjlab.viewer.viser.latent_bar_panel import LatentBarPanel
from mjlab.viewer.viser.overlays import ViserTermOverlays


class _Scene:
  env_idx = 1
  debug_visualization_enabled = False
  needs_update = False
  show_contact_points = False
  show_contact_forces = False

  def clear_debug_all(self) -> None:
    pass

  def clear(self) -> None:
    pass


class _OverlayEnv:
  def __init__(self, unwrapped) -> None:
    self._unwrapped = unwrapped

  @property
  def unwrapped(self):
    return self._unwrapped


def _make_panel() -> tuple[LatentBarPanel, Mock]:
  server = Mock()
  html_handle = Mock()
  server.gui.add_html.return_value = html_handle
  panel = LatentBarPanel(server, label="student l_hat", value_range=(-3.0, 3.0))
  return panel, html_handle


def test_rejects_invalid_display_range():
  server = Mock()
  with pytest.raises(ValueError, match="increasing"):
    LatentBarPanel(server, label="latent", value_range=(3.0, -3.0))


def test_renders_heatmap_bars_and_summary():
  panel, html_handle = _make_panel()
  values = np.linspace(-3.0, 3.0, 16)
  panel.update(values, env_idx=2)

  content = html_handle.content
  assert "student l_hat" in content
  assert "env #2" in content
  assert "d0" in content and "d15" in content
  assert "-3.00" in content and "+3.00" in content
  assert "norm" in content


def test_clear_and_cleanup():
  panel, html_handle = _make_panel()
  panel.clear()
  assert "Waiting" in html_handle.content
  panel.cleanup()
  html_handle.remove.assert_called_once()


def test_term_overlays_create_latent_tab_when_env_publishes_vector():
  server = Mock()
  html_handle = Mock()
  server.gui.add_html.return_value = html_handle
  tabs = Mock()
  tabs.add_tab.return_value = nullcontext()
  latent = torch.zeros(2, 16)
  unwrapped = SimpleNamespace(
    cfg=SimpleNamespace(
      viewer=ViewerConfig(
        latent_bar_attr="latent",
        latent_bar_label="test latent",
      )
    ),
    latent=latent,
  )
  overlays = ViserTermOverlays(
    server=server,
    env=_OverlayEnv(unwrapped),
    scene=_Scene(),
    frame_time=1 / 30,
  )
  overlays.setup_tabs(tabs)

  assert overlays.latent_bar_panel is not None
  latent[1] = torch.linspace(-2.0, 2.0, 16)
  overlays.update(paused=False)
  assert "env #1" in html_handle.content
