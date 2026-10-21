"""Live latent-vector heatmap and bar panel for the Viser viewer."""

from __future__ import annotations

import html

import numpy as np
import viser


class LatentBarPanel:
  """Render a fixed-range latent vector as a heatmap and vertical bars."""

  def __init__(
    self,
    server: viser.ViserServer,
    label: str,
    value_range: tuple[float, float] = (-3.0, 3.0),
  ) -> None:
    low, high = value_range
    if low >= high:
      raise ValueError(f"latent display range must be increasing, got {(low, high)}.")
    self._label = label
    self._low = float(low)
    self._high = float(high)
    self._html = server.gui.add_html("")
    self.clear()

  def clear(self) -> None:
    """Show a waiting state until the first latent is published."""
    self._html.content = (
      '<div style="padding:0.6em;color:#999;font-size:0.85em;">'
      f"Waiting for {html.escape(self._label)}…"
      "</div>"
    )

  def cleanup(self) -> None:
    self._html.remove()

  def update(self, values: np.ndarray, env_idx: int) -> None:
    """Render one latent vector for the selected environment."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
      self.clear()
      return
    finite = np.isfinite(vector)
    if not finite.all():
      vector = np.where(finite, vector, 0.0)

    n = int(vector.size)
    columns = 4
    rows = (n + columns - 1) // columns
    span = self._high - self._low
    normalized = np.clip((vector - self._low) / span, 0.0, 1.0)

    cells = []
    for dim, value in enumerate(vector):
      color = self._color(value)
      cells.append(
        f'<div title="l[{dim}]={value:.4f}" '
        'style="display:flex;flex-direction:column;align-items:center;'
        "justify-content:center;border-radius:4px;min-height:38px;"
        "padding:2px;background:"
        f'{color};color:#fff;font-family:monospace;font-size:0.72em;">'
        f"<span>d{dim}</span><strong>{value:+.2f}</strong>"
        "</div>"
      )
    heatmap = (
      f'<div style="display:grid;grid-template-columns:repeat({columns},'
      f"1fr);grid-template-rows:repeat({rows},1fr);gap:3px;"
      f'max-width:{columns * 64}px;margin:0 auto 8px auto;">'
      + "".join(cells)
      + "</div>"
    )

    bars = []
    for dim, _ in enumerate(normalized):
      value = float(vector[dim])
      magnitude = abs(value) / max(abs(self._low), abs(self._high), 1e-12)
      magnitude = min(magnitude, 1.0) * 50.0
      color = self._color(value)
      if value >= 0.0:
        bar = (
          '<div style="position:absolute;left:5px;right:5px;bottom:50%;'
          f'height:{magnitude:.1f}%;background:{color};border-radius:2px 2px 0 0;">'
          "</div>"
        )
      else:
        bar = (
          '<div style="position:absolute;left:5px;right:5px;top:50%;'
          f'height:{magnitude:.1f}%;background:{color};border-radius:0 0 2px 2px;">'
          "</div>"
        )
      bars.append(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'min-width:20px;flex:1;">'
        '<div style="position:relative;width:100%;height:54px;background:#222;'
        'border-radius:3px;overflow:hidden;">'
        '<div style="position:absolute;left:0;right:0;top:50%;'
        'border-top:1px solid #777;"></div>'
        f"{bar}</div>"
        f'<span style="font-size:0.58em;color:#aaa;">{dim}</span>'
        "</div>"
      )
    bar_chart = (
      '<div style="display:flex;gap:2px;align-items:flex-end;">'
      + "".join(bars)
      + "</div>"
    )

    norm = float(np.linalg.norm(vector))
    safe_label = html.escape(self._label)
    header = (
      '<div style="font-size:0.82em;color:#ddd;margin-bottom:6px;">'
      f"<strong>{safe_label}</strong> · env #{env_idx} · "
      f"min {vector.min():+.2f} · max {vector.max():+.2f} · "
      f"norm {norm:.2f}</div>"
    )
    self._html.content = (
      '<div style="padding:0.35em;font-family:monospace;">'
      f"{header}{heatmap}{bar_chart}</div>"
    )

  def _color(self, value: float) -> str:
    """Map negative values to blue and positive values to red."""
    negative = abs(min(self._low, 0.0))
    positive = max(self._high, 0.0)
    if value >= 0.0:
      strength = min(value / max(positive, 1e-12), 1.0)
      color = (80.0 + 150.0 * strength, 80.0 - 30.0 * strength, 80.0)
    else:
      strength = min(abs(value) / max(negative, 1e-12), 1.0)
      color = (80.0, 80.0 - 20.0 * strength, 80.0 + 170.0 * strength)
    return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"
