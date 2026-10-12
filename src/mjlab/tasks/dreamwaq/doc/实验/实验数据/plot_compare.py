"""Plot cross-run comparisons of the DreamWaQ experiments.

Usage (from the repository root)::

    uv run python src/mjlab/tasks/dreamwaq/doc/实验数据/plot_compare.py

Writes PNGs into ``figures/`` next to this script:

    kl_recon_curves.png      Loss/cenet_kl and Loss/cenet_recon vs iteration
    recon_terms_window.png   per-term reconstruction error (mean_pct) in a
                             matched iteration window
"""

from __future__ import annotations

import csv
import pathlib
import statistics

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

DATA = pathlib.Path(__file__).resolve().parent
FIGURES = DATA / "figures"

# (run directory, label, colour, bar-chart window)
RUNS = [
  ("2026-10-12_10-10-21", "beta=0", "#7f8c8d", (111, 186)),
  ("2026-10-12_11-19-48", "beta=0.05", "#c0392b", (400, 790)),
  ("2026-10-12_10-29-54", "beta=0.1", "#2980b9", (400, 790)),
]
TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
)


def load(run: str) -> list[dict[str, str]]:
  with (DATA / run / "metrics.csv").open() as f:
    return list(csv.DictReader(f))


def series(rows, tag: str) -> tuple[list[int], list[float]]:
  xs = [int(r["step"]) for r in rows if r.get(tag)]
  ys = [float(r[tag]) for r in rows if r.get(tag)]
  return xs, ys


def mean_in_window(rows, tag: str, window: tuple[int, int]) -> float:
  values = [
    float(r[tag])
    for r in rows
    if r.get(tag) and window[0] <= int(r["step"]) <= window[1]
  ]
  return statistics.fmean(values) if values else float("nan")


def plot_curves() -> None:
  fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
  for run, label, color, _ in RUNS:
    rows = load(run)
    for ax, tag, title in zip(
      axes,
      ("Loss/cenet_kl", "Loss/cenet_recon"),
      ("CENet KL loss", "CENet reconstruction loss"),
      strict=True,
    ):
      xs, ys = series(rows, tag)
      ax.plot(xs, ys, label=label, color=color, lw=1.3)
      ax.set_xlabel("iteration")
      ax.set_title(title)
      ax.set_yscale("log")
  axes[1].annotate(
    "beta=0.05 blow-up\n(iteration 806)",
    xy=(806, 300),
    xytext=(400, 1200),
    arrowprops={"arrowstyle": "->", "color": "#c0392b"},
    color="#c0392b",
    fontsize=8,
  )
  axes[0].legend(fontsize=8)
  fig.tight_layout()
  FIGURES.mkdir(exist_ok=True)
  fig.savefig(FIGURES / "kl_recon_curves.png", dpi=150)
  plt.close(fig)


def plot_term_bars() -> None:
  width = 0.8 / len(RUNS)
  fig, ax = plt.subplots(figsize=(9, 3.6))
  for index, (run, label, color, window) in enumerate(RUNS):
    rows = load(run)
    values = [
      mean_in_window(rows, f"Metrics/recon_{term}_mean_pct", window) for term in TERMS
    ]
    offsets = [i + index * width for i in range(len(TERMS))]
    ax.bar(
      offsets,
      values,
      width=width,
      label=f"{label} (iters {window[0]}-{window[1]})",
      color=color,
    )
  ax.set_xticks([i + 0.4 - width / 2 for i in range(len(TERMS))])
  ax.set_xticklabels(TERMS, fontsize=8)
  ax.set_ylabel("recon error [% of signal std]")
  ax.legend(fontsize=8)
  fig.tight_layout()
  FIGURES.mkdir(exist_ok=True)
  fig.savefig(FIGURES / "recon_terms_window.png", dpi=150)
  plt.close(fig)


if __name__ == "__main__":
  plot_curves()
  plot_term_bars()
  print(f"figures written to {FIGURES}")
