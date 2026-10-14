"""Plot the CENet gradient-routing A/B pair (beta=0, 600 iterations).

Usage (from the repository root)::

    uv run python src/mjlab/tasks/dreamwaq/doc/实验/实验数据/plot_grad_routing.py

Writes ``figures/grad_routing.png`` next to this script. Raw curves are drawn
faintly and the solid line is a trailing 20-iteration mean.
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

# (run directory, label, colour)
RUNS = [
  ("2026-10-12_13-02-09", "CENet takes PPO gradient", "#7f8c8d"),
  ("2026-10-12_14-17-48", "CENet detached (aux loss only)", "#c0392b"),
]
PANELS = (
  ("Train/mean_reward", "Mean reward"),
  ("Train/mean_episode_length", "Mean episode length [steps]"),
  ("Episode_Termination/illegal_contact", "Illegal-contact terminations"),
  ("Curriculum/terrain_levels/mean", "Mean terrain level"),
  ("Metrics/velocity_est_error", "Velocity estimate error [m/s]"),
  ("Loss/cenet_recon", "CENet reconstruction loss"),
)
SMOOTH = 20


def load(run: str) -> list[dict[str, str]]:
  with (DATA / run / "metrics.csv").open() as f:
    return list(csv.DictReader(f))


def series(rows: list[dict[str, str]], tag: str) -> tuple[list[int], list[float]]:
  xs = [int(r["step"]) for r in rows if r.get(tag)]
  ys = [float(r[tag]) for r in rows if r.get(tag)]
  return xs, ys


def trailing_mean(ys: list[float], window: int = SMOOTH) -> list[float]:
  return [statistics.fmean(ys[max(0, i - window + 1) : i + 1]) for i in range(len(ys))]


def main() -> None:
  fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
  for run, label, color in RUNS:
    rows = load(run)
    for ax, (tag, title) in zip(axes.flat, PANELS, strict=True):
      xs, ys = series(rows, tag)
      ax.plot(xs, ys, color=color, alpha=0.18, lw=0.7)
      ax.plot(xs, trailing_mean(ys), color=color, lw=1.6, label=label)
      ax.set_title(title, fontsize=10)
      ax.set_xlabel("iteration", fontsize=8)
      ax.grid(alpha=0.25, lw=0.5)
  axes.flat[0].legend(fontsize=8)
  fig.suptitle("CENet gradient routing A/B (beta=0, seed 42, 4096 envs)", fontsize=11)
  fig.tight_layout()
  FIGURES.mkdir(exist_ok=True)
  out = FIGURES / "grad_routing.png"
  fig.savefig(out, dpi=150)
  plt.close(fig)
  print(f"figure written to {out}")


if __name__ == "__main__":
  main()
