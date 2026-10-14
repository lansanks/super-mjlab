"""Export DreamWaQ run scalars (TensorBoard event files) into this folder.

Usage (from the repository root)::

    uv run python src/mjlab/tasks/dreamwaq/doc/实验/实验数据/export_scalars.py \
      [RUN ...]

Without arguments every run under ``logs/rsl_rl/go1_dreamwaq/`` is exported to
``<run>/`` next to this script:

    metrics.csv        one row per logged iteration, one column per scalar tag
    summary.csv        per tag: n, first/last, min/max with their steps
    source.txt         provenance (source directory, event file, export time)
    config_agent.yaml  training configuration of the run
    config_env.yaml    environment configuration of the run
    code_state.diff    ``git diff`` the runner stored when the run started

The CSVs are snapshots: re-run this script to refresh a run that is still
training. Checkpoints (``model_*.pt``) and the exported ONNX policy stay in the
run directory under ``logs/rsl_rl/go1_dreamwaq/<run>/``.
"""

from __future__ import annotations

import csv
import pathlib
import shutil
import sys
from datetime import datetime
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = pathlib.Path.cwd()
SOURCE_ROOT = REPO / "logs/rsl_rl/go1_dreamwaq"
DEST_ROOT = pathlib.Path(__file__).resolve().parent


def export_run(run: str) -> None:
  """Write the CSVs and configuration copies of a single run."""
  run_dir = SOURCE_ROOT / run
  event_files = sorted(run_dir.glob("events.out.tfevents.*"))
  if not event_files:
    print(f"skip {run}: no event file")
    return
  target: Any = str(event_files[0]) if len(event_files) == 1 else str(run_dir)
  accumulator = EventAccumulator(target)
  accumulator.Reload()
  tags = sorted(accumulator.Tags()["scalars"])
  series = {tag: accumulator.Scalars(tag) for tag in tags}
  by_step = {
    tag: {event.step: event.value for event in events} for tag, events in series.items()
  }
  steps = sorted({step for values in by_step.values() for step in values})

  out_dir = DEST_ROOT / run
  out_dir.mkdir(parents=True, exist_ok=True)

  with (out_dir / "metrics.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", *tags])
    for step in steps:
      row = [
        step,
        *[
          f"{v:.6g}" if (v := by_step[tag].get(step)) is not None else ""
          for tag in tags
        ],
      ]
      writer.writerow(row)

  with (out_dir / "summary.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
      [
        "tag",
        "n",
        "first_step",
        "last_step",
        "first",
        "last",
        "min",
        "min_step",
        "max",
        "max_step",
      ]
    )
    for tag in tags:
      events = series[tag]
      values = [event.value for event in events]
      min_index = min(range(len(values)), key=lambda i: values[i])
      max_index = max(range(len(values)), key=lambda i: values[i])
      writer.writerow(
        [
          tag,
          len(events),
          events[0].step,
          events[-1].step,
          f"{values[0]:.6g}",
          f"{values[-1]:.6g}",
          f"{values[min_index]:.6g}",
          events[min_index].step,
          f"{values[max_index]:.6g}",
          events[max_index].step,
        ]
      )

  (out_dir / "source.txt").write_text(
    f"run dir : {run_dir}\n"
    f"events  : {', '.join(p.name for p in event_files)}\n"
    f"exported: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
    f"tags    : {len(tags)}\n"
    f"steps   : {steps[0]}..{steps[-1]} ({len(steps)} logged iterations)\n"
    "note    : metrics.csv/summary.csv 由 TensorBoard event 文件导出；"
    "run 若仍在训练，这里是快照。\n"
  )
  for name, target_name in (
    ("agent.yaml", "config_agent.yaml"),
    ("env.yaml", "config_env.yaml"),
  ):
    source = run_dir / "params" / name
    if source.exists():
      shutil.copy2(source, out_dir / target_name)
  diffs = sorted((run_dir / "git").glob("*.diff"))
  if diffs:
    shutil.copy2(diffs[0], out_dir / "code_state.diff")
  print(f"{run}: {len(tags)} tags, {len(steps)} iterations -> {out_dir.name}/")


def main() -> None:
  if not SOURCE_ROOT.is_dir():
    raise SystemExit(
      f"{SOURCE_ROOT} not found; run this script from the repository root."
    )
  runs = sys.argv[1:] or sorted(p.name for p in SOURCE_ROOT.iterdir() if p.is_dir())
  for run in runs:
    export_run(run)


if __name__ == "__main__":
  main()
