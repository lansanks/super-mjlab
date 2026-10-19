"""Compare HiMLoco stress results with the archived DreamWaQ evaluation."""

from __future__ import annotations

import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
HIM_RESULTS = HERE / "results.json"
DREAMWAQ_RESULTS = (
  HERE.parents[4] / "dreamwaq/doc/实验/实验数据/stress_test/results.json"
)
OUTPUT_PNG = HERE / "comparison_with_dreamwaq.png"
OUTPUT_CSV = HERE / "comparison_with_dreamwaq.csv"
OUTPUT_SUMMARY = HERE / "comparison_with_dreamwaq.json"

CONDITIONS = [
  "clean",
  "push1x",
  "push2x",
  "push3x",
  "push4x",
  "noise2x",
  "noise2x_bias3x",
  "push2x_noise2x",
  "noise4x",
  "noise4x_bias8x",
  "push2x_noise4x",
]
POLICIES = {
  "him_tau01": "him_tau01_3000",
  "him_tau3": "him_tau3_3000",
  "dreamwaq_b004": "D_det_b004",
  "velocity_baseline": "vel_baseline",
}
LABELS = {
  "him_tau01": "HIM tau=0.1 (new)",
  "him_tau3": "HIM tau=3.0 (prev)",
  "dreamwaq_b004": "DreamWaQ beta=0.04",
  "velocity_baseline": "Velocity baseline",
}
COLORS = {
  "him_tau01": "#d62728",
  "him_tau3": "#ff7f0e",
  "dreamwaq_b004": "#1f77b4",
  "velocity_baseline": "#555555",
}


def load_rows(path, level: int | None, policy_names: dict[str, str]):
  rows = json.loads(path.read_text(encoding="utf-8"))
  out = {}
  for key, policy_name in policy_names.items():
    selected = [
      row
      for row in rows
      if row["policy"] == policy_name and (level is None or row.get("level") == level)
    ]
    out[key] = {row["condition"]: row for row in selected}
  return out


def main() -> None:
  him = load_rows(
    HIM_RESULTS,
    None,
    {
      "him_tau01": POLICIES["him_tau01"],
      "him_tau3": POLICIES["him_tau3"],
    },
  )
  dream = load_rows(
    DREAMWAQ_RESULTS,
    3,
    {
      "dreamwaq_b004": POLICIES["dreamwaq_b004"],
      "velocity_baseline": POLICIES["velocity_baseline"],
    },
  )
  all_rows = {**him, **dream}

  table = []
  for condition in CONDITIONS:
    row = {"condition": condition}
    for key in ("him_tau01", "him_tau3", "dreamwaq_b004", "velocity_baseline"):
      row[key] = int(all_rows[key][condition]["falls"])
      row[f"{key}_reward"] = float(all_rows[key][condition]["reward_per_step"])
    table.append(row)

  with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)

  totals = {
    key: sum(int(row[key]) for row in table)
    for key in ("him_tau01", "him_tau3", "dreamwaq_b004", "velocity_baseline")
  }
  total_reduction = {
    key: 1.0 - totals["him_tau01"] / totals[key]
    for key in ("him_tau3", "dreamwaq_b004", "velocity_baseline")
  }
  push4_falls = {key: int(all_rows[key]["push4x"]["falls"]) for key in all_rows}
  push4_reduction = {
    key: 1.0 - push4_falls["him_tau01"] / push4_falls[key]
    for key in ("him_tau3", "dreamwaq_b004", "velocity_baseline")
  }
  summary = {
    "protocol": {
      "terrain_level": 3,
      "num_envs": 256,
      "duration_s": 20,
      "seed": 20261012,
      "steps": 1000,
      "push_period_steps": 50,
    },
    "total_falls_all_conditions": totals,
    "new_him_total_reduction_vs": total_reduction,
    "push4_falls": push4_falls,
    "new_him_push4_reduction_vs": push4_reduction,
  }
  OUTPUT_SUMMARY.write_text(
    json.dumps(summary, indent=1),
    encoding="utf-8",
  )

  plt.rcParams.update({"font.size": 10})
  fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
  push_conditions = ["clean", "push1x", "push2x", "push3x", "push4x"]
  x = [0, 1, 2, 3, 4]
  for key in ("him_tau01", "him_tau3", "dreamwaq_b004", "velocity_baseline"):
    values = [all_rows[key][condition]["falls"] for condition in push_conditions]
    axes[0].plot(
      x,
      values,
      marker="o",
      linewidth=2,
      markersize=6,
      color=COLORS[key],
      label=LABELS[key],
    )
    for xv, value in zip(x, values, strict=True):
      axes[0].annotate(
        str(value),
        (xv, value),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
        fontsize=7,
      )
  axes[0].set_xticks(x, ["clean", "1x", "2x", "3x", "4x"])
  axes[0].set_xlabel("Push multiplier")
  axes[0].set_ylabel("Falls / 256 envs / 20 s")
  axes[0].set_title("Push dose response (terrain level 3)")
  axes[0].grid(alpha=0.25)
  axes[0].legend(fontsize=8)

  noise_conditions = [
    "noise2x",
    "noise2x_bias3x",
    "push2x_noise2x",
    "noise4x",
    "noise4x_bias8x",
    "push2x_noise4x",
  ]
  short_labels = ["n2", "n2+b3", "p2+n2", "n4", "n4+b8", "p2+n4"]
  width = 0.2
  positions = range(len(noise_conditions))
  for index, key in enumerate(
    ("him_tau01", "him_tau3", "dreamwaq_b004", "velocity_baseline")
  ):
    values = [all_rows[key][condition]["falls"] for condition in noise_conditions]
    offsets = [p + (index - 1.5) * width for p in positions]
    axes[1].bar(
      offsets,
      values,
      width=width,
      color=COLORS[key],
      label=LABELS[key],
    )
  axes[1].set_xticks(list(positions), short_labels)
  axes[1].set_ylabel("Falls / 256 envs / 20 s")
  axes[1].set_title("Observation noise and combined disturbances")
  axes[1].grid(axis="y", alpha=0.25)
  axes[1].legend(fontsize=8)

  fig.tight_layout()
  fig.savefig(OUTPUT_PNG, bbox_inches="tight")
  plt.close(fig)
  print(json.dumps(summary, indent=1))
  print(f"[INFO] saved {OUTPUT_PNG}")
  print(f"[INFO] saved {OUTPUT_CSV}")
  print(f"[INFO] saved {OUTPUT_SUMMARY}")


if __name__ == "__main__":
  main()
