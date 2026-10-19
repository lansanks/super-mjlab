"""Measure SRMA teacher action sensitivity to external latent interventions."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import cast

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tsne_latent import build_env, load_policy  # noqa: E402

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--level", type=int, default=5)
  parser.add_argument("--num-envs", type=int, default=700)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--warmup", type=int, default=60)
  parser.add_argument("--sample-every", type=int, default=4)
  parser.add_argument("--command-x", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  env, num_types, level, type_names = build_env(
    "srma",
    args.num_envs,
    args.level,
    args.seed,
  )
  try:
    device = env.unwrapped.device
    policy = load_policy(
      env,
      "srma",
      device,
      str(pathlib.Path(args.checkpoint).resolve()),
    )
    command_term = env.unwrapped.command_manager.get_term("twist")
    assert command_term is not None
    command = cast(torch.Tensor, command_term.command)
    command[:] = torch.tensor(
      [args.command_x, 0.0, 0.0],
      device=device,
    )

    obs, _ = env.reset()
    terrain = env.unwrapped.scene.terrain
    assert terrain is not None
    terrain_types = terrain.terrain_types.clone()
    env_ids = torch.arange(env.num_envs, device=device)
    if env.num_envs % num_types != 0:
      raise ValueError(
        f"num_envs={env.num_envs} must be divisible by {num_types} terrain types."
      )
    # With contiguous terrain-type blocks and env_idx % num_types assignment,
    # +num_types stays on the same terrain and +1 changes terrain type.
    same_ids = (env_ids + num_types) % env.num_envs
    different_ids = (env_ids + 1) % env.num_envs
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)

    records: dict[str, list[float]] = {
      "zero": [],
      "same_terrain": [],
      "different_terrain": [],
    }
    by_terrain = {
      name: {"zero": [], "same_terrain": [], "different_terrain": []}
      for name in type_names
    }
    torch.manual_seed(args.seed)
    for step in range(args.warmup + args.steps):
      command[:] = torch.tensor(
        [args.command_x, 0.0, 0.0],
        device=device,
      )
      with torch.no_grad():
        baseline_actions = policy(obs)
        latent = policy.last_external_output.latent
        zero_actions = policy.action_mean_with_latent(
          obs,
          torch.zeros_like(latent),
        )
        same_actions = policy.action_mean_with_latent(
          obs,
          latent[same_ids],
        )
        different_actions = policy.action_mean_with_latent(
          obs,
          latent[different_ids],
        )

      deltas = {
        "zero": (baseline_actions - zero_actions).norm(dim=-1),
        "same_terrain": (baseline_actions - same_actions).norm(dim=-1),
        "different_terrain": (baseline_actions - different_actions).norm(dim=-1),
      }
      if step >= args.warmup and (step - args.warmup) % args.sample_every == 0:
        for key, values in deltas.items():
          records[key].append(float(values[alive].mean()))
          for terrain_id, terrain_name in enumerate(type_names):
            mask = alive & (terrain_types == terrain_id)
            if bool(mask.any()):
              by_terrain[terrain_name][key].append(float(values[mask].mean()))

      obs, _, dones, _ = env.step(baseline_actions)
      alive[dones > 0] = False

    summary = {key: float(np.mean(values)) for key, values in records.items()}
    summary["different_minus_same"] = (
      summary["different_terrain"] - summary["same_terrain"]
    )
    per_terrain = {
      name: {key: float(np.mean(values)) for key, values in terrain_values.items()}
      for name, terrain_values in by_terrain.items()
    }
    payload = {
      "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
      "level": level,
      "num_envs": args.num_envs,
      "command_x": args.command_x,
      "summary": summary,
      "per_terrain": per_terrain,
      "raw": records,
    }
    out = args.out or OUTPUT_DIR / f"latent_intervention_L{level}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    labels = ["zero", "same terrain", "different terrain"]
    values = [
      summary["zero"],
      summary["same_terrain"],
      summary["different_terrain"],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=180)
    bars = ax.bar(labels, values, color=["#777777", "#2a9d8f", "#e76f51"])
    for bar, value in zip(bars, values, strict=True):
      ax.annotate(
        f"{value:.3f}",
        (bar.get_x() + bar.get_width() / 2, value),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=9,
      )
    ax.set_ylabel("Mean action L2 delta")
    ax.set_title(f"SRMA latent intervention, terrain level {level}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    print("[INFO] intervention summary:", summary, flush=True)
    print("[INFO] per terrain:", per_terrain, flush=True)
    print(f"[INFO] saved {out}", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
