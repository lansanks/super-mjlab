"""Closed-loop SRMA teacher evaluation with latent interventions."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import cast

import torch
from tsne_latent import build_env, load_policy  # noqa: E402

MODES = ("normal", "zero_latent", "different_terrain_latent")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent


def evaluate(
  checkpoint: str,
  level: int,
  num_envs: int,
  steps: int,
  seed: int,
  command_x: float,
) -> list[dict[str, float | str | int]]:
  env, num_types, _, _ = build_env("srma", num_envs, level, seed)
  try:
    device = env.unwrapped.device
    policy = load_policy(env, "srma", device, checkpoint)
    command_term = env.unwrapped.command_manager.get_term("twist")
    assert command_term is not None
    command = cast(torch.Tensor, command_term.command)
    ids = torch.arange(env.num_envs, device=device)
    different_ids = (ids + 1) % env.num_envs
    results = []

    for mode in MODES:
      torch.manual_seed(seed)
      obs, _ = env.reset()
      total_reward = 0.0
      falls = 0
      tracking_errors: list[float] = []
      peak_joint_vels: list[float] = []

      for _ in range(steps):
        command[:] = torch.tensor([command_x, 0.0, 0.0], device=device)
        with torch.no_grad():
          if mode == "normal":
            actions = policy(obs)
          elif mode == "zero_latent":
            latent_dim = policy.external_encoder.latent_dim
            actions = policy.action_mean_with_latent(
              obs,
              torch.zeros(env.num_envs, latent_dim, device=device),
            )
          else:
            policy(obs)
            latent = policy.last_external_output.latent
            actions = policy.action_mean_with_latent(
              obs,
              latent[different_ids],
            )

        obs, rewards, dones, extras = env.step(actions)
        total_reward += float(rewards.sum())
        falls += int((dones > 0).sum())
        log = extras.get("log", {})
        if "Metrics/twist/error_vel_xy" in log:
          tracking_errors.append(float(log["Metrics/twist/error_vel_xy"]))
        if "Episode_Metrics/peak_joint_vel" in log:
          peak_joint_vels.append(float(log["Episode_Metrics/peak_joint_vel"]))

      results.append(
        {
          "level": level,
          "mode": mode,
          "reward_per_step": total_reward / (env.num_envs * steps),
          "falls_per_env_20s": falls / env.num_envs * (1000.0 / steps),
          "error_vel_xy": sum(tracking_errors) / max(len(tracking_errors), 1),
          "peak_joint_vel": max(peak_joint_vels, default=float("nan")),
        }
      )
    return results
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--levels", default="3,5,7")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--command-x", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  levels = [int(value) for value in args.levels.split(",")]
  checkpoint = str(pathlib.Path(args.checkpoint).resolve())
  rows: list[dict[str, float | str | int]] = []
  for level in levels:
    rows.extend(
      evaluate(
        checkpoint,
        level,
        args.num_envs,
        args.steps,
        args.seed,
        args.command_x,
      )
    )

  out = args.out or OUTPUT_DIR / "closed_loop_eval.json"
  out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
  for row in rows:
    print(row, flush=True)
  print(f"[INFO] saved {out}", flush=True)


if __name__ == "__main__":
  main()
