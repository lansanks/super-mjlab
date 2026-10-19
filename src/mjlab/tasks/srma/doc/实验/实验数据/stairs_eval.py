"""Evaluate the teacher on ascending and inverted stair terrains by level."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import cast

import torch
from tsne_latent import build_env, load_policy

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent


def evaluate_type(
  env,
  policy,
  terrain_type_id: int,
  level: int,
  steps: int,
  warmup: int,
  command_x: float,
  step_height: float,
) -> dict[str, float | int | str]:
  terrain = env.unwrapped.scene.terrain
  assert terrain is not None
  terrain.terrain_levels[:] = level
  terrain.terrain_types[:] = terrain_type_id
  terrain.env_origins[:] = terrain.terrain_origins[level, terrain_type_id]

  obs, _ = env.reset()
  device = env.unwrapped.device
  command_term = env.unwrapped.command_manager.get_term("twist")
  assert command_term is not None
  command = cast(torch.Tensor, command_term.command)
  command[:] = torch.tensor([command_x, 0.0, 0.0], device=device)
  alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
  survival_steps = torch.zeros(env.num_envs, device=device)
  forward_distance = torch.zeros(env.num_envs, device=device)
  velocity_sum = torch.zeros(env.num_envs, device=device)
  stopped_steps = torch.zeros(env.num_envs, device=device)
  falls = 0
  tracking_errors: list[float] = []
  robot = env.unwrapped.scene["robot"]

  for step in range(steps):
    command[:] = torch.tensor([command_x, 0.0, 0.0], device=device)
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, extras = env.step(actions)
    forward_velocity = robot.data.root_link_lin_vel_b[:, 0]
    alive_before = alive
    survival_steps += alive_before.float()
    forward_distance += forward_velocity * env.unwrapped.step_dt * alive_before.float()
    velocity_sum += forward_velocity * alive_before.float()
    if step >= warmup:
      stopped_steps += ((forward_velocity < 0.1) & alive_before).float()
    log = extras.get("log", {})
    if "Metrics/twist/error_vel_xy" in log:
      tracking_errors.append(float(log["Metrics/twist/error_vel_xy"]))
    falls += int(((dones > 0) & alive_before).sum())
    alive &= dones <= 0

  measured_steps = survival_steps.clamp_min(1.0)
  post_warmup_steps = (survival_steps - warmup).clamp_min(1.0)
  terrain_names = list(env.unwrapped.scene.terrain.cfg.terrain_generator.sub_terrains)
  return {
    "type": terrain_names[terrain_type_id],
    "level": level,
    "step_height_m": step_height,
    "falls_per_env_20s": falls / env.num_envs * (1000.0 / steps),
    "mean_survival_steps": float(survival_steps.mean()),
    "mean_forward_velocity": float((velocity_sum / measured_steps).mean()),
    "mean_forward_distance_m": float(forward_distance.mean()),
    "stop_fraction": float((stopped_steps / post_warmup_steps).mean()),
    "error_vel_xy": float(sum(tracking_errors) / max(len(tracking_errors), 1)),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--levels", default="3,5,7,9")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--warmup", type=int, default=40)
  parser.add_argument("--command-x", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  levels = [int(value) for value in args.levels.split(",")]
  rows = []
  for level in levels:
    env, num_types, _, type_names = build_env(
      "srma",
      args.num_envs,
      level,
      args.seed,
    )
    try:
      if env.num_envs % num_types != 0:
        raise ValueError("num_envs must be divisible by terrain type count.")
      policy = load_policy(
        env,
        "srma",
        env.unwrapped.device,
        str(pathlib.Path(args.checkpoint).resolve()),
      )
      for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
        type_id = type_names.index(terrain_name)
        terrain = env.unwrapped.scene.terrain
        assert terrain is not None
        terrain_origins = terrain.terrain_origins
        assert terrain_origins is not None
        level_count = terrain_origins.shape[0]
        step_height = 0.25 * level / max(level_count - 1, 1)
        rows.append(
          evaluate_type(
            env,
            policy,
            type_id,
            level,
            args.steps,
            args.warmup,
            args.command_x,
            step_height,
          )
        )
    finally:
      env.close()

  out = args.out or OUTPUT_DIR / "stairs_eval.json"
  out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
  for row in rows:
    print(row, flush=True)
  print(f"[INFO] saved {out}", flush=True)


if __name__ == "__main__":
  main()
