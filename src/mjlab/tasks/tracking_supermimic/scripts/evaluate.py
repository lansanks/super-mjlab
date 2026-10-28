"""Evaluate a SuperMimic G1 checkpoint once on every motion clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking_supermimic.mdp import MAX_STEP_REWARD
from mjlab.tasks.tracking_supermimic.runner import (
  TrackingSuperMimicRunner,
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, default=Path("eval_supermimic.json"))
  args = parser.parse_args()

  cfg = load_env_cfg("Mjlab-Tracking-SuperMimic-Unitree-G1")
  motion_cfg = cfg.commands["motion"]
  motion_cfg.motion_file = str(args.motion_file)
  motion_cfg.pose_range = {}
  motion_cfg.velocity_range = {}
  motion_cfg.joint_position_range = (0.0, 0.0)
  motion_cfg.sampling_mode = "all"
  cfg.auto_reset = False
  # Run each clip to its end: the metric divides by the full clip length, so a
  # 10 s episode cap would stop the numerator at 500 steps and deflate the score.
  cfg.episode_length_s = int(1e9)

  # The command loader needs the number of clips before the environment can
  # assign one clip per environment.
  from mjlab.tasks.tracking_supermimic.motion import (
    LafanMotionLibrary,
  )

  motion = LafanMotionLibrary(args.motion_file, "cpu")
  cfg.scene.num_envs = motion.num_clips
  env = ManagerBasedRlEnv(cfg, device=args.device)
  checkpoint = torch.load(
    args.checkpoint,
    map_location=args.device,
    weights_only=False,
  )
  runner_cfg = dict(checkpoint.get("config", {}))
  runner_cfg.update(
    {
      "obs_history": runner_cfg.get("obs_history", 10),
      "goal_frames": runner_cfg.get("goal_frames", 10),
      "horizon": runner_cfg.get("horizon", 32),
      "logger": "none",
    }
  )
  runner = TrackingSuperMimicRunner(
    env,
    runner_cfg,
    device=args.device,
  )
  runner.load(str(args.checkpoint), map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  returns = torch.zeros(env.num_envs, device=env.device)
  failures = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  previous_action = torch.zeros(env.num_envs, 29, device=env.device)
  max_steps = int(motion.clip_len.max().item()) + 2
  for _ in range(max_steps):
    action = policy(None)
    _, _, terminated, timed_out, _ = env.step(action)
    next_state = runner.state()
    reward = runner.mdp.reward(
      next_state,
      runner.command.time_steps,
      action,
      previous_action,
    )
    active = ~completed
    returns += torch.where(active, reward, torch.zeros_like(reward))
    failures |= active & terminated
    completed |= active & (terminated | timed_out)
    previous_action = action.detach()

    reset_ids = (terminated | timed_out).nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
      env.reset(env_ids=reset_ids)
      previous_action[reset_ids] = 0.0
    if completed.all():
      break

  denominator = (runner.motion.clip_len - 1).float() * MAX_STEP_REWARD
  normalized = returns / denominator
  result = {
    "normalized_return": float(normalized.mean()),
    "failed_count": int(failures.sum()),
    "completed_count": int(completed.sum()),
    "per_clip": {
      name: {
        "return": float(normalized[index]),
        "failed": bool(failures[index]),
        "completed": bool(completed[index]),
      }
      for index, name in enumerate(runner.motion.names)
    },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(
    f"normalized return {result['normalized_return']:.4f}; "
    f"failed {result['failed_count']}/{env.num_envs}"
  )
  env.close()


if __name__ == "__main__":
  main()
