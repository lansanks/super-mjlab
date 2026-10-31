"""Export a saved SuperMimic checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--task",
    default="Mjlab-Tracking-SuperMimic-Unitree-G1",
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  checkpoint = torch.load(
    args.checkpoint,
    map_location=args.device,
    weights_only=False,
  )
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  motion_cfg = cfg.commands["motion"]
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise TypeError(f"Task {args.task} does not use a motion command.")
  motion_cfg.motion_file = str(args.motion_file)

  runner_cfg = dict(checkpoint.get("config", {}))
  runner_cfg.update({"logger": "none", "compile": False})
  env = ManagerBasedRlEnv(cfg, device=args.device)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError(f"Task {args.task} does not define a custom runner.")
  runner = runner_cls(
    env,
    runner_cfg,
    device=args.device,
  )
  runner.load(str(args.checkpoint), map_location=args.device)

  output = args.output
  if output is None:
    output = args.checkpoint.parent / f"{args.checkpoint.parent.name}.onnx"
  runner.export_policy_to_onnx(
    str(output.parent),
    output.name,
    verbose=args.verbose,
  )
  env.close()


if __name__ == "__main__":
  main()
