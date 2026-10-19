"""Robustness stress test for the two HiMLoco 3000-iteration checkpoints.

The evaluation protocol matches the archived DreamWaQ stress test:
level-3 terrain, 256 environments, 1000 control steps, paired seeds and pushes
every 1 s. Each checkpoint is evaluated with the observation scales used during
its training. Observation-noise multipliers are therefore converted into the
scaled observation units for the latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-HIM-Rough-Unitree-Go1"
NUM_ENVS = 256
LEVEL = 3
SEED = 20261012
PUSH_PERIOD = 50
STEPS = 1000

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT = OUTPUT_DIR / "results.json"

POLICIES = (
  (
    "him_tau01_3000",
    "logs/rsl_rl/go1_himloco/2026-10-17_18-49-36/model_2999.pt",
    "scaled",
  ),
  (
    "him_tau3_3000",
    "logs/rsl_rl/go1_himloco/2026-10-16_22-19-10/model_2999.pt",
    "legacy",
  ),
)

CONDITIONS = (
  ("clean", 0.0, 0.0, 0.0),
  ("push1x", 1.0, 0.0, 0.0),
  ("push2x", 2.0, 0.0, 0.0),
  ("push3x", 3.0, 0.0, 0.0),
  ("push4x", 4.0, 0.0, 0.0),
  ("noise2x", 0.0, 2.0, 0.0),
  ("noise2x_bias3x", 0.0, 2.0, 3.0),
  ("push2x_noise2x", 2.0, 2.0, 0.0),
  ("noise4x", 0.0, 4.0, 0.0),
  ("noise4x_bias8x", 0.0, 4.0, 8.0),
  ("push2x_noise4x", 2.0, 4.0, 0.0),
)

NOISE_WIDTHS = {
  "legacy": (0.2, 0.05, 0.01, 1.5),
  "scaled": (0.05, 0.05, 0.01, 0.075),
}
NOISE_SLICES = ((0, 3), (3, 6), (6, 18), (18, 30))
PUSH_WIDTHS = (0.5, 0.5, 0.4, 0.52, 0.52, 0.78)
JOINT_POS_SLICE = (6, 18)
TRAIN_BIAS = 0.015


def _restore_legacy_obs_scales(env_cfg) -> None:
  """Undo the official observation scaling used by the latest checkpoint."""
  for group in ("actor", "critic", "him_input", "him_state"):
    terms = env_cfg.observations[group].terms
    terms["base_ang_vel"].scale = None
    terms["joint_vel"].scale = None
    if "command" in terms:
      terms["command"].scale = None
  env_cfg.observations["critic"].terms["base_lin_vel"].scale = None


def build_env(observation_mode: str):
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.episode_length_s = 1e9
  for group in ("actor", "him_input", "him_state"):
    env_cfg.observations[group].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  env_cfg.curriculum = {}
  if observation_mode == "legacy":
    _restore_legacy_obs_scales(env_cfg)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  n_rows, n_cols = terrain.terrain_origins.shape[:2]
  terrain.terrain_levels[:] = LEVEL
  terrain.terrain_types[:] = torch.arange(NUM_ENVS, device=device) % n_cols
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]
  wrapper = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
  return wrapper, device, n_rows, n_cols


def make_noise(tensor: torch.Tensor, gen: torch.Generator, scale, mode: str):
  noise = torch.zeros_like(tensor)
  widths = NOISE_WIDTHS[mode]
  for (lo, hi), width in zip(NOISE_SLICES, widths, strict=True):
    noise[..., lo:hi] = (
      torch.rand(tensor.shape[:-1] + (hi - lo,), generator=gen, device="cpu").to(
        tensor.device
      )
      * 2.0
      - 1.0
    ) * (width * scale)
  return noise


def corrupt(
  obs,
  gen: torch.Generator,
  noise_scale: float,
  bias: torch.Tensor,
  bias_scale: float,
  mode: str,
):
  out = obs.clone()
  for key in ("actor", "him_input"):
    tensor = out[key]
    shape = tensor.shape
    flat = tensor.reshape(shape[0], -1, 45)
    if noise_scale > 0:
      flat = flat + make_noise(flat, gen, noise_scale, mode)
    if bias_scale > 0:
      lo, hi = JOINT_POS_SLICE
      flat[..., lo:hi] += bias.unsqueeze(1)
    out[key] = flat.reshape(shape)
  return out


def apply_push(env: RslRlVecEnvWrapper, gen: torch.Generator, scale: float) -> None:
  robot = env.unwrapped.scene["robot"]
  n = env.num_envs
  delta = torch.zeros(n, 6, device=env.unwrapped.device)
  for i, width in enumerate(PUSH_WIDTHS):
    delta[:, i] = (
      torch.rand(n, generator=gen, device="cpu").to(delta.device) * 2.0 - 1.0
    ) * (width * scale)
  ids = torch.arange(n, device=env.unwrapped.device)
  robot.write_root_link_velocity_to_sim(
    robot.data.root_link_vel_w[ids] + delta, env_ids=ids
  )


def run_cell(
  env: RslRlVecEnvWrapper,
  runner,
  device: str,
  checkpoint: str,
  mode: str,
  condition,
  steps: int,
):
  name, push_scale, noise_scale, bias_scale = condition
  runner.load(
    checkpoint,
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  torch.manual_seed(SEED)
  obs, _ = env.reset()
  gen = torch.Generator(device="cpu").manual_seed(SEED)
  bias = (
    (torch.rand(NUM_ENVS, 12, generator=gen) * 2.0 - 1.0) * (TRAIN_BIAS * bias_scale)
  ).to(device)

  falls = 0
  reward_sum = 0.0
  for step in range(steps):
    obs_perturbed = corrupt(obs, gen, noise_scale, bias, bias_scale, mode)
    with torch.no_grad():
      actions = policy(obs_perturbed)
    obs, reward, _, _ = env.step(actions)
    falls += int(env.unwrapped.termination_manager.terminated.sum().item())
    reward_sum += float(reward.sum().item())
    if push_scale > 0 and (step + 1) % PUSH_PERIOD == 0:
      apply_push(env, gen, push_scale)

  return {
    "condition": name,
    "falls": falls,
    "falls_per_env_20s": falls / NUM_ENVS * (1000.0 / steps),
    "reward_per_step": reward_sum / (NUM_ENVS * steps),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=STEPS)
  parser.add_argument(
    "--only",
    default=None,
    help="Comma-separated condition names; default runs all conditions.",
  )
  parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()

  conditions = CONDITIONS
  if args.only:
    wanted = set(args.only.split(","))
    conditions = tuple(c for c in CONDITIONS if c[0] in wanted)

  results = []
  policies_by_mode: dict[str, list[tuple[str, str]]] = {}
  for name, checkpoint, mode in POLICIES:
    policies_by_mode.setdefault(mode, []).append((name, checkpoint))

  for mode, policies in policies_by_mode.items():
    started = time.time()
    env, device, n_rows, n_cols = build_env(mode)
    runner_cls = load_runner_cls(TASK)
    assert runner_cls is not None
    runner = runner_cls(env, asdict(load_rl_cfg(TASK)), device=device)
    print(
      f"[INFO] mode={mode} terrain={n_rows}x{n_cols}, "
      f"env ready in {time.time() - started:.1f}s",
      flush=True,
    )

    for condition in conditions:
      for policy_name, checkpoint in policies:
        started = time.time()
        row = run_cell(
          env,
          runner,
          device,
          checkpoint,
          mode,
          condition,
          args.steps,
        )
        row.update(
          {
            "level": LEVEL,
            "policy": policy_name,
            "observation_mode": mode,
            "checkpoint": checkpoint,
            "wall_s": round(time.time() - started, 1),
          }
        )
        results.append(row)
        print(
          f"L{LEVEL} {row['condition']:16s} {policy_name:15s} "
          f"falls={row['falls']:5d} "
          f"({row['falls_per_env_20s']:6.3f}/env/20s) "
          f"rew={row['reward_per_step']:7.3f} [{row['wall_s']}s]",
          flush=True,
        )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
  print(f"[INFO] saved {args.out}")


if __name__ == "__main__":
  main()
