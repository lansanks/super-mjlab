"""Robustness stress test for the four DreamWaQ checkpoints.

Training-like terrain (curriculum rows = difficulty), all envs pinned to a
fixed level; paired design: one env instance per level, same seed per cell.

Perturbation axes (scaled relative to the training ranges):
  push  : velocity kick every 1 s (x,y +-0.5, z +-0.4, roll/pitch +-0.52, yaw +-0.78) * k
  noise : actor/cenet-history noise (ang_vel 0.2, gravity 0.05, jpos 0.01, jvel 1.5) * k
  bias  : constant joint-pos encoder bias (+-0.015 rad * k)
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from dataclasses import asdict, replace

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-DreamWaQ-Rough-Unitree-Go1"
NUM_ENVS = 256
SEED = 20261012
PUSH_PERIOD = 50

BASELINE = ("vel_baseline", "logs/rsl_rl/go1_velocity/2026-10-12_18-19-35/model_599.pt")

CKPTS = [
  ("A_cpl_b0", "logs/rsl_rl/go1_dreamwaq/2026-10-12_13-02-09/model_599.pt"),
  ("B_det_b0", "logs/rsl_rl/go1_dreamwaq/2026-10-12_14-17-48/model_599.pt"),
  ("D_det_b004", "logs/rsl_rl/go1_dreamwaq/2026-10-12_16-06-44/model_599.pt"),
  ("C_det_b01", "logs/rsl_rl/go1_dreamwaq/2026-10-12_15-07-42/model_599.pt"),
]

CONDITIONS = [
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
]

NOISE_SLICES = [(0, 3, 0.2), (3, 6, 0.05), (6, 18, 0.01), (18, 30, 1.5)]
PUSH_WIDTHS = [0.5, 0.5, 0.4, 0.52, 0.52, 0.78]
JOINT_POS_SLICE = (6, 18)
TRAIN_BIAS = 0.015


def build_env(level: int, noise_device: str):
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg = replace(env_cfg)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.episode_length_s = 1e9  # no timeouts
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.observations["cenet_input"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)  # pushes are injected manually
  env_cfg.curriculum = {}  # freeze difficulty at spawn

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  n_rows, n_cols = terrain.terrain_origins.shape[:2]
  terrain.terrain_levels[:] = level
  terrain.terrain_types[:] = torch.arange(NUM_ENVS, device=device) % n_cols
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]
  wrapper = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
  return wrapper, device, n_rows, n_cols


def make_noise(t, gen, scale):
  noise = torch.zeros_like(t)
  for lo, hi, width in NOISE_SLICES:
    noise[..., lo:hi] = (
      torch.rand(t.shape[:-1] + (hi - lo,), generator=gen, device="cpu").to(t.device)
      * 2.0
      - 1.0
    ) * (width * scale)
  return noise


def corrupt(obs, gen, noise_scale, bias, bias_scale):
  out = obs.clone()
  for key in ("actor", "cenet_input"):
    t = out[key]
    shape = t.shape
    flat = t.reshape(shape[0], -1, 45) if t.dim() > 2 or shape[-1] == 270 else t
    if noise_scale > 0:
      flat = flat + make_noise(flat, gen, noise_scale)
    if bias_scale > 0:
      lo, hi = JOINT_POS_SLICE
      offset = bias.unsqueeze(1) if flat.dim() == 3 else bias
      flat[..., lo:hi] = flat[..., lo:hi] + offset
    out[key] = flat.reshape(shape)
  return out


def apply_push(env, gen, scale):
  robot = env.unwrapped.scene["robot"]
  n = env.num_envs
  delta = torch.zeros(n, 6, device=env.unwrapped.device)
  for i, w in enumerate(PUSH_WIDTHS):
    delta[:, i] = (
      torch.rand(n, generator=gen, device="cpu").to(delta.device) * 2.0 - 1.0
    ) * (w * scale)
  ids = torch.arange(n, device=env.unwrapped.device)
  robot.write_root_link_velocity_to_sim(
    robot.data.root_link_vel_w[ids] + delta, env_ids=ids
  )


def load_baseline_policy(env, device, path):
  from rsl_rl.models import MLPModel

  obs = env.get_observations()
  policy = MLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=12,
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
  missing, unexpected = policy.load_state_dict(sd, strict=False)
  assert not missing and not unexpected, (missing, unexpected)
  return policy.eval().to(device)


def run_cell(env, runner, device, policy_path, cond, steps, baseline=False):
  cname, push_scale, noise_scale, bias_scale = cond
  if baseline:
    policy = load_baseline_policy(env, device, policy_path)
  else:
    runner.load(policy_path, load_cfg={"actor": True}, strict=True, map_location=device)
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
    obs_p = corrupt(obs, gen, noise_scale, bias, bias_scale)
    with torch.no_grad():
      actions = policy(obs_p)
    obs, rew, dones, extras = env.step(actions)
    falls += int(env.unwrapped.termination_manager.terminated.sum().item())
    reward_sum += float(rew.sum().item())
    if push_scale > 0 and (step + 1) % PUSH_PERIOD == 0:
      apply_push(env, gen, push_scale)
  return {
    "condition": cname,
    "falls": falls,
    "falls_per_env_20s": falls / NUM_ENVS * (1000.0 / steps),
    "reward_per_step": reward_sum / (NUM_ENVS * steps),
  }


def main():
  mode = "calibrate" if "--calibrate" in sys.argv else "full"
  lv = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--levels=")), None)
  levels = [3, 4, 5] if mode == "calibrate" else [int(lv)] if lv else [3]
  steps = 500 if mode == "calibrate" else 1000
  only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
  conditions = [CONDITIONS[0]] if mode == "calibrate" else CONDITIONS
  if only:
    wanted = set(only.split(","))
    conditions = [c for c in conditions if c[0] in wanted]

  runner = None
  results = []
  for level in levels:
    t0 = time.time()
    env, device, n_rows, n_cols = build_env(level, "cuda:0")
    if runner is None:
      runner_cls = load_runner_cls(TASK)
      runner = runner_cls(env, asdict(load_rl_cfg(TASK)), device=device)
      print(
        f"[INFO] env built ({n_rows} rows x {n_cols} cols) in {time.time() - t0:.1f}s",
        flush=True,
      )
    policies = [BASELINE] if "--baseline-only" in sys.argv else CKPTS
    for cond in conditions:
      for pname, path in policies:
        t0 = time.time()
        row = run_cell(
          env, runner, device, path, cond, steps, baseline=pname.startswith("vel_")
        )
        row.update(
          {"level": level, "policy": pname, "wall_s": round(time.time() - t0, 1)}
        )
        results.append(row)
        print(
          f"L{level} {row['condition']:16s} {pname:11s} falls={row['falls']:5d} "
          f"({row['falls_per_env_20s']:6.3f}/env/20s) rew={row['reward_per_step']:7.3f} "
          f"[{row['wall_s']}s]",
          flush=True,
        )
  out = pathlib.Path(
    "/tmp/stress_results_calib.json"
    if mode == "calibrate"
    else "/tmp/stress_results.json"
  )
  out.write_text(json.dumps(results, indent=1))
  print(f"[INFO] saved {out}")


if __name__ == "__main__":
  main()
