"""HIM 隐空间 t-SNE 可视化（复刻论文 Fig. 4 的风格）。

做法：把训练好的 Go1 HIM 策略加载进 rough 地形环境，每个环境固定在
**同一种地形类型 + 同一难度等级**上，用策略跑一段 rollout，取出源编码器
对历史 ``o_t^H`` 编码出的 16 维隐向量 ``l``，用 t-SNE 降到 2 维后按
地形类型上色。如果 latent 已经具备地形分类能力，同色点应当聚成簇、不同色
的簇应当分得开（论文 Fig. 4(a) 的样子）。

用法（在仓库根目录）::

  uv run python src/mjlab/tasks/himloco/doc/实验/实验数据/tsne_latent.py
  uv run python src/mjlab/tasks/himloco/doc/实验/实验数据/tsne_latent.py \
    --checkpoint logs/rsl_rl/go1_himloco/2026-10-16_18-49-56/model_2999.pt \
    --level 3 --num-envs 448 --steps 400

输出：同目录下 ``tsne_him_<checkpoint>_L<level>.png``（图）和同名 ``.npz``
（2 维嵌入 + 地形标签，方便重画/二次分析）。

注意：mjlab 的 venv 里没有 scikit-learn，本脚本内置了一份精确（O(n^2)）
t-SNE 实现，参数与 sklearn 默认值一致（perplexity 30、early exaggeration
12、250 步后关掉、学习率 n/48、动量 0.5 -> 0.8、PCA 初始化）。
"""

from __future__ import annotations

import argparse
import math
import pathlib
import time
from dataclasses import asdict
from typing import Any, cast

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import (  # noqa: E402
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)

# 两种 latent：HIM 的隐响应 l̂（论文 Fig. 4(a)）与 DreamWaQ CENet 的后验均值 μ
# （论文 Fig. 4(b) 的 regression 对照）。两者的 env、地形、观测口径完全一致。
TASK_BY_KIND = {
  "him": "Mjlab-HIM-Rough-Unitree-Go1",
  "cenet": "Mjlab-DreamWaQ-Rough-Unitree-Go1",
}
DEFAULT_CKPT_BY_KIND = {
  "him": "logs/rsl_rl/go1_himloco/2026-10-16_18-49-56/model_2999.pt",
  "cenet": "logs/rsl_rl/go1_dreamwaq/2026-10-12_16-06-44/model_599.pt",
}
LABEL_BY_KIND = {
  "him": "HIM latent l (ours)",
  "cenet": "DreamWaQ CENet mu",
  "him_isaac": "HIM latent l (Isaac Gym, original)",
}
# 评估时需要关掉噪声的观测组（actor 与对应编码器的输入）。
CLEAN_GROUPS = {
  "him": ("actor", "him_input", "him_state"),
  "cenet": ("actor", "cenet_input"),
}
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# 环境与 rollout
# --------------------------------------------------------------------------- #
def build_env(
  kind: str, num_envs: int, level: int, seed: int
) -> tuple[RslRlVecEnvWrapper, int, int, list[str]]:
  """构建 rough 地形环境，并把地形类型/等级钉死。

  Returns:
    (env, num_types, level, type_names)：``num_types`` 是地形类型数，
    每种类型分到 ``num_envs // num_types`` 个环境（列 = 类型，行 = 难度）。
  """
  # 用训练配置：curriculum=True 时每种地形恰好占一列（列号 = 地形类型），
  # 这正是 stress_eval.py 钉地形用的口径；play 配置是 5x5 随机地形，列号
  # 并不代表地形类型。观测噪声/推力在这里手动关掉。
  cfg = load_env_cfg(TASK_BY_KIND[kind], play=False)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 1e9  # 不因超时重置
  cfg.curriculum = {}  # 冻结课程，难度由 --level 指定
  cfg.events.pop("push_robot", None)
  for group in CLEAN_GROUPS[kind]:
    if group in cfg.observations:
      cfg.observations[group].enable_corruption = False

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.seed(seed)
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  generator = terrain_cfg.terrain_generator
  assert generator is not None
  type_names = list(generator.sub_terrains.keys())

  n_rows, n_cols = terrain.terrain_origins.shape[:2]
  assert n_cols == len(type_names), (n_cols, type_names)
  terrain.terrain_levels[:] = min(level, n_rows - 1)
  terrain.terrain_types[:] = torch.arange(num_envs, device=device) % n_cols
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]

  task = TASK_BY_KIND[kind]
  wrapper = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)
  return wrapper, n_cols, level, type_names


def load_policy(env: RslRlVecEnvWrapper, kind: str, device: str, checkpoint: str):
  """加载 checkpoint 里的 actor，返回确定性策略。"""
  task = TASK_BY_KIND[kind]
  runner_cls = load_runner_cls(task)
  assert runner_cls is not None, f"任务 {task} 没有注册 runner_cls"
  runner = runner_cls(env, asdict(load_rl_cfg(task)), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def latent_of(policy, kind: str) -> torch.Tensor:
  """取出上一次 Actor 前向里缓存的隐向量（HIM: l̂；CENet: 后验均值 μ）。"""
  if kind == "him":
    output = getattr(policy, "last_him_output", None)
    assert output is not None, "Actor 还没有跑过 HIM 前向"
    return output.latent
  output = getattr(policy, "last_cenet_output", None)
  assert output is not None, "Actor 还没有跑过 CENet 前向"
  return output.mu


def collect_latents(
  env: RslRlVecEnvWrapper,
  policy,
  kind: str,
  num_types: int,
  warmup: int,
  steps: int,
  sample_every: int,
  max_raw: int,
  command_x: float | None,
  seed: int,
) -> tuple[
  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
  """跑 rollout，收集 ``(latent, terrain_type)`` 样本。

  每个采样步对每个环境取一个 l（来自 Actor 前向里估计器的缓存），
  采样间隔 ``sample_every`` 用来降低时间相关性。环境一旦终止（摔倒）就
  永久剔除：自动重置会把它的地形类型重新随机化，标签不再可信。

  Returns:
    ``(latents, terrain_types, env_ids, steps, joints, air_times)``。env_ids 用来在
    画图时区分 "按地形聚类" 与 "按环境聚类"；steps/joints/air_times 供相位分析
    （``phase_check.py``）使用，与 latent 是同一时刻的采样。
  """
  device = env.unwrapped.device
  robot = env.unwrapped.scene["robot"]
  contact = env.unwrapped.scene["feet_ground_contact"]
  torch.manual_seed(seed)
  obs, _ = env.reset()
  terrain = env.unwrapped.scene.terrain
  assert terrain is not None
  terrain_types = terrain.terrain_types.clone()
  term = env.unwrapped.command_manager.get_term("twist")
  assert term is not None
  command = cast(torch.Tensor, term.command)
  if command_x is not None:
    command[:] = torch.tensor([command_x, 0.0, 0.0], device=device)

  alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
  env_ids_all = torch.arange(env.num_envs, device=device)
  latents: list[torch.Tensor] = []
  labels: list[torch.Tensor] = []
  env_ids: list[torch.Tensor] = []
  step_ids: list[torch.Tensor] = []
  joints: list[torch.Tensor] = []
  air_times: list[torch.Tensor] = []
  collected = 0
  for step in range(warmup + steps):
    if command_x is not None:
      command[:] = torch.tensor([command_x, 0.0, 0.0], device=device)
    with torch.no_grad():
      actions = policy(obs)
      latent = latent_of(policy, kind)
      # 相位分析用：与 latent 同一时刻的关节角与足端接触/腾空时间。
      joint_pos = robot.data.joint_pos.clone()
      contact_data = contact.data
      air_time = torch.cat(
        [
          t.reshape(env.num_envs, -1)
          for t in (
            contact_data.current_air_time,
            contact_data.current_contact_time,
          )
          if t is not None
        ],
        dim=-1,
      )
    obs, _, dones, _ = env.step(actions)
    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
      alive[done_ids] = False
    if step < warmup or (step - warmup) % sample_every != 0:
      continue
    if collected >= max_raw:
      break
    keep = alive
    latents.append(latent[keep].detach().cpu())
    labels.append(terrain_types[keep].detach().cpu())
    env_ids.append(env_ids_all[keep].detach().cpu())
    step_ids.append(torch.full((int(keep.sum()),), step, dtype=torch.long))
    joints.append(joint_pos[keep].detach().cpu())
    air_times.append(air_time[keep].detach().cpu())
    collected += int(keep.sum())

  excluded = int((~alive).sum())
  print(
    f"[INFO] rollout 完成：{collected} 个样本，{excluded}/{env.num_envs} 个环境"
    f"因终止被剔除（地形类型会在重置后重新随机，必须丢弃）",
    flush=True,
  )
  return (
    torch.cat(latents),
    torch.cat(labels),
    torch.cat(env_ids),
    torch.cat(step_ids),
    torch.cat(joints),
    torch.cat(air_times),
  )


def subsample(
  latents: torch.Tensor,
  labels: torch.Tensor,
  env_ids: torch.Tensor,
  steps: torch.Tensor,
  joints: torch.Tensor,
  air_times: torch.Tensor,
  num_types: int,
  max_samples: int,
  seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """按地形类型均衡地随机下采样（t-SNE 是 O(n^2)，点数不宜过多）。"""
  rng = np.random.default_rng(seed)
  per_type = max(max_samples // num_types, 2)
  keep: list[np.ndarray] = []
  for t in range(num_types):
    idx = (labels == t).nonzero(as_tuple=False).squeeze(-1).numpy()
    if len(idx) > per_type:
      idx = rng.choice(idx, size=per_type, replace=False)
    keep.append(idx)
  sel = np.concatenate(keep)
  rng.shuffle(sel)
  return (
    latents.numpy()[sel],
    labels.numpy()[sel],
    env_ids.numpy()[sel],
    steps.numpy()[sel],
    joints.numpy()[sel],
    air_times.numpy()[sel],
  )


# --------------------------------------------------------------------------- #
# t-SNE（与 sklearn 默认参数一致的精确实现，无第三方依赖）
# --------------------------------------------------------------------------- #
def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
  sq = (x * x).sum(dim=1, keepdim=True)
  d2 = sq + sq.T - 2.0 * (x @ x.T)
  return d2.clamp_min_(0.0)


def _joint_probabilities(
  x: torch.Tensor, perplexity: float, tol: float = 1e-5, max_iter: int = 100
) -> torch.Tensor:
  """按 perplexity 对每个点做二分搜索，得到对称联合概率 P。"""
  n = x.shape[0]
  d2 = _pairwise_sq_dists(x)
  beta = torch.ones(n, dtype=x.dtype)
  target = math.log(perplexity)
  p = torch.zeros_like(d2)
  for _ in range(max_iter):
    p = torch.exp(-d2 * beta[:, None])
    p.fill_diagonal_(0.0)
    sum_p = p.sum(dim=1, keepdim=True)
    entropy = torch.log(sum_p).squeeze(-1) + (
      beta * (d2 * p).sum(dim=1) / sum_p.squeeze(-1)
    )
    diff = entropy - target
    if bool(diff.abs().max() < tol):
      break
    beta = torch.where(diff > 0, beta * 2.0, beta * 0.5)
  p = p / p.sum(dim=1, keepdim=True)
  p = (p + p.T) / (2.0 * n)
  return p.clamp_min_(1e-12)


def tsne(
  x: np.ndarray,
  perplexity: float = 30.0,
  n_iter: int = 1000,
  early_exaggeration: float = 12.0,
  early_iters: int = 250,
  seed: int = 0,
) -> np.ndarray:
  """精确 t-SNE（van der Maaten & Hinton 2008），返回 (n, 2) 嵌入。

  用 float64 计算（n^2 的距离矩阵在 float32 下长时间迭代可能溢出），
  更新公式与 sklearn 的 ``_gradient_descent`` 一致：动量 + 自适应增益，
  学习率 ``max(n / early_exaggeration / 4, 50)``。
  """
  xt = torch.as_tensor(np.asarray(x, dtype=np.float64))
  n = xt.shape[0]
  p = _joint_probabilities(xt, perplexity)

  # PCA 初始化（sklearn 的默认做法：标准化后再缩放到 1e-4 量级）。
  xc = xt - xt.mean(dim=0, keepdim=True)
  _, _, vh = torch.linalg.svd(xc, full_matrices=False)
  y = xc @ vh[:2].T
  y = y / y.std().clamp_min_(1e-12) * 1e-4

  lr = max(n / early_exaggeration / 4.0, 50.0)
  momentum = 0.5
  velocity = torch.zeros_like(y)
  gains = torch.ones_like(y)
  t0 = time.time()
  for it in range(n_iter):
    exaggeration = early_exaggeration if it < early_iters else 1.0
    inv = 1.0 / (1.0 + _pairwise_sq_dists(y))
    inv.fill_diagonal_(0.0)
    q = inv / inv.sum()
    weights = (p * exaggeration - q) * inv
    grad = 4.0 * (weights.sum(dim=1, keepdim=True) * y - weights @ y)
    gains = torch.where(grad * velocity > 0, gains * 0.8, gains + 0.2)
    gains.clamp_min_(0.01)
    velocity = momentum * velocity - lr * gains * grad
    y = y + velocity
    y = y - y.mean(dim=0, keepdim=True)
    if it == early_iters:
      momentum = 0.8
    if (it + 1) % 250 == 0:
      # q 的对角线恒为 0（inv 的对角线被置零），夹一个下界只影响诊断量。
      kl = float((p * (p / q.clamp_min(1e-30)).log()).sum())
      print(
        f"[INFO] t-SNE 迭代 {it + 1}/{n_iter}  KL={kl:.3f} "
        f"max|y|={float(y.abs().max()):.1f}",
        flush=True,
      )
  print(f"[INFO] t-SNE 用时 {time.time() - t0:.1f}s", flush=True)
  return y.detach().to(torch.float32).numpy()


# --------------------------------------------------------------------------- #
# 画图
# --------------------------------------------------------------------------- #
def neighbour_purity(embedding: np.ndarray, labels: np.ndarray) -> float:
  """2 维嵌入里的 1-NN 同标签比例，作为"分得开不开"的粗略量化。"""
  d2 = np.sum((embedding[:, None, :] - embedding[None, :, :]) ** 2, axis=-1)
  np.fill_diagonal(d2, np.inf)
  nn = d2.argmin(axis=1)
  return float((labels[nn] == labels).mean())


def plot(
  embedding: np.ndarray,
  labels: np.ndarray,
  env_ids: np.ndarray,
  type_names: list[str],
  kind: str,
  level: int,
  checkpoint: str,
  out_path: pathlib.Path,
) -> None:
  """复刻论文 Fig. 4 的样式：按地形类型上色的散点图（图内文字用英文）。"""
  fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
  cmap = plt.get_cmap("tab10")
  for t, name in enumerate(type_names):
    m = labels == t
    ax.scatter(
      embedding[m, 0],
      embedding[m, 1],
      s=8,
      alpha=0.75,
      color=cmap(t % 10),
      label=f"{name} (n={int(m.sum())})",
      linewidths=0,
    )
  ax.set_title(
    f"{LABEL_BY_KIND[kind]} t-SNE — {pathlib.Path(checkpoint).name}, "
    f"terrain level {level}\n"
    f"each point = one (history → latent), n={len(labels)}; 1-NN same terrain "
    f"{neighbour_purity(embedding, labels):.2f} / same env "
    f"{neighbour_purity(embedding, env_ids):.2f}",
    fontsize=9,
  )
  ax.set_xlabel("Dimension 1")
  ax.set_ylabel("Dimension 2")
  ax.set_xticks([])
  ax.set_yticks([])
  ax.legend(fontsize=8, loc="best", framealpha=0.9)
  fig.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  print(f"[INFO] 图已保存：{out_path}", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--latent",
    choices=sorted(TASK_BY_KIND),
    default="him",
    help="要可视化的隐向量：him = HIM 的 l̂，cenet = DreamWaQ CENet 的 μ",
  )
  parser.add_argument(
    "--checkpoint", default=None, help="默认按 --latent 取各自 run 的最终 checkpoint"
  )
  parser.add_argument("--level", type=int, default=3, help="地形难度等级（行）")
  parser.add_argument("--num-envs", type=int, default=700)
  parser.add_argument("--steps", type=int, default=800, help="采样阶段步数（50 Hz）")
  parser.add_argument("--warmup", type=int, default=60, help="丢弃的起步步数")
  parser.add_argument("--sample-every", type=int, default=4)
  parser.add_argument("--max-raw", type=int, default=120000, help="最多收集的原始样本")
  parser.add_argument(
    "--max-samples", type=int, default=5000, help="t-SNE 实际使用点数"
  )
  parser.add_argument(
    "--command-x",
    type=float,
    default=1.0,
    help="固定的前进速度指令；给 -1 表示保留环境自身的随机指令",
  )
  parser.add_argument("--perplexity", type=float, default=30.0)
  parser.add_argument("--tsne-iters", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", default=None, help="输出图片路径（默认自动命名）")
  parser.add_argument(
    "--replot", default=None, help="不跑仿真，直接从已有 .npz 重画图（改样式用）"
  )
  parser.add_argument(
    "--store-raw",
    action="store_true",
    help="把密集采样的原始数组也写进 npz（相位分析用；文件会大很多）",
  )
  args = parser.parse_args()

  if args.replot:
    d = np.load(args.replot, allow_pickle=True)
    src = pathlib.Path(args.replot)
    out = pathlib.Path(args.out) if args.out else src.with_suffix(".png")
    plot(
      d["embedding"],
      d["labels"],
      d["env_ids"],
      [str(x) for x in d["type_names"]],
      str(d["kind"]) if "kind" in d else args.latent,
      int(d["level"]) if "level" in d else args.level,
      str(d["checkpoint"]) if "checkpoint" in d else src.stem,
      out,
    )
    return

  kind = args.latent
  ckpt_arg = args.checkpoint or DEFAULT_CKPT_BY_KIND[kind]
  t0 = time.time()
  env, num_types, level, type_names = build_env(
    kind, args.num_envs, args.level, args.seed
  )
  device = env.unwrapped.device
  ckpt = pathlib.Path(ckpt_arg)
  assert ckpt.exists(), f"找不到 checkpoint：{ckpt}"
  policy = load_policy(env, kind, device, str(ckpt))
  print(f"[INFO] 环境与策略就绪（{time.time() - t0:.1f}s）", flush=True)

  command_x = None if args.command_x < 0 else args.command_x
  latents, labels, env_ids, steps, joints, air_times = collect_latents(
    env,
    policy,
    kind,
    num_types,
    warmup=args.warmup,
    steps=args.steps,
    sample_every=args.sample_every,
    max_raw=args.max_raw,
    command_x=command_x,
    seed=args.seed,
  )
  x, y, e, st, jp, at = subsample(
    latents,
    labels,
    env_ids,
    steps,
    joints,
    air_times,
    num_types,
    args.max_samples,
    args.seed,
  )
  print(f"[INFO] t-SNE 输入：{x.shape[0]} 个样本，{x.shape[1]} 维", flush=True)

  embedding = tsne(
    x, perplexity=args.perplexity, n_iter=args.tsne_iters, seed=args.seed
  )
  out = (
    pathlib.Path(args.out)
    if args.out
    else OUTPUT_DIR / f"tsne_{kind}_{ckpt.stem}_L{level}.png"
  )
  plot(embedding, y, e, type_names, kind, level, str(ckpt), out)
  saved: dict[str, Any] = {
    "embedding": embedding,
    "labels": y,
    "env_ids": e,
    "steps": st,
    "joint_pos": jp,
    "air_time": at,
    "kind": kind,
    "level": level,
    "checkpoint": str(ckpt),
    "type_names": np.array(type_names),
    "latent": x,
  }
  if args.store_raw:
    saved.update(
      latent_raw=latents.numpy(),
      labels_raw=labels.numpy(),
      env_ids_raw=env_ids.numpy(),
      steps_raw=steps.numpy(),
      joint_pos_raw=joints.numpy(),
      air_time_raw=air_times.numpy(),
    )
  np.savez(out.with_suffix(".npz"), **saved)
  print(f"[INFO] 嵌入数据已保存：{out.with_suffix('.npz')}", flush=True)
  print(f"[INFO] 总用时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
  main()
