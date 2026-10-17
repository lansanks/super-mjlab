"""把 latent 拆成"相位成分"和"慢变成分"，看地形信息藏在哪里。

输入是 ``tsne_latent.py --store-raw`` 产出的 ``.npz``::

  uv run python src/mjlab/tasks/himloco/doc/实验/实验数据/latent_decompose.py \
    src/mjlab/tasks/himloco/doc/实验/实验数据/tsne_cenet_phase_L3.npz

做三件事：

1. **相位回归**：latent ≈ A·[cos φ, sin φ] + b（φ = Hilbert 步态相位），残差即
   "相位解释不了的部分"；对残差重新做 t-SNE，看地形是否在其中聚簇；
2. **环境平均**：每个环境把几百个采样点的 latent 求平均（相位在多个周期上被平均掉，
   只剩慢变成分），得到 ~700 个"每环境一个点"的样本再做 t-SNE——这正是论文
   Fig. 4 那种"每个环境一个点"的画法更可能对应的口径；
3. 打印三种口径下的地形 1-NN 可分性（原始 / 相位残差 / 环境平均）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from phase_check import attach_phase, env_phases, phase_r2, purity  # noqa: E402
from tsne_latent import tsne  # noqa: E402


def env_means(
  latent: np.ndarray, env_ids: np.ndarray, labels: np.ndarray, min_samples: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """每个环境的 latent 平均，返回 (mean_latent, terrain_label, env_ids)。"""
  means, labs, kept = [], [], []
  for e in np.unique(env_ids):
    m = env_ids == e
    if int(m.sum()) < min_samples:
      continue
    means.append(latent[m].mean(axis=0))
    labs.append(labels[m][0])
    kept.append(e)
  return np.stack(means), np.array(labs), np.array(kept)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("npz", help="tsne_latent.py --store-raw 产出的 .npz")
  parser.add_argument("--tsne-iters", type=int, default=500)
  parser.add_argument("--out", default=None)
  parser.add_argument(
    "--env-mean-only",
    action="store_true",
    help="只画每环境平均后的 t-SNE（单图，秒级完成）",
  )
  args = parser.parse_args()

  src = pathlib.Path(args.npz)
  d = np.load(src, allow_pickle=True)
  if "latent_raw" not in d:
    raise SystemExit("需要 *_raw 数据，请用 tsne_latent.py --store-raw 采集")
  names = [str(v) for v in d["type_names"]]
  chance = 1.0 / len(names)
  kind = str(d["kind"]) if "kind" in d else src.stem
  tag = pathlib.Path(str(d["checkpoint"])).stem if "checkpoint" in d else src.stem

  # ---- 相位 ----
  table = env_phases(d["joint_pos_raw"], d["env_ids_raw"], d["steps_raw"])
  phase, _ = attach_phase(d["env_ids"], d["steps"], table)
  ok = np.isfinite(phase)
  raw = d["latent"][ok].astype(np.float64)
  labels = d["labels"][ok]
  phase_r2_value, residual = phase_r2(raw, np.unwrap(phase[ok]))

  # ---- 环境平均 ----
  raw_all = d["latent_raw"].astype(np.float64)
  mean_lat, mean_lab, mean_env = env_means(raw_all, d["env_ids_raw"], d["labels_raw"])
  n_env = len(mean_lat)

  print(f"[INFO] {kind} / {tag}")
  print(
    f"[INFO] 相位解释 {phase_r2_value:.1%} 的 latent 方差；用于比较的样本 {ok.sum()}"
  )
  p_raw = purity(raw, labels)
  p_res = purity(residual, labels)
  p_mean = purity(mean_lat, mean_lab)

  if args.env_mean_only:
    emb_mean = tsne(mean_lat, n_iter=args.tsne_iters, seed=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    cmap = plt.get_cmap("tab10")
    for t, nm in enumerate(names):
      m = mean_lab == t
      ax.scatter(
        emb_mean[m, 0],
        emb_mean[m, 1],
        s=18,
        alpha=0.85,
        color=cmap(t % 10),
        label=f"{nm} (n={int(m.sum())})",
        linewidths=0,
      )
    ax.set_title(
      f"{kind}: per-environment average, t-SNE (slow component)\n"
      f"n={n_env} envs, terrain 1-NN {p_mean:.2f} (chance {chance:.2f})",
      fontsize=10,
    )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    out = (
      pathlib.Path(args.out)
      if args.out
      else src.with_suffix("").with_name(src.stem + "_envmean.png")
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    np.savez(
      out.with_suffix(".npz"),
      embedding=emb_mean,
      labels=mean_lab,
      env_ids=mean_env,
      latent=mean_lat,
      type_names=np.array(names),
      kind=kind,
    )
    print(
      f"[INFO] {kind} 环境平均 t-SNE：n={n_env}, 地形 1-NN={p_mean:.3f} "
      f"(random {chance:.3f})"
    )
    print(f"[INFO] 图已保存：{out}")
    print(f"[INFO] 嵌入数据已保存：{out.with_suffix('.npz')}")
    return
  print(f"[INFO] 地形 1-NN 可分性（随机 {chance:.3f}）：")
  print(f"        原始 latent        {p_raw:.3f}")
  print(f"        相位残差           {p_res:.3f}")
  print(f"        环境平均（{n_env} 个） {p_mean:.3f}")

  # ---- 三张 t-SNE ----
  emb_res = tsne(residual, n_iter=args.tsne_iters, seed=0)
  emb_mean = tsne(mean_lat, n_iter=args.tsne_iters, seed=0)
  emb_raw = d["embedding"]

  fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), dpi=160)
  cmap = plt.get_cmap("tab10")

  def draw(ax, emb, lab, title):
    for t, nm in enumerate(names):
      m = lab == t
      ax.scatter(
        emb[m, 0],
        emb[m, 1],
        s=6,
        alpha=0.75,
        color=cmap(t % 10),
        label=nm,
        linewidths=0,
      )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

  draw(
    axes[0],
    emb_raw,
    d["labels"],
    f"{kind}: raw latent t-SNE\nterrain 1-NN {p_raw:.2f} (chance {chance:.2f})",
  )
  draw(
    axes[1],
    emb_res,
    labels,
    f"{kind}: after removing the phase component\n"
    f"terrain 1-NN {p_res:.2f} (chance {chance:.2f})",
  )
  draw(
    axes[2],
    emb_mean,
    mean_lab,
    f"{kind}: per-environment average (slow component, n={n_env})\n"
    f"terrain 1-NN {p_mean:.2f} (chance {chance:.2f})",
  )
  axes[0].legend(fontsize=7, loc="best")

  out = (
    pathlib.Path(args.out)
    if args.out
    else src.with_suffix("").with_name(src.stem + "_decompose.png")
  )
  fig.savefig(out, bbox_inches="tight")
  plt.close(fig)
  summary = {
    "kind": kind,
    "checkpoint": tag,
    "phase_variance_explained": phase_r2_value,
    "terrain_purity_raw": p_raw,
    "terrain_purity_phase_removed": p_res,
    "terrain_purity_env_mean": p_mean,
    "n_envs_averaged": n_env,
    "chance": chance,
  }
  out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
  print(f"[INFO] 图已保存：{out}")
  print(f"[INFO] 指标已保存：{out.with_suffix('.json')}")


if __name__ == "__main__":
  main()
