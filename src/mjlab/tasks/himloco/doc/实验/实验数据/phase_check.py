"""按步态相位给 latent 上色，检验"latent 是否被相位主导"。

输入是 ``tsne_latent.py --store-raw`` 产出的 ``.npz``（含密集原始样本的
关节角与足端接触时间）::

  uv run python src/mjlab/tasks/himloco/doc/实验/实验数据/phase_check.py \
    src/mjlab/tasks/himloco/doc/实验/实验数据/tsne_him_k32_phase_L3.npz

做法：

1. 按环境取出第一条腿 calf 关节（关节 2）的**连续**角度序列，做 Hilbert 变换
   得到瞬时相位（步态周期 ≈ 2π / 平均角频率）；
2. 把相位映射回 t-SNE 用的那批子采样点（按 (env_id, step) 配对）；
3. 画三联图：t-SNE 按地形上色 / t-SNE 按相位上色（周期色环）/ latent 的
   PC1-PC2 按相位上色；
4. 打印三个量：相位对 latent 的方差解释率 R²、以及"原始 / 去掉相位成分后"
   的地形 1-NN 可分性——如果 R² 很高而地形可分性没有变化，就说明 latent 主要
   在编码相位，地形（若有）藏在残差里。
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

CALF_JOINT = 2  # 关节顺序 = 执行器顺序：FR_hip, FR_thigh, FR_calf, ...


def env_phases(joints: np.ndarray, env_ids: np.ndarray, steps: np.ndarray) -> dict:
  """对每个环境做 Hilbert 相位估计，返回 {env_id: {"steps", "phase", "period_s"}}。"""
  dt = 0.02  # 采样间隔 = 1 个策略步（--sample-every 1）
  out: dict[int, dict[str, np.ndarray | float]] = {}
  for e in np.unique(env_ids):
    m = env_ids == e
    order = np.argsort(steps[m])
    s = steps[m][order]
    x = joints[m][order, CALF_JOINT].astype(np.float64)
    if len(x) < 12:
      continue
    if len(x) >= 5:  # 轻度平滑，抑制单步噪声
      x = np.convolve(x, np.ones(3) / 3.0, mode="same")
    phase = np.angle(hilbert(x - x.mean()))
    dphase = np.diff(np.unwrap(phase))
    omega = float(np.median(np.abs(dphase)))
    period_s = float(2.0 * np.pi / (omega / dt)) if omega > 1e-9 else float("nan")
    out[int(e)] = {"steps": s, "phase": phase, "period_s": period_s}
  return out


def attach_phase(
  env_ids: np.ndarray, steps: np.ndarray, table: dict
) -> tuple[np.ndarray, np.ndarray]:
  """把相位按 (env_id, step) 映射到子采样点；没匹配到的返回 NaN。"""
  phase = np.full(len(steps), np.nan)
  period = np.full(len(steps), np.nan)
  for i, (e, s) in enumerate(zip(env_ids.tolist(), steps.tolist(), strict=True)):
    entry = table.get(int(e))
    if entry is None:
      continue
    j = np.searchsorted(entry["steps"], s)
    if j < len(entry["steps"]) and entry["steps"][j] == s:
      phase[i] = entry["phase"][j]
      period[i] = entry["period_s"]
  return phase, period


def purity(a: np.ndarray, labels: np.ndarray) -> float:
  d2 = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1)
  np.fill_diagonal(d2, np.inf)
  return float((labels[d2.argmin(axis=1)] == labels).mean())


def phase_r2(x: np.ndarray, phase: np.ndarray) -> tuple[float, np.ndarray]:
  """把 x 对 [cos φ, sin φ] 做最小二乘，返回 R² 与残差。"""
  f = np.stack([np.cos(phase), np.sin(phase), np.ones_like(phase)], axis=-1)
  coef, *_ = np.linalg.lstsq(f, x, rcond=None)
  resid = x - f @ coef
  r2 = 1.0 - float((resid**2).sum() / (x**2).sum())
  return r2, resid


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("npz", help="tsne_latent.py --store-raw 产出的 .npz")
  parser.add_argument("--out", default=None, help="输出图片（默认同名 .png 加 _phase）")
  args = parser.parse_args()

  src = pathlib.Path(args.npz)
  d = np.load(src, allow_pickle=True)
  if "latent_raw" not in d:
    raise SystemExit(
      "这个 npz 没有 *_raw 数据，请用 tsne_latent.py --store-raw 重新采集"
    )
  embedding, labels = d["embedding"], d["labels"]
  env_ids, steps = d["env_ids"], d["steps"]
  names = [str(v) for v in d["type_names"]]

  table = env_phases(d["joint_pos_raw"], d["env_ids_raw"], d["steps_raw"])
  ctx = "checkpoint" in d and pathlib.Path(str(d["checkpoint"])).stem or src.stem

  phase, period = attach_phase(env_ids, steps, table)
  ok = np.isfinite(phase)
  print(f"[INFO] 有效相位样本 {ok.sum()}/{len(phase)}（来自 {len(table)} 个环境）")
  med_period = float(np.nanmedian(period))
  print(f"[INFO] 步态周期（FR calf Hilbert 相位）≈ {med_period:.3f} s")

  p2 = np.unwrap(phase[ok])
  latent_r2, resid = phase_r2(d["latent"][ok].astype(np.float64), p2)
  emb_r2, _ = phase_r2(embedding[ok].astype(np.float64), p2)
  print(f"[INFO] 相位解释的方差：latent {latent_r2:.1%}，t-SNE 嵌入 {emb_r2:.1%}")
  p_raw = purity(d["latent"][ok].astype(np.float64), labels[ok])
  p_res = purity(resid, labels[ok])
  print(
    f"[INFO] 地形 1-NN 可分性：原始 {p_raw:.3f} → 去掉相位成分 {p_res:.3f}"
    f"（随机 {1.0 / len(names):.3f}）"
  )

  fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), dpi=160)
  cmap = plt.get_cmap("tab10")
  for t, nm in enumerate(names):
    m = labels == t
    axes[0].scatter(
      embedding[m, 0],
      embedding[m, 1],
      s=6,
      alpha=0.75,
      color=cmap(t % 10),
      label=nm,
      linewidths=0,
    )
  axes[0].set_title("t-SNE coloured by terrain type", fontsize=10)
  axes[0].legend(fontsize=7, loc="best")

  sc = axes[1].scatter(
    embedding[ok, 0],
    embedding[ok, 1],
    s=6,
    alpha=0.9,
    c=phase[ok],
    cmap="hsv",
    linewidths=0,
  )
  axes[1].set_title(
    f"t-SNE coloured by gait phase (FR calf, period ≈ {med_period:.2f} s)\n"
    f"phase explains {emb_r2:.1%} of the 2-D embedding variance",
    fontsize=10,
  )
  fig.colorbar(sc, ax=axes[1], shrink=0.8, label="phase [rad]")

  x = d["latent"][ok].astype(np.float64)
  x = x - x.mean(0)
  u, s, _ = np.linalg.svd(x, full_matrices=False)
  pc = u * s
  sc2 = axes[2].scatter(
    pc[:, 0], pc[:, 1], s=6, alpha=0.9, c=phase[ok], cmap="hsv", linewidths=0
  )
  axes[2].set_title(
    f"latent PC1-PC2 coloured by gait phase\n"
    f"phase explains {latent_r2:.1%} of the 16-D latent variance\n"
    f"terrain 1-NN purity: {p_raw:.2f} → {p_res:.2f} after removing phase",
    fontsize=10,
  )
  fig.colorbar(sc2, ax=axes[2], shrink=0.8, label="phase [rad]")
  for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

  out = (
    pathlib.Path(args.out)
    if args.out
    else src.with_suffix("").with_name(src.stem + "_phase.png")
  )
  fig.savefig(out, bbox_inches="tight")
  plt.close(fig)
  summary = {
    "checkpoint": ctx,
    "period_s": med_period,
    "phase_variance_explained_latent": latent_r2,
    "phase_variance_explained_tsne": emb_r2,
    "terrain_purity_raw": p_raw,
    "terrain_purity_phase_removed": p_res,
    "n_phase_samples": int(ok.sum()),
  }
  out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
  print(f"[INFO] 图已保存：{out}")
  print(f"[INFO] 指标已保存：{out.with_suffix('.json')}")


if __name__ == "__main__":
  main()
