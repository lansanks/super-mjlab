"""把两份 latent t-SNE 嵌入并排画成论文 Fig. 4 风格的双联图。

输入是 ``tsne_latent.py`` 产出的 ``.npz``（按命令行顺序从左到右排列）::

  uv run python src/mjlab/tasks/himloco/doc/实验/实验数据/tsne_compare.py \
    .../tsne_him_model_2999_L3.npz .../tsne_cenet_model_599_L3.npz \
    --out .../tsne_him_vs_cenet_L3.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL_BY_KIND = {
  "him": "HIM latent l (ours)",
  "cenet": "CENet mu (regression-like)",
  "him_isaac": "HIM latent l (Isaac Gym, original)",
}


def purity(embedding: np.ndarray, labels: np.ndarray) -> float:
  d2 = np.sum((embedding[:, None, :] - embedding[None, :, :]) ** 2, axis=-1)
  np.fill_diagonal(d2, np.inf)
  return float((labels[d2.argmin(axis=1)] == labels).mean())


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("npz", nargs="+", help="tsne_latent.py 产出的 .npz（按顺序并排）")
  parser.add_argument("--out", required=True, help="输出图片路径")
  parser.add_argument(
    "--tags",
    default=None,
    help="每个面板的短标签，逗号分隔（默认用 latent 种类 + 文件名）",
  )
  args = parser.parse_args()

  files = [pathlib.Path(p) for p in args.npz]
  tags = args.tags.split(",") if args.tags else None
  if tags is not None and len(tags) != len(files):
    raise SystemExit(f"--tags 给了 {len(tags)} 个，但输入有 {len(files)} 个文件")
  fig, axes = plt.subplots(
    1, len(files), figsize=(6.2 * len(files), 5.2), dpi=200, squeeze=False
  )
  cmap = plt.get_cmap("tab10")
  for i, (ax, path) in enumerate(zip(axes[0], files, strict=True)):
    d = np.load(path, allow_pickle=True)
    embedding, labels = d["embedding"], d["labels"]
    names = [str(x) for x in d["type_names"]]
    kind = str(d["kind"]) if "kind" in d else path.stem.split("_")[1]
    tag = tags[i] if tags is not None else None
    if tag is None:
      ckpt = str(d["checkpoint"]) if "checkpoint" in d else path.stem
      tag = f"{LABEL_BY_KIND.get(kind, kind)} — {pathlib.Path(ckpt).stem}"
    for t, name in enumerate(names):
      m = labels == t
      ax.scatter(
        embedding[m, 0],
        embedding[m, 1],
        s=6,
        alpha=0.75,
        color=cmap(t % 10),
        label=name,
        linewidths=0,
      )
    ax.set_title(
      f"{tag}\n"
      f"n={len(labels)}, 1-NN same terrain {purity(embedding, labels):.2f} "
      f"(chance {1.0 / len(names):.2f})",
      fontsize=10,
    )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
  axes[0][-1].legend(fontsize=7, loc="best", framealpha=0.9)
  out = pathlib.Path(args.out)
  fig.savefig(out, bbox_inches="tight")
  plt.close(fig)
  print(f"[INFO] 对比图已保存：{out}", flush=True)


if __name__ == "__main__":
  main()
