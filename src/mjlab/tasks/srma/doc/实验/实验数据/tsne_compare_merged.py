"""Compare DreamWaQ, HIM and SRMA latents after merging terrain classes.

The merged evaluation removes flat terrain, combines ascending/descending
stairs, and combines ascending/descending slopes. Rough and wave remain
separate, leaving four balanced classes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tsne_latent import neighbour_purity, pca_metrics, tsne  # noqa: E402

MERGED_GROUPS = {
  "stairs": ("pyramid_stairs", "pyramid_stairs_inv"),
  "slope": ("hf_pyramid_slope", "hf_pyramid_slope_inv"),
  "rough": ("random_rough",),
  "wave": ("wave_terrain",),
}
MERGED_NAMES = tuple(MERGED_GROUPS)
COLORS = ("#d62728", "#1f77b4", "#8c564b", "#e377c2")


def prepare_data(
  path: pathlib.Path, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
  data = np.load(path, allow_pickle=True)
  latent = np.asarray(data["latent"], dtype=np.float64)
  labels = np.asarray(data["labels"])
  type_names = [str(value) for value in data["type_names"]]
  env_ids = np.asarray(data["env_ids"])

  indices_by_name = {name: i for i, name in enumerate(type_names)}
  merged_indices: list[np.ndarray] = []
  for group in MERGED_NAMES:
    source_labels = [indices_by_name[name] for name in MERGED_GROUPS[group]]
    merged_indices.append(np.flatnonzero(np.isin(labels, source_labels)))
  per_group = min(len(indices) for indices in merged_indices)
  rng = np.random.default_rng(seed)
  selected: list[np.ndarray] = []
  merged_labels: list[np.ndarray] = []
  for group_index, indices in enumerate(merged_indices):
    chosen = rng.choice(indices, size=per_group, replace=False)
    selected.append(chosen)
    merged_labels.append(np.full(per_group, group_index, dtype=np.int64))
  chosen = np.concatenate(selected)
  terrain = np.concatenate(merged_labels)
  order = rng.permutation(len(chosen))
  chosen = chosen[order]
  terrain = terrain[order]
  label_lookup: dict[int, int] = {}
  for merged_index, group in enumerate(MERGED_NAMES):
    for name in MERGED_GROUPS[group]:
      label_lookup[indices_by_name[name]] = merged_index
  remapped = np.array([label_lookup[int(label)] for label in labels[chosen]])
  assert np.array_equal(remapped, terrain)
  metrics = {
    "n": int(len(chosen)),
    "chance": 1.0 / len(MERGED_NAMES),
    **pca_metrics(latent[chosen]),
    "raw_1nn_terrain": neighbour_purity(latent[chosen], remapped, k=1),
    "raw_5nn_terrain": neighbour_purity(latent[chosen], remapped, k=5),
  }
  return latent[chosen], remapped, env_ids[chosen], metrics


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("npz", nargs="+", help="three tsne_latent.py output files")
  parser.add_argument("--labels", default=None, help="comma-separated panel labels")
  parser.add_argument("--out", required=True, type=pathlib.Path)
  parser.add_argument("--tsne-iters", type=int, default=1000)
  parser.add_argument("--perplexity", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  paths = [pathlib.Path(value) for value in args.npz]
  labels = (
    args.labels.split(",")
    if args.labels
    else [path.stem for path in paths]
  )
  if len(labels) != len(paths):
    raise SystemExit("--labels must match the number of input files")

  fig, axes = plt.subplots(
    1,
    len(paths),
    figsize=(6.4 * len(paths), 5.4),
    dpi=180,
    squeeze=False,
  )
  summary: dict[str, Any] = {
    "merged_classes": list(MERGED_NAMES),
    "source_groups": MERGED_GROUPS,
    "perplexity": args.perplexity,
    "tsne_iterations": args.tsne_iters,
    "panels": {},
  }
  saved: dict[str, np.ndarray] = {}
  for index, (ax, path, label) in enumerate(
    zip(axes[0], paths, labels, strict=True)
  ):
    latent, terrain, env_ids, metrics = prepare_data(path, args.seed)
    print(
      f"[INFO] {label}: n={len(latent)}, raw 1-NN={metrics['raw_1nn_terrain']:.3f}",
      flush=True,
    )
    embedding = tsne(
      latent,
      perplexity=args.perplexity,
      n_iter=args.tsne_iters,
      seed=args.seed,
    )
    metrics["tsne_1nn_terrain"] = neighbour_purity(embedding, terrain, k=1)
    metrics["tsne_5nn_terrain"] = neighbour_purity(embedding, terrain, k=5)
    metrics["tsne_1nn_environment"] = neighbour_purity(
      embedding, env_ids, k=1
    )
    for class_index, (name, color) in enumerate(
      zip(MERGED_NAMES, COLORS, strict=True)
    ):
      mask = terrain == class_index
      ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        s=7,
        alpha=0.72,
        color=color,
        label=f"{name} (n={int(mask.sum())})",
        linewidths=0,
      )
    ax.set_title(
      f"{label}\n"
      f"t-SNE 1-NN terrain {metrics['tsne_1nn_terrain']:.2f}, "
      f"raw 1-NN {metrics['raw_1nn_terrain']:.2f} "
      f"(chance {metrics['chance']:.2f})",
      fontsize=9,
    )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    summary["panels"][label] = metrics
    saved[f"embedding_{index}"] = embedding
    saved[f"terrain_{index}"] = terrain
    saved[f"env_ids_{index}"] = env_ids
    saved[f"latent_{index}"] = latent
    saved[f"name_{index}"] = np.array(label)

  axes[0][-1].legend(fontsize=8, loc="best", framealpha=0.9)
  axes[0][0].figure.suptitle(
    "Level-5 latent t-SNE after removing flat and merging terrain direction",
    fontsize=12,
  )
  fig.tight_layout()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, bbox_inches="tight")
  plt.close(fig)
  np.savez(args.out.with_suffix(".npz"), **saved)  # type: ignore
  summary_path = args.out.with_suffix(".json")
  summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
  print(json.dumps(summary, indent=1))
  print(f"[INFO] saved {args.out}")
  print(f"[INFO] saved {args.out.with_suffix('.npz')}")
  print(f"[INFO] saved {summary_path}")


if __name__ == "__main__":
  main()
