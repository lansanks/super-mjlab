# DreamWaQ 实验数据（从 TensorBoard event 文件导出的标量）

每个 run 一个子目录，内容：

```
<run>/metrics.csv         宽表：第一列 step，之后每个标量 tag 一列；缺失处留空
<run>/summary.csv         每个 tag 的 n / first_step / last_step / first / last / min / min_step / max / max_step
<run>/source.txt          来源目录、event 文件名、导出时间、tag 数与迭代区间
<run>/config_agent.yaml   该 run 的训练配置（β、AdaBoot、迭代数、种子……）
<run>/config_env.yaml     该 run 的环境配置（地形、num_envs、观测组……）
<run>/code_state.diff     run 启动时 runner 记录的 git diff（当时的工作区代码状态）
```

## 刷新

```bash
# 全部 run（会覆盖同名目录里的 CSV）
uv run python src/mjlab/tasks/dreamwaq/doc/实验/实验数据/export_scalars.py

# 只导某几个
uv run python src/mjlab/tasks/dreamwaq/doc/实验/实验数据/export_scalars.py \
  2026-10-12_10-29-54

# 重新生成 figures/ 下的跨 run 对照图
uv run python src/mjlab/tasks/dreamwaq/doc/实验/实验数据/plot_compare.py
```

- 在仓库根目录执行；数据源是 `logs/rsl_rl/go1_dreamwaq/<run>/events.out.tfevents.*`。
- **训练还在进行时导出的是快照**，隔一段时间重跑即可。
- checkpoint（`model_*.pt`）和导出的 ONNX 不在这个目录里，仍在
  `logs/rsl_rl/go1_dreamwaq/<run>/`。
- `2026-10-08_22-03-36` 是修复前的早期冒烟（1 轮、14 个 tag），仅留档，未纳入对比。
- `figures/` 里的对照图由 `plot_compare.py` 生成，改 run 列表或迭代窗口改脚本顶部的 `RUNS` 即可。
- CENet 梯度路径（吃/不吃 PPO 梯度）的 A/B 对照图由
  `plot_grad_routing.py` 生成（`figures/grad_routing.png`），数据为
  `2026-10-12_13-02-09` 与 `2026-10-12_14-17-48` 两个 run。

## 读法与画图

指标口径、各 run 的平台值对照见 `../实验记录.md`（第 3、4 节）。最小画图示例：

```python
import csv, pathlib, matplotlib.pyplot as plt

DATA = pathlib.Path("src/mjlab/tasks/dreamwaq/doc/实验/实验数据")


def load(run, tag):
  rows = list(csv.DictReader(open(DATA / run / "metrics.csv")))
  return [int(r["step"]) for r in rows if r[tag]], [
    float(r[tag]) for r in rows if r[tag]
  ]


for run, label in [("2026-10-12_10-10-21", "β=0"), ("2026-10-12_10-29-54", "β=0.1")]:
  plt.plot(*load(run, "Loss/cenet_kl"), label=label)
plt.xlabel("iteration")
plt.ylabel("Loss/cenet_kl")
plt.legend()
plt.show()
```
