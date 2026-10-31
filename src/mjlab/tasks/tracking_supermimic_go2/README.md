# Unitree Go2 周期动作 SuperMimic

这个任务复用 G1 的 SuperMimic 训练闭环，跟踪桌面 `data/datasets` 中的
90 条 Go2 周期轨迹。周期类动作默认延拓到约 5 秒；空翻保留原始长度，
以避免首尾姿态不连续。

任务 ID：

```text
Mjlab-Tracking-SuperMimic-Unitree-Go2
```

## 转换动作

```bash
uv run python -m mjlab.tasks.tracking_supermimic_go2.scripts.convert_go2_cycles \
  --input-dir /home/linkchen/桌面/data/datasets \
  --output data/go2_cycles/motion.npz \
  --target-duration 5 \
  --drop-infeasible
```

转换器会完成：

- 100 Hz 到 50 Hz 重采样；
- 每腿 `[HFE, KFE, HAA]` 到 Go2 `[HAA, -HFE, -KFE]` 的映射；
- 根高度落地标定；
- 周期动作延拓；
- 13 个 Go2 body 的 FK、位置和速度；
- 关节限位、力矩和周期接缝检查。

`datasets_park` 中是早期试验数据，默认不作为输入。两条 1.5 m/s trot
超过 Go2 执行器能力，`--drop-infeasible` 会过滤这些片段。

## 训练

论文尺度默认配置：

```bash
uv run train Mjlab-Tracking-SuperMimic-Unitree-Go2 \
  --env.commands.motion.motion-file data/go2_cycles/motion.npz
```

8 GB 显存可以先使用小配置验证闭环：

```bash
uv run train Mjlab-Tracking-SuperMimic-Unitree-Go2 \
  --env.commands.motion.motion-file data/go2_cycles/motion.npz \
  --env.scene.num-envs 128 \
  --agent.wm-hidden 512 512 512 \
  --agent.policy-encoder-hidden 512 256 128 \
  --agent.policy-decoder-hidden 512 512 512 \
  --agent.wm-batch 128 \
  --agent.wm-batches 4 \
  --agent.policy-batch 256 \
  --agent.buffer-steps 500 \
  --agent.max-iterations 5000
```

## 播放和评测

```bash
uv run play Mjlab-Tracking-SuperMimic-Unitree-Go2 \
  --checkpoint-file <checkpoint> \
  --motion-file data/go2_cycles/motion.npz

uv run python -m mjlab.tasks.tracking_supermimic.scripts.evaluate \
  --task Mjlab-Tracking-SuperMimic-Unitree-Go2 \
  --checkpoint <checkpoint> \
  --motion-file data/go2_cycles/motion.npz
```

## ONNX 导出

SuperMimic runner 在每次保存 checkpoint 时会自动在同目录生成与运行目录同名的
ONNX 文件。例如：

```text
logs/rsl_rl/go2_cycles_tracking_supermimic/<run>/<run>.onnx
```

已有 checkpoint 可以单独导出：

```bash
uv run python -m mjlab.tasks.tracking_supermimic.scripts.export_onnx \
  --task Mjlab-Tracking-SuperMimic-Unitree-Go2 \
  --checkpoint <checkpoint.pt> \
  --motion-file data/go2_cycles/motion.npz \
  --device cpu
```

导出的策略接口是固定的：

```text
observation_history: [1, 10, 48]
goal:                [1, 390]
actions:             [1, 12]
```

观测和目标归一化 buffer 已包含在 ONNX 图中。

