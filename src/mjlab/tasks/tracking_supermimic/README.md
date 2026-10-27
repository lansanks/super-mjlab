# G1 LAFAN1 全身跟踪（SuperMimic）

这个任务把 `~/桌面/知识库/supermimic_vibed` 中的 G1 LAFAN1 实验迁移到
mjlab，同时保留 SuperMimic 的模型式训练闭环：

1. 在 mjlab G1 tracking 环境中采集少量真实物理转移；
2. 用短自回归窗口拟合确定性的世界模型；
3. 在世界模型内部想象 32 步，并通过 BPTT 更新策略；
4. 用想象奖励和 GAE 风格目标训练 Critic。

任务 ID：

```text
Mjlab-Tracking-SuperMimic-Unitree-G1
```

## 动作转换

输入是 G1-retargeted LAFAN1 的 40 段 CSV。先把它们转换成包含片段边界和
14 个身体 FK 轨迹的单个 50 Hz NPZ：

```bash
uv run python -m mjlab.tasks.tracking_supermimic.scripts.convert_lafan1 \
  --input-dir data/lafan1_g1 \
  --output data/lafan1_g1/motion.npz
```

转换器会执行：

- 30 Hz 到 50 Hz 重采样；
- 根节点四元数球面插值；
- 根节点和关节位置的差分速度；
- 用 G1 MJCF 计算 14 个被跟踪身体的 FK 轨迹；
- 保存每个动作片段的起止帧，供重置和评测使用。

快速验证时可以使用 `--limit-clips 1`。

LAFAN1 数据使用 Ubisoft La Forge 的 CC BY-NC-ND 4.0 许可，仓库不会下载或
重新分发动作数据。

## 训练

默认配置已经对齐参考复现：

- 4096 个并行环境；
- World Model `2048 × 2048 × 2048`；
- Policy `2048 × 1024 × 512 × 256`；
- 每次更新 10 个 WM batch，每个 batch 2048；
- 4096 条想象轨迹，长度 32；
- ReplayBuffer 1000 步；
- 总迭代 320,000。

动作文件默认指向 `data/lafan1_g1/motion.npz`，所以论文尺度直接运行：

```bash
uv run train Mjlab-Tracking-SuperMimic-Unitree-G1
```

参考实现使用 RTX PRO 6000 级别显卡。8 GB RTX 4060 不能使用论文尺度网络和
batch，必须显式覆盖为小显存版本：

```bash
uv run train Mjlab-Tracking-SuperMimic-Unitree-G1 \
  --env.scene.num-envs 128 \
  --agent.wm-hidden 512 512 512 \
  --agent.policy-encoder-hidden 512 256 128 \
  --agent.policy-decoder-hidden 512 512 512 \
  --agent.wm-batch 128 \
  --agent.wm-batches 4 \
  --agent.policy-batch 128 \
  --agent.buffer-steps 500
```

小显存版本不是论文原参数，只是用于验证训练闭环；论文结果使用上面的
4096 环境、2048 宽网络配置。

## 评测

```bash
uv run python -m mjlab.tasks.tracking_supermimic.scripts.evaluate \
  --checkpoint logs/rsl_rl/g1_lafan1_tracking_supermimic/<run>/model_014000.pt \
  --motion-file data/lafan1_g1/motion.npz
```

每段动作从第 0 帧完整播放一次。高度或倾斜超过训练终止阈值会记为失败；
正常播放到片段结尾属于 timeout，不记失败。

## 与参考仓库的主要差异

- 真实仿真直接使用 mjlab + MuJoCo-Warp，不再经过 Isaac Lab/Newton。
- 机器人直接使用 mjlab 自带的 G1 MJCF，不需要 URDF 转 USD。
- 为匹配参考跟踪设置，训练关闭自碰撞。
- 世界模型上下文 `xi` 包括脚部摩擦、关节编码器偏置和躯干质心偏移。
- 策略、世界模型、ReplayBuffer、解析奖励和可微 FK 都放在本任务目录内，
  不修改已有 mjlab 任务。
