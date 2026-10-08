# DreamWaQ 在 mjlab 中的复现指南

> 论文：*DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain
> Imagination via Deep Reinforcement Learning* (ICRA 2023)
> 原版代码：`paper-reading-note/dreamwaq/DreamWaQ/`（Isaac Gym + rsl_rl 1.0.2）
> 本仓库：mjlab（MuJoCo Warp + rsl_rl 5.5.1，manager-based API）

---

## 0. 结论先行

DreamWaQ 相对标准 PPO 只多了三样东西：

1. **CENet**：一个共享 encoder、两个 head（速度估计 + β-VAE 上下文 `z`）、一个
   decoder 的估计器网络；输入是 5 帧本体感受历史 `o_t^H`（225 维）。
2. **新增损失** `L_CE = L_est + L_VAE`，与 PPO 的 surrogate/value loss 一起反传。
3. **AdaBoot**：用 `p_boot = 1 - tanh(CV(R))` 决定这一轮策略是否"自举"（吃估计
   速度），早期训练改为吃仿真真值速度。

mjlab 本身已经提供了 90% 的地基：非对称 actor-critic、地形课程、域随机化、
PPO、观测历史缓冲、ONNX 导出。缺的只有上面三块，已经把可运行的骨架写进
`src/mjlab/tasks/dreamwaq/`：

| 文件 | 作用 |
| --- | --- |
| `src/mjlab/tasks/dreamwaq/env_cfg.py` | DreamWaQ 的 Go1 粗糙地形环境配置（观测/奖励/DR/地形） |
| `src/mjlab/tasks/dreamwaq/cenet.py` | `CENet` + `CENetActorModel`（rsl_rl 5.x 模型接口） |
| `src/mjlab/tasks/dreamwaq/dreamwaq_ppo.py` | `DreamWaQPPO`（`L_CE` + AdaBoot） |
| `src/mjlab/tasks/dreamwaq/rl_cfg.py` | RL 超参与类名注册（`class_name` 指向上面两个类） |
| `src/mjlab/tasks/dreamwaq/__init__.py` | 注册任务 `Mjlab-DreamWaQ-Rough-Unitree-Go1` |
| `src/mjlab/tasks/dreamwaq/mdp/rewards.py` | 论文 Table I 里 mjlab 没有的奖励项 |
| `tests/test_dreamwaq_task.py` | 不需要 GPU 的单元测试（观测布局 / CENet / 一次完整 PPO 更新） |

**已验证**（本机 CPU + MuJoCo Warp CPU 后端，2~4 个环境）：

* 环境可创建、可 step，观测形状 `actor=(N,5,45)`、`priv=(N,3)`、`critic=(N,259)`；
* 一次完整 rollout + PPO 更新，`cenet / cenet_est / cenet_rec / cenet_kl / pboot`
  都出现在日志里，`L_CE` 能反传到 CENet；
* checkpoint 保存/加载、`uv run play` 的推理路径、ONNX 导出（输入 `obs_history`
  225 维 → 输出 12 维动作）。

**未验证**（这台机器没有 NVIDIA 驱动，训练必须在你自己的 GPU 上跑）：真实收敛性、
地形课程推进速度、sim-to-real。

---

## 1. 复现路线图

| 阶段 | 目标 | 验收标准 |
| --- | --- | --- |
| M0 环境 | 跑通 `Mjlab-DreamWaQ-Rough-Unitree-Go1` | 观测形状对、奖励量级 ≈ 论文、地形/DR 生效 |
| M1 估计器 | 只训 CENet（可先冻结策略） | `L_est` 单调下降，估计速度与真值相关性 > 0.8 |
| M2 策略 | 打开联合训练 | episode return 上升、`pboot` 随训练上升 |
| M3 地形 | 打开地形课程 | terrain level 上升、能上下 10 级楼梯 |
| M4 部署 | 导出 ONNX 播放/上机 | 单次输入 225 维历史 → 12 维动作，行走稳定 |

---

## 2. 论文组件 → mjlab 落点

| 论文中的东西 | mjlab 里的对应物 | 说明 |
| --- | --- | --- |
| `o_t = [ω, g, c, θ, θ̇, a_{t-1}]`（45 维） | `observations["actor"]` 的 6 个 term | 去掉 mjlab 默认的 `base_lin_vel` 和 `height_scan` |
| `o_t^H`（H=5，225 维） | `ObservationGroupCfg(history_length=5, flatten_history_dim=False)` | 得到 `(N,5,45)`，帧序 old→new |
| `p_t = [ṽ_t, z_t, o_t]`（64 维） | `CENetActorModel.get_latent()` | 估计器输出拼在策略输入前 |
| `s_t = [o_t, v_t, d_t, h_t]` | `observations["critic"]` | 含真值速度、height_scan、脚部接触 |
| CENet（128×64×19 / 64×128×45） | `CENet` | encoder 128→64，速度头 3，latent 16，decoder 45 |
| `L_CE` | `CENetActorModel.cenet_loss()` | 加在 `DreamWaQPPO.update()` 里 |
| AdaBoot | `DreamWaQPPO._update_adaboot()` | 每轮算 `p_boot`，Coin flip 决定是否吃估计速度 |
| PPO（clip 0.2, γ 0.99, λ 0.95, lr 1e-3） | `rl_cfg.py` | 与论文一致 |
| 4096 envs × 24 steps × 1000 iters | `--env.scene.num-envs 4096` + `num_steps_per_env=24` + `max_iterations=1000` | 论文用 3060Ti 约 1 小时 |
| 地形：slope/rough/stairs up/down/discrete，10 级 | `dreamwaq_terrains_cfg()` | 按论文比例 0.1/0.1/0.35/0.35/0.1 |
| 域随机化 Table II | `env_cfg.py` 的 `events` | 负载 / Kp,Kd / 力矩上限 / CoM / 摩擦 / 延迟 |

---

## 3. 三块核心代码怎么读

### 3.1 观测：`env_cfg.py`

```python
"actor": ObservationGroupCfg(
  terms=proprio_terms,          # ω, g, c, θ, θ̇, a_{t-1}
  enable_corruption=not play,   # 训练加噪声，play 不加
  history_length=5,
  flatten_history_dim=False,    # -> (N, 5, 45)，oldest→newest
),
"priv":  真值 base_lin_vel（×2，"AdaBoot 的老师" + L_est 回归目标）
"critic": o_t + 真值速度 + height_scan + 脚部接触（无噪声）
```

几个容易忽略但必须对齐的点：

* **观测 scale**：`ω×0.25`、`θ̇×0.05`、命令 `×[2,2,0.25]`、`v×2.0`。这是
  legged_gym 的历史约定，也是论文里"所有观测零均值单位方差 → 先验取 N(0,I)"
  的前提，少了这步 KL 会乱。
* **噪声幅度**：mjlab 的管线是 `compute → noise → clip → scale`，所以这里写的
  `Unoise(-0.2, 0.2)`（角速度）在乘完 scale 后等效于 legged_gym 的 0.05，和
  原版一致。
* **奖励 dt**：mjlab 默认 `scale_rewards_by_dt=True`（奖励 × 0.02）。论文权重是
  **每步**的，所以这里显式设 `cfg.scale_rewards_by_dt = False`，否则所有惩罚项
  会缩小 50 倍。
* **延迟**：`[0, 15] ms` 由 actuator 的 `delay_max_lag = round(0.015 / dt)` 实现
  （Go1 任务物理步长 5 ms → 3 步），比在观测上加延迟更接近真实总线延迟。

### 3.2 CENet：`cenet.py`

```
o^H (225) ─► encoder 128 ─► 64 ─┬─► velocity_head ─► ṽ (3)   ── L_est (MSE vs 真值)
                                ├─► latent μ (16) / logσ² (16) ─► 采样 z (16)
                                └────────────────────────────────┐
                                       decoder 64 ─►128 ─► o_{t+1} (45)  ── L_rec
```

* 策略输入是 `[o_t, v_t, z_t]`，其中 `v_t` 训练时可能是真值（AdaBoot 决定），
  推理时永远是 `ṽ_t`。
* 采样用重参数化（`μ + σ·ε`），推理/导出用 `μ`（`encode_deterministic`），所以
  ONNX 是确定性的。
* 损失在 `cenet_loss()` 里：
  `L_CE = MSE(ṽ, v) + MSE(o_{t+1}, ô_{t+1}) + β·KL(q(z|o^H) ‖ N(0,I))`，
  KL 用逐样本对角高斯闭式解再取 mean。

**下一帧目标怎么进 buffer**：rsl_rl 的 `RolloutStorage.observations` 是一个
TensorDict，mini-batch generator 会对整个 TensorDict 做索引。于是
`DreamWaQPPO.process_env_step()` 在写入 transition 之前，把"下一时刻的最新帧"
塞进 `storage.observations["dwq_next_obs"]`。这样不需要自定义 storage 类，也
不需要改 rsl_rl。

### 3.3 DreamWaQPPO：`dreamwaq_ppo.py`

* `construct_algorithm()` 负责按 `class_name` 构造 CENet actor、MLP critic、
  storage，并给 storage 增加 `dwq_next_obs` 缓冲。
* `update()` 是 rsl_rl 5.5.1 `PPO.update` 的拷贝 + 三项改动：
  1. 每步多算一个 `L_CE` 并加到总 loss；
  2. 把 `cenet/cenet_est/cenet_rec/cenet_kl/pboot/bootstrap` 写进 loss_dict（会
     自动被 logger 打到 W&B/TensorBoard）；
  3. 更新前调用 `_update_adaboot()`。
* `_update_adaboot()`：
  `CV = std(R)/|mean(R)|`，`p_boot = clamp(1 - tanh(CV))`，然后**每轮抛一次硬币**；
  硬币为"自举"时 `actor.use_privileged_velocity=False`（吃估计值），否则为
  `True`（吃仿真真值速度）。论文没写清是 per-iteration 还是 per-env，这里做成
  per-iteration（`adaboot_cfg.smoothing` 可选做 EMA 平滑）。
* `adaboot_cfg={"enabled": False}` 就是论文的消融 **DreamWaQ w/o AdaBoot**
  （恒吃估计值）。

---

## 4. 怎么跑

> 训练需要 CUDA GPU（mjlab 只在 macOS/CPU 上支持评估）。

```bash
# 0) 先在 CPU 上做无仿真单元测试（秒级）
uv run pytest tests/test_dreamwaq_task.py -q

# 1) 2 轮冒烟训练（确认环境、显存、日志都正常）
uv run train Mjlab-DreamWaQ-Rough-Unitree-Go1 \
  --env.scene.num-envs 64 --agent.max-iterations 2 \
  --agent.logger tensorboard --agent.upload-model False

# 2) 正式训练（论文配置）
uv run train Mjlab-DreamWaQ-Rough-Unitree-Go1 \
  --env.scene.num-envs 4096 --agent.max-iterations 1000

# 3) 回放 / 录像（--checkpoint-file 指向 logs/rsl_rl/go1_dreamwaq/<run>/model_1000.pt）
uv run play Mjlab-DreamWaQ-Rough-Unitree-Go1 \
  --checkpoint-file logs/rsl_rl/go1_dreamwaq/<run>/model_1000.pt

# 4) 也可以用内置 dummy agent 检查 MDP（零动作/随机动作）
uv run play Mjlab-DreamWaQ-Rough-Unitree-Go1 --agent zero
uv run play Mjlab-DreamWaQ-Rough-Unitree-Go1 --agent random
```

观测/动作维度自检：actor `(N,5,45)`、priv `(N,3)`、critic `(N,259)`；
动作 12 维；ONNX 输入 `obs_history` 225 维。

---

## 5. 分层验证清单（照这个顺序 debug 最快）

**M0 环境**

1. `uv run pytest tests/test_dreamwaq_task.py -q` 必须全绿。
2. `uv run play Mjlab-DreamWaQ-Rough-Unitree-Go1 --agent zero` 看机器人是否
   站立/下落自然（说明 PD、默认姿态、地形都对）。
3. 看 reward 曲线：tracking 项应接近 1（命令 0 时），`base_height`、`lin_vel_z`
   量级应为 1e-2~1e0，而不是 1e-5（否则就是 dt 缩放没关）。

**M1 估计器**

在 `cenet_loss()` 里临时把 `β` 拉大/关掉 KL 观察各项趋势；正常情况：
`cenet_est` 快速降到 < 0.05（×2 scale 后速度量级约 1 m/s），`cenet_rec` 缓降，
`cenet_kl` 稳定在个位数。若 `cenet_kl` 一开始就爆大，多半是观测没有 scale 到
单位量级。

**M2 策略**

`surrogate` 应震荡、`value` 下降、`entropy loss`（动作分布熵，不是 CENet 的项）
随训练下降；`pboot` 应从 ~0.5 缓慢上升到 ~0.9+，`bootstrap` 列显示这一轮是否
吃了估计值（1=吃估计，0=吃真值）。如果 `pboot` 长期在 0 附近，说明各环境回报
差异大，多半是训练不稳定（或 CV 的定义需要调整）。

**M3 地形**

在 W&B 里看 terrain level/难度分布；`curriculum["terrain_levels"]` 用的是
mjlab 的 `terrain_levels_vel`（走够距离才升级，= legged_gym 的 game-inspired
curriculum）。

**M4 部署**

ONNX 输入名 `obs_history`，形状 `(1,225)`，顺序是 `[o_t, o_{t-1}, …, o_{t-4}]`
（最新帧在最前，代码里用 `flip(1)` 保证）；输出 `actions` 12 维。上机时注意：
历史缓冲、观测 scale、动作 scale（`GO1_ACTION_SCALE`）、PD 增益、控制频率
（50 Hz）都必须与训练一致。

---

## 6. 与原版 Isaac Gym 代码的差异（复现的"坑"清单）

| 项 | 原版 DreamWaQ | 本复现 | 影响 |
| --- | --- | --- | --- |
| 奖励缩放 | legged_gym 每步权重 | `scale_rewards_by_dt=False` + 论文权重 | 不关会差 50 倍 |
| decoder 目标 | 代码里 `decode_target = obs_batch`（历史已含 `o_t`，存在信息泄漏） | 严格的 `o_{t+1}` | 更符合论文 Eq.(7)，但也更容易欠拟合 |
| 速度 logvar | 原版 `logvar_vel = self.encode_mean_vel(...)`（应是笔误，复用了 mean 头） | 只用确定性速度头 | 与原版行为等价但更干净 |
| KL 项 | `-0.5*sum(...)` 再整体 `/num_mini_batches` | 逐样本 KL 后取 mean | 数值量纲不同，β 需要按需调整 |
| AdaBoot | 公开代码里没实现 | 论文公式的 per-iteration 实现 | 需要自己验证消融 |
| 仿真器 | Isaac Gym（trimesh 地形、PhysX 接触） | MuJoCo Warp（heightfield/primitive） | 接触/摩擦不同，奖励 target（如 foot clearance）需重调 |
| 机器人 | Unitree A1（Kp=20/Kd=0.5 训练，真机 28/0.7） | Unitree Go1（mjlab 自带 MJCF + 电机模型） | 想完全对齐需要移植 A1 MJCF（见下） |
| 地形种类 | smooth/rough slope, stairs up/down, discrete | 同名比例已实现（`dreamwaq_terrains_cfg`） | 曲线形态会不同 |
| 扰动 | 有随机扰动力 `d_t`（privileged obs） | mjlab 的 `push_robot` interval 事件 | 可关掉做干净对照 |

另外两个工程细节：

* `H=5` 表示"5 帧共 225 维"（`[o_t..o_{t-4}]`），不是 6 帧。
* mjlab 的 `body_mass` DR 会打印一条警告（只改质量不改惯量）。论文的
  "payload [-1,2] kg" 就是"在 CoM 处加一个点质量"，用它是合理的；想物理更严谨
  可以换 `dr.pseudo_inertia(alpha_range=...)`。

---

## 7. 想更进一步

1. **换成 A1**：把 A1 的 MJCF 放进 `src/mjlab/asset_zoo/robots/`，改
   `INIT_STATE`、`GO1_ACTION_SCALE` 类似的动作缩放，`BASE_HEIGHT_TARGET` 改到
   0.25 左右。
2. **做消融实验**：`adaboot_cfg.enabled=False`（w/o AdaBoot）、去掉 latent 头只留
   速度估计（EstimatorNet baseline）、去掉估计器（Baseline）。
3. **复现论文的对比曲线**：同一 seed、同一地形课程、同一奖励，比较
   1e8 步内的 episode return 与最大推力/存活率。
4. **加视觉/地形先验**：mjlab 的 `terrain_scan` 已经在 critic 里，把 raycast 观测
   接到 actor 就变成"oracle"上界，可与 DreamWaQ 对比。
5. **上真机**：导出 ONNX + 维护 5 帧历史缓冲 + 50 Hz 推理；注意部署侧历史顺序
   与观测 scale 必须逐项对齐（最容易错的是命令 ×2、角速度 ×0.25）。

---

## 8. 参考

* 论文 PDF / 笔记：`paper-reading-note/dreamwaq/`
* 原版代码：`paper-reading-note/dreamwaq/DreamWaQ/{legged_gym,rsl_rl-1.0.2}`
* mjlab 任务模板：`src/mjlab/tasks/velocity/config/go1/`
* rsl-rl 5.5.1 模型接口：`.venv/lib/python3.13/site-packages/rsl_rl/models/mlp_model.py`
