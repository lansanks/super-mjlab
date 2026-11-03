# SuperMimic（Unitree G1 × LAFAN1）工程架构

面向"已经跑通、现在要接手维护"的读者。回答三个问题：**谁在驱动谁**、**每一步数据从哪来**、
**和 `知识库/supermimic_vibed` 相比改了什么**。

---

## 一句话

SuperMimic 不是一个"接进 rsl_rl 的算法插件"，而是**一个自带完整 MDP 和训练循环的 runner**：
mjlab 在这里只被当作「GPU 仿真器 + 动作/指令/事件管理器 + 任务注册表」使用，
观测、奖励、终止、世界模型、策略、Critic、ReplayBuffer 全部由本目录自己实现，
rsl_rl 的 PPO 一行都不参与。

---

## 1. 三层视图

| 层 | 位置 | 你在这里要动的概率 |
|---|---|---|
| **A. 框架层**（仿真 + manager + 环境生命周期） | `src/mjlab/{sim,entity,scene,managers,envs,rl,scripts}` | 低。除非要新传感器/新 action term |
| **B. 任务注册层**（把任务挂到 `train`/`play` CLI 上） | `src/mjlab/tasks/registry.py` + 本目录 `__init__.py` | 低 |
| **C. SuperMimic 任务层**（MDP + 世界模型 + 训练循环） | 本目录 | **高。日常改动几乎都在这一层** |

层 C 对层 A 的依赖只有四个口子：

```
仿真状态读出    robot.data.root_link_pose_w / joint_pos / geom_friction / body_ipos
动作写入        action_manager (JointPositionAction, scale=G1_ACTION_SCALE)
重置与随机化    command_manager("motion") + event_manager(encoder_bias / foot_friction / base_com)
episode 边界    termination_manager (只在 cfg.terminations 里留了 4 项)
```

---

## 2. 启动链路（CLI → 对象图）

```
uv run train Mjlab-Tracking-SuperMimic-Unitree-G1
  │
  ├─ console script: train = mjlab.scripts.train:main
  ├─ import mjlab.tasks  ──► import_packages() 递归 import 所有 task 包
  │                          └─► tracking_supermimic/__init__.py
  │                              register_mjlab_task(env_cfg, play_env_cfg, rl_cfg, runner_cls)
  ├─ tyro 解析 TrainConfig  (env: ManagerBasedRlEnvCfg, agent: TrackingSuperMimicRunnerCfg)
  ├─ ManagerBasedRlEnv(cfg)              # 建 Scene → MjSpec → MjModel → MuJoCo-Warp；装 8 个 manager
  ├─ RslRlVecEnvWrapper(env)             # 只是让 runner 能拿到 env.unwrapped
  ├─ runner_cls = load_runner_cls(...)   # = TrackingSuperMimicRunner
  └─ runner.learn(max_iterations=320_000)
```

**运行期对象图**（谁持有谁）：

```
TrackingSuperMimicRunner
├── self.wrapper      RslRlVecEnvWrapper（训练中几乎不用）
├── self.env          = wrapper.unwrapped → ManagerBasedRlEnv
├── self.robot        env.scene["robot"]（Entity）
├── self.command      env.command_manager["motion"] → LafanMotionCommand
├── self.motion       command.motion → LafanMotionLibrary（NPZ 里的全部参考轨迹）
├── self.mdp          MDP(kinematics, motion, default_joint_pos)   ← 解析式 obs/reward/termination/critic 特征
├── self.kinematics   G1Kinematics（从 G1 MJCF 现搭的可微 FK，与仿真模型是两份）
├── self.world_model  WorldModel(133 → 70)
├── self.policy       Policy(goal 730 → z 32 → concat(obs_hist 990) → 29)
├── self.critic       Critic(914 → 1)
└── self.buffer       ReplayBuffer(1000 steps × num_envs)
```

---

## 3. 目录职责

| 文件 | 职责 | 对应参考实现 |
|---|---|---|
| `__init__.py` | 注册 task id、env cfg、runner cls | —（新增） |
| `constants.py` | 29 关节名、14 被跟踪 body、`CONTROL_DT=0.02`、默认 NPZ 路径 | `robot.py` 的一部分 |
| `env_cfgs.py` | 在 `unitree_g1_flat_tracking_env_cfg` 上改出 SuperMimic 环境配置 | `env.py` 的 `make_sim/TrackingSceneCfg` |
| `mdp.py` | 解析式观测 / 奖励 / 终止 / Critic 特征 / 状态展平 / ξ 归一化 | `mdp.py`（逻辑等价，仅风格） |
| `kinematics.py` | 读 G1 MJCF 现搭的**可微** FK，输出 14 个 body 的 p/R/v/ω | `kinematics.py`（等价） |
| `motion.py` | 多片段 NPZ 参考库 + `goal()` 未来窗口 + 1 s bin 表 | `motion.py`（`from_dir` 换成读 NPZ） |
| `motion_command.py` | **新的** CommandTerm：多片段自适应采样、重置写 sim、把 ghost 接到 viewer | 参考实现里由 `TrackingSim.reset/sample_frames` 承担 |
| `nets.py` | WorldModel / Policy / Critic / RunningNorm / amp 包装 | `networks.py`（等价） |
| `buffer.py` | `(time, env)` 布局的 GPU 环形缓冲区 | `buffer.py`（等价） |
| `rotations.py` | quat/mat/6D/so3/log/yaw 等批处理工具 | `rotations.py`（等价） |
| `runner.py` | **训练循环主体**：采集、WM 拟合、BPTT、Critic、日志、存档、推理策略 | `train.py` + `env.py` 合并重写 |
| `rl_cfg.py` | 超参 dataclass（`RslRlBaseRunnerCfg` 子类，只当参数包用） | `train.py` 的 argparse |
| `terminations.py` | 3 个 manager 侧终止项（anchor 高度/倾角、片段结束） | `env.py` 的 `timeout` 逻辑 |
| `scripts/convert_lafan1.py` | 40 段 30 Hz CSV → 单个 50 Hz NPZ（含 FK） | `motion.py` 的离线部分 |
| `scripts/evaluate.py` | 每段动作完整播一遍、算归一化回报 | `evaluate.py` |
| `README.md` / 本文件 | 使用说明 / 架构说明 | —（新增） |

---

## 4. 训练循环

### 4.1 一个 iteration 的骨架（`runner.py:learn`，L721）

```
iteration i:
  lr = cosine(2e-4 → 0.03× @250k)                       ← 三个 optimizer 共享
  step_acc += 1/utd(0.4); collect(int(step_acc))        ← 每 iter 采 2~3 个 sim step
  for _ in range(wm_batches=10): _world_model_update()  ← 10×2048 个 8 步窗口
  if i >= wm_only_iters(500):
      _policy_update()                                  ← 4096 条 × 32 步想象 + BPTT
  if i % 50 == 0: _log();  if i % 2000 == 0: save()
```

对应论文 Figure 2 的三个 panel：**collect → fit world model → improve policy**。
`wm_only_iters=500` 是"先让世界模型像样，再开策略"，`utd=0.4` 是"每 2.5 个仿真步做 1 次更新"。

### 4.2 采集一步的时序（`_collect_step`，L304）

```
 1. 观测          o_t = add_noise(mdp.obs(s_t, a_{t-1}, ξ, f_t))         # 99 维
 2. 历史          obs_hist ← where(fresh, o_t×10, shift(obs_hist, o_t))  # 10×99
 3. 目标          g_t = motion.goal(f_t, 10)                            # 730 维
 4. 归一化统计    policy.obs_norm.update(o_t); policy.goal_norm.update(g_t)
 5. 动作          a_t = π(obs_hist, g_t)  (+0.1·N(0,1)，clip ±20)
 6. 仿真          env.step(a_t)   ← 4 次 5 ms 物理子步（decimation=4）
 7. 状态/奖励     s_{t+1}, f_{t+1}；r = mdp.reward(s_{t+1}, f_{t+1}, a_t, a_{t-1})
 8. 入 buffer     (s, s1, a, a_prev, ξ, r, f, ep_step, done)
 9. 统计          wm.input_norm.update(...); wm.output_norm.update(...)
10. 手动 reset    done 的 env → env.reset(env_ids)；同步刷新 state/ξ/history/return
11. 写历史        obs_hist ← shift(obs_hist, obs(s_{t+1}, a_t, ξ, f_{t+1}))  ← 见 §8.1
```

> 训练时 `cfg.auto_reset = False`（`env_cfgs.py:107`），`env.step` 不做自动重置，
> 由第 10 步手动 reset。这是 mjlab 文档里给"自己写训练循环"的人准备的正规姿势。

### 4.3 想象 rollout 的时序（`_policy_update`，L451）

```
从 buffer 抽 4096 个"还能再走 32 帧"的起点 (s_0, f_0, ξ, a_{-1}) + 10 帧观测历史
冻结 WM 和 Critic 的梯度
for t in 0..31:                                  ← 全在 GPU 上，无仿真
    a_t     = π(obs_hist, g_t)
    s_{t+1} = WM.step(s_t, a_t, normalize(ξ))     ← 预测"heading 局部系下的状态增量"
    o_{t+1} = add_noise(mdp.obs(s_{t+1}, a_t, ξ, f_t+1))
    obs_hist← shift；g_{t+1} = motion.goal(f_{t+1})
J = Σ γ^t · c_t · r_t  +  γ^32 · c_32 · V(s_32, g_32, normalize(ξ))   # c_t = Π(1-d̂)
loss = -J.mean() → backward 穿过 32 步 WM（BPTT）→ clip(1.0) + clip_value(0.1) → Adam
用同一批 rollout 的 (V_t, r_t, d_t) 反算 GAE(λ=0.95) 目标 → Critic 拟合 5 epoch × 4 minibatch
```

**梯度只流到 π**：`_set_requires_grad(world_model, False)` 在 rollout 前后开合（L467/L554），
所以世界模型是"被冻结的可微环境"，这正是论文 §3.3 的 stop-gradient 约定。

---

## 5. 张量与维度（`constants/mdp/nets` 里硬编码，改维度会连环失效）

| 符号 | 维度 | 组成 |
|---|---|---|
| 状态 `s` | 71 | p(3) + quat(4) + v(3) + ω(3) + q(29) + q̇(29) |
| 不变特征 `invariant_features(s)` | 71 | z(1) + 6D 倾角(6) + 体坐标速度(3) + 体坐标角速度(3) + q(29) + qd(29) |
| 观测 `o` | 99 | (q−q₀−bias)(29) + q̇(29) + ω_body(3) + 重力_body(3) + a_prev(29) + 参考相对姿态 6D(6) |
| 目标 `g` | 730 | 10 帧 × [Δp(3) + 相对旋转 6D(6) + v_local(3) + ω_local(3) + q(29) + 0.1·q̇(29)] |
| 仿真上下文 `ξ` | 33 | 摩擦(1) + 关节编码器偏置(29) + 躯干质心偏移(3) |
| WorldModel 输入 | 133 | 不变特征 71 + 动作 29 + normalize(ξ) 33 |
| WorldModel 输出 | 70 | Δp(3) + so3_log(3) + Δv(3) + Δω(3) + Δq(29) + Δq̇(29) |
| Critic 输入 | 914 | 不变特征 71 + 113 + 目标 730 |
| Buffer 单条 | ~949 B | 1000 × 4096 × 949 B ≈ **3.6 GiB**（论文尺度，仅 buffer） |

`MAX_STEP_REWARD = (4×1.0 + 2×0.5) × 0.02 = 0.1`：满分行每步上限，用于把回报归一化到 ~[0,1]。

---

## 6. 与参考实现对照

| 参考实现 | 本工程 | 变化性质 |
|---|---|---|
| `train.py`（352 行，脚本式全局变量） | `runner.py` 的 `TrackingSuperMimicRunner`（类） | 重写为 mjlab 的 runner 协议：`__init__(env, cfg, log_dir, device)` / `learn` / `save` / `load` / `get_inference_policy` |
| `env.py::TrackingSim`（Isaac Lab 3 + Newton） | `motion_command.py`（重置/采样/写 sim） + `runner._collect_step`（观测/噪声/入 buffer） + `terminations.py` | 拆成"环境该管的"和"训练循环该管的"两半 |
| `env.py` 里手写 friction/com 写入 | mjlab 的 `event_manager`（`foot_friction` / `base_com` / `encoder_bias`） | 交给框架，`sim_context()` 只负责读回来 |
| `motion.py::MotionLib.from_dir`（启动时读 40 个 CSV 做 FK） | `scripts/convert_lafan1.py` 离线转 NPZ + `motion.py` 读 NPZ | 离线化，启动快、可复现、易上传 wandb |
| 单文件 motion + 自写 `sample_frames` | `LafanMotionCommand`（`MotionCommand` 子类）覆盖 `_resample_command` / `_update_command` / `gui` | **关键改造**：从"一条连续轨迹"变成"多片段 + 片段边界" |
| `robot.py`（G1 cfg / action scale / 关节序） | `mjlab.asset_zoo.robots.unitree_g1.g1_constants` | 复用，`G1_ACTION_SCALE` 同样是 `0.25·effort/σ_kp` |
| `evaluate.py` | `scripts/evaluate.py` | 逻辑相同，改用 task registry 取 cfg |
| — | `torch.compile` 10 个热点、viewer 片段下拉框、`tests/test_tracking_supermimic.py` | 本工程新增 |

**纯搬运、逻辑等价**的文件：`mdp.py`、`nets.py`、`buffer.py`、`rotations.py`、`kinematics.py`。
差异基本只有 2 空格缩进、类型注解、变量改名（`S→state`、`H→horizon`、`xi→sim_context`）。

---

## 7. 八条设计取舍

1. **为什么奖励/终止/观测不走 manager？**
   因为 SuperMimic 要把同一份解析式 MDP **同时用在两个地方**：真仿真一步，和世界模型里 32 步。
   manager 的 term 绑定在 `env` 上（读 `env.scene`、`env.command_manager`），没法在"想象的
   (s, f, a)"上求值。所以 `mdp.py` 写成纯函数式张量算子，`cfg.rewards = {}` 清空 manager 奖励。

2. **那 manager 还留着干什么？**
   `cfg.terminations` 仍是 episode 边界的唯一权威（高度/倾角/片段结束/超时）；
   `action_manager` 负责把策略输出变成 `q₀ + scale·a − bias` 的关节目标；
   `event_manager` 负责域随机化；`command_manager` 负责片段采样和步进。
   **奖励和观测归 runner，边界和动作归 manager** —— 这是本工程最需要记住的一条分界线。

3. **为什么保留 `cfg.observations`？**
   `ManagerBasedRlEnv` 需要至少一个观测组来构造空间、`play.py` 也要走 `VecEnv` 接口。
   训练时这些组会被算出来再丢掉（开销很小，因为 `enable_corruption=False`）。
   真正的策略输入是 `runner` 里自己拼的 `obs_hist`，**两者不是一回事**（见 §8.6）。

4. **为什么 `auto_reset=False`？**
   世界模型需要一个干净的 `(s, s1)` 转移；自动重置会让 `step()` 返回重置后的观测，
   把终止态吞掉。训练时手动 reset，`play` 时（`env_cfgs.py:107`）再打开自动重置。

5. **为什么世界模型预测"增量"而不是"下一状态"？**
   输入是 heading 局部系、去掉了全局 xy 和 yaw。人形机器人朝哪走、走多远不影响动力学，
   这样 40 段动作共享同一套动力学，样本效率高（论文 §3.2 的核心动机）。
   输出端 `Δp/Δθ/Δv/Δω` 都在体/航向系里，落地时再用 `heading`、`rot` 转回世界系（`nets.py:step`）。

6. **为什么 reward 里锚点被"投影"到参考高度？**
   `mdp.py:162` 的 `anchor = cat(p[..., :2], reference_p[..., 2:3])`：只跟踪 xy，
   z 用参考的，再叠一个 `yaw` 对齐的 `delta_rot`。这样"机器人跳起来"不会被当成"高度错了"，
   只比较**相对姿态**（BeyondMimic 的做法）。

7. **为什么有软终止？**
   `soft_termination` 用两条 sigmoid 代替硬阈值，让生存权重 `c_t` 在想象中对动作**可微**；
   硬阈值（`termination`）只在 manager 里用于真实 episode 边界。

8. **为什么 `torch.compile` 打的是"方法"而不是整个 runner？**
   一个 iteration 的瓶颈是 8 步 WM rollout 和 32 步想象里成百上千个小 kernel 的启动开销，
   编译 `WorldModel.step` / `Policy.forward` / `mdp.obs` / `motion.goal` 等 10 个热函数即可，
   `RunningNorm` 的 buffer 更新留在 eager 里。首次迭代有几秒编译预热。

---

## 8. 我发现的偏差（按影响排序，供你决定要不要改）

### 8.1 `obs_hist` 在每个采集步被 push 了两次（`runner.py:318` 和 `:392`）

- 步首 push 当前状态**带噪声**的观测；`env.step` 之后又 push 同一状态**不带噪声**的观测。
- 结果：策略看到的 10 帧历史 = 5 个状态、每个重复两遍、噪声/无噪声交替，有效时间窗减半。
- 参考实现只 push 一次（`train.py:273`，每步一个带噪观测，10 帧 10 个不同状态）。
- 而 `_policy_update` 从 buffer 重建历史（`:458-464`）是**10 帧无重复**的，推理（`:865-878`）也是
  10 帧无重复。于是"采集 / 想象 / 推理"三种历史分布互不相同。
- 建议：删掉 `:386-392` 与 `:412` 的重置写历史，只保留步首那次（`fresh` 已处理重置）。
  删之前建议先跑个 2k iter 的对照，确认对 `wm`/`sim_ret` 没有副作用。

### 8.2 Critic 的 bootstrap 用了未归一化的 ξ，而 GAE target 用了归一化 ξ

- 引导项 `:538-546` → `mdp.critic_features(state, frame, sim_context)` 传的是**原始 ξ**（摩擦 ≈0.3~1.2）
- 拟合目标 `:576-578` → `all_context = normalized_context...`（摩擦 ≈−1~1）
- Critic 的 `RunningNorm` 统计量是从**归一化**版本学来的，于是自举项喂进去的是分布外输入，
  这个值还要乘 `γ³²≈0.73` 参与策略梯度。
- 参考实现两处都用 `xi_n`（`train.py:209` 与 `:230`），所以这几乎肯定是搬运时的笔误。
- 建议：`:545` 处把 `sim_context` 换成 `normalized_context`。

### 8.3 World-model 损失的归一化分母不同

`runner.py:440` 用 `mask.sum()`（有效步数），参考实现 `train.py:108` 用 batch 大小。
mjlab 这版在窗口里有 done 时更"正确"，但和参考实现的数值不可逐位对比。知道即可。

### 8.4 训练 episode 被 10 s 截断，但奖励按"整段动作"归一化

`env_cfgs.py:106` 的 `episode_length_s=10.0` → 500 步上限，而 LAFAN1 片段普遍更长；
评测脚本为此把 `episode_length_s` 改成 `1e9`（`scripts/evaluate.py:37`）。
训练时看 `sim_ret` 会被这个截断影响，和历史 `train_log.csv` 对比时要注意。

### 8.5 `encoder_bias` 的符号约定与参考实现相反

- mjlab：`joint_pos_biased = q + bias`，而动作目标里 `target = q₀ + scale·a − bias`（`actions.py:223`），
  物理上稳态为 `q ≈ q₀ + scale·a − bias`。
- 本工程的观测仍沿用参考实现的 `q − q₀ − bias`，于是稳态观测相对"理想"多出 `−2·bias`。
- `|bias| ≤ 0.01 rad`，即偏差 ≤ `0.02 rad ≈ 1.1°`；因为 ξ 里带着它、世界模型可以吸收，
  实际影响很小。但要改成更"物理"的写法时，记得观测要改成 `q_biased − q₀`。

### 8.6 训练时 manager 的观测组是死代码

`cfg.observations` 每步都被 `ObservationManager.compute()` 算一遍（`env_cfgs.py:62-82`），
但 runner 从 `env.step()` 的返回值里只取 `terminated/time_outs`，观测丢弃。
不影响正确性；如果哪天要压低 CPU 开销可以考虑精简成最小一组。

### 8.7 文档与仓库垃圾

- `docs/source/changelog.rst` 里还写着已经在 `7db26cb` 删掉的
  `Mjlab-Velocity-SuperMimic-Unitree-Go1*`（两个变体），现在已无对应代码。
- 根目录 `supermimic-torch-compile.patch` 的内容已经被 `76472d7` 提交进代码，
  该文件是残留，可以直接删。
- `logs/rsl_rl/go1_velocity_supermimic*` 是已删任务的旧日志。

---

## 9. 常用命令与调试入口

```bash
# 1) 数据：40 段 CSV → 50 Hz NPZ（含 14 body FK）
uv run python -m mjlab.tasks.tracking_supermimic.scripts.convert_lafan1 \
  --input-dir data/lafan1_g1 --output data/lafan1_g1/motion.npz

# 2) 训练（论文尺度）
uv run train Mjlab-Tracking-SuperMimic-Unitree-G1

# 3) 小显存验证闭环（README 里那串覆盖参数）
uv run train Mjlab-Tracking-SuperMimic-Unitree-G1 \
  --env.scene.num-envs 128 --agent.wm-hidden 512 512 512 \
  --agent.policy-encoder-hidden 512 256 128 --agent.policy-decoder-hidden 512 512 512 \
  --agent.wm-batch 128 --agent.wm-batches 4 --agent.policy-batch 128 --agent.buffer-steps 500

# 4) 可视化（viser，带 ghost 和片段下拉框）
uv run play Mjlab-Tracking-SuperMimic-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_lafan1_tracking_supermimic/<run>/model_014000.pt \
  --motion-file data/lafan1_g1/motion.npz

# 5) 逐片段评测
uv run python -m mjlab.tasks.tracking_supermimic.scripts.evaluate \
  --checkpoint <ckpt> --motion-file data/lafan1_g1/motion.npz

# 6) 单测
uv run pytest tests/test_tracking_supermimic.py
```

**想加日志**：`_log()`（`runner.py:674`）是唯一出口，行字典的 key 就是 `train_log.csv` 的列。
**想改超参**：先看 `rl_cfg.py`（dataclass 默认值），CLI 覆盖走 tyro 的 `--agent.*`。
**想换机器人/动作库**：改 `constants.py` 的 `G1_JOINT_NAMES` / `TRACKED_BODIES`，
并确认 `motion.command` 里 `LafanMotionCommand.__init__` 的顺序断言通过。
