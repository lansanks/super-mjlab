# CENet 接入 DreamWaQ 训练链路：实施方案

> 本文只描述"怎么改"，不含任何代码改动。
> 依据只有两处：论文 `src/mjlab/tasks/dreamwaq/doc/dreamwaq.pdf` 和当前仓库代码。
> 当前仓库的 actor 观测组（45 维）和 critic 观测组（特权观测）保持现状，不要求与论文完全一致。

---

## 0. 口径确认（先定死，后面全部按这个算）

| 项目 | 结论 | 依据 |
| --- | --- | --- |
| Actor 网络输入 | **64 = 45 + 3 + 16** | 论文 II-B.1：`πφ(at \| ot, vt, zt)`；Fig.1 里 `Current obs (o_t)` 的箭头同样汇入 Actor。45 = 当前帧本体观测，3 = v̂，16 = z |
| CENet 编码器输入 | **270 = 6 × 45** | `o_t^H = [o_t, o_{t-1}, ..., o_{t-5}]`，H = 5 表示"过去 5 帧 + 当前 1 帧"，共 6 帧（此前按 5 帧 225 的理解作废） |
| 编码器原始输出 | **35 = 3 + 16 + 16** | 3 = v̂；16 = μ；16 = logσ²。论文 Fig.2 标注的 "128×64×19" 是概念输出（v̂ + z），后验参数单列 |
| 解码器输出 | **45** | 重建下一帧观测 `õ_{t+1}`；论文 Fig.2 标注的 "48" 按笔误处理 |
| Actor MLP | 64 → 512 → 256 → 128 → 12 | 论文 Fig.1 "Policy Network (Actor) 512×256×128×12" |
| Actor / Critic 观测组 | 保持现状 | 本仓库 actor = 45 维、critic = 特权观测；新增内容一律放进独立观测组，不碰这两个组 |

---

## 1. 一句话概括

**CENet 不做成一个单独训练的模块，而是直接挂在 Actor 模型内部**：

- 每个 control step，Actor 前向时顺带跑一遍 CENet 编码器，拿到 `v̂` 和 `z`，拼成 `[o_t, v̂, z]` 再进 MLP 出动作；
- 训练时 CENet 沿 PPO 一起更新：策略梯度经过 `[o_t, v̂, z]` 回传到编码器（v̂、z 不 detach），辅助损失 `L_CE = L_est + L_VAE` 与 PPO 的损失相加后一次 backward、同一个优化器更新（论文未直接说明梯度路径，这是本复现的实现选择，见 4.2）；
- 因为 CENet 是 Actor 的子模块，它天然被同一个 Adam 优化器、同一份 `actor_state_dict`、同一个 ONNX 导出覆盖，不需要给 rsl_rl 动存储结构、也不需要第二套优化器。

---

## 2. 数据流总览

```
环境每个 control step 产出 5 个观测组（全部会被 RolloutStorage 逐步存下来）：

  actor        (45)   当前帧本体观测，喂 Actor 的 o_t 分支（训练时带噪声）
  critic       (...)  特权观测（不变）
  cenet_input  (270)  6 帧本体观测历史 o_t..o_{t-5}，只喂 CENet 编码器
  cenet_target_o (45) 干净的本体观测，训练时错一格当作 o_{t+1} 的重建目标
  cenet_target_v (3)  仿真真值基座线速度 v_t，只用于 L_est

                              │
                              ▼
                  RolloutStorage（按 (时间, 环境) 存全部观测组）
                              │
                              ▼
   ┌──────────────────────── PPO update 的每个 minibatch ────────────────────────┐
   │                                                                              │
   │  o^H(270) ──→ CENet 编码器 270→128→64 ──┬──→ v̂ (3)                            │
   │                                         └──→ μ, logσ² (16+16) ──→ z = μ+σε   │
   │                                                                              │
   │  [ o_t(45), v̂(3), z(16) ] = 64 ──→ MLP 512→256→128 ──→ 12 维动作分布         │
   │                                                    │                         │
   │                                                    ├─→ PPO loss（surrogate + value - entropy）
   │                                                    │                         │
   │  z(16) ──→ CENet 解码器 16→64→128 ──→ õ_{t+1}(45)  │                         │
   │                                                    │                         │
   │  L_est = MSE(v̂, v_gt)                             │                         │
   │  L_VAE = MSE(õ_{t+1}, o_{t+1}) + β·KL(q(z|o^H) ‖ N(0,I))                     │
   │                                                    │                         │
   │  total = PPO loss + λ·(L_est + L_VAE) ─────────────┘                         │
   │                              │                                               │
   │                              ▼                                               │
   │        一次 backward → 一个 Adam 更新（actor + CENet + critic 全在一起）       │
   └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 环境侧：新增 3 个观测组（只增不改）

改动文件：`src/mjlab/tasks/dreamwaq/mdp/observations.py`、`src/mjlab/tasks/dreamwaq/dreamwaq_env_cfg.py`。

先加一个函数 `dreamwaq_proprio(env)`：按当前 actor 组的同一顺序拼出 45 维的本体观测向量（角速度、重力投影、关节角、关节速度、上一步动作、速度指令），保证 CENet 看到的 `o_t` 和 Actor 看到的 `o_t` 是同一份数据。

然后加三个观测组（现有的 `actor`、`critic` 组一个字不动）：

| 观测组 | 内容 | 形状 | 关键配置 |
| --- | --- | --- | --- |
| `cenet_input` | `dreamwaq_proprio` 一个 term | (N, 270) | `history_length=6`、`flatten_history_dim=True`、训练时 `enable_corruption=True`（噪声范围照抄 actor 各项） |
| `cenet_target_o` | 同一个 term，干净 | (N, 45) | `enable_corruption=False` |
| `cenet_target_v` | `base_lin_vel`（`robot.data.root_link_lin_vel_b`，仿真真值） | (N, 3) | `enable_corruption=False` |

三个要点：

1. **用单个 term + history，而不是复制 6 个 term**。mjlab 的 history 缓冲按"旧 → 新"排列，单个 45 维 term 配 `history_length=6` 展平后是 `[o_{t-5}, ..., o_t]`（按帧拼接）；模型里 flip 一下就是论文的 `[o_t, ..., o_{t-5}]` 顺序。如果复制 6 个 term，会变成"按 term 分段"的拼接顺序，不是论文的顺序。
2. **episode 重置行为**：mjlab 在每条 episode 重置时用首帧回填整段历史，前 5 步不会串到上一条 episode 的数据，这个语义符合论文，不用自己写 buffer。
3. 这三个组**不路由给任何网络**，只是被 RolloutStorage 存下来，给辅助损失当输入和目标用。

---

## 4. 模型侧：CENet + Actor 合成一个模型

建议新文件：`src/mjlab/tasks/dreamwaq/rl/cenet.py`。

### 4.1 `CENet(nn.Module)`

- 编码器主干：270 → 128 → 64（ELU），输出头 64 → 35，拆成 v̂(3)、μ(16)、logσ²(16)；
- 采样：`z = μ + exp(0.5·logσ²) ⊙ ε`（重参数化，训练时采样）；
- 解码器：16 → 64 → 128 → 45（最后一层线性）；
- KL 用对角高斯的闭式解，只加在 z 这一路，v̂ 那一路没有任何正则。

### 4.2 `DreamwaqActorModel(MLPModel)`

关键决定：**继承 rsl_rl 的 `MLPModel`，只覆写需要的钩子**，动作分布（GaussianDistribution）、log_prob、KL、save/load、optimizer 接线全部复用现成逻辑。要覆写的只有四处：

| 覆写点 | 做什么 |
| --- | --- |
| `_get_latent_dim()` | 返回 45 + 3 + 16 = 64（本来只返回 45） |
| `get_latent(obs)` | 从 `obs["actor"]` 取 o_t 并归一化；从 `obs["cenet_input"]` 取 270 维历史 → CENet 编码器 → v̂、z；返回 `cat([o_t, v̂, z])`。同时把 (v̂, μ, logσ², z) 缓存到实例上给损失用 |
| `update_normalization(obs)` | 除了 actor 自己的 45 维 normalizer，还要更新 CENet 的 45 维 normalizer（把 270 维历史 reshape 成 (·, 6, 45) 后逐帧统计） |
| `as_onnx()` / `as_jit()` | 导出时必须把 CENet 一起装进去，否则默认导出器只会导出一个"没有 CENet 的空壳" |

推理/导出接口建议：**输入 270 维历史 → 内部取最新一帧当 o_t → 输出 12 维动作均值**。部署端只需要维护一个 45 维 × 6 帧的 ring buffer。

一个重要细节（本复现已拍板）：**v̂、z 不 detach，CENet 沿 PPO 一起更新**。策略梯度会经过 `[o_t, v̂, z]` 回传到编码器；`L_CE` 与 PPO 损失相加后一次 backward、同一个优化器更新，编码器同时服务"估计准"和"策略回报"两个目标（端到端）。

需要如实标注的是：论文对"策略梯度是否回传编码器"没有任何直接表述——II-C 只说 CENet 由 `L_CE` 优化，III-B 只说三个网络同步训练。因此"一起更新"是本复现的实现选择，不是论文结论；按此实现即可，不再做 detach 开关或 A/B。

---

## 5. 算法侧：损失函数在哪一步接进来

建议新文件：`src/mjlab/tasks/dreamwaq/rl/algorithm/dreamwaq_ppo.py`，`DreamwaqPPO(PPO)`。

### 5.1 rollout 阶段（收集数据）

每个 control step：

1. 环境算出 5 个观测组；
2. `alg.act(obs)`：Actor 前向 —— CENet 只需要编码器（此处不跑解码器），得到 `[o_t, v̂, z]`，出动作分布并采样动作；critic 照常算 V；
3. `env.step(action)` 得到 `o_{t+1}`、reward、done；
4. 把这一步的全部观测组、动作、log_prob、V 存进 storage。

### 5.2 update 阶段（算损失）

分三步：

**第一步：准备重建目标（覆写 `compute_returns`）**

训练需要一个 `o_{t+1}` 作为重建目标，但它不能放在"当前步"里（当前步只有 o_t）。做法：

1. storage 里 `cenet_target_o` 记录了每个 transition 的干净 o_t，形状 (T, N, 45)；
2. 沿时间维整体前移一格，最后一行用 `compute_returns(obs)` 拿到的当前观测补上 —— 它正好是最后一步的下一帧；
3. 结果写回 storage 的观测字典（例如新 key `cenet_recon_target`），同时把 `1 - dones` 写成 mask key；
4. 因为 rsl_rl 的 minibatch 采样器是对整个观测字典做扁平化 + 打乱切片，新增的 key 会自动跟着 batch 对齐。

注意：**不能等进了 minibatch 再移位**，因为 batch 已经打乱，时间相邻关系丢失。这一步是整个接入里最容易写错的地方。

同理，done 的 transition 其"下一帧"其实是 reset 后的观测，重建项必须用 mask 屏蔽；`L_est` 和 KL 用的是当步数据，仍然是有效的，不需要屏蔽。

**第二步：每个 minibatch 里算三项损失并合并**

```text
L_est  = MSE(v̂, v_gt)                                    # v_gt 来自 cenet_target_v
oe     = normalize(o_{t+1})                               # 与 CENet 输入同一套归一化
L_rec  = MSE(õ_{t+1}, oe) · mask                          # mask 屏蔽 done 步
L_kl   = 0.5 · Σ(σ² + μ² − 1 − ln σ²)                     # 对角高斯到 N(0,I) 的闭式 KL
L_CE   = L_est + β · L_kl + L_rec
total  = L_PPO + λ · L_CE
```

- `L_PPO` = surrogate + value_loss_coef · value_loss − entropy_coef · entropy（rsl_rl 现有逻辑不变）；
- 一次 `backward()`，一次 optimizer step。CENet 参数在 actor 内，`chain(actor.parameters(), critic.parameters())` 自动包含，梯度裁剪、保存、加载都不用改；
- `loss_dict` 里追加 `cenet_est`、`cenet_recon`、`cenet_kl`，runner 会自动记进 wandb。

**第三步：更新归一化统计**

PPO 原有的 `update_normalization` 照常调用，但我们的覆写要同时更新 actor 的 45 维 normalizer 和 CENet 的 normalizer。

### 5.3 归一化为什么必须做

论文明确写了"所有观测归一化到零均值单位方差"，这是 KL 先验取标准正态分布 N(0,I) 的前提。所以：

- `cenet_input` 的 270 维（按帧拆成 45 维）过 running normalizer 再进编码器；
- 重建目标 `o_{t+1}` 用同一套 normalizer 变换后再算 MSE；
- actor 自己的 o_t 输入照旧走 `obs_normalization=True`。

---

## 6. 配置与注册（当前链路缺的一环）

目前 `src/mjlab/tasks/dreamwaq/__init__.py` 只导入了 runner，没有调用 `register_mjlab_task`，所以 `list-envs` 里看不到 DreamWaQ 任务，需要补齐：

| 文件 | 内容 |
| --- | --- |
| `config/<robot>/rl_cfg.py`（新） | actor：`hidden_dims=(512,256,128)`、`obs_normalization=True`、`class_name="mjlab.tasks.dreamwaq.rl.cenet:DreamwaqActorModel"`；critic：`(512,256,128)`；algorithm：`class_name="...dreamwaq_ppo:DreamwaqPPO"`，`learning_rate=1e-3` |
| `config/<robot>/__init__.py`（新） | `register_mjlab_task(..., runner_cls=DreamwaqOnPolicyRunner)` |
| `obs_groups` | 增加 `"cenet": ("cenet_input",)` 之类的条目；其余三组不用路由给网络，storage 会自动存 |

如果希望 β、λ、λ_est 这类超参能从 rl_cfg 传入，最省事的做法是给 `src/mjlab/rl/config.py` 的 `RslRlModelCfg` / `RslRlPpoAlgorithmCfg` 各加一个可选的 `dreamwaq_cfg: dict | None = None` 字段，并在 `MjlabOnPolicyRunner.__init__` 的"None 值清理"列表里带上它（否则 critic 的构造函数会收到这个 None 参数）。不想动公共配置的话，也可以把这几项直接写死在 dreamwaq 的类里，等需要调参再外提。

另外两处小配套：

- `rl/runner.py`：ONNX metadata 目前按 actor 组生成，DreamWaQ 下应改成按 `cenet_input` 组生成（history_length=6），部署端才知道要喂 6 帧；
- play 模式：现在 play 时会关掉 actor 组的噪声，`cenet_input` 组也要同步关掉。

---

## 7. 容易踩的坑（重点看这一节）

1. **重建目标的移位时机**：必须在 update 之前对整段 storage 做移位并注入观测字典，不能在 minibatch 里移位（见 5.2 第一步）。
2. **done 步的"下一帧"是 reset 后的观测**：重建项要 mask 掉，否则 decoder 会被要求去重建一个不相干的帧。
3. **history 顺序**：mjlab 给的是旧 → 新 `[o_{t-5}, ..., o_t]`，论文写的是新 → 旧 `[o_t, ..., o_{t-5}]`。在模型里 flip 一次即可，选择一种后写死，不要前后不一致。
4. **归一化**：不做归一化就套 N(0,I) 先验，KL 会变成一个尺度错位的正则项，训练会变得不可控（论文原话就是为了这个才归一化）。
5. **v̂、z 不 detach（已定）**：CENet 沿 PPO 一起更新，PPO 梯度经 `[o_t, v̂, z]` 回传编码器，`L_CE` 与 PPO 损失相加后一次 backward、同一个优化器。论文没有直接说明这一点（属本复现的实现选择），但实现上不要再加 detach、也不要拆成两套优化器。
6. **ONNX 导出**：默认的 `as_onnx()` 只吃 45 维 actor 观测，会导出一个完全不包含 CENet 的模型；必须覆写，导出输入是 270 维历史、输出 12 维动作，并且经 play 环境验证一遍。
7. **优化器与学习率**：CENet 参数在 actor 内，自动共用 PPO 的优化器；论文是全网络固定 Adam 1e-3，而 mjlab 默认 `schedule="adaptive"` 会连 CENet 的学习率一起调整，想严格对齐就把 schedule 设为 `fixed`。
8. **旧 checkpoint 不兼容**：actor 的 state_dict 会多出 `cenet.*` 的键，旧模型无法 strict 加载，需要从头训。
9. **显存/内存**：每步多存 270 + 45 + 3 维，4096 envs × 24 steps 大约 100 MB 量级/轮，可接受。
10. **噪声一致性（可选）**：`cenet_input` 和 `actor` 是两个组，各自独立采样噪声，数值上不会完全相同（分布相同）。论文里没有这个要求，先接受；将来若要严格一致，再考虑让两者共享同一份噪声实例。

---

## 8. 论文没写死、先定默认值的超参

| 项 | 建议默认 | 说明 |
| --- | --- | --- |
| β（KL 权重） | 1.0 | β-VAE，论文未给取值 |
| λ（L_CE 相对 PPO 的权重） | 1.0 | 论文未给；影响辅助损失和策略损失的量级平衡 |
| L_est 单独权重 | 1.0 | 与重建项同量级；速度单位是 m/s，量级约 O(1)，可后续再调 |
| z 的用法 | 策略用 μ；VAE 重建采样 z；推理 μ | 见《论文未明确之处.md》第 2 条 |
| 策略梯度是否回传 CENet | 回传（不 detach，已定） | 论文未直接说明；本复现选择端到端一起更新 |
| 学习率调度 | 论文固定 1e-3；mjlab 默认 adaptive | 二选一，建议先跟论文 fixed |
| done 步的重建项 | mask | 见坑 2 |
| 观测噪声 | `cenet_input` 与 actor 同分布 | 见坑 10 |

---

## 9. 验证步骤

1. **静态检查**：`make check`（format + lint + type），维度断言写进模型构造函数（270、35、45、64）。
2. **不跑仿真的单测**：用假 obs TensorDict 构造模型和算法 —— 检查前向形状、`L_est/L_rec/L_kl` 可反向且 encoder 有梯度、移位逻辑（手工造 T=3 的序列验证目标对齐）、done mask 生效。
3. **小规模冒烟训练**：小 env 数跑 2 次迭代，确认 loss_dict 出现 `cenet_est / cenet_recon / cenet_kl` 且数值有限、v̂ 的误差在下降。
4. **导出验证**：导出 ONNX，确认输入 270、输出 12，并在 play 环境跑通推理。
5. **回归**：确认没有动到公共代码；若改了 `src/mjlab/rl/config.py` 等公共文件，跑一遍 velocity 任务相关单测。

---

## 10. 范围外（按当前要求暂不做）

- AdaBoot：先恒用 v̂（即论文的 "w/o AdaBoot" 版本）。模型里把"这一步用 v̂ 还是用真值 v"做成一个可替换点，将来算出 p_boot 后按环境掩码替换即可，不需要改链路结构。
- 奖励函数、课程学习：保持现状，不动。
