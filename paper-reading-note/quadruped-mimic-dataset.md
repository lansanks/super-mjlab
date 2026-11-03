# 四足 mimic 任务的数据集调研（2026-10-28）

目标：为 mjlab 新增一个与 `tracking_supermimic`（G1 × LAFAN1）同构的四足任务，
先确定参考动作数据从哪来。

## 1. 任务的硬性需求

从 `src/mjlab/tasks/tracking_supermimic` 反推，数据必须满足：

| 需求 | 具体约束 | 来源 |
|---|---|---|
| 形态匹配 | 12 DoF 腿足机器人（每条腿 3 关节），能被 Go1 跟踪 | `asset_zoo/robots/unitree_go1` |
| 世界系根轨迹 | 需要 `body_pos_w` / `body_quat_w`，不能是原地（in-place）动作 | `motion.py` |
| 全量 FK | 需要被跟踪 body 的世界位姿与线/角速度 | `convert_lafan1.py` |
| 帧率 | `CONTROL_DT = 0.02`（50 Hz） | `constants.py` |
| 库结构 | 多片段 + 片段边界（`clip_start` / `clip_len` / `clip_id`） | `motion.py` |
| 规模 | 要能支撑"从无结构 mocap 学技能"，几十秒不够 | SuperMimic 论文 |

NPZ schema（`convert_lafan1.py` 的 `np.savez_compressed`）：
`names, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
body_ang_vel_w, body_names, clip_start, clip_len, clip_id`。

## 2. 候选对比（均已实际探测）

| 数据集 | 规模 | 格式 | 是否已重定向到四足 | 许可 | 可达性 |
|---|---|---|---|---|---|
| **MANN dog mocap**（Zhang et al., SIGGRAPH 2018） | 51 段 / 41 min / 60 fps | BVH（21 关节，cm，世界系根轨迹） | 否（需自己做 IK） | 数据 CC BY-NC 4.0，仅研究用途 | 官方站当前不可达；**Wayback 镜像 33 MB 已验证可下** |
| **DeepMimic / motion_imitation dog clips**（Peng et al., RSS 2020） | 10 段 / **30.6 s** | DeepMimic JSON（3 位置 + 4 四元数 + 12 关节角） | 是（Laikago/A1 形态） | 仓库无 LICENSE，随论文发布 | 可用，raw.githubusercontent |
| **NJU-RLC Go2 重定向数据**（Fu et al., Nat Commun 2026） | 20 min，带 walk/pace/trot/canter/jump 标签 | 与 MANN 同源但已对齐 Go2 | **是（Go2）** | 论文声明公开 | Code Ocean DOI `10.24433/CO.8261908.v1`，本机 403（需账号/网络） |
| **Tencent "Lifelike Agility and Play"** | 未核实 | — | 未知 | 未核实 | figshare 24968946，本机 403 |
| **Barkour**（DeepMind） | — | — | — | — | 只发布机器人模型与 benchmark，**未发布 mocap** |
| **LocoMuJoCo** | — | — | — | MIT | 有 4 个四足环境，但 mocap 只有 humanoid（ANYmal/Spot 仅纯 RL） |
| LAFAN1 / AMASS | 大 | 人体 BVH | 人体，与四足形态不符 | — | 已在本仓库（G1 用） |

结论：**MANN 是唯一"足够大 + 形态正确 + 可下载"的四足 mocap**。

## 3. MANN 数据实测结果

已下载并逐帧解析（`/tmp/mann`，51 个 BVH）：

- **41.0 min / 51 段**，全部 60 fps，`Frame Time = 0.0167`。
- **单一骨架，21 关节**（含 End Site 共 27）：
  `Hips, Spine, Spine1, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand,
  RightShoulder, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot,
  RightUpLeg, RightLeg, RightFoot, Tail, Tail1`
  前爪 = `LeftHand/RightHand`，后爪 = `LeftFoot/RightFoot`，与 NJU 脚本注释一致。
- 单位 **cm**，根带宽世界系平移与 ZXY 欧拉角（`CHANNELS 6`）。
- 根高 0.06–1.21 m（含趴卧、跳跃），站立约 0.46 m。
- 关键点：踝关节 `LeftFoot` 在爪底上方约 7 cm，**爪底 = End Site**；按 End Site 算，
  地面接触帧的爪高 ≈ 0（验证通过），所以重定向应以 End Site / 减固定偏移为准。
- 运动量（按根水平速度统计）：

| 速度档 | 段数 | 时长 |
|---|---|---|
| < 0.15 m/s（站/坐/卧） | 11 | 10.1 min |
| 0.15–0.5 m/s | 23 | 26.2 min |
| 0.5–1.0 m/s | 9 | 3.7 min |
| 1.0–2.0 m/s | 5 | 0.7 min |
| > 2.0 m/s | 3 | 0.2 min |

即 41 min 里真正在移动（> 0.3 m/s）的约 **13 min**，> 0.6 m/s 约 **7.6 min**，
其余是原地/静止/趴卧。NJU 论文也明确说他们剔除了 stretching、body shaking 等段。

### 下载方式

官方 `https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip`
本机连不上（TLS 断连 / 502）。可用归档：

```
http://web.archive.org/web/20260524053129id_/https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
```

33,422,409 B，解压 51 个 `.bvh`，共 83 MB。**不要用在 MDP 里**：
`erwincoumans/motion_imitation` 的 `data/motions/*.json` 只是 30.6 s 的演示切片。

## 4. 重定向到 Go1 的可行性

mjlab 的 Go1 与 NJU 论文使用的 Go2 几何几乎一致，NJU 的重定向代码（MIT）可直接借用：

| 量 | 狗（MANN） | Go1（mjlab XML） | Go2（NJU 脚本常量） |
|---|---|---|---|
| 肩–髋跨 | 0.424 m | 0.1881 × 2 = 0.376 m | 0.3868 m |
| 大腿 + 小腿 | ≈ 0.41 m（Arm→Hand / UpLeg→Foot） | 0.213 + 0.213 = 0.426 m | 0.213 + 0.213 |
| 站立根高 | ≈ 0.46 m | ≈ 0.30 m | — |

Go1 关节限位：`hip ±0.863`、`thigh [-0.686, 4.501]`、`calf [-2.818, -0.888]`；
脚端有 site `FR/FL/RR/RL`（calf 下 0.213 m，球半径 0.023）。

流程：
1. 解析 BVH → 每帧世界位姿（已有验证脚本 `/tmp/bvh_fk.py`）；
2. 取 5 个关键点（躯干中心 + 4 个爪底），按 x/y/z 三轴分别缩放；
3. 每条腿 3 DoF → 3 维足端目标，解析 IK（hip 外摆 + thigh + calf），超限位则裁剪；
4. 60 → 50 Hz 重采样（四元数 slerp，位置差分求速度）；
5. 用 Go1 MJCF 算 13 个 body 的 FK 轨迹，写成上面的 NPZ；
6. `TRACKED_BODIES = ("trunk", "FR_hip", "FR_thigh", "FR_calf", ..., "RL_calf")`（13 个；
   也可退化为 trunk + 4 个脚端，便于先跑通）。

## 5. 建议的推进顺序

1. **冒烟**：先用 `motion_imitation` 的 10 段（30.6 s）跑通"四足版 NPZ → 训练一步 → play"，
   确认四足任务骨架（观测/奖励维度、世界模型输入维度）无误；数据量不足以训练。
2. **正式**：MANN 全量重定向（41 min，必要时裁掉静止段），作为论文规模实验的数据集。
3. 若能拿到 NJU 的 Go2 重定向标签数据（Code Ocean），可直接对比，省一次 IK 调参。

## 6. 风险

- MANN 数据 **CC BY-NC 4.0**，本仓库现有 LAFAN1 也是 CC BY-NC-ND，一致；不能商用/再分发。
- 41 min 中静止段占多数，需分段或对采样做加权，否则根轨迹奖励会被"站着"主导。
- 狗的步态（pace/canter）超出多数四足控制器的常见速度区间，Go1 的关节限位会裁剪部分姿态，
  需要检查裁剪率；过高则要放宽缩放或只保留 walk/trot 段。
- 原始 BVH 无接触标签，脚底接触要从 End Site 高度阈值自己推。

---

# 实施记录（2026-10-28）

## 已完成

1. **数据落位**：51 个 BVH → `data/mann_raw/`（80 MB，`data/` 已在 `.gitignore` 中）。
2. **Go2 资产移植**：`src/mjlab/asset_zoo/robots/unitree_go2/`
   - `xmls/go2.xml`：由 Menagerie 版本改造（去掉 `<motor>`，加 IMU 传感器 / 相机 / 脚端 site，碰撞体按 mjlab 的 `.*_collision\d*` 与 `^[FR][LR]_foot_collision$` 命名）。
   - `xmls/assets/`：16 个 obj（28 MB）。
   - `go2_constants.py`：12 个位置执行器（hip/thigh kp 15.9、knee kp 35.76；effort 23.7/45.43），`INIT_STATE` 站高 0.278 m，已在 `asset_zoo/robots/__init__.py` 注册。
3. **转换器**：`src/mjlab/tasks/tracking_supermimic_go2/scripts/convert_mann_bvh.py`
   - 纯 numpy BVH 解析 + FK（27 关节，含 End Site 爪底）
   - 逐段重贴地（地面 = 四爪底 5% 分位）
   - 分段筛选：站立（相对髋高 >0.35 m）且运动（>0.15 m/s），补 1 s 空档，段长 ≥2 s，前后留 0.3 s，绝对髋高 <0.70 m 剔除台子
   - 5 关键点 → Go2 比例缩放（腿长用站立帧估，避免趴卧段把尺度放大到 2.4 倍）
   - 每腿 3-DoF 解析 IK + 关节限位裁剪 + 单帧跳变修补
   - 60 → 50 Hz 重采样，MuJoCo FK 出 13 个 body 的位姿/速度，写 mjlab NPZ

## 产出

`data/mann_go2/motion.npz`：**127 段 / 48,810 帧 / 16.3 min / 76 MB**

| 检查项 | 结果 |
|---|---|
| 关节限位越界 | **0 / 585,720** |
| 基座高度 | 均值 0.268 m（0.078–0.623） |
| 脚端高度 | p1 −0.052 / p50 +0.038 / p99 +0.186 m |
| 基座速度 | 均值 0.62，p99 3.06，最大 4.76 m/s |
| 段长 | 平均 384 帧，127–2045 |

## 已知问题

- **残留在段内的关节跳变**：单帧最大 0.90 rad（45 rad/s），`joint_vel` p99 12.1 rad/s。修补器能压掉大部分，少数快速步态段仍有毛刺，后续要么对目标做时域低通，要么在 task 层对奖励做鲁棒化。
- **姿态偏蹲**：垂直缩放按"地面一致"取（狗 0.379 m → Go2 0.278 m），所以参考动作比狗更弯腿；这是有意的（机器人用自己的自然站姿复现落足位置），但看起来会比狗"蹲"。
- 重新贴地用的是整段 5% 分位，跨台子/跳下的段会保留跳落过程（已用绝对高度过滤掉大部分）。
