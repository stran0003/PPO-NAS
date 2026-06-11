# GAE Debug 与训练诊断指南

> 如何用 `gae_debug.xlsx` 和 `output/plots/*.png` 判断训练是否健康、定位问题、调整超参数。

---

## 1. 你手上有哪些诊断工具

| 工具 | 路径 | 更新频率 | 用途 |
|---|---|---|---|
| 终端日志 | 控制台 | 每 5 轮 | 快速扫一眼整体趋势 |
| Excel 表 | `output/gae_debug.xlsx` | 每 5 轮覆盖 | 深入看每一步的 GAE 分配 |
| Pareto 图 | `output/plots/pareto.png` | 训练结束/`--analyze` | 看最终搜到的最优架构分布 |
| 训练曲线 | `output/plots/training.png` | 同上 | Reward/Loss/Entropy/KL 趋势 |
| 架构分布图 | `output/plots/distribution.png` | 同上 | 看 Controller 偏好哪些选择 |
| GAE 热力图 | `output/plots/gae_heatmap.png` | 同上 | V(s)/δ/A 的全局分布 |

---

## 2. 终端日志 — 最常用的第一眼

```
[   0][WARM] R=-15.0[-66,-4]±12.3 | res=5.0[4,22]dB | aL=-0.002 cL=6.33 | H=1.61 KL=0.001 | P=1 | 45s
[   5][WARM] R=-18.2[-60,-6]±10.8 | res=6.1[4,20]dB | aL=0.001 cL=5.21 | H=1.55 KL=0.003 | P=3 | 42s
[  10]       R=-22.5[-58,-8]±9.2  | res=7.5[4,19]dB | aL=0.003 cL=4.10 | H=1.42 KL=0.005 | P=5 | 40s
```

### 逐个字段解读

| 字段 | 正常范围 | 异常信号 | 怎么调 |
|---|---|---|---|
| `R=均值[min,max]±std` | min 和 max 间隔 >10 | 间隔<5，全是差架构 | 增大 `acc_weight` 或等更多轮 |
| `res=均值[min,max]dB` | max-min 应该拉开 | 全挤在 18-22 | 搜索还没找到好方向，加大 temp/entropy_coef |
| `aL` (actor_loss) | −0.01 ~ 0.01 | 持续 >0.05 或 <−0.05 | KL 可能飙升，降 lr |
| `cL` (critic_loss) | 持续下降 | 不降反升或卡住 | V(s) 没学到，增大 `value_loss_coef` 或等更多轮 |
| `H` (entropy) | 1.6→0.5 缓慢下降 | 骤降到 <0.3 | 探索过早枯竭，提高 `temp_final` 或 `entropy_coef` |
| `KL` | <0.03 | >0.05 持续 | 策略变化太大，降 lr 或增大 `clip_epsilon` |
| `P` (Pareto 大小) | 持续增长 | 长期不涨 | 搜不到更好的架构了，加大探索 |

### 最关键的判断：res 的 [min, max] 范围

```
res=[18,22]  → 全是垃圾，信号太弱 → 增大 acc_weight, 多跑几轮
res=[10,22]  → 开始出现中等架构 → 方向对了，继续
res=[5,20]   → 稳定搜到好架构 → 可以开始加 param_weight
res=[4,12]   → 搜索收敛到优质区域 → 考虑切换到 Phase 2
```

---

## 3. Excel 表 — 深入诊断 GAE

打开 `output/gae_debug.xlsx`，每轮 17×7≈119 行（warmup 期 17 个架构×7 步）。

### 3.1 先看最后一步（t=6）

最后一步 `delta = reward - V(s_6)`，最直接反映 Critic 是否学会了估值。

| 看什么 | 怎么算 | 判断 |
|---|---|---|
| `A_norm` 的符号是否和 reward 正相关 | 筛选 reward 高的行，看 A_norm 是否 >0 | 应是：高 reward → A_norm>0；低 reward → A_norm<0 |
| `V(s)` 是否接近 reward | 同一轮里 V(s_6) vs reward | 训练初期 V(s)≈0 正常，后期应接近 reward 均值 |
| `delta` 是否集中 | delta 的标准差 | 标准差大说明 Critic 估值不准 |

### 3.2 看中间步骤（t=0~5）

| 看什么 | 哪里看 | 判断 |
|---|---|---|
| A_raw 是否随 t 递减到 0 | 同一架构从 t=6 往 t=0 看 A_raw | γλ 衰减意味着 t=6 的 δ 传到 t=0 只剩 ~69%。但如果中间步 δ 也为正，A_0 可能反而更大。关键是看 A_raw 在好/差架构间是否有区分度 |
| 信号是否能传到前面 | A_raw t=0 的绝对值 | 如果 t=0 的 A_raw ≈ 0 但 t=6 很大 → 信号没传到前面 → 增大 γ 或 λ |
| 中间步 δ 是否忽正忽负 | 同一架构 δ 列 | 好架构大部分 δ>0，差架构大部分 δ<0 |

### 3.3 按 conv_type 分组比较

Excel 筛选 `conv_type` 列，看 grouped / 1x1 / skip 的 A_norm 均值：

| 发现 | 结论 | 行动 |
|---|---|---|
| skip 的 A_norm 系统性为负 | skip 拖累性能 | 临时去掉 skip |
| grouped 的 A_norm 普遍 > 1x1 | Controller 应该更倾向 grouped | 正常，等 Controller 自己学会 |
| 三种类型 A_norm 无差异 | 信号太弱 | 增大 acc_weight |

### 3.4 找"好架构"的共同特征

按 `A_norm` 降序排列，看前 20% 的行：

1. 哪些层选了 grouped？哪些选了 1x1？
2. kernel 集中在哪些值？
3. spline 集中在哪些值？

如果这些分布和你的基线架构一致，说明 **Controller 正在学会**。

---

## 4. 训练曲线图 — 看长期趋势

`output/plots/training.png` 四张子图：

### Reward 曲线
```
✅ 正常: 总体上升，锯齿波动（每轮 16 个架构质量参差正常）
⚠️ 异常: 持续不涨，说明信号太弱，Controller 没学到
🔧 调法: 增大 acc_weight，提高 temp_init，检查 reward 函数
```

### Loss 曲线 (Actor + Critic)
```
✅ 正常: Actor loss 在 0 附近振荡，Critic loss 持续下降
⚠️ 异常: Actor loss 突然飙升 → KL 爆炸，策略崩了
🔧 调法: 降 lr，减小 clip_epsilon（收紧更新幅度）
⚠️ 异常: Critic loss 不降 → Critic 没学到 V(s)
🔧 调法: 增大 value_loss_coef，等更多轮
```

### Entropy 曲线
```
✅ 正常: 从 ~1.6 缓慢下降到 ~0.5，平稳不骤降
⚠️ 异常: 骤降到 0.2 以下 → 探索过早枯竭
🔧 调法: 提高 temp_final (如 0.5→0.7)，增大 entropy_coef
⚠️ 异常: 一直居高不下 → 太随机，学不到东西
🔧 调法: 降低 temp_init，减小 entropy_coef
```

### KL 曲线
```
✅ 正常: 先升后降，峰值 <0.05
⚠️ 异常: 持续飙升 >0.1 → 策略突变，可能崩溃
🔧 调法: 降 lr，增大 clip_epsilon
⚠️ 异常: 一直 ≈0 → 策略没变化，没学到东西
🔧 调法: 增大 lr，增加 ppo_epochs
```

---

## 5. GAE 热力图 — 全局扫一眼

`output/plots/gae_heatmap.png` 三列子图：

### V(s) 热力图
- 行 = 16 个架构，列 = 7 个时间步
- 颜色深浅应随训练从均匀（随机初始化）→ 有规律（Critic 学到了）

### δ (delta) 热力图  
- 暖色 = δ>0（这一步做对了），冷色 = δ<0（这一步做错了）
- **好架构应整行偏暖**，差架构整行偏冷
- 如果颜色全白（接近 0），信号太弱

### A_norm 热力图
- **Warm 注入那轮应该有 1 行特别暖**（基线架构）
- 其他行有暖有冷 = 正常，全白 = 信号弱

---

## 6. 诊断流程图

```
训练跑起来
    │
    ▼
看终端: res=[min,max] 拉开了吗？
    │
    ├── 没拉开（全挤在 18-22）
    │   → 增大 acc_weight (3→5)
    │   → 提高 temp_init (2→3)
    │   → 多跑 50 轮再看
    │
    ├── 拉开了但有波动
    │   → 开 Excel: 好架构的 A_norm 是否 >0？
    │   │
    │   ├── 是 → 正常，继续跑
    │   │
    │   └── 否 → 信号反了！
    │       → 检查 reward 函数：res 越小,R 应该越大
    │       → 确认 compute_reward 的 perf_score = -res
    │
    └── 拉开了且 P 在涨
        → 看 Excel: 好架构的共同特征是否接近基线？
        │
        ├── 是 → 🎉 搜索在正确轨道上
        │   → 等参数量相关的 res 出现后加 param_weight
        │
        └── 否 → 好架构的特征和基线不一致
            → 增加 inject_iterations
            → 减小 inject_interval (让基线更频繁示范)
```

---

## 7. 参数速查表

| 现象 | 调哪个参数 | 方向 |
|---|---|---|
| res 全挤在一起 | `acc_weight` | ↑ 增大 (3→5→8) |
| 好架构太少 | `temp_init` / `rollouts_per_iter` | ↑ |
| 探索过早枯竭 | `temp_final` / `entropy_coef` | ↑ |
| KL 飙升 | `lr` / `clip_epsilon` | ↓ lr / ↓ ε |
| Critic 不学习 | `value_loss_coef` | ↑ |
| 信号传不到前面 | `gamma` / `gae_lambda` | ↑ (0.99→0.995) |
| 训练太慢 | `rollouts_per_iter` / `ppo_epochs` | ↑ |
| 对参数量不敏感 | `param_weight_final` | ↑ |
| 基线学不会 | `inject_iterations` / `inject_interval` | ↑ 轮数 / ↓ 间隔 |

---

## 8. 日常检查清单

每跑 100 轮做一次：

- [ ] 终端日志：res= 的 [min,max] 范围是否在扩大
- [ ] Excel：打开最新一轮，按 A_norm 降序，好架构的特征是否一致
- [ ] 训练曲线：Entropy 是否平稳下降，KL 是否在 0.05 以下
- [ ] Pareto 图：前沿是否在向左下角移动
- [ ] 架构分布：Controller 是否对某些选择有了偏好（不再是均匀分布）
