# PPO-NAS 前向过程完整详解

> 基于 `libs/ppo.py`、`libs/model/controller.py`、`libs/reward.py`、`libs/environment.py`

---

## 目录

1. [总体框图](#1-总体框图)
2. [阶段一：采样](#2-阶段一采样-forward)
3. [阶段二：评估与奖励](#3-阶段二评估与奖励)
4. [阶段三：GAE 优势计算](#4-阶段三gae-优势计算)
5. [阶段四：PPO 更新循环](#5-阶段四ppo-更新循环)
6. [阶段五：反向传播与梯度流](#6-阶段五反向传播与梯度流)
7. [完整迭代总结](#7-完整迭代总结)

---

## 1. 总体框图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PPO-NAS 一轮迭代总览                              │
│                                                                         │
│  ┌──────────────────────┐                                               │
│  │   Controller (θ)     │  Actor-Critic LSTM                            │
│  │  ┌─────────────────┐ │                                               │
│  │  │  共享 LSTM (2层) │ │                                               │
│  │  │  hidden_dim=128  │ │                                               │
│  │  └────┬──────┬─────┘ │                                               │
│  │       │      │        │                                               │
│  │   Actor头  Critic头   │                                               │
│  │   (4个)    (MLP→1)   │                                               │
│  └──────┬──────┬────────┘                                               │
│         │      │                                                        │
│    ┌────┘      └────┐                                                   │
│    ▼                ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ ① 采样 (forward)                                                 │  │
│  │   batch_size=16, LSTM逐时间步生成7层配置                          │  │
│  │   → 16个架构, log_probs_old(16,7), values_old(16,7)              │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ ② 评估 & 奖励                                                    │  │
│  │   每个架构 → train(arch) → nmse, num_params                      │  │
│  │   → compute_reward(nmse, params) → 16个标量奖励                   │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ ③ GAE 优势计算                                                   │  │
│  │   从后往前: A_t = δ_t + γλ·A_{t+1}                               │  │
│  │   → advantages(16,7), returns(16,7)                              │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ ④ PPO 更新 ×4 epochs                                             │  │
│  │   每 epoch: 打乱→分4个minibatch→evaluate()重算→                   │  │
│  │   actor_loss + 0.5*critic_loss − 0.01*entropy                    │  │
│  │   → loss.backward() → optimizer.step()                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  重复 total_iterations=500 次                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段一：采样 (forward)

**代码位置**: `libs/model/controller.py::forward()` [line 92]

### 2.1 输入

| 参数 | 值 | 含义 |
|------|-----|------|
| `batch_size` | 16 | 并行生成16个架构 |
| `deterministic` | False | 随机采样（非argmax） |

### 2.2 LSTM 时间步展开

Controller 按固定的7层结构逐时间步生成架构：

```
t=0 (L1-线性):  start_token → LSTM → h₀
                ├─ actor_conv(h₀):  Linear(128→2) → logits → softmax → P(conv)
                │   → sampled c₀ ∈ {0(grouped), 1(1x1)}
                │   → log P(c₀) 计入 log_prob
                │
                ├─ actor_kernel(h₀): Linear(128→5) → logits → softmax → P(kernel)
                │   → sampled k₀ ∈ {0..4}
                │   → 仅当 c₀=0(grouped) 时计入 log_prob，否则 k₀=-1
                │
                ├─ actor_init(h₀): Linear(128→3) → logits → softmax → P(init)
                │   → sampled i₀ ∈ {0..2}
                │   → log P(i₀) 计入 log_prob
                │
                └─ critic(h₀): Linear→ReLU→Linear→ReLU→Linear(1) → V(s₀)

                每步总 log_prob = log P(c) + log P(k)×mask + log P(i)
                每步总 entropy  = H(P_c) + H(P_k)×mask + H(P_i)

t=1 (L2-线性):  encode(arch[0], 21维) → LSTM → h₁
                重复上述采样...
                ...
t=3 (L4-LUT):   只采样 spline (1个分布, 5选1)
                每步总 log_prob = log P(spline)
                每步总 entropy  = H(P_spline)
                ...
t=6 (L7-线性):  最后一层采样完毕
```

### 2.3 输出张量

| 变量 | 维度 | 含义 | 梯度状态 |
|------|------|------|----------|
| `architectures` | list[16] of list[7] | 16个完整架构配置 | 无(纯数据) |
| `log_probs` | (16, 7) | 每步采样动作的log概率 | 有(连着LSTM) |
| `values` | (16, 7) | 每步Critic估值 V(s) | 有(连着LSTM) |
| `entropies` | (16, 7) | 每步策略分布的熵 | 有(连着LSTM) |
| `actions` | list[16][7] | 每步各头的动作索引 | 无(纯索引) |

### 2.4 动作解码

`decode_actions(t, indices)` 将动作索引转为层配置 dict：

```python
# 线性层例: indices=[1, 2, 0]
→ {"type":"linear", "kernel":7, "conv":"1x1", "init":"center_spike"}
# 1x1时 kernel 强制为1，k_idx被忽略

# LUT层例: indices=[3]
→ {"type":"lut", "spline":32}
```

---

## 3. 阶段二：评估与奖励

**代码位置**: `libs/ppo.py::train_one_iteration()` [line 61-77]

### 3.1 评估架构

```python
for arch in architectures:                    # 16个架构逐个评估
    metrics = evaluate_architecture(arch)      # 构建→训练→评估
    nmse = metrics["nmse"]                    # 归一化均方误差(dB), 越小越好
    params = metrics["num_params"]            # 参数量
```

`evaluate_architecture` 调用链：
```
evaluate_architecture(arch)
  → train(arch)          # libs/model/train.py
    → run_model(arch, cfgs)
      → build_model(arch, cfgs)  → real_model(cfgs, configs)
      → 训练模型
      → 评估模型
      → return {"nmse": res, "num_params": num_params}
```

### 3.2 奖励函数设计 ⭐

**代码位置**: `libs/reward.py::compute_reward()` [line 19]

#### 当前策略：简单线性加权

```
R = -nmse × acc_weight  −  param_weight × (num_params / target_params)
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `acc_weight` | 1.0 | 性能权重 |
| `param_weight` | 0.1 | 参数量惩罚权重 |
| `target_params` | 100000 | 目标参数量 |

#### 数值举例

```
架构A: nmse=-32, params=120000
  perf_score = -(-32) × 1.0 = 32.0
  param_ratio = 120000 / 100000 = 1.2
  param_penalty = 0.1 × 1.2 = 0.12
  R_A = 32.0 - 0.12 = 31.88   ✅ 高分

架构B: nmse=-28, params=80000
  R_B = 28.0 - 0.08 = 27.92   ❌ 性能差，分低

架构C: nmse=-32, params=200000
  R_C = 32.0 - 0.20 = 31.80   ⚠️ 性能好但太大，略低于A
```

#### 设计要点

- **NMSE取负号**：NMSE是负数（如-32dB），取负后变成正数，性能越好分数越高
- **参数量惩罚**：超过目标按比例扣分，鼓励搜索轻量模型
- **延迟奖励**：整个架构生成完才给一次奖励（terminal reward），中间步骤奖励为0
- **奖励是PPO唯一的学习信号**：Controller完全靠这个标量知道"什么样的架构好"

---

## 4. 阶段三：GAE 优势计算

**代码位置**: `libs/ppo.py::_compute_gae()` [line 116]

### 4.1 为什么需要GAE

NAS 是序列决策——Controller分7步生成架构，但奖励只在最后一步给。GAE 把终端奖励**沿时间轴反向回传**，让每一步都知道自己的贡献。

### 4.2 计算过程（NAS特化版）

```python
gamma = 0.99       # 折扣因子
gae_lambda = 0.95  # GAE平滑参数

# 输入: rewards (16,)  每个架构一个终端奖励
#       values  (16,7) 每步的Critic估值

for t in reversed(range(7)):          # t=6,5,4,3,2,1,0
    if t == 6:                          # 最后一步
        δ = reward - V(s_6)             # 直接用奖励减价值
        A_6 = δ
    else:                               # 中间步骤 (奖励=0)
        δ = 0.99 × V(s_{t+1}) - V(s_t)  # 纯价值差
        A_t = δ + 0.99×0.95 × A_{t+1}   # 累积未来的优势
```

### 4.3 直观理解

```
假设一个架构最终奖励 R=31.88

t=6: V(s_6)=30 → A_6 = 31.88-30 = +1.88  (最后一步贡献了正的)
t=5: V(s_5)=29, V(s_6)=30
     δ_5 = 0.99×30-29 = 0.7  (Critic认为状态在变好)
     A_5 = 0.7 + 0.94×1.88 = 2.47  (累积了t=6的正优势)
t=4: ...
     A_4 = δ_4 + 0.94×2.47
...
```

- A_t > 0：这一步的决策比Critic预期的好 → 应该增加这个动作的概率
- A_t < 0：这一步的决策比Critic预期的差 → 应该降低这个动作的概率

### 4.4 后处理

```python
returns = advantages + values           # 每步的"真实价值"=优势+估值
advantages = (A - mean(A)) / std(A)     # 标准化到零均值单位方差
```

标准化后 advantages 均值=0，正负各半——这正是 Actor loss 接近零的原因。

### 4.5 关键：全部 detach

进入PPO更新循环前，`log_probs`, `values`, `advantages`, `returns` 全部 detach，切断与旧计算图的连接，作为**固定标签**使用。

---

## 5. 阶段四：PPO 更新循环

**代码位置**: `libs/ppo.py::_ppo_update()` [line 156]

### 5.1 循环结构

```
epoch 1: indices = randperm(16) → [3,12,0,7,9,2,14,5,1,8,11,15,6,10,4,13]
         minibatch 1: idx=[3,12,0,7]     → evaluate() → loss → backward()
         minibatch 2: idx=[9,2,14,5]     → evaluate() → loss → backward()
         minibatch 3: idx=[1,8,11,15]    → evaluate() → loss → backward()
         minibatch 4: idx=[6,10,4,13]    → evaluate() → loss → backward()

epoch 2: indices = randperm(16) → [14,1,9,6,...]  ← 重新打乱
         ...同上...

epoch 3-4: 继续打乱...
```

每个 epoch 重新排列顺序 → 同批16个样本以不同组合出现在minibatch中 → 更充分的梯度利用。

### 5.2 重新评估 (evaluate)

```python
# controller.evaluate() 内部:
for t in range(7):
    h_t = LSTM(encode(arch[t-1]))     # 用当前参数重新前向
    logits = actor_conv(h_t)          # 当前Actor头输出
    dist = Categorical(logits=logits)
    lp = dist.log_prob(action_old[t]) # 用旧动作查新分布的概率
    ent = dist.entropy()              # 当前分布的熵
    V = critic(h_t)                   # 当前Critic估值

# 输出:
new_log_probs  (4,7)  ← 当前策略下旧动作的概率
new_values     (4,7)  ← 当前Critic的估值
new_entropies  (4,7)  ← 当前分布的熵
```

### 5.3 Actor Loss

```python
ratio = exp(new_log_probs - old_log_probs)           # (4,7) 策略变化比

surr1 = ratio × advantages                           # 不加约束
surr2 = clamp(ratio, 0.8, 1.2) × advantages           # clip约束
actor_loss = -mean(min(surr1, surr2))                 # 标量
```

**min 的安全阀机制**：

| advantage | ratio倾向 | min选谁 | 效果 |
|-----------|----------|---------|------|
| + | ratio>1 (更倾向) | surr2 (被clip) | 不过度增加概率 |
| + | ratio<1 (变冷淡) | surr1 (未被clip) | 正常梯度 |
| − | ratio<1 (变冷淡) | surr2 (被clip) | 不过度降低概率 |
| − | ratio>1 (更倾向) | surr1 (未被clip) | 正常梯度 |

### 5.4 Critic Loss

```python
critic_loss = MSE(new_values, returns)
```

`returns = advantages + values_old` 是GAE算的实际价值。Critic的目标是让 `new_values` 尽量接近 `returns`。

### 5.5 熵正则化

```python
entropy = new_entropies.mean()
```

熵衡量策略的随机程度。减去熵 = 鼓励高熵 = 鼓励探索。

### 5.6 总损失

```python
loss = actor_loss + 0.5 × critic_loss - 0.01 × entropy
```

### 5.7 梯度更新

```python
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(max=0.5)       # 梯度裁剪，防止爆炸
optimizer.step()
```

---

## 6. 阶段五：反向传播与梯度流

**代码位置**: `libs/ppo.py::_ppo_update()` [line 214-217]

### 6.1 计算图结构

```
                      ┌─────────────────────────────────┐
                      │     controller.parameters()      │
                      │  ┌───────────────────────────┐   │
                      │  │  input_proj (Linear 21→128)│   │
                      │  │  LSTM (2层, hidden=128)    │   │
                      │  │                            │   │
                      │  │  actor_conv  (128→2)       │   │
                      │  │  actor_kernel(128→5)       │   │
                      │  │  actor_init  (128→1)       │   │
                      │  │  actor_spline(128→4)       │   │
                      │  │                            │   │
                      │  │  critic (128→128→64→1)     │   │
                      │  └───────────────────────────┘   │
                      └─────────────────────────────────┘
```

### 6.2 各项对参数的梯度贡献

```
                        LSTM+input_proj    Actor头     Critic头
                        ───────────────    ───────     ────────
actor_loss                   ✅              ✅           ❌
  (ratio × adv, clipped)

critic_loss                  ✅              ❌           ✅
  (MSE(V_new, returns))

entropy (−ent_coef)           ✅              ✅           ❌
  (−0.01 × H(π_new))
═══════════════════════════════════════════════════════════
总计                         ✅✅✅            ✅✅          ✅
```

### 6.3 逐项梯度流详解

#### Actor Loss 的梯度流

```
actor_loss = -mean(min(ratio×adv, clip(ratio)×adv))
                        │
                   ratio = exp(new_log_probs − old_log_probs)
                        │              │             │
                   有梯度          detach(无梯度)   detach(无梯度)
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    actor_conv     actor_kernel    actor_init
         │              │              │
         └──────────────┼──────────────┘
                        ▼
                      LSTM
                        │
                        ▼
                   input_proj

Actor头参数 ← 直接梯度
LSTM+input_proj ← 间接梯度(通过h_t)
Critic头 ← ❌ 不在此计算图中
```

#### Critic Loss 的梯度流

```
critic_loss = MSE(new_values, returns)
                   │           │
               有梯度     detach(无梯度)
                   │
                   ▼
              critic 头 (3层MLP)
                   │
                   ▼
                  h_t
                   │
                   ▼
                 LSTM
                   │
                   ▼
              input_proj

Critic头参数 ← 直接梯度
LSTM+input_proj ← 间接梯度(通过h_t)
Actor头 ← ❌ 不在此计算图中
```

#### Entropy 的梯度流

```
entropy = mean(H(π_new))
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
actor_conv  actor_  actor_init   (每个Categorical.entropy())
    │      kernel       │
    └─────────┼─────────┘
              ▼
            h_t → LSTM → input_proj

Actor头 ← 直接梯度 (推动分布向高熵方向)
LSTM+input_proj ← 间接梯度
Critic头 ← ❌ 不在此计算图中
```

### 6.4 第一轮 epoch 的特殊情况

```
epoch 1: ratio = 1.0 (因为 θ_new = θ_old)
         advantages 零均值 → actor_loss 均值为 0
         → Actor loss 不贡献有效梯度
         
         但: critic_loss ≠ 0 (Critic 是随机初始化的)
              entropy ≠ 0
         → LSTM被critic_loss和entropy更新
         → h_t 变了

epoch 2: h_t 变化 → new_log_probs ≠ old_log_probs → ratio ≠ 1.0
         → actor_loss 开始贡献梯度
         → Actor头被更新
```

这就是为什么第一轮 epoch 不会浪费——Critic 和 Entropy 帮 LSTM "热身"，为 Actor 的后续优化铺路。

### 6.5 梯度裁剪

```python
clip_grad_norm_(self.controller.parameters(), max_grad=0.5)
```

所有参数的梯度拼成一个大向量 g，若 ||g||₂ > 0.5，则按比例缩小：

```
g ← g × (0.5 / ||g||₂)
```

防止某一步梯度爆炸导致策略崩溃。

---

## 7. 完整迭代总结

```
┌──────────────────────────────────────────────────────────────────┐
│                    一次完整的 train_one_iteration                 │
│                                                                  │
│  ① forward(16)                                                   │
│     LSTM 逐时间步采样 → 16个架构                                  │
│     记录: log_probs_old (16,7), values_old (16,7)               │
│                                                                  │
│  ② 评估 16个架构                                                 │
│     for arch in architectures:                                   │
│         metrics = train(arch)           # 训练 → nmse, params    │
│         r = -nmse - 0.1×(params/100K)  # 奖励函数               │
│     → rewards (16,)                                              │
│                                                                  │
│  ③ GAE 计算                                                     │
│     for t in reversed(range(7)):                                 │
│         A_t = δ_t + 0.94×A_{t+1}                                 │
│     → advantages (16,7), returns (16,7)                          │
│     → 全部 detach()                                              │
│                                                                  │
│  ④ PPO 更新 ×4                                                   │
│     for epoch in range(4):                                       │
│         indices = randperm(16)                                    │
│         for start in 0,4,8,12:   # 4个minibatch                  │
│             evaluate(archs_batch) → new_log_probs, V_new, H_new  │
│             ratio = exp(new - old)                                │
│             actor_loss  = -mean(min(ratio×A, clip(ratio)×A))     │
│             critic_loss = MSE(V_new, returns)                     │
│             loss = actor + 0.5×critic - 0.01×H_new               │
│             loss.backward()                                       │
│             clip_grad_norm(0.5)                                   │
│             optimizer.step()                                      │
│                                                                  │
│  ⑤ 返回 stats, architectures, rewards, nmse_list, params_list   │
└──────────────────────────────────────────────────────────────────┘

重复 500 次 → 搜索完成 → 输出 Pareto 前沿
```

---

## 附录：关键超参数速查

| 参数 | 值 | 作用 |
|------|-----|------|
| `rollouts_per_iter` | 16 | 每轮采样架构数 |
| `ppo_epochs` | 4 | 同一批数据的重复利用率 |
| `mini_batch_size` | 4 | 每次梯度更新用的样本数 |
| `clip_epsilon` | 0.2 | PPO策略更新幅度上限 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE偏差-方差权衡 |
| `lr` | 3e-4 | Adam学习率 |
| `value_loss_coef` | 0.5 | Critic损失权重 |
| `entropy_coef` | 0.01 | 熵正则化强度 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `total_iterations` | 500 | 总迭代轮数 |
