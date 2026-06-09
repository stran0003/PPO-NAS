"""
Actor-Critic 控制器
===================
用 LSTM 逐层生成架构，Actor 输出动作概率，Critic 输出状态价值。

结构:
    输入(上一层动作) → [共享LSTM] → h_t ─┬→ [Actor头们] → 多个独立动作分布
                                        └→ [Critic头] → 状态价值 V(s)

每个时间步的采样:
    线性层: 从 3 个独立分布各采一个 → kernel(5选1) + conv(2选1) + init(3选1)
    LUT层:  从 1 个分布采样      → spline(5选1)

    该步的 log_prob = 各分布 log_prob 之和
    该步的 entropy  = 各分布 entropy 之和
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_space import (
    LAYER_TYPES, encode_layer_action,
    get_action_dims, decode_actions, get_feature_dim,
    KERNEL_SIZES, CONV_TYPES, SPLINE_COUNTS, INIT_STRATEGIES,
)


class Controller(nn.Module):
    """
    Actor-Critic LSTM 控制器。

    用法:
        ctrl = Controller(hidden_dim=128)
        archs, log_probs, values, entropies, actions = ctrl.forward(4)
    """

    def __init__(self, hidden_dim=128, num_lstm_layers=2, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = len(LAYER_TYPES)  # 7

        # -- 输入投影: 特征维度从 search_space 动态读取 --
        feat_dim = get_feature_dim()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)

        # -- 共享 LSTM --
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # -- 起始符（t=0 时用）--
        self.start_token = nn.Parameter(torch.zeros(1, hidden_dim))

        # -- Actor 头: 维度从 search_space 动态读取 --
        self.actor_kernel = nn.Linear(hidden_dim, len(KERNEL_SIZES))
        self.actor_conv   = nn.Linear(hidden_dim, len(CONV_TYPES))
        self.actor_init   = nn.Linear(hidden_dim, len(INIT_STRATEGIES))
        self.actor_spline = nn.Linear(hidden_dim, len(SPLINE_COUNTS))

        # -- Critic 头: h → scalar --
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # -- 温度（控制探索程度）--
        self.register_buffer('temperature', torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _init_hidden(self, batch_size, device):
        h = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    # ─── 生成架构（采样）─────────────────────────────────────────

    def forward(self, batch_size=1, deterministic=False):
        """
        生成一批架构。每个时间步从多个独立分布各采样一个动作。

        返回:
            architectures: 架构列表，每个是 7 个 dict 的列表
            log_probs:     (B, 7) — 每步总 log 概率
            values:        (B, 7) — 每步状态价值
            entropies:     (B, 7) — 每步总熵
            all_actions:   每步各头的动作编号
                           [batch][step] = [conv_idx, kernel_idx_or_-1, init_idx]  (线性, kernel=-1表示没采样)
                           [batch][step] = [spline_idx]                            (LUT)
        """
        device = next(self.parameters()).device
        hidden = self._init_hidden(batch_size, device)

        architectures = [[] for _ in range(batch_size)]
        log_probs = torch.zeros(batch_size, self.num_layers, device=device)
        values = torch.zeros(batch_size, self.num_layers, device=device)
        entropies = torch.zeros(batch_size, self.num_layers, device=device)
        all_actions = [[None for _ in range(self.num_layers)] for _ in range(batch_size)]

        prev_x = None

        for t in range(self.num_layers):
            layer_type = LAYER_TYPES[t]

            # ── LSTM 一步 ──
            if prev_x is None:
                lstm_in = self.start_token.unsqueeze(0).expand(1, batch_size, self.hidden_dim)
            else:
                proj = self.input_proj(prev_x)
                lstm_in = proj.unsqueeze(0)

            out, hidden = self.lstm(lstm_in, hidden)
            h_t = out.squeeze(0)  # (B, hidden_dim)

            # ── Actor: 线性层: conv → (grouped时选kernel) → init ──
            if layer_type == "linear":
                step_log_prob = torch.zeros(batch_size, device=device)
                step_entropy = torch.zeros(batch_size, device=device)
                step_actions = []

                # 1) 卷积类型 (grouped / 1x1 / skip)
                logits_c = self.actor_conv(h_t) / self.temperature
                dist_c = torch.distributions.Categorical(logits=logits_c)
                if deterministic:
                    c = torch.argmax(logits_c, dim=-1)
                else:
                    c = dist_c.sample()
                step_log_prob += dist_c.log_prob(c)
                step_entropy += dist_c.entropy()
                step_actions.append(c)  # 索引 0

                # 2) 卷积核: 只有 grouped 有效, 1x1→k=-1, skip→k=-1
                logits_k = self.actor_kernel(h_t) / self.temperature
                dist_k = torch.distributions.Categorical(logits=logits_k)
                if deterministic:
                    k = torch.argmax(logits_k, dim=-1)
                else:
                    k = dist_k.sample()

                mask_need_kernel = (c == 0)  # CONV_TYPES[0] = "grouped"
                step_log_prob += dist_k.log_prob(k) * mask_need_kernel.float()
                step_entropy += dist_k.entropy() * mask_need_kernel.float()
                k_final = torch.where(mask_need_kernel, k, torch.tensor(-1, device=device))
                step_actions.append(k_final)  # 索引 1

                # 3) 初始化策略: skip 不计入 (无参数不需要初始化)
                logits_i = self.actor_init(h_t) / self.temperature
                dist_i = torch.distributions.Categorical(logits=logits_i)
                if deterministic:
                    i = torch.argmax(logits_i, dim=-1)
                else:
                    i = dist_i.sample()

                mask_has_weights = (c != 2)  # CONV_TYPES[2] = "skip" → 无权重
                step_log_prob += dist_i.log_prob(i) * mask_has_weights.float()
                step_entropy += dist_i.entropy() * mask_has_weights.float()
                i_final = torch.where(mask_has_weights, i, torch.tensor(-1, device=device))
                step_actions.append(i_final)  # 索引 2

            else:
                # LUT 层: 只选样条数
                step_log_prob = torch.zeros(batch_size, device=device)
                step_entropy = torch.zeros(batch_size, device=device)
                step_actions = []

                logits = self.actor_spline(h_t) / self.temperature
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    a = torch.argmax(logits, dim=-1)
                else:
                    a = dist.sample()
                step_log_prob += dist.log_prob(a)
                step_entropy += dist.entropy()
                step_actions.append(a)

            log_probs[:, t] = step_log_prob
            entropies[:, t] = step_entropy

            # ── Critic ──
            values[:, t] = self.critic(h_t).squeeze(-1)

            # ── 解码动作 → 层配置 → 下一时间步的输入 ──
            feature_vecs = []
            for b in range(batch_size):
                # 取出第 b 个样本在各头的动作编号
                indices = [head_actions[b].item() for head_actions in step_actions]
                all_actions[b][t] = indices

                layer_action = decode_actions(t, indices)
                architectures[b].append(layer_action)

                if t < self.num_layers - 1:
                    feature_vecs.append(encode_layer_action(layer_action))

            if t < self.num_layers - 1:
                prev_x = torch.tensor(feature_vecs, device=device, dtype=torch.float32)

        return architectures, log_probs, values, entropies, all_actions

    # ─── 重新评估（PPO 更新用）─────────────────────────────────

    def evaluate(self, architectures, action_seqs):
        """
        给定一批已有的 (架构, 动作序列)，重新计算 log_probs, values, entropies。

        参数:
            architectures: batch 中每个架构的完整 7 层配置
            action_seqs:   batch 中每个架构每步各头的动作编号
                           线性层: [conv_idx, kernel_idx_or_-1, init_idx]
                           LUT层:  [spline_idx]

        返回:
            log_probs: (B, 7) — 总 log 概率
            values:    (B, 7) — 状态价值
            entropies: (B, 7) — 总熵
        """
        B = len(architectures)
        T = self.num_layers
        device = next(self.parameters()).device
        hidden = self._init_hidden(B, device)

        log_probs = torch.zeros(B, T, device=device)
        values = torch.zeros(B, T, device=device)
        entropies = torch.zeros(B, T, device=device)
        prev_x = None

        for t in range(T):
            layer_type = LAYER_TYPES[t]

            # LSTM 一步
            if prev_x is None:
                lstm_in = self.start_token.unsqueeze(0).expand(1, B, self.hidden_dim)
            else:
                lstm_in = self.input_proj(prev_x).unsqueeze(0)

            out, hidden = self.lstm(lstm_in, hidden)
            h_t = out.squeeze(0)

            if layer_type == "linear":
                # 1) 卷积类型
                logits_c = self.actor_conv(h_t) / self.temperature
                dist_c = torch.distributions.Categorical(logits=logits_c)
                c_t = torch.tensor([action_seqs[b][t][0] for b in range(B)],
                                   device=device, dtype=torch.long)
                lp = dist_c.log_prob(c_t)
                ent = dist_c.entropy()

                # 2) 卷积核: 只有 grouped 才计入
                logits_k = self.actor_kernel(h_t) / self.temperature
                dist_k = torch.distributions.Categorical(logits=logits_k)
                k_t = torch.tensor([action_seqs[b][t][1] for b in range(B)],
                                   device=device, dtype=torch.long)
                mask_k = (k_t >= 0)
                lp += dist_k.log_prob(k_t.clamp(min=0)) * mask_k.float()
                ent += dist_k.entropy() * mask_k.float()

                # 3) 初始化策略: skip 不计入
                logits_i = self.actor_init(h_t) / self.temperature
                dist_i = torch.distributions.Categorical(logits=logits_i)
                i_t = torch.tensor([action_seqs[b][t][2] for b in range(B)],
                                   device=device, dtype=torch.long)
                mask_i = (i_t >= 0)  # skip → i_idx=-1 → mask=0
                lp += dist_i.log_prob(i_t.clamp(min=0)) * mask_i.float()
                ent += dist_i.entropy() * mask_i.float()

            else:
                # LUT 层
                logits = self.actor_spline(h_t) / self.temperature
                dist = torch.distributions.Categorical(logits=logits)
                a_t = torch.tensor([action_seqs[b][t][0] for b in range(B)],
                                   device=device, dtype=torch.long)
                lp = dist.log_prob(a_t)
                ent = dist.entropy()

            log_probs[:, t] = lp
            entropies[:, t] = ent
            values[:, t] = self.critic(h_t).squeeze(-1)

            if t < T - 1:
                feature_vecs = [encode_layer_action(architectures[b][t]) for b in range(B)]
                prev_x = torch.tensor(feature_vecs, device=device, dtype=torch.float32)

        return log_probs, values, entropies

    def set_temperature(self, temp):
        self.temperature.fill_(temp)
