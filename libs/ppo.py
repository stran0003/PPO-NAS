"""
PPO 训练器
==========
实现 PPO 的核心算法:
  1. GAE (Generalized Advantage Estimation) 计算优势函数
  2. Clipped Surrogate Objective 更新 Actor
  3. MSE Loss 更新 Critic
  4. 熵正则化鼓励探索
"""

import time
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOTrainer:
    """
    PPO 训练器。

    用法:
        trainer = PPOTrainer(controller, config)
        stats = trainer.train_one_iteration(eval_fn)
    """

    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

        # 优化器 (Actor + Critic 共享)
        self.optimizer = torch.optim.Adam(
            controller.parameters(),
            lr=config.get("lr", 3e-4),
            eps=1e-5,
        )

        # 训练状态
        self.iteration = 0
        self.stats_history = []

    def train_one_iteration(self, eval_fn=None):
        """
        执行一轮 PPO 训练:
          1. 采样架构
          2. 评估 → 得奖励
          3. 计算 GAE 优势
          4. PPO 更新（多轮）

        返回:
            stats, architectures, rewards, nmse_list, params_list
        """
        config = self.config
        t_start = time.time()

        # ── 0. 温度退火 ──
        temp_init = config.get("temp_init", 1.0)
        temp_final = config.get("temp_final", 0.5)
        total_iter = config.get("total_iterations", 500)
        if total_iter > 0:
            progress = min(1.0, self.iteration / total_iter)
            temp = temp_init + (temp_final - temp_init) * progress
            self.controller.set_temperature(temp)

        # ── 1. 采样 ──
        n_rollout = config.get("rollouts_per_iter", 16)
        architectures, log_probs, values, entropies, actions = \
            self.controller.forward(n_rollout)

        # ── 2. 评估 ──
        rewards = []
        nmse_list = []
        params_list = []
        for arch in architectures:
            if eval_fn is not None:
                metrics = eval_fn(arch)
            else:
                # 模拟评估
                from .model.dpd_model import evaluate_architecture
                metrics = evaluate_architecture(arch)

            nmse = metrics.get("nmse", -30.0)
            params = metrics.get("num_params", 100000)

            from .reward import compute_reward
            r = compute_reward(nmse, params, config.get("reward", {}),
                               iteration=self.iteration,
                               total_iterations=config.get("total_iterations", 500))

            rewards.append(r)
            nmse_list.append(nmse)
            params_list.append(params)

        rewards = torch.tensor(rewards, device=values.device)

        # ── 3. GAE 计算优势 ──
        advantages, returns = self._compute_gae(rewards, values)

        # ── 3.5 GAE 调试日志 ──
        log_interval = config.get("log_interval", 5)
        self._write_gae_log(architectures, nmse_list, params_list, log_interval)

        # ── 4. PPO 更新 ──
        stats = {}
        for epoch in range(config.get("ppo_epochs", 4)):
            batch_stats = self._ppo_update(
                architectures, actions, log_probs, values,
                advantages, returns, entropies,
            )
            for k, v in batch_stats.items():
                stats[k] = stats.get(k, 0) + v

        # 平均
        for k in stats:
            stats[k] /= config.get("ppo_epochs", 4)

        # ── 5. 统计 ──
        stats["iteration"] = self.iteration
        stats["mean_reward"] = rewards.mean().item()
        stats["reward_min"] = rewards.min().item()
        stats["reward_max"] = rewards.max().item()
        stats["reward_std"] = rewards.std().item()
        stats["mean_nmse"] = sum(nmse_list) / len(nmse_list)
        stats["nmse_min"] = min(nmse_list)
        stats["nmse_max"] = max(nmse_list)
        stats["mean_params"] = sum(params_list) / len(params_list)
        stats["mean_entropy"] = entropies.mean().item()
        stats["elapsed"] = time.time() - t_start

        self.stats_history.append(stats)
        self.iteration += 1

        return stats, architectures, rewards.tolist(), nmse_list, params_list

    # ─── GAE ────────────────────────────────────────────────────

    def _compute_gae(self, rewards, values):
        """
        Generalized Advantage Estimation.

        δ_t = r_{t+1} + γ·V(s_{t+1}) - V(s_t)    (TD error)
        A_t = Σ (γλ)^l · δ_{t+l}

        注意: NAS 中只在最后一步给奖励, 中间步骤 r=0。
        所以 δ_{T-1} = reward - V(s_{T-1})
            δ_{t}   = γ·V(s_{t+1}) - V(s_t)    (t < T-1)
        """
        gamma = self.config.get("gamma", 0.99)
        gae_lambda = self.config.get("gae_lambda", 0.95)

        B, T = values.shape
        advantages = torch.zeros(B, T, device=values.device)
        deltas = torch.zeros(B, T, device=values.device)

        # 从后往前累加
        gae = torch.zeros(B, device=values.device)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0  # 终止状态价值为0
                delta = rewards - values[:, t]
            else:
                next_value = values[:, t + 1]
                delta = gamma * next_value - values[:, t]

            deltas[:, t] = delta
            gae = delta + gamma * gae_lambda * gae
            advantages[:, t] = gae

        # 保存原始优势（归一化前）供调试用
        raw_advantages = advantages.clone()

        # returns = advantage + value (用于 Critic 训练)
        returns = advantages + values

        # 归一化优势（稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 存储调试数据
        self._gae_debug = {
            "rewards": rewards,
            "values": values,
            "deltas": deltas,
            "raw_advantages": raw_advantages,
            "advantages": advantages,
            "returns": returns,
        }

        return advantages, returns

    # ─── GAE 调试日志 ─────────────────────────────────────────

    def _write_gae_log(self, architectures, nmse_list, params_list, log_interval=5):
        """
        将每步的 r, V(s), δ, A_raw, A_norm 写入 CSV 和 Excel 文件。
        每 log_interval 轮写一次，每次覆盖 Excel（保留最新完整数据），
        CSV 追加写入（保留历史）。
        """
        if self.iteration % log_interval != 0:
            return

        debug = self._gae_debug
        if debug is None:
            return

        B, T = debug["values"].shape
        out_dir = self.config.get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "gae_debug.csv")
        xlsx_path = os.path.join(out_dir, "gae_debug.xlsx")

        columns = [
            "iteration", "arch_idx", "step",
            "reward", "nmse", "num_params",
            "V(s)", "delta", "A_raw", "A_norm", "returns",
            "conv_type", "kernel"
        ]

        # 收集本轮数据
        rows = []
        for b in range(B):
            arch = architectures[b]
            r = debug["rewards"][b].item()
            nmse = nmse_list[b]
            n_param = params_list[b]
            for t in range(T):
                layer = arch[t]
                conv_type = layer.get("conv", "LUT")
                kernel = layer.get("kernel", layer.get("spline", "—"))
                rows.append([
                    self.iteration, b, t,
                    round(r, 4), round(nmse, 4), n_param,
                    round(debug["values"][b, t].item(), 4),
                    round(debug["deltas"][b, t].item(), 4),
                    round(debug["raw_advantages"][b, t].item(), 4),
                    round(debug["advantages"][b, t].item(), 4),
                    round(debug["returns"][b, t].item(), 4),
                    conv_type, kernel,
                ])

        # CSV: 追加写入
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(columns)
            writer.writerows(rows)

        # Excel: 覆盖写入（只保留最新一轮，文件不会无限膨胀）
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=columns)
            df.to_excel(xlsx_path, index=False, sheet_name=f"iter_{self.iteration}")
        except Exception:
            pass  # pandas 不可用时跳过

    # ─── PPO 更新 ───────────────────────────────────────────────

    def _ppo_update(self, architectures, actions_old, log_probs_old,
                    values_old, advantages, returns, entropies_old):
        """一次 PPO epoch 的参数更新。"""
        config = self.config
        clip_eps = config.get("clip_epsilon", 0.2)
        vf_coef = config.get("value_loss_coef", 0.5)
        ent_coef = config.get("entropy_coef", 0.01)
        max_grad = config.get("max_grad_norm", 0.5)
        mini_batch = config.get("mini_batch_size", 4)

        # 必须 detach rollout 数据，否则多次 backward 会冲突
        log_probs_old = log_probs_old.detach()
        values_old = values_old.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        B = len(architectures)
        T = advantages.shape[1]
        device = advantages.device

        # 随机打乱
        indices = torch.randperm(B)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl = 0.0

        for start in range(0, B, mini_batch):
            idx = indices[start:start + mini_batch]
            b_B = len(idx)

            # 重新评估当前策略下的值
            batch_archs = [architectures[i] for i in idx]
            batch_actions = [actions_old[i] for i in idx]
            new_log_probs, new_values, new_entropies = \
                self.controller.evaluate(batch_archs, batch_actions)

            old_log_probs = log_probs_old[idx]
            old_values = values_old[idx]
            batch_adv = advantages[idx]
            batch_ret = returns[idx]

            # -- Actor 损失 (Clipped Surrogate) --
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # -- Critic 损失 (MSE) --
            critic_loss = F.mse_loss(new_values, batch_ret)

            # -- 熵正则化 --
            entropy = new_entropies.mean()

            # -- 总损失 --
            loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

            # -- 反向传播 --
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.controller.parameters(), max_grad)
            self.optimizer.step()

            # 记录
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            with torch.no_grad():
                total_kl += (old_log_probs - new_log_probs).mean().item()

        n_batches = max(1, B // mini_batch)
        return {
            "actor_loss": total_actor_loss / n_batches,
            "critic_loss": total_critic_loss / n_batches,
            "approx_kl": total_kl / n_batches,
        }

    # ─── 保存/加载 ──────────────────────────────────────────────

    def save(self, path):
        torch.save({
            "iteration": self.iteration,
            "controller": self.controller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.controller.load_state_dict(ckpt["controller"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.iteration = ckpt["iteration"]
