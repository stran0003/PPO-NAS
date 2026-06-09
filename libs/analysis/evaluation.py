"""
评估工具
========
包含: 渐进淘汰(NOSH)、排序相关性验证、指标追踪。

这些是辅助工具，搜索主流程不依赖它们。
"""

import math
import time
import json
import numpy as np


# ============================================================
# NOSH 渐进淘汰
# ============================================================

class NOSH:
    """
    非均匀连续折半 (Non-uniform Successive Halving)。

    思想: 先给所有候选少量训练预算，差的淘汰，好的加预算继续。
    可以节省 60-70% 的总评估时间。

    用法:
        nosh = NOSH(stages=[50, 150, 300, 1000])
        survivors = nosh.run(architectures, eval_fn)
    """

    def __init__(self, stages=(50, 150, 300), keep_ratio=0.5):
        """
        stages:     每阶段的训练 epoch 数
        keep_ratio: 每阶段保留的比例
        """
        self.stages = stages
        self.keep_ratio = keep_ratio

    def run(self, architectures, eval_fn, verbose=True):
        """
        运行渐进淘汰。

        eval_fn(arch, epochs) → dict: 评估函数，返回 {"nmse": ..., "num_params": ...}
        """
        candidates = list(architectures)
        eliminated = []

        for stage_idx, epochs in enumerate(self.stages):
            if verbose:
                print(f"NOSH 阶段 {stage_idx+1}: {len(candidates)} 个候选, {epochs} epochs")

            # 评估所有候选
            for arch in candidates:
                metrics = eval_fn(arch, epochs)
                arch["_nmse"] = metrics.get("nmse", 0)
                arch["_params"] = metrics.get("num_params", 0)

            # 按性能排序 (NMSE 越小越好)
            candidates.sort(key=lambda a: a.get("_nmse", 0))

            # 淘汰
            n_keep = max(1, int(len(candidates) * self.keep_ratio))
            if stage_idx == len(self.stages) - 1:
                n_keep = len(candidates)  # 最后阶段全保留

            eliminated.extend(candidates[n_keep:])
            candidates = candidates[:n_keep]

            if verbose:
                best = candidates[0].get("_nmse", 0)
                print(f"  → 保留 {len(candidates)}, 最佳 NMSE={best:.2f}")

        return candidates


# ============================================================
# 排序相关性验证
# ============================================================

def check_ranking_correlation(architectures, eval_fn, short_epochs=300, full_epochs=1000):
    """
    验证短训练排名的可靠性。

    对同一批架构分别做短训练和完整训练，
    计算两种排名之间的 Spearman 相关性。

    返回值 > 0.7 说明短训练代理可靠。
    """
    print(f"验证短训练代理: {short_epochs} vs {full_epochs} epochs")

    short_results = []
    full_results = []

    for i, arch in enumerate(architectures):
        s = eval_fn(arch, short_epochs)
        f = eval_fn(arch, full_epochs)
        short_results.append(s.get("nmse", 0))
        full_results.append(f.get("nmse", 0))

    # Spearman 秩相关
    # 用 numpy 简单计算
    s_rank = np.argsort(np.argsort(short_results))
    f_rank = np.argsort(np.argsort(full_results))

    n = len(s_rank)
    d2 = np.sum((s_rank - f_rank) ** 2)
    spearman = 1 - (6 * d2) / (n * (n**2 - 1)) if n > 1 else 0

    # Pearson
    pearson = np.corrcoef(short_results, full_results)[0, 1]

    print(f"  Spearman ρ = {spearman:.3f}")
    print(f"  Pearson  r = {pearson:.3f}")
    print(f"  {'✅ 可靠' if spearman > 0.7 else '⚠️ 偏低，建议提高 short_epochs'}")

    return spearman, pearson


# ============================================================
# 指标追踪
# ============================================================

class Tracker:
    """追踪搜索过程中的各项指标。"""

    def __init__(self):
        self.data = {
            "rewards": [],
            "nmse": [],
            "params": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "pareto_size": [],
        }

    def update(self, stats, pareto_size=0):
        self.data["rewards"].append(stats.get("mean_reward", 0))
        self.data["nmse"].append(stats.get("mean_nmse", 0))
        self.data["params"].append(stats.get("mean_params", 0))
        self.data["actor_loss"].append(stats.get("actor_loss", 0))
        self.data["critic_loss"].append(stats.get("critic_loss", 0))
        self.data["entropy"].append(stats.get("mean_entropy", 0))
        self.data["pareto_size"].append(pareto_size)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    def summary(self):
        print(f"追踪了 {len(self.data['rewards'])} 轮")
        if self.data["rewards"]:
            print(f"  最终奖励 (MA20): {np.mean(self.data['rewards'][-20:]):.3f}")
            print(f"  最佳 NMSE:        {min(self.data['nmse']):.2f}")
            print(f"  最小参数量:       {min(self.data['params']):.0f}")
