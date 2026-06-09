"""
奖励函数
========
计算一个完整架构的奖励值。

架构生成完毕后，一次性给奖励（延迟奖励）。
PPO 内部用 GAE 把这个终端奖励分配到每一步。

MODIFY HERE: 修改 compute_reward() 函数来改变奖励计算方式。
"""

import math


# ============================================================
# MODIFY HERE: 奖励计算
# ============================================================

def _get_param_weight(config, iteration):
    """
    计算当前迭代的 param_weight (支持自动退火).

    param_weight_init:     初始值 (前期不管参数量)
    param_weight_final:    最终值 (后期约束参数量)
    param_weight_anneal_start: 从第几轮开始退火
    param_weight_anneal_iters: 退火持续多少轮

    例如: init=0.0, final=0.1, start=50, iters=150
      iter≤50:  weight=0.0
      iter=125: weight=0.05 (中期)
      iter≥200: weight=0.1 (到达终值)
    """
    pw_init = config.get("param_weight_init", 0.0)
    pw_final = config.get("param_weight_final", 0.1)
    pw_start = config.get("param_weight_anneal_start", 50)
    pw_iters = config.get("param_weight_anneal_iters", 150)

    if iteration is None or pw_iters <= 0:
        return config.get("param_weight", pw_final)

    if iteration < pw_start:
        return pw_init
    elif iteration >= pw_start + pw_iters:
        return pw_final
    else:
        progress = (iteration - pw_start) / pw_iters
        return pw_init + (pw_final - pw_init) * progress


def compute_reward(nmse, num_params, config=None, iteration=None, total_iterations=None):
    """
    根据模型性能和参数量计算奖励。

    参数:
        nmse:             性能指标 (dB, 越小越好, 如 -32.5 表示 res=32.5dB)
        num_params:       参数量 (越小越好)
        config:           配置字典, 取 reward 部分
        iteration:        当前迭代轮数 (用于 ADF 退火)
        total_iterations: 总迭代轮数 (用于 ADF 退火)

    返回: 奖励值 (越大越好)

    策略选择 (config["type"]):

    ── simple (简单线性加权) ──
        R = -nmse * acc_weight − param_weight × (num_params / target_params)
        优点: 简单直观, 参数量越接近 target 惩罚越小
        缺点: 对极端参数量没有硬性约束

    ── adf (退火期望函数) ──
        R = (-nmse) × acc_weight × window(num_params)
        window(p) = max(0, 1 − |p − tau| / delta)

        其中 tau 和 delta 可随时间退火:
          tau(t)    = tau_init    + (tau_final    − tau_init)    × t/T
          delta(t)  = delta_init  + (delta_final  − delta_init)  × t/T

        优点: 在 tau 周围形成一个"奖励窗口", 离太远直接零分
        退火: 初期 delta 大(宽容), 后期收紧, 引导搜索收敛到目标参数量
    """
    if config is None:
        config = {}

    reward_type = config.get("type", "simple")
    acc_weight = config.get("acc_weight", 1.0)

    # 性能得分: nmse 越小 → -nmse 越大 → 越好
    perf = -nmse * acc_weight

    # ── 根据策略计算奖励 ──
    if reward_type == "adf":
        param_penalty = _adf_penalty(
            num_params, config, iteration, total_iterations
        )
        reward = perf - param_penalty
    else:
        # simple (默认), 支持 param_weight 自动退火
        param_weight = _get_param_weight(config, iteration)
        target_params = config.get("target_params", 20000)
        param_ratio = num_params / target_params
        param_penalty = param_weight * param_ratio
        reward = perf - param_penalty

    return reward


def _adf_penalty(params, config, iteration, total_iterations):
    """
    ADF 参数量惩罚项。

    在目标区间 [tau−delta, tau+delta] 内 → 惩罚 = 0 (只看性能)
    超出区间 → 惩罚随距离线性增长

    退火: tau 和 delta 随训练从 init 线性过渡到 final
    """
    tau_init = config.get("adf_tau_init", 30000)
    tau_final = config.get("adf_tau_final", 15000)
    delta_init = config.get("adf_delta_init", 30000)
    delta_final = config.get("adf_delta_final", 5000)
    penalty_slope = config.get("adf_penalty_slope", 0.5)  # 超出窗口后每 1000 参数扣多少

    # 线性退火
    if iteration is not None and total_iterations is not None and total_iterations > 0:
        progress = min(1.0, iteration / total_iterations)
        tau = tau_init + (tau_final - tau_init) * progress
        delta = delta_init + (delta_final - delta_init) * progress
    else:
        tau = tau_init
        delta = delta_init

    distance = abs(params - tau)

    if distance <= delta:
        return 0.0           # 在窗口内，零惩罚
    else:
        return penalty_slope * (distance - delta) / 1000.0


# ============================================================
# Pareto 前沿管理（用于跟踪搜索过程中的最优架构）
# ============================================================

class ParetoFront:
    """
    维护搜索过程中发现的 Pareto 最优架构集合。

    支配关系: A 支配 B 当且仅当
      - A 的 NMSE ≤ B 的 NMSE (越小越好)
      - A 的参数量 ≤ B 的参数量 (越小越好)
      - 至少一个严格不等
    """

    def __init__(self):
        self.archs = []  # list of (nmse, params, architecture)

    def add(self, nmse, params, architecture):
        """尝试添加一个架构到前沿。"""
        # 检查是否被已有架构支配
        for n, p, _ in self.archs:
            if n <= nmse and p <= params and (n < nmse or p < params):
                return False  # 被支配，不添加

        # 移除被新架构支配的旧架构
        self.archs = [(n, p, a) for n, p, a in self.archs
                       if not (nmse <= n and params <= p and (nmse < n or params < p))]

        self.archs.append((nmse, params, architecture))
        return True

    def get_best_by_nmse(self):
        """返回 NMSE 最好的架构。"""
        if not self.archs:
            return None
        return min(self.archs, key=lambda x: x[0])

    def get_best_by_params(self):
        """返回参数量最小的架构。"""
        if not self.archs:
            return None
        return min(self.archs, key=lambda x: x[1])

    def __len__(self):
        return len(self.archs)

    def __iter__(self):
        return iter(sorted(self.archs, key=lambda x: x[1]))
