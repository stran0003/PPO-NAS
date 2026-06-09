"""
搜索空间定义
============
DPD 七层混合模型的架构搜索空间。

固定结构:
    L1(线性) → L2(线性) → L3(线性) → L4(LUT非线性) → L5(线性) → L6(线性) → L7(线性)

每个时间步的决策:
    线性层: 同时选择 kernel_size + conv_type + init_strategy (3个独立分布)
    LUT层:  选择 spline_count (1个分布)

MODIFY HERE: 修改下面的列表可以改变搜索范围
"""

import random

# ============================================================
# MODIFY HERE: 搜索范围 — 改这里就可以调整搜索空间
# ============================================================

KERNEL_SIZES = [11, 15, 21, 27, 33]       # 卷积核大小
CONV_TYPES = ["grouped", "1x1", "skip"]  # 卷积类型: 分组卷积/合路层/直通
SPLINE_COUNTS = [8, 16, 32, 64]          # LUT 样条个数
INIT_STRATEGIES = ["center_spike"]        # 初始化策略

# ============================================================
# 层类型定义（固定）
# ============================================================

LAYER_TYPES = ["linear", "linear", "linear", "lut", "linear", "linear", "linear"]


# ============================================================
# 动作维度: 每个时间步有几个独立的选择，每个选择有多少选项
# ============================================================

def get_action_dims(layer_idx: int) -> list:
    """
    返回第 layer_idx 层需要做几个独立选择，每个选择有多少个选项。

    线性层: [len(KERNEL), len(CONV), len(INIT)]
    LUT层:  [len(SPLINE)]
    """
    if LAYER_TYPES[layer_idx] == "linear":
        return [len(KERNEL_SIZES), len(CONV_TYPES), len(INIT_STRATEGIES)]
    else:
        return [len(SPLINE_COUNTS)]


def get_feature_dim() -> int:
    """动态计算编码后的特征向量维度。"""
    return (2 +                          # 层类型 (linear/LUT)
            len(KERNEL_SIZES) +          # 卷积核 one-hot
            len(CONV_TYPES) +            # 卷积类型 one-hot
            len(INIT_STRATEGIES) +       # 初始化 one-hot
            len(SPLINE_COUNTS) +         # 样条数 one-hot
            2 +                          # LUT初始化预留
            2)                           # 归一化预留


def decode_actions(layer_idx: int, action_indices: list) -> dict:
    """
    把一组动作编号解码成完整的层配置。

    线性层: action_indices = [conv_idx, kernel_idx_or_neg1, init_idx_or_neg1]
        - grouped → kernel_idx 有效, init_idx 有效
        - 1x1     → kernel_idx=-1 (核固定为1), init_idx 有效
        - skip    → kernel_idx=-1, init_idx=-1 (直通无参数)

    LUT层:  action_indices = [spline_idx]
    """
    if LAYER_TYPES[layer_idx] == "linear":
        c_idx = action_indices[0]
        k_idx = action_indices[1]
        i_idx = action_indices[2]
        conv = CONV_TYPES[c_idx]

        if conv == "skip":
            kernel = 0
            init = "none"
        elif conv == "1x1":
            kernel = 1
            init = INIT_STRATEGIES[i_idx] if i_idx >= 0 else "center_spike"
        else:  # grouped
            kernel = KERNEL_SIZES[k_idx]
            init = INIT_STRATEGIES[i_idx] if i_idx >= 0 else "center_spike"

        return {"type": "linear", "kernel": kernel, "conv": conv, "init": init}
    else:
        return {
            "type": "lut",
            "spline": SPLINE_COUNTS[action_indices[0]],
        }


# ============================================================
# 特征编码
# ============================================================

def encode_layer_action(action: dict) -> list:
    """
    把一个层的完整配置编码成特征向量（给 LSTM 输入用）。
    维度 = get_feature_dim()，随搜索空间自动适配。

    action 示例:
      {"type": "linear", "kernel": 15, "conv": "grouped", "init": "center_spike"}
      {"type": "lut", "spline": 16}
      {"type": "linear", "kernel": 0, "conv": "skip", "init": "none"}
    """
    vec = []

    # 层类型 (2)
    vec += [1, 0] if action["type"] == "linear" else [0, 1]
    # 卷积核 (len(KERNEL_SIZES))
    k = action.get("kernel", max(KERNEL_SIZES))
    vec += [1 if k == ks else 0 for ks in KERNEL_SIZES]
    # 卷积类型 (len(CONV_TYPES)，动态)
    ct = action.get("conv", CONV_TYPES[0])
    vec += [1 if ct == c else 0 for c in CONV_TYPES]
    # 初始化 (len(INIT_STRATEGIES))
    init = action.get("init", INIT_STRATEGIES[0])
    vec += [1 if init == s else 0 for s in INIT_STRATEGIES]
    # 样条数 (len(SPLINE_COUNTS))
    sp = action.get("spline", SPLINE_COUNTS[0])
    vec += [1 if sp == s else 0 for s in SPLINE_COUNTS]
    # LUT 初始化预留 (2)
    vec += [1, 0]
    # 归一化预留 (2)
    vec += [1, 0]

    return vec


# ============================================================
# 辅助函数
# ============================================================

def random_architecture() -> list:
    """生成一个随机架构，每个属性独立随机选择。"""
    arch = []
    for i, ltype in enumerate(LAYER_TYPES):
        if ltype == "linear":
            arch.append({
                "type": "linear",
                "kernel": random.choice(KERNEL_SIZES),
                "conv": random.choice(CONV_TYPES),
                "init": random.choice(INIT_STRATEGIES),
            })
        else:
            arch.append({
                "type": "lut",
                "spline": random.choice(SPLINE_COUNTS),
            })
    return arch


def architecture_to_str(arch: list) -> str:
    """架构 → 可读字符串。"""
    parts = []
    for i, a in enumerate(arch):
        if a["type"] == "linear":
            if a.get("conv") == "skip":
                parts.append(f"L{i+1}(skip)")
            else:
                parts.append(f"L{i+1}({a.get('conv','?')},k={a.get('kernel','?')},init={a.get('init','?')})")
        else:
            parts.append(f"L{i+1}(LUT,sp={a.get('spline','?')})")
    return " -> ".join(parts)


def estimate_params(arch: list) -> int:
    """粗略估算参数量。skip 层参数量为 0。"""
    total = 0
    for a in arch:
        if a["type"] == "linear":
            conv = a.get("conv", "grouped")
            if conv == "skip":
                total += 0
            elif conv == "1x1":
                total += 64 * 64
            else:
                k = a.get("kernel", 15)
                total += k * 8 * 64
        else:
            total += a.get("spline", 16) * 256
    return total
