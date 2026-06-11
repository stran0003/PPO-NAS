"""
Visualization Tools
===================
Generate Pareto front plots, training curves, architecture distribution plots.

Requires: pip install matplotlib
If matplotlib is not installed, functions skip gracefully with a message.
"""

import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Color palette
C = {
    "blue": "#2b6cb0",
    "red": "#e53e3e",
    "green": "#38a169",
    "yellow": "#d69e2e",
    "purple": "#6b46c1",
    "grey": "#a0aec0",
    "dark": "#1a365d",
}

if HAS_MPL:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 12,
        "axes.grid": True, "grid.alpha": 0.3,
    })


def need_mpl(fn):
    """Decorator: skip if matplotlib is not installed."""
    def wrap(*args, **kwargs):
        if not HAS_MPL:
            print(f"[Skip] {fn.__name__}: matplotlib not installed")
            return None
        return fn(*args, **kwargs)
    return wrap


# ============================================================
# Pareto Front Plot
# ============================================================

@need_mpl
def plot_pareto(arch_data, save_path="output/plots/pareto.png"):
    """
    Plot Pareto front: X=params, Y=-NMSE (higher is better)

    arch_data: list of {"nmse": float, "num_params": int, "architecture": str}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # All evaluated architectures
    nmse_vals = [d["nmse"] for d in arch_data]
    param_vals = [d["num_params"] for d in arch_data]
    ax.scatter(param_vals, nmse_vals, c=C["grey"], alpha=0.4, s=20, label="Evaluated")

    # Pareto front (nmse越小越好 → 左下角最优)
    front = _find_pareto(arch_data)
    if front:
        fp = [d["num_params"] for d in front]
        fn = [d["nmse"] for d in front]
        idx = np.argsort(fp)
        ax.plot([fp[i] for i in idx], [fn[i] for i in idx],
                "-o", color=C["blue"], markersize=6, label=f"Pareto Front ({len(front)})")

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("NMSE (dB) [lower is better]")
    ax.set_title("Pareto Front: Performance vs Parameters")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Pareto front -> {save_path}")


def _find_pareto(data):
    """Simple O(n^2) Pareto front extraction."""
    front = []
    for d in data:
        dominated = False
        for other in data:
            if (other["nmse"] <= d["nmse"] and other["num_params"] <= d["num_params"]
                    and (other["nmse"] < d["nmse"] or other["num_params"] < d["num_params"])):
                dominated = True
                break
        if not dominated:
            front.append(d)
    return front


# ============================================================
# Training Curves
# ============================================================

@need_mpl
def plot_training(stats_list, save_path="output/plots/training.png"):
    """
    Plot training curves: reward, losses, entropy, KL divergence.

    stats_list: list of dict, one dict per iteration
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def ma(data, w=10):
        if len(data) <= w:
            return [np.mean(data[:max(1, i+1)]) for i in range(len(data))]
        return [np.mean(data[max(0, i-w+1):i+1]) for i in range(len(data))]

    iters = list(range(len(stats_list)))

    # Reward
    ax = axes[0, 0]
    r = [s.get("mean_reward", 0) for s in stats_list]
    ax.plot(iters, r, alpha=0.3, color=C["blue"])
    ax.plot(iters, ma(r), color=C["blue"], linewidth=1.5)
    ax.set_title("Mean Reward")
    ax.set_ylabel("Reward")

    # Actor / Critic Loss
    ax = axes[0, 1]
    al = [s.get("actor_loss", 0) for s in stats_list]
    cl = [s.get("critic_loss", 0) for s in stats_list]
    ax.plot(iters, ma(al), color=C["blue"], label="Actor")
    ax.plot(iters, ma(cl), color=C["red"], label="Critic")
    ax.set_title("Loss")
    ax.legend()

    # Entropy
    ax = axes[1, 0]
    ent = [s.get("mean_entropy", 0) for s in stats_list]
    ax.plot(iters, ent, alpha=0.3, color=C["green"])
    ax.plot(iters, ma(ent), color=C["green"], linewidth=1.5)
    ax.set_title("Policy Entropy (exploration)")
    ax.set_xlabel("Iteration")

    # KL Divergence
    ax = axes[1, 1]
    kl = [s.get("approx_kl", 0) for s in stats_list]
    ax.plot(iters, ma(kl), color=C["purple"], linewidth=1.5)
    ax.axhline(y=0.02, color=C["grey"], linestyle=":", alpha=0.5)
    ax.set_title("KL Divergence (update magnitude)")
    ax.set_xlabel("Iteration")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Training curves -> {save_path}")


# ============================================================
# Architecture Distribution
# ============================================================

@need_mpl
def plot_distribution(arch_data, save_path="output/plots/distribution.png"):
    """
    Plot distributions of architecture choices: kernel size, conv type, spline count.
    所有 bins 从 search_space 自动读取，修改搜索空间后无需手动调整。
    """
    from ..model.search_space import KERNEL_SIZES, CONV_TYPES, SPLINE_COUNTS

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── Kernel size distribution ──
    ax = axes[0]
    kernels = [a["kernel"] for d in arch_data
               for a in d.get("actions", []) if "kernel" in a]
    if kernels:
        # 自适应 bins：以每个 kernel 值为中心，边界取相邻值的中点
        bin_edges = _value_centered_bins(KERNEL_SIZES)
        ax.hist(kernels, bins=bin_edges, color=C["blue"], alpha=0.7, edgecolor="white")
        ax.set_xticks(KERNEL_SIZES)
    ax.set_title("Kernel Size Distribution")
    ax.set_xlabel("Kernel Size")

    # ── Convolution type ──
    ax = axes[1]
    conv_counts = {ct: 0 for ct in CONV_TYPES}
    for d in arch_data:
        for a in d.get("actions", []):
            if a.get("conv") in conv_counts:
                conv_counts[a["conv"]] += 1
    if any(conv_counts.values()):
        colors = [C["blue"], C["green"], C["yellow"], C["purple"]][:len(CONV_TYPES)]
        ax.bar(CONV_TYPES, [conv_counts[ct] for ct in CONV_TYPES],
               color=colors, alpha=0.7)
    ax.set_title("Convolution Type")

    # ── Spline count distribution ──
    ax = axes[2]
    splines = [a["spline"] for d in arch_data
               for a in d.get("actions", []) if "spline" in a]
    if splines:
        bin_edges = _value_centered_bins(SPLINE_COUNTS)
        ax.hist(splines, bins=bin_edges, color=C["yellow"], alpha=0.7, edgecolor="white")
        ax.set_xticks(SPLINE_COUNTS)
    ax.set_title("LUT Spline Count Distribution (L4)")
    ax.set_xlabel("Spline Count")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Architecture distribution -> {save_path}")


# ============================================================
# GAE Debug Heatmap
# ============================================================

@need_mpl
def plot_gae_heatmap(csv_path="output/gae_debug.csv",
                     save_path="output/plots/gae_heatmap.png"):
    """
    从 gae_debug.csv 读取 GAE 调试数据，画出最近一轮的热力图。
    三列子图：V(s) | δ (delta) | A_norm (归一化优势)

    行 = 16 个架构，列 = 7 个时间步。
    """
    import csv as _csv
    if not os.path.exists(csv_path):
        print(f"[Skip] GAE heatmap: {csv_path} not found")
        return

    # 读取最后一轮数据
    rows = []
    with open(csv_path, "r") as f:
        reader = _csv.DictReader(f)
        last_it = None
        for row in reader:
            it = int(row["iteration"])
            if last_it is None or it > last_it:
                last_it = it
                rows = [row]
            elif it == last_it:
                rows.append(row)

    if not rows:
        print("[Skip] GAE heatmap: no data")
        return

    B = 16  # rollouts_per_iter
    T = 7   # LAYER_TYPES
    V_mat = np.full((B, T), np.nan)
    D_mat = np.full((B, T), np.nan)
    A_mat = np.full((B, T), np.nan)

    for row in rows:
        b = int(row["arch_idx"])
        t = int(row["step"])
        if b < B and t < T:
            V_mat[b, t] = float(row["V(s)"])
            D_mat[b, t] = float(row["delta"])
            A_mat[b, t] = float(row["A_norm"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [f"V(s) — Iter {last_it}", f"δ (delta) — Iter {last_it}",
              f"A_norm — Iter {last_it}"]
    matrices = [V_mat, D_mat, A_mat]
    cmaps = ["Blues", "RdBu_r", "RdBu_r"]

    for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Architecture Index")
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"L{i+1}" for i in range(T)])
        ax.set_yticks(range(B))
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] GAE heatmap -> {save_path}")


@need_mpl
def plot_gae_heatmap_series(csv_path="output/gae_debug.csv",
                             save_path="output/plots/gae_heatmap_series.png",
                             num_panels=4):
    """
    从 gae_debug.csv 读取数据, 等间距选取 num_panels 轮,
    画出 A_norm 热力图序列, 展示 GAE 信用分配随训练的变化趋势。

    每行 = 一个迭代轮次, 每列 = 时间步 (L1~L7)。
    """
    import csv as _csv
    if not os.path.exists(csv_path):
        print(f"[Skip] GAE series: {csv_path} not found")
        return

    # 读取所有轮次的数据, 按 iteration 分组
    from collections import defaultdict
    iter_data = defaultdict(list)  # iter → list of rows
    with open(csv_path, "r") as f:
        for row in _csv.DictReader(f):
            it = int(row["iteration"])
            iter_data[it].append(row)

    if not iter_data:
        print("[Skip] GAE series: no data")
        return

    # 等间距选取 num_panels 轮
    all_iters = sorted(iter_data.keys())
    if len(all_iters) <= num_panels:
        selected = all_iters
    else:
        step = max(1, len(all_iters) // (num_panels - 1))
        selected = [all_iters[i] for i in range(0, len(all_iters), step)]
        selected = selected[:num_panels]

    B = 16
    T = 7
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, it in zip(axes, selected):
        rows = iter_data[it]
        mat = np.full((B, T), np.nan)
        for row in rows:
            b = int(row["arch_idx"])
            t = int(row["step"])
            if b < B and t < T:
                mat[b, t] = float(row["A_norm"])

        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_title(f"Iter {it}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Arch Index" if ax == axes[0] else "")
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"L{i+1}" for i in range(T)])
        ax.set_yticks(range(0, B, 4))

    plt.colorbar(im, ax=axes[-1], shrink=0.8, label="A_norm")
    plt.suptitle("GAE Advantage Evolution (A_norm)", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] GAE series ({len(selected)} panels) -> {save_path}")


def _value_centered_bins(values):
    """给定一组离散值，返回以每个值为中心的直方图 bin 边界。

    例如 values=[3,5,7] → bins=[2,4,6,8]，每个 bin 中心对准一个值。
    """
    sorted_vals = sorted(values)
    edges = []
    for i, v in enumerate(sorted_vals):
        if i == 0:
            left = v - (sorted_vals[1] - v) / 2 if len(sorted_vals) > 1 else v - 0.5
        else:
            left = (sorted_vals[i-1] + v) / 2
        edges.append(left)
    # 最后一个右边界
    if len(sorted_vals) > 1:
        right = sorted_vals[-1] + (sorted_vals[-1] - sorted_vals[-2]) / 2
    else:
        right = sorted_vals[-1] + 0.5
    edges.append(right)
    return edges
