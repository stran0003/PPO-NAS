"""
PPO-NAS 主入口
==============
基于 PPO 强化学习的神经架构搜索，用于 DPD 七层混合模型。

用法:
    python main.py                     # 默认配置运行
    python main.py --config my.yaml    # 指定配置文件
    python main.py --mock              # 模拟评估（快速测试）
    python main.py --resume 100        # 从第100轮恢复
    python main.py --analyze           # 只生成图表（不搜索）
    python main.py -n 50               # 只跑 50 轮

快速开始:
    bash auto.sh
"""

import os
import sys
import random
import json
import argparse
import logging

import yaml
import numpy as np
import torch

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libs.environment import NASEnvironment
from libs.analysis import plot_pareto, plot_training, plot_distribution

logger = logging.getLogger(__name__)


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="PPO-NAS: DPD 七层混合模型架构搜索")

    p.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    p.add_argument("--resume", "-r", type=str, default=None, help="从 checkpoint 恢复")
    p.add_argument("--mock", "-m", action="store_true", help="模拟评估模式")
    p.add_argument("--analyze", "-a", action="store_true", help="只分析/画图，不搜索")
    p.add_argument("--iterations", "-n", type=int, default=None, help="覆盖总轮数")
    p.add_argument("--seed", "-s", type=int, default=None, help="随机种子")
    p.add_argument("--device", "-d", default=None, choices=["cpu", "cuda", "auto"])
    p.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    return p.parse_args()


# ============================================================
# 加载配置
# ============================================================

def load_config(path):
    """加载 YAML 配置文件。"""
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 搜索
# ============================================================

def run_search(args):
    """运行 PPO-NAS 搜索。"""
    config = load_config(args.config)

    # 命令行覆盖
    if args.mock:
        config.setdefault("evaluation", {})["mock"] = True
    if args.device:
        config.setdefault("global", {})["device"] = args.device
    if args.seed is not None:
        config.setdefault("global", {})["seed"] = args.seed

    # 设置随机种子
    seed = config.get("global", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print("PPO-NAS: DPD 七层混合模型架构搜索")
    print("=" * 60)
    print(f"设备:       {config.get('global', {}).get('device', 'auto')}")
    print(f"奖励类型:   {config.get('reward', {}).get('type', 'simple')}")
    print(f"总轮数:     {config.get('ppo', {}).get('total_iterations', 500)}")
    print(f"模拟模式:   {config.get('evaluation', {}).get('mock', True)}")
    print(f"PPO ε:      {config.get('ppo', {}).get('clip_epsilon', 0.2)}")
    print(f"学习率:     {config.get('ppo', {}).get('lr', 3e-4)}")
    print()

    # 创建环境
    env = NASEnvironment(config)

    # 恢复 checkpoint
    if args.resume:
        print(f"从 checkpoint 恢复: {args.resume}")
        env.load_checkpoint(args.resume)

    # 开始搜索
    env.run(total_iterations=args.iterations)

    # 搜索结束，生成图表
    print("\n" + "=" * 60)
    print("生成分析图表...")
    _save_plots(env)
    print("完成!")


# ============================================================
# 分析模式（只画图，不搜索）
# ============================================================

def run_analysis(args):
    """从已有结果生成图表。"""
    config = load_config(args.config)
    output_dir = config.get("global", {}).get("output_dir", "output") if "config" in dir() else "output"

    pareto_file = os.path.join(output_dir, "pareto_front.json")
    stats_file = os.path.join(output_dir, "training_stats.json")

    if not os.path.exists(pareto_file):
        print(f"找不到结果文件: {pareto_file}")
        print("请先运行搜索: python main.py")
        return

    with open(pareto_file, "r") as f:
        pareto_data = json.load(f)

    stats = []
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)

    print(f"加载了 {len(pareto_data)} 个 Pareto 前沿架构")
    print(f"加载了 {len(stats)} 轮训练统计")

    plot_pareto(pareto_data, os.path.join(output_dir, "plots", "pareto.png"))
    if stats:
        plot_training(stats, os.path.join(output_dir, "plots", "training.png"))
    if pareto_data:
        plot_distribution(pareto_data, os.path.join(output_dir, "plots", "distribution.png"))

    # GAE 热力图（如果 gae_debug.csv 存在）
    from libs.analysis.plotting import plot_gae_heatmap
    plot_gae_heatmap(
        csv_path=os.path.join(output_dir, "gae_debug.csv"),
        save_path=os.path.join(output_dir, "plots", "gae_heatmap.png"),
    )


def _save_plots(env):
    """保存分析图表。"""
    out = env.output_dir
    plot_dir = os.path.join(out, "plots")

    # 收集数据
    pareto_data = []
    for nmse, params, arch in env.pareto:
        from libs.model.search_space import architecture_to_str
        pareto_data.append({
            "nmse": nmse,
            "num_params": params,
            "architecture": architecture_to_str(arch),
            "actions": arch,
        })

    plot_pareto(pareto_data, os.path.join(plot_dir, "pareto.png"))
    plot_training(env.trainer.stats_history, os.path.join(plot_dir, "training.png"))

    if pareto_data:
        plot_distribution(pareto_data, os.path.join(plot_dir, "distribution.png"))

    from libs.analysis.plotting import plot_gae_heatmap
    plot_gae_heatmap(
        csv_path=os.path.join(out, "gae_debug.csv"),
        save_path=os.path.join(plot_dir, "gae_heatmap.png"),
    )


# ============================================================
# 入口
# ============================================================

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # GPU 信息
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.analyze:
        run_analysis(args)
    else:
        run_search(args)


if __name__ == "__main__":
    main()
