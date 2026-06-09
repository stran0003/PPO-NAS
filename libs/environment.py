"""
NAS 搜索环境
============
把控制器、PPO训练、评估、日志串联成完整的搜索流程。
"""

import os
import json
import time
import logging

from .model.controller import Controller
from .model.dpd_model import evaluate_architecture
from .model.search_space import architecture_to_str
from .ppo import PPOTrainer
from .reward import ParetoFront

logger = logging.getLogger(__name__)


class NASEnvironment:
    """
    完整的 PPO-NAS 搜索环境。

    用法:
        env = NASEnvironment(config_dict)
        env.run()  # 开始搜索
    """

    def __init__(self, config):
        self.config = config

        # -- 控制器 --
        ctrl_cfg = config.get("controller", {})
        self.controller = Controller(
            hidden_dim=ctrl_cfg.get("hidden_dim", 128),
            num_lstm_layers=ctrl_cfg.get("num_layers", 2),
            dropout=ctrl_cfg.get("dropout", 0.1),
        )

        # 移到设备
        device = config.get("global", {}).get("device", "cpu")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.controller.to(device)

        # -- PPO 训练器 --
        ppo_cfg = config.get("ppo", {})
        ppo_cfg["reward"] = config.get("reward", {})
        ppo_cfg["short_epochs"] = config.get("evaluation", {}).get("short_epochs", 300)
        ppo_cfg["output_dir"] = config.get("global", {}).get("output_dir", "output")
        ppo_cfg["log_interval"] = config.get("global", {}).get("log_interval", 5)
        self.trainer = PPOTrainer(self.controller, ppo_cfg)

        # -- Pareto 前沿 --
        self.pareto = ParetoFront()

        # -- 目录: log_nas/日期/时间/ --
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        base_dir = os.path.join("log_nas", date_str, time_str)
        self.output_dir = os.path.join(base_dir, "output")
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # -- 日志 --
        self.log_interval = config.get("global", {}).get("log_interval", 5)
        self.save_interval = config.get("global", {}).get("save_interval", 50)

    def run(self, total_iterations=None):
        """
        运行完整的 PPO-NAS 搜索。

        参数:
            total_iterations: 覆盖配置文件中的迭代轮数
            eval_fn:          自定义评估函数 (默认用模拟评估)
        """
        total_iter = total_iterations or self.config.get("ppo", {}).get("total_iterations", 500)
        mock_mode = self.config.get("evaluation", {}).get("mock", True)

        logger.info(f"输出目录: {os.path.abspath(self.output_dir)}")
        logger.info(f"开始 PPO-NAS 搜索: {total_iter} 轮, "
                     f"设备={self.device}, 模拟模式={mock_mode}")
        logger.info(f"控制器参数量: {sum(p.numel() for p in self.controller.parameters()):,}")

        eval_fn = None if mock_mode else evaluate_architecture
        t_start = time.time()

        # ── Warm Start: 注入已知好架构 ──
        warm_cfg = self.config.get("warm_start", {})
        if warm_cfg.get("enabled", False) and warm_cfg.get("architecture"):
            logger.info("Warm Start: 注入基线架构到 Pareto 前沿...")
            baseline = warm_cfg["architecture"]
            metrics = evaluate_architecture(baseline)
            nmse = metrics.get("nmse", 99)
            params = metrics.get("num_params", 0)
            added = self.pareto.add(nmse, params, baseline)
            logger.info(f"  基线架构: res={nmse:.2f}dB, params={params}, "
                        f"加入Pareto={'是' if added else '否(被支配)'}")

        for it in range(total_iter):
            # 一轮 PPO 训练
            stats, architectures, _, nmse_list, params_list = \
                self.trainer.train_one_iteration(eval_fn)

            # 更新 Pareto 前沿（直接复用 PPO 评估阶段的结果，无需重复训练）
            for arch, nmse, params in zip(architectures, nmse_list, params_list):
                self.pareto.add(nmse, params, arch)

            # 日志
            if it % self.log_interval == 0:
                self._log(it, stats)

            # 保存 checkpoint
            if it % self.save_interval == 0 and it > 0:
                self.save_checkpoint(it)

        # 完成
        total_time = time.time() - t_start
        logger.info(f"搜索完成! 总耗时: {total_time:.0f}s ({total_time/3600:.1f}h)")
        logger.info(f"Pareto 前沿大小: {len(self.pareto)}")

        self.save_results()
        self.save_checkpoint("final")

    def _log(self, it, stats):
        logger.info(
            f"[{it:4d}] "
            f"R={stats['mean_reward']:.1f}[{stats.get('reward_min',0):.0f},{stats.get('reward_max',0):.0f}]±{stats.get('reward_std',0):.1f} | "
            f"res={stats['mean_nmse']:.1f}[{stats.get('nmse_min',99):.0f},{stats.get('nmse_max',99):.0f}]dB | "
            f"aL={stats.get('actor_loss',0):.3f} cL={stats.get('critic_loss',0):.3f} | "
            f"H={stats.get('mean_entropy',0):.2f} KL={stats.get('approx_kl',0):.4f} | "
            f"P={len(self.pareto)} | "
            f"{stats.get('elapsed',0):.0f}s"
        )

    def save_checkpoint(self, tag):
        path = os.path.join(self.checkpoint_dir, f"ckpt_{tag}.pt")
        self.trainer.save(path)
        logger.info(f"Checkpoint 已保存: {path}")

    def load_checkpoint(self, tag):
        path = os.path.join(self.checkpoint_dir, f"ckpt_{tag}.pt")
        self.trainer.load(path)
        logger.info(f"Checkpoint 已加载: {path} (第 {self.trainer.iteration} 轮)")

    def save_results(self):
        """保存搜索结果。"""
        # Pareto 前沿架构
        pareto_data = []
        for nmse, params, arch in self.pareto:
            pareto_data.append({
                "architecture": architecture_to_str(arch),
                "nmse": nmse,
                "num_params": params,
                "actions": arch,
            })

        with open(os.path.join(self.output_dir, "pareto_front.json"), "w") as f:
            json.dump(pareto_data, f, indent=2, ensure_ascii=False)

        # 训练统计
        with open(os.path.join(self.output_dir, "training_stats.json"), "w") as f:
            json.dump(self.trainer.stats_history, f, indent=2, ensure_ascii=False)

        logger.info(f"结果已保存到 {self.output_dir}/")
