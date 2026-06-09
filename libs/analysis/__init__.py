"""
分析与可视化
============
- evaluation: NOSH 渐进淘汰、排序相关性
- plotting:   画图工具
"""
from .evaluation import NOSH, check_ranking_correlation, Tracker
from .plotting import plot_pareto, plot_training, plot_distribution
