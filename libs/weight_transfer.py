"""
权重共享 / 权重复用
====================
加速搜索: 新架构中未改变的层直接复用旧架构的权重。

用法:
    transfer = WeightTransfer()
    new_weights = transfer.transfer(old_arch, old_weights, new_arch)

目前是简化实现，只做逐层比对和标记。
"""


def find_common_prefix(old_arch, new_arch):
    """
    找出两个架构从头开始有多少层完全相同。

    返回: 相同层数 (int)
    """
    n = 0
    for a1, a2 in zip(old_arch, new_arch):
        if _layers_equal(a1, a2):
            n += 1
        else:
            break
    return n


def _layers_equal(a1, a2):
    """判断两层配置是否相同。"""
    if a1["type"] != a2["type"]:
        return False
    if a1["type"] == "linear":
        return (a1.get("kernel") == a2.get("kernel") and
                a1.get("conv") == a2.get("conv"))
    else:
        return a1.get("spline") == a2.get("spline")


class WeightTransfer:
    """
    权重迁移管理器。

    维护一个已训练架构的缓存。
    新架构生成后，从缓存中找最相似的"父架构"来复用权重。

    TODO: 接入真实模型训练后，实现真正的权重拷贝逻辑。
    """

    def __init__(self, cache_size=20):
        self.cache = {}         # arch_str → (architecture, state_dict)
        self.max_size = cache_size

    def find_parent(self, new_arch):
        """找公共前缀最长的已缓存架构。"""
        best_key = None
        best_prefix = -1
        for key, (old_arch, _) in self.cache.items():
            prefix = find_common_prefix(old_arch, new_arch)
            if prefix > best_prefix:
                best_prefix = prefix
                best_key = key
        if best_key:
            return self.cache[best_key]
        return None

    def add(self, architecture, state_dict=None):
        """把训练好的架构加入缓存。"""
        from .model.search_space import architecture_to_str
        key = architecture_to_str(architecture)
        if len(self.cache) >= self.max_size:
            # 删掉最早加入的
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = (architecture, state_dict)
