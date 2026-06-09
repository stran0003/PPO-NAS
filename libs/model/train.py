import torch
from torch import nn


class real_model(nn.Module):
    def __init__(self, cfgs, layer_configs_nas):
        super(real_model, self).__init__()
        # 在这里依据传进来的cfgs, layer_configs_nas配置每一层，然后初始化
        # 

    def initialize_grouped(self, layer):
        # 这里对卷积层（分组卷积层）做初始化
        return None
    def initialize_FC(self, layer):
        # 这里对合路层（1*1卷积层）做初始化
        return None
    def forward(self, x, mm):
        # mm可不用理会，其实LUT层我们会用一个nn.ModuleDict来存至少两个LUT层，然后根据mm进行切换，有的数据用LUT[0]有的用LUT[1]

        return x
    def getOptimizers(self, model, LR_stepSize, LR):
        # 这里用来生成模型的优化器和调度器
        optimizer = None
        scheduler = None
        return optimizer, scheduler
    
def train(architecture):
    def build_model(architecture: list, _cfgs):
        """
        把搜索到的架构（7个 layer action 的列表）转成模型配置并构建模型。

        architecture 格式:
        [{"type":"linear","kernel":5,"conv":"grouped","init":"xavier"},  # L1
        {"type":"linear","kernel":3,"conv":"1x1","init":"kaiming"},     # L2
        ...
        {"type":"lut","spline":16}]                                    # L4

        MODIFY HERE: 如果模型结构变了，在这里改配置映射逻辑。
        """
        configs = []
        for i, action in enumerate(architecture):
            cfg = {
                "type": action["type"],
                "name": f"layer_{i}",
                "in_channels": 64,
                "out_channels": 64,
            }
            if action["type"] == "linear":
                cfg["kernel"] = action.get("kernel", 5)
                cfg["conv"] = action.get("conv", "grouped")
                cfg["init"] = action.get("init", "xavier")
                if cfg["conv"] == "skip":
                    cfg["groups"] = 0  # 直通层：无参数
                elif cfg["conv"] == "1x1":
                    cfg["groups"] = 2
                else:
                    cfg["groups"] = 64
            else:
                cfg["spline"] = action.get("spline", 16)
            configs.append(cfg)

        return real_model(_cfgs, configs)
    def run_model(architecture, _cfgs):
        """
        这里的步骤大致是
        1.数据处理，准备好双频的数据，64个通道，长为B的tx和rx数据
        2.用build_model函数初始化模型
        3.训练
        4.推理以及画图
        """
        res, num_params = 1, 1
        return {'nmse':res, 'num_params':num_params}
    """
    这里接下来就是复杂的数据的训练过程
    首先，在rl_nas_research目录下面会有一个configs文件夹，然后训练的配置参数_cfgs将存在这里面
    然后，libs/model文件夹下面将会多放入Model.py文件，然后libs/analysis下面将会多存放analysis.py,std_lib.py,utils.py
    这里新存放的文件都是为了服务我的训练用的
    剩下的流程就是：
    1.配置好_cfgs参数,
    2.跑run_model函数 要跑多少论由_cfgs来配置
    """
    _cfgs = None
    metrics = run_model(architecture, _cfgs)
    return metrics