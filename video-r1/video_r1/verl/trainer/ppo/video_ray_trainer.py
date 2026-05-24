# Copyright 2025 Video-R1 implementation on verl 0.8
# Licensed under the Apache License, Version 2.0
"""Video-R1 训练器 —— 子类化 verl RayPPOTrainer。

Phase 1(基础 GRPO):
    fit() 不覆盖,直接走基类 → 等价于 verl 默认 GRPO 行为
    我们在这一阶段调试 dataset / reward / yaml,确保链路通

Phase 2(T-GRPO):
    覆盖 fit() 加入双 rollout 逻辑:
        1. 对视频样本额外做一次"帧序打乱"的 rollout
        2. 比较 normal_acc 和 shuffled_acc
        3. 如 normal_acc >= 0.8 * shuffled_acc → 答对的样本 reward + 0.3
    参照 verl 自带的 REMAX 分支模式(verl-main/verl/trainer/ppo/ray_trainer.py:1361-1388)
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class VideoR1RayPPOTrainer(RayPPOTrainer):
    """T-GRPO 训练器(Phase 1 透传,Phase 2 覆盖 fit())。"""

    pass

    # TODO Phase 2:取消下面注释,完整复制 verl fit() 代码(含中文注释),
    # 在 REMAX 分支后加 T-GRPO 双 rollout 分支
    #
    # def fit(self):
    #     ...
