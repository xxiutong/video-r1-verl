# Copyright 2025 Video-R1 implementation on verl 0.8
# Licensed under the Apache License, Version 2.0
"""Video-R1 reward function —— 5 题型 accuracy + format + 长度奖励。

通过 yaml 配置 custom_reward_function.path/name 注入:
    custom_reward_function:
      path: /data/zyt/LLM1/video-r1-verl/video-r1/video_r1/verl/utils/reward_score/video_r1.py
      name: compute_score

reward 组成(对应原版 Video-R1 grpo.py):
    accuracy(5 题型分发):
        - multiple choice:字符串完全相等 → 0/1
        - numerical:四舍五入到 2 位相等 → 0/1
        - OCR:1 - WER → 连续
        - free-form:ROUGE F → 连续
        - regression:1 - 相对误差 → 连续
    format:<think>...</think><answer>...</answer> 严格匹配 → 0/1
    length_bonus:答对 + 320 <= length <= 512 → +0.2
    (T-GRPO 时序奖励 +0.3 不在这里,在 trainer.fit() 中按 rollout 比较后叠加)
"""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Video-R1 总 reward(不含 T-GRPO 时序奖励)。

    Args:
        data_source: 数据来源标识(如 "NeXT-QA/..." 或 "Math/...")
        solution_str: 模型生成的完整文本(含 <think>...</think><answer>X</answer>)
        ground_truth: 标准答案(含 <answer> 标签)
        extra_info: dict,必含 problem_type;可能还有 data_type、problem 等

    Returns:
        float:单条样本的总分(accuracy + format + length_bonus)
    """
    # TODO: 实现 5 题型 accuracy(参考 Video-R1-main/src/r1-v/src/open_r1/grpo.py:66-165)
    # TODO: format 奖励
    # TODO: length 奖励
    raise NotImplementedError("TODO: 移植 grpo.py 的 accuracy_reward + format_reward + len_control")
