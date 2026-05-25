# Copyright 2025 Video-R1 implementation on verl 0.8
# Licensed under the Apache License, Version 2.0
"""Video-R1 reward function —— 5 题型 accuracy + format + length。

通过 yaml 配置注入:
    custom_reward_function:
      path: /data/zyt/LLM1/video-r1-verl/video-r1/video_r1/verl/utils/reward_score/video_r1.py
      name: compute_score

接口(由 verl 的 reward_manager 逐条调用):
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict
        - solution_str:  模型生成的完整回答(含 <think>...</think><answer>X</answer>)
        - ground_truth:  标准答案(含 <answer>...</answer> 标签)
        - extra_info:    {problem_type, data_type, index, ...},reward 函数必读 problem_type
    返回:
        {"score": float, "accuracy": float, "format": float, "length": float}
        - score:    总分 = accuracy + format + length(给 advantage 用)
        - 其他键:    分项指标,reward_manager 自动塞进 non_tensor_batch,上 wandb

reward 组成(对应原版 Video-R1 grpo.py):
    1. accuracy(5 题型分发,grpo.py:66-165 移植):
        - multiple choice → 字符串完全相等 → 0/1
        - numerical       → 四舍五入到 2 位相等(+ 小数点/逗号特征一致) → 0/1
        - OCR             → 1 - WER → 连续 [0, 1]
        - free-form       → ROUGE-1/2/L F 值平均 → 连续 [0, 1]
        - regression      → 1 - 相对误差 → 连续 [0, 1]
    2. format(grpo.py:168-173 移植):
        re.fullmatch(<think>...</think><answer>...</answer>) → 0/1
    3. length(原版在 trainer 算,我们近似用 word count):
        accuracy > 0.1 且 token 数在 [320, 512] → +0.2
        ⚠️ Phase 1 用 len(solution_str.split()) 近似 token 数,
        Phase 2 在 trainer.fit() override 时换成 completion_mask.sum(1) 精确值
"""
import re


# ============================================================
# 可调常量
# ============================================================
LENGTH_BONUS_MIN_TOKENS = 320
LENGTH_BONUS_MAX_TOKENS = 512
LENGTH_BONUS_VALUE = 0.2
ACCURACY_THRESHOLD_FOR_LENGTH = 0.1   # 配合原版:答对(accuracy > 0.1)才给长度奖


# ============================================================
# 4 个辅助函数 —— 原版 grpo.py:68-106 移植
# ============================================================
def extract_answer(text: str) -> str:
    """从文本里抠出 <answer>...</answer> 标签内的内容,没找到返回空串。"""
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def normalize_number(num_str: str):
    """数字字符串 → float。去掉千分位逗号。失败返回 None。"""
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None


def wer(reference: str, hypothesis: str) -> float:
    """词错误率(Word Error Rate)。

    标准编辑距离 DP:
        把 hypothesis 改成 reference 需要"增/删/改"多少个词,再除以 reference 词数。
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m, n = len(ref_words), len(hyp_words)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[m][n] / max(1, m)


def compute_rouge_score(reference: str, hypothesis: str, use_stemmer: bool = True) -> float:
    """ROUGE-1/2/L F 值的平均(自由文本相似度)。需要 rouge_score 包。"""
    # 懒导入,避免没装 rouge_score 时其他题型也跑不了
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
    )
    scores = scorer.score(reference, hypothesis)
    return (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3


# ============================================================
# Reward 分项
# ============================================================
def accuracy_reward(problem_type: str, output_ans: str, gt_ans: str) -> float:
    """按 5 题型分发判分,返回 [0.0, 1.0] 的分数。"""
    try:
        if problem_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

        elif problem_type == "numerical":
            # 检"小数点/逗号"特征一致(防止整数答案蒙混成小数题)
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_n = normalize_number(gt_ans)
            out_n = normalize_number(output_ans)
            if gt_n is None or out_n is None:
                return 0.0
            return 1.0 if round(gt_n, 2) == round(out_n, 2) else 0.0

        elif problem_type == "OCR":
            error_rate = wer(gt_ans, output_ans)
            return max(0.0, min(1.0, 1 - error_rate))

        elif problem_type == "free-form":
            score = compute_rouge_score(gt_ans, output_ans)
            return max(0.0, min(1.0, score))

        elif problem_type == "regression":
            gt_n = normalize_number(gt_ans)
            out_n = normalize_number(output_ans)
            if gt_n is None or out_n is None:
                return 0.0
            rel_diff = (abs(out_n - gt_n) + 1e-9) / (abs(gt_n) + 1e-9)
            return max(0.0, min(1.0, 1 - rel_diff))

        else:
            return 0.0
    except Exception as e:
        print(f"[accuracy_reward] error for problem_type={problem_type}: {e}")
        return 0.0


def format_reward(solution_str: str) -> float:
    """检查输出是否严格符合 <think>...</think><answer>...</answer> 格式。"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # strip 应对模型可能输出前后空白
    return 1.0 if re.fullmatch(pattern, solution_str.strip(), re.DOTALL) else 0.0


def length_bonus(solution_str: str, accuracy_score: float) -> float:
    """长度奖励:答对(accuracy > 0.1)且长度在合理区间 → +0.2。

    Phase 1 近似:用 len(solution_str.split()) 当 token 数。
    Phase 2 在 trainer.fit() 用 completion_mask.sum(1) 精确值替换。
    """
    if accuracy_score <= ACCURACY_THRESHOLD_FOR_LENGTH:
        return 0.0
    approx_token_count = len(solution_str.split())   # word count,近似 token 数
    if LENGTH_BONUS_MIN_TOKENS <= approx_token_count <= LENGTH_BONUS_MAX_TOKENS:
        return LENGTH_BONUS_VALUE
    return 0.0


# ============================================================
# 主入口 —— verl reward_manager 调用这个
# ============================================================
def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None) -> dict:
    """Video-R1 总 reward(单条样本)。

    Args:
        data_source: 数据来源(如 "NeXT-QA/30_60_s_nextqa")
        solution_str: 模型完整生成(含 <think>...</think><answer>X</answer>)
        ground_truth: 标准答案(含 <answer>...</answer> 标签)
        extra_info: 必含 "problem_type" 用于分发判分

    Returns:
        dict with keys:
            score:    float, 总分(给 advantage 用)
            accuracy: float [0, 1]
            format:   float, 0 或 1
            length:   float, 0 或 0.2
    """
    extra_info = extra_info or {}
    problem_type = extra_info.get("problem_type", "multiple choice")

    output_ans = extract_answer(solution_str)
    gt_ans = extract_answer(ground_truth)

    acc = accuracy_reward(problem_type, output_ans, gt_ans)
    fmt = format_reward(solution_str)
    length = length_bonus(solution_str, acc)

    return {
        "score": acc + fmt + length,
        "accuracy": acc,
        "format": fmt,
        "length": length,
    }
