# Copyright 2025 Video-R1 implementation on verl 0.8
# Licensed under the Apache License, Version 2.0
"""Video-R1 数据集 —— 子类化 verl RLHFDataset,直接读 Video-R1 JSON(无需预处理成 parquet)。

通过 yaml 配置 data.custom_cls.path/name 注入:
    data:
      custom_cls:
        path: /data/zyt/LLM1/video-r1-verl/video-r1/video_r1/verl/utils/dataset/my_rl_dataset.py
        name: VideoR1Dataset

设计原则:
    - 只覆盖 _read_files_and_tokenize 一个方法,读 JSON + 字段重组
    - __getitem__ / _build_messages / 过滤等沿用 verl 基类
    - 媒体路径只存字符串(dict 形式),不预加载 PIL.Image
      → 下游 qwen-vl-utils.process_vision_info 真正要用时才按路径加载

Video-R1 JSON 字段 → verl 期望字段映射:
    problem (+ options)                  →  prompt[0].content(含 <image>/<video> 占位)
    path + data_type=="image"            →  images = [{"image": abs_path}]
    path + data_type=="video"            →  videos = [{"video": abs_path}]
    solution                             →  reward_model.ground_truth
    data_source                          →  data_source
    problem_type / data_type / problem_id →  extra_info(reward 函数可读)
"""
import json
from pathlib import Path

import datasets as hf_datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', "
    "'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, "
    "and then give your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) "
                       "within the <answer> </answer> tags.",
    "numerical":       " Please provide the numerical value (e.g., 42 or 3.14) "
                       "within the <answer> </answer> tags.",
    "OCR":             " Please transcribe text from the image/video clearly and provide your text "
                       "answer within the <answer> </answer> tags.",
    "free-form":       " Please provide your text answer within the <answer> </answer> tags.",
    "regression":      " Please provide the numerical value (e.g., 42 or 3.14) "
                       "within the <answer> </answer> tags.",
}


def _absolute_media_path(rel_path: str, data_root: str) -> str:
    """Video-R1 JSON 里的 path 是相对路径(./NeXT-QA/... 这种),转成绝对路径。"""
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    return str(Path(data_root) / rel_path)


def _build_question(item: dict) -> str:
    """拼问题文本:problem (+ options if multiple choice) + 题型指令。"""
    if item["problem_type"] == "multiple choice":
        question = item["problem"] + "Options:\n"
        for op in item["options"]:
            question += op + "\n"
    else:
        question = item["problem"]
    return QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[item["problem_type"]]


def _transform_one(item: dict, data_root: str) -> dict:
    """单条 Video-R1 JSON → verl RLHFDataset 期望的行格式。

    返回 dict 的所有键(无论是不是当前样本类型用得到)都要存在,
    因为 HF Dataset.from_list 会基于第一条记录推断 schema,
    后续记录必须有相同的列(可以是空列表)。
    """
    data_type = item["data_type"]                       # "image" or "video"
    placeholder = "<image>" if data_type == "image" else "<video>"
    text = _build_question(item)
    abs_path = _absolute_media_path(item["path"], data_root)
    # 这里因为video-r1数据集里文本部分没加图或者视频的占位，这里手动补上。
    record = {
        "prompt": [{"role": "user", "content": placeholder + text}],
        "data_source": item.get("data_source", "video_r1"),
        "reward_model": {
            "ground_truth": item["solution"],            # 含 <answer>...</answer> 标签
            "style": "rule",                              # verl 约定:rule-based reward
        },
        "extra_info": {
            "problem_type": item["problem_type"],
            "data_type": data_type,
            "index": item.get("problem_id", -1),
        },
        "images": [],
        "videos": [],
    }

    if data_type == "image":
        # 存 dict 含路径,基类 _build_messages 会展开成 {"type":"image","image":path}
        # qwen-vl-utils 下游会按路径自动加载 PIL.Image
        # 这个数据集都是单个图像或者单个视频貌似，所以这里暂时只写一个默认。
        record["images"] = [{"image": abs_path}]
    elif data_type == "video":
        record["videos"] = [{"video": abs_path}]
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return record


class VideoR1Dataset(RLHFDataset):
    """直接从 Video-R1 JSON 加载数据,无需预处理成 parquet。
    """

    def _read_files_and_tokenize(self):
        """读 Video-R1 JSON,转格式,设置 self.dataframe。

        媒体根目录 = train JSON 所在的目录(Video-R1 JSON 里的 path 是 './XXX/yyy' 这种
        相对于 JSON 所在目录的路径,直接用 parent 即可)。
        """
        records = []
        for json_path in self.data_files:
            data_root = str(Path(json_path).parent)
            print(f"VideoR1Dataset: loading {json_path} (media root = {data_root})")
            with open(json_path, "r") as f:
                items = json.load(f)
            for item in items:
                records.append(_transform_one(item, data_root))

        self.dataframe: hf_datasets.Dataset = hf_datasets.Dataset.from_list(records)

        # 这种写法也应该可以
        # dataframes = []
        # for json_path in self.data_files:
        #     data_root = str(Path(json_path).parent)
        #     df = hf_datasets.load_dataset("json", data_files=json_path)["train"]
        #     df = df.map(
        #         lambda x: _transform_one(x, data_root),
        #         remove_columns=df.column_names,   # 删掉原 Video-R1 字段
        #     )
        #     dataframes.append(df)
        # self.dataframe = hf_datasets.concatenate_datasets(dataframes)


        total = len(self.dataframe)
        print(f"loaded {total} samples")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} out of {total}")

        # ---- 过滤超长 prompt(基类提供的方法,需要 tokenizer/processor)----
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
