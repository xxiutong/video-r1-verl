#!/usr/bin/env python3
# Copyright 2025 Video-R1 implementation on verl 0.8
"""扫描 Video-R1 JSON 全量,按维度统计分布。

用法:
    # 全部数据集分布
    python datasets/preprocess/inspect_data.py

    # 只看某个数据集文件夹
    python datasets/preprocess/inspect_data.py --folder NeXT-QA

    # 看 video 类型样本的题型分布
    python datasets/preprocess/inspect_data.py --data-type video
"""
import argparse
import json
from collections import Counter


DEFAULT_INPUT = "/data/zyt/datas_video_r1/Video-R1-260k.json"


def parse_args():
    p = argparse.ArgumentParser(description="Video-R1 JSON 全量分布扫描")
    p.add_argument("--input", default=DEFAULT_INPUT, help=f"输入 JSON(默认: {DEFAULT_INPUT})")
    p.add_argument("--folder", default=None, help="只看某个顶层文件夹(如 NeXT-QA / Math)")
    p.add_argument("--data-type", default=None, help="只看某个 data_type(image / video)")
    p.add_argument("--problem-type", default=None, help="只看某个 problem_type")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"Total samples: {len(data):,}\n")

    # 过滤
    filters = []
    if args.folder:
        before = len(data)
        data = [x for x in data if x["path"].lstrip("./").startswith(args.folder + "/")]
        filters.append(f"folder={args.folder}: {before:,} → {len(data):,}")
    if args.data_type:
        before = len(data)
        data = [x for x in data if x.get("data_type") == args.data_type]
        filters.append(f"data_type={args.data_type}: {before:,} → {len(data):,}")
    if args.problem_type:
        before = len(data)
        data = [x for x in data if x.get("problem_type") == args.problem_type]
        filters.append(f"problem_type={args.problem_type}: {before:,} → {len(data):,}")

    if filters:
        print("=== 过滤 ===")
        for f in filters:
            print(f"  {f}")
        print(f"After filters: {len(data):,}\n")

    if not data:
        print("(过滤后无样本)")
        return

    # 维度统计
    print("=== By data_type ===")
    for k, v in Counter(x.get("data_type", "?") for x in data).most_common():
        print(f"  {k:8} : {v:,}")

    print("\n=== By problem_type ===")
    for k, v in Counter(x.get("problem_type", "?") for x in data).most_common():
        print(f"  {k:20} : {v:,}")

    print("\n=== By (data_type × problem_type) ===")
    for k, v in Counter(
        (x.get("data_type", "?"), x.get("problem_type", "?")) for x in data
    ).most_common():
        print(f"  {k[0]:8} × {k[1]:20} : {v:,}")

    print("\n=== By 顶层文件夹 ===")
    for k, v in Counter(x["path"].lstrip("./").split("/")[0] for x in data).most_common():
        print(f"  {k:30} : {v:,}")

    print("\n=== By data_source(前 20)===")
    src_counts = Counter(x.get("data_source", "?") for x in data).most_common(20)
    for k, v in src_counts:
        print(f"  {k:50} : {v:,}")


if __name__ == "__main__":
    main()
