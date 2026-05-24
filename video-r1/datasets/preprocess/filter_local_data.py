#!/usr/bin/env python3
# Copyright 2025 Video-R1 implementation on verl 0.8
"""按"本地媒体文件是否存在"过滤 Video-R1 JSON,可选打乱 + 分层均衡 + 截断。

使用场景:
    我们只下了部分 zip,全量 Video-R1-260k.json 里大部分样本的 path 指向没下的 zip。
    跑训练前用这个脚本过滤一遍,产出只含"本地有媒体"的小 JSON。

工作流:
    1. 加载全量 JSON
    2. 按本地文件存在性过滤
    3. 可选:打乱(默认开,避免按原顺序偏)
    4. 可选:分层均衡(--balance,按 data_type × problem_type 等量采样)
    5. 可选:截断到前 N 条(--limit)
    6. 写出

例子:
    # 全保留(每次覆盖输出文件):
    python datasets/preprocess/filter_local_data.py

    # 截到 500 条,默认 shuffle:
    python datasets/preprocess/filter_local_data.py --limit 500

    # 分层均衡,每类等量取,总共 ~100:
    python datasets/preprocess/filter_local_data.py --limit 100 --balance

    # 关掉 shuffle(按原顺序取前 N):
    python datasets/preprocess/filter_local_data.py --limit 500 --no-shuffle
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATA_ROOT = "/data/zyt/datas_video_r1"
DEFAULT_INPUT = f"{DEFAULT_DATA_ROOT}/Video-R1-260k.json"
DEFAULT_OUTPUT = f"{DEFAULT_DATA_ROOT}/Video-R1-260k-local.json"


def parse_args():
    p = argparse.ArgumentParser(description="过滤 Video-R1 JSON,只留本地有媒体的样本")
    p.add_argument("--input", default=DEFAULT_INPUT, help=f"输入 JSON(默认: {DEFAULT_INPUT})")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help=f"输出 JSON(默认: {DEFAULT_OUTPUT})")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help=f"媒体根目录(默认: {DEFAULT_DATA_ROOT})")
    p.add_argument("--limit", type=int, default=-1, help="截断到 N 条(-1 = 不截断,保留全部过滤后样本)")
    p.add_argument("--shuffle", action="store_true", default=True, help="过滤后打乱(默认开)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="关掉打乱")
    p.add_argument("--seed", type=int, default=42, help="打乱用的随机种子(默认 42)")
    p.add_argument("--balance", action="store_true",
                   help="按 (data_type × problem_type) 分层均衡采样(需配合 --limit)")
    return p.parse_args()


def stratified_sample(items, limit, key_fn, rng):
    """按 key 分桶,每桶等量采样,总数尽量靠近 limit。

    小桶取完所有,大桶补齐剩余配额。
    """
    buckets = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)

    n_buckets = len(buckets)
    if n_buckets == 0:
        return []

    quota_per_bucket = max(1, limit // n_buckets)
    selected = []
    remaining_buckets = []

    # 第一轮:每桶取 quota,小桶全取
    for k, vs in buckets.items():
        rng.shuffle(vs)
        if len(vs) <= quota_per_bucket:
            selected.extend(vs)
        else:
            selected.extend(vs[:quota_per_bucket])
            remaining_buckets.append((k, vs[quota_per_bucket:]))

    # 第二轮:还差多少,从大桶轮流补
    short = limit - len(selected)
    if short > 0 and remaining_buckets:
        round_robin = [v for _, vs in remaining_buckets for v in vs]
        rng.shuffle(round_robin)
        selected.extend(round_robin[:short])

    rng.shuffle(selected)
    return selected[:limit]


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.data_root):
        print(f"ERROR: 媒体根目录不存在: {args.data_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"Total samples in JSON: {len(data):,}")

    # ---- 逐条检查 path 是否在本地 ----
    print(f"\nChecking media presence under {args.data_root} ...")
    kept = []
    missing = 0
    for i, item in enumerate(data):
        if (i + 1) % 50000 == 0:
            print(f"  scanned {i + 1:,} / {len(data):,}")
        rel = item["path"]
        if rel.startswith("./"):
            rel = rel[2:]
        if (Path(args.data_root) / rel).exists():
            kept.append(item)
        else:
            missing += 1

    print(f"\n=== 过滤结果 ===")
    print(f"  本地有媒体: {len(kept):,}")
    print(f"  缺媒体(跳过): {missing:,}")

    if not kept:
        print("\nERROR: 一条都没保留,检查 --data-root 路径是否正确、zip 是否解压。", file=sys.stderr)
        sys.exit(2)

    # ---- 打乱(默认开)----
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(kept)
        print(f"\nShuffled (seed={args.seed})")

    # ---- 截断 / 分层均衡 ----
    if args.limit > 0 and args.limit < len(kept):
        if args.balance:
            kept = stratified_sample(
                kept, args.limit,
                key_fn=lambda x: (x["data_type"], x.get("problem_type", "?")),
                rng=rng,
            )
            print(f"--balance --limit {args.limit}: 分层均衡后 {len(kept)} 条")
        else:
            kept = kept[: args.limit]
            print(f"--limit {args.limit}: 截到前 {len(kept)} 条")

    # ---- 分布统计 ----
    print(f"\n=== 最终样本 (data_type × problem_type) ===")
    by_type = Counter((x["data_type"], x.get("problem_type", "?")) for x in kept)
    for (dt, pt), n in sorted(by_type.items()):
        print(f"  {dt:6}  ×  {pt:18}  :  {n}")

    print(f"\n=== 最终样本按数据集文件夹 ===")
    by_folder = Counter(x["path"].lstrip("./").split("/")[0] for x in kept)
    for folder, n in sorted(by_folder.items(), key=lambda kv: -kv[1]):
        print(f"  {folder:30}  :  {n}")

    # ---- 写出(覆盖式)----
    print(f"\nWriting: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(kept, f, ensure_ascii=False)
    print(f"Done. {len(kept):,} samples written to {args.output}")
    print(f"\n训练时把 data.train_files 指向: {args.output}")


if __name__ == "__main__":
    main()
