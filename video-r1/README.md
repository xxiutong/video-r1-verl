# Video-R1 on verl

基于 [verl](https://github.com/volcengine/verl) 0.8 复现 [Video-R1](https://github.com/tulerfeng/Video-R1) 论文的 T-GRPO 视频推理训练。

## 仓库布局

```
video-r1-verl/                                  # 顶层 monorepo
├── verl-main/                                  # verl 源码(不修改)
└── video-r1/                                   # 本项目
    ├── video_r1/                               # Python 包,镜像 verl 目录结构
    │   └── verl/
    │       ├── trainer/
    │       │   ├── main_ppo.py                 # Hydra 入口 + VideoR1TaskRunner
    │       │   ├── ppo/video_ray_trainer.py    # VideoR1RayPPOTrainer(T-GRPO)
    │       │   └── config/
    │       │       └── ppo_trainer.yaml        # 训练配置(改自 verl 默认)
    │       └── utils/
    │           ├── dataset/my_rl_dataset.py    # VideoR1Dataset(读 JSON)
    │           └── reward_score/video_r1.py    # 5 题型 reward
    ├── datasets/
    │   └── preprocess/
    │       └── filter_local_data.py            # 按本地媒体存在性过滤 JSON
    ├── scripts/
    │   └── train_video_r1.sh                   # 训练启动
    ├── requirements.txt
    └── README.md
```

## 设计原则

- **verl 源码不动一行** —— 全部通过子类化扩展
- **完整外壳** —— 自带 main_ppo / video_ray_trainer / my_rl_dataset / reward_score,镜像 verl 路径
- **走官方扩展点** —— `data.custom_cls`、`custom_reward_function`、`run_ppo(task_runner_class=...)`

## 启动

```bash
cd /data/zyt/LLM1/video-r1-verl/video-r1
bash scripts/train_video_r1.sh
```

## 数据

放在 `/data/zyt/datas_video_r1/`:
- `Video-R1-260k.json` —— RL 训练数据
- `NeXT-QA/` —— 视频数据(解压后)
- `Math/` —— 图像数据(解压后)

## 进度

- [x] 目录骨架
- [ ] VideoR1Dataset 实现
- [ ] reward 函数(5 题型 + format + length)
- [ ] yaml 配置(Video-R1 专属覆盖)
- [ ] 跑通基础 GRPO
- [ ] T-GRPO 双 rollout 实现
- [ ] 验证 + 调参
