# video-r1-verl

在 [verl](https://github.com/volcengine/verl) 上复现 [Video-R1](https://github.com/tulerfeng/Video-R1)
论文的 T-GRPO 视频推理训练。

> Video-R1 原版基于 TRL 实现,本仓库基于verl框架实现。
> 不对verl源码进行修改,所有 Video-R1 专属功能(数据加载、reward、T-GRPO fit)都以
> 子类化的方式放在 `video-r1/` 下,跟 verl 并列。

## 上游

- **Video-R1**: <https://github.com/tulerfeng/Video-R1>(原论文 + 代码 + 数据集)
- **verl**: <https://github.com/verl-project/verl> 由于8.0dev版本变更相比于0.7.1release变更较多，直接基于0.8dev实现(bf4b152e5d178eed2c6bdb79620111442f82689c)
- **数据集和COT-SFT模型**: 均来自Video-R1

## 目录布局

```
video-r1-verl/
├── verl-main/                                 verl 0.8
└── video-r1/
    ├── video_r1/verl/
    │   ├── trainer/main_ppo.py                训练入口
    │   ├── trainer/ppo/video_ray_trainer.py   T-GRPO fit
    │   ├── trainer/config/ppo_trainer.yaml    配置
    │   ├── utils/dataset/my_rl_dataset.py     
    │   ├── utils/reward_score/video_r1.py     reward 函数
    │   └── experimental/agent_loop/video_r1_agent_loop.py   帧打乱 rollout
    ├── datasets/preprocess/                   数据预处理
    └── scripts/train_video_r1.sh              启动脚本
```

## 快速开始

```bash
# 1. 下载SFT模型和RL数据集到本地(从 HF 或 ModelScope,见上游链接)

# 2. 启动训练
bash scripts/train_video_r1.sh
```


## Acknowledgements

We sincerely appreciate the contributions of the open-source community.
The related projects are as follows:
[Video-R1](https://github.com/tulerfeng/Video-R1),
[verl](https://github.com/verl-project/verl),
[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1).

## Citations

```bibtex
@article{feng2025video,
  title={Video-R1: Reinforcing Video Reasoning in MLLMs},
  author={Feng, Kaituo and Gong, Kaixiong and Li, Bohao and Guo, Zonghao and Wang, Yibing and Peng, Tianshuo and Wang, Benyou and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2503.21776},
  year={2025}
}
```
