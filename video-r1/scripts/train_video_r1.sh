#!/bin/bash
# Video-R1 on verl 0.8 训练启动脚本(Phase 1:基础 GRPO,T-GRPO 后续在 trainer.fit() 加)
# 参考:verl-main/examples/grpo_trainer/run_qwen2_5_vl-7b_seq_balance.sh
set -x

# 指定用 GPU 0、2、3、4(物理卡 ID)
# CUDA_VISIBLE_DEVICES 后,这 4 张卡会被 verl 看到为逻辑卡 0、1、2、3
# trainer.n_gpus_per_node 仍设 4(对应逻辑卡数)
export CUDA_VISIBLE_DEVICES=0,2,3,4

# 强制离线模式,避免 transformers/HF Hub 偷偷尝试联网(已通过 modelscope 离线下完)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ENGINE=${1:-vllm}

# === 路径 ===
DATA_ROOT=/data/zyt/datas_video_r1
# 训练用过滤后的小 JSON(filter_local_data.py 产出),里面只含本地有媒体的样本
TRAIN_FILE=$DATA_ROOT/Video-R1-260k-local.json
VAL_FILE=$DATA_ROOT/Video-R1-260k-local.json

VIDEO_R1_HOME=/data/zyt/LLM1/video-r1-verl/video-r1
# VideoR1Dataset 在 main_ppo.py 里静态 import,不走 verl 的 data.custom_cls 路径
REWARD_FN_PATH=$VIDEO_R1_HOME/video_r1/verl/utils/reward_score/video_r1.py

# 让 Python 能 import 到 video_r1 包(verl 假设已在环境里)
export PYTHONPATH=$VIDEO_R1_HOME:$PYTHONPATH

# === 模型 ===
# Video-R1 论文的 SFT 冷启动模型,本地路径(从 modelscope 离线下载)
MODEL_PATH=${MODEL_PATH:-/home/zyt/.cache/modelscope/hub/models/Video-R1/Qwen2.5-VL-7B-COT-SFT}

# === 启动 ===
python3 -m video_r1.verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    +algorithm.tgrpo.enable=True \
    +algorithm.tgrpo.bonus_value=0.3 \
    +algorithm.tgrpo.ratio_threshold=0.8 \
    +algorithm.tgrpo.accuracy_threshold=0.1 \
    \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=4 \
    data.max_prompt_length=16384 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.video_key=videos \
    \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='video_r1_verl' \
    trainer.experiment_name='qwen2_5_vl_7b_phase1' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    $@
