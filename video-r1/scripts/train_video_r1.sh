#!/bin/bash
# Video-R1 on verl 0.8 训练启动脚本
# 参考:verl-main/examples/grpo_trainer/run_qwen2_5_vl-7b_seq_balance.sh
set -x

ENGINE=${1:-vllm}

# === 路径配置 ===
DATA_ROOT=/data/zyt/datas_video_r1
TRAIN_FILE=$DATA_ROOT/Video-R1-260k.json
# TODO: 如果有 val 文件再补上,目前只用 train
VAL_FILE=$DATA_ROOT/Video-R1-260k.json

VIDEO_R1_HOME=/data/zyt/LLM1/video-r1-verl/video-r1
DATASET_CLS_PATH=$VIDEO_R1_HOME/video_r1/verl/utils/dataset/my_rl_dataset.py
REWARD_FN_PATH=$VIDEO_R1_HOME/video_r1/verl/utils/reward_score/video_r1.py

# === 模型 ===
MODEL_PATH=Video-R1/Qwen2.5-VL-7B-COT-SFT    # 或者本地路径

# === 启动 ===
python3 -m video_r1.verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=4 \
    data.max_prompt_length=16384 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.video_key=videos \
    data.custom_cls.path=$DATASET_CLS_PATH \
    data.custom_cls.name=VideoR1Dataset \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='video_r1_verl' \
    trainer.experiment_name='qwen2_5_vl_7b_tgrpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    $@
