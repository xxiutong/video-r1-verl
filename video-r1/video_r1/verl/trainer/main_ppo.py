# Copyright 2025 Video-R1 implementation on verl 0.8
# Licensed under the Apache License, Version 2.0
"""Video-R1 训练入口 —— 子类化 verl TaskRunner,把 trainer 类换成 VideoR1RayPPOTrainer。

启动方式:
    python -m video_r1.verl.trainer.main_ppo \\
        algorithm.adv_estimator=grpo \\
        data.train_files=/data/zyt/datas_video_r1/Video-R1-260k.json \\
        data.custom_cls.path=.../my_rl_dataset.py \\
        data.custom_cls.name=VideoR1Dataset \\
        custom_reward_function.path=.../video_r1.py \\
        custom_reward_function.name=compute_score \\
        actor_rollout_ref.model.path=Video-R1/Qwen2.5-VL-7B-COT-SFT \\
        ...

设计:
    - VideoR1TaskRunner 完整复制 verl TaskRunner.run() 的流程,
      仅把最后一段 trainer = RayPPOTrainer(...) 换成 VideoR1RayPPOTrainer(...)
    - VideoR1RayPPOTrainer 在 Phase 1 是 RayPPOTrainer 的透传(等同基础 GRPO),
      Phase 2 覆盖 fit() 加入 T-GRPO 双 rollout
    - 这样 Phase 1 → Phase 2 切换无需改 main_ppo.py / TaskRunner 结构
"""
import os
import socket

import hydra
import ray

from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset,
    create_rl_sampler,
    run_ppo,
)
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

from video_r1.verl.trainer.ppo.video_ray_trainer import VideoR1RayPPOTrainer

# 副作用 import:让 @register("video_r1_shuffled_agent") 在 agent loop 注册表里登记
# Phase 2 在 fit() 里把 batch.non_tensor_batch["agent_name"] 切到 "video_r1_shuffled_agent"
# 来触发 T-GRPO 的打乱-帧 rollout
from video_r1.verl.experimental.agent_loop import video_r1_agent_loop  # noqa: F401


class VideoR1TaskRunner(TaskRunner):
    """Video-R1 自定义 TaskRunner。

    继承 verl TaskRunner,完整复制 run() 的逻辑,
    只在实例化 trainer 那一段把 RayPPOTrainer 换成 VideoR1RayPPOTrainer。
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)

        self.add_teacher_model_resource_pool(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        # 根据mapping和资源确定pool，比如所有Worker对应global_pool，这个global_pool中所有资源都使用所有卡
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # === Video-R1 唯一改动:把 RayPPOTrainer 换成 VideoR1RayPPOTrainer ===
        # Initialize the PPO trainer.
        trainer = VideoR1RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Hydra 入口 —— 把 VideoR1TaskRunner 传给 run_ppo,run_ppo 内部负责启 Ray 集群 + 远程调用 .run()。"""
    runner_cls = ray.remote(num_cpus=1)(VideoR1TaskRunner)
    run_ppo(config, task_runner_class=runner_cls)


if __name__ == "__main__":
    main()
