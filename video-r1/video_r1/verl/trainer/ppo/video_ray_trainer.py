# Copyright 2025 Video-R1 implementation on verl 0.8
# Licensed under the Apache License, Version 2.0
"""Video-R1 训练器 —— 子类化 verl RayPPOTrainer,加入 T-GRPO 双 rollout。

T-GRPO 核心思想:
    1. 正常 rollout(视频帧顺序)— verl 默认流程
    2. 对视频样本额外做一次"帧顺序打乱"的 rollout
    3. 比较两次的 mean accuracy:正常 >= 0.8 × 打乱 → 模型用到了时序信息
       → 给答对的视频样本 reward + 0.3

实现路径:
    - 沿用 verl 默认 rollout 流程做"正常 rollout"
    - 在 reward 计算后,插入 T-GRPO 分支:
        * 把 gen_batch 复制一份,agent_name 改成 "video_r1_shuffled_agent"
        * 调 self.async_rollout_manager.generate_sequences(shuffled_gen)
        * VideoR1ShuffledAgentLoop 内部 randperm 视频帧序后再生成
        * 算 shuffled reward,做对比,改 reward_tensor
    - 后面 advantage/loss 用改过的 reward_tensor

配置(在 yaml 或命令行打开):
    algorithm:
      tgrpo:
        enable: True              # 启用 T-GRPO
        bonus_value: 0.3          # 触发后给答对样本加多少分
        ratio_threshold: 0.8      # normal_acc >= ratio * shuffled_acc 才触发
        accuracy_threshold: 0.1   # accuracy > 这个才算"答对"
"""
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    extract_reward,
    should_save_ckpt_esi,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.tracking import Tracking


# ============================================================
# T-GRPO 配置默认值(yaml 没设时用)
# ============================================================
_TGRPO_DEFAULTS = {
    "enable": False,
    "bonus_value": 0.3,
    "ratio_threshold": 0.8,
    "accuracy_threshold": 0.1,
}


def _get_tgrpo_config(config) -> dict:
    """从 config.algorithm.tgrpo 读 T-GRPO 配置,缺的字段用默认。"""
    raw = config.algorithm.get("tgrpo", {})
    merged = dict(_TGRPO_DEFAULTS)
    if raw:
        merged.update(
            OmegaConf.to_container(raw, resolve=True) if hasattr(raw, "keys") else dict(raw)
        )
    return merged


def _has_video_samples(batch: DataProto) -> bool:
    """检查 batch 里是否有视频样本(extra_info.data_type == "video")。"""
    extra_infos = batch.non_tensor_batch.get("extra_info", None)
    if extra_infos is None:
        return False
    for info in extra_infos:
        if isinstance(info, dict) and info.get("data_type") == "video":
            return True
    return False


def _video_mask(batch: DataProto) -> np.ndarray:
    """返回长度 len(batch) 的布尔数组,True 表示视频样本。"""
    extra_infos = batch.non_tensor_batch.get("extra_info", None)
    if extra_infos is None:
        return np.zeros(len(batch), dtype=bool)
    return np.array(
        [isinstance(info, dict) and info.get("data_type") == "video" for info in extra_infos],
        dtype=bool,
    )


class VideoR1RayPPOTrainer(RayPPOTrainer):
    """T-GRPO 训练器 —— 在 verl RayPPOTrainer 之上加视频帧打乱的双 rollout。

    覆盖 fit() 全部 —— 把 verl 原版逻辑复制过来,只在 reward 计算后插一段
    T-GRPO 分支(仿照 REMAX 的双 rollout 模式)。其他流程一字不动。
    """

    # =========================================================
    # T-GRPO 辅助方法
    # =========================================================
    def _run_tgrpo_shuffled_rollout(self, gen_batch: DataProto) -> DataProto:
        """对 gen_batch 做"帧打乱"版本的 rollout,返回 shuffled 完成后的 batch。

        gen_batch 是 repeat(n) 之前的原始 batch。流程:
            1. deepcopy gen_batch
            2. 把所有样本的 agent_name 改成 "video_r1_shuffled_agent"
               (image 样本经过这个 agent 时也会跑,只是没视频可打乱,等价正常 rollout
                —— 小浪费,但实现最简洁)
            3. repeat(n) 同正常 rollout
            4. 调 self.async_rollout_manager.generate_sequences
        """
        shuffled_gen = deepcopy(gen_batch)
        b = len(shuffled_gen)
        shuffled_gen.non_tensor_batch["agent_name"] = np.array(
            ["video_r1_shuffled_agent"] * b, dtype=object
        )
        shuffled_gen = shuffled_gen.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
        )
        return self.async_rollout_manager.generate_sequences(shuffled_gen)

    def _compute_tgrpo_shuffled_reward(
        self, batch: DataProto, shuffled_gen_output: DataProto
    ) -> DataProto:
        """把 shuffled 生成结果套上原 batch 的 prompt 数据,跑一次 reward。

        batch 是 repeat(n) 之后的 batch(已含 prompt 信息)。
        shuffled_gen_output 只有生成部分的 token + 元数据。
        union 后塞进 _compute_reward_colocate。
        """
        shuffled_batch = deepcopy(batch)
        # 清掉原本的 responses / rm_scores 等,准备换成 shuffled 版本
        keys_to_clear = [
            k
            for k in ("responses", "response_mask", "rm_scores", "token_level_scores")
            if k in shuffled_batch.batch
        ]
        if keys_to_clear:
            shuffled_batch.pop(batch_keys=keys_to_clear)
        shuffled_batch = shuffled_batch.union(shuffled_gen_output)
        if "response_mask" not in shuffled_batch.batch.keys():
            shuffled_batch.batch["response_mask"] = compute_response_mask(shuffled_batch)
        return self._compute_reward_colocate(shuffled_batch)

    def _apply_tgrpo_bonus(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict,
        shuffled_reward_batch: DataProto,
        tgrpo_cfg: dict,
    ):
        """对比 normal vs shuffled accuracy,触发条件满足时给答对视频样本加 bonus。

        reward_tensor 形状 (B, L) 稀疏张量,每行只在"最后有效 token"位置非零。
        我们对视频 + 答对的样本,在那个位置加 bonus_value。

        Returns:
            (reward_tensor, tgrpo_metrics_dict)
        """
        normal_acc = np.asarray(
            reward_extra_infos_dict.get("accuracy", []), dtype=np.float32
        )
        shuffled_acc = np.asarray(
            shuffled_reward_batch.non_tensor_batch.get("accuracy", []), dtype=np.float32
        )

        if len(normal_acc) == 0 or len(shuffled_acc) == 0:
            return reward_tensor, {
                "tgrpo/triggered": 0.0,
                "tgrpo/normal_acc_mean": 0.0,
                "tgrpo/shuffled_acc_mean": 0.0,
            }

        normal_mean = float(normal_acc.mean())
        shuffled_mean = float(shuffled_acc.mean())
        triggered = normal_mean >= tgrpo_cfg["ratio_threshold"] * shuffled_mean

        tgrpo_metrics = {
            "tgrpo/normal_acc_mean": normal_mean,
            "tgrpo/shuffled_acc_mean": shuffled_mean,
            "tgrpo/triggered": 1.0 if triggered else 0.0,
            "tgrpo/eligible_count": 0.0,
        }

        if not triggered:
            return reward_tensor, tgrpo_metrics

        # 触发:视频样本 + accuracy > threshold → 加 bonus
        video_mask = _video_mask(batch)
        correct_mask = normal_acc > tgrpo_cfg["accuracy_threshold"]
        eligible = video_mask & correct_mask

        if not eligible.any():
            return reward_tensor, tgrpo_metrics

        # 找每行的非零 token 位置(verl 把 reward 塞到 valid_response_length-1 那个位置)
        # 用 argmax 取布尔张量第一个 True 的列下标 —— 每行只有一个非零位置
        nonzero_col = (reward_tensor != 0).int().argmax(dim=1)

        bonus = float(tgrpo_cfg["bonus_value"])
        eligible_indices = np.nonzero(eligible)[0]
        for i in eligible_indices:
            reward_tensor[i, nonzero_col[i]] += bonus

        tgrpo_metrics["tgrpo/eligible_count"] = float(eligible.sum())
        return reward_tensor, tgrpo_metrics

    # =========================================================
    # 主训练循环 —— 复制 verl RayPPOTrainer.fit(),仅在 reward 后插 T-GRPO 分支
    # =========================================================
    def fit(self):
        """
        The training loop of PPO.

        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        tgrpo_cfg = _get_tgrpo_config(self.config)
        if tgrpo_cfg["enable"]:
            print(f"[VideoR1RayPPOTrainer] T-GRPO ENABLED with config: {tgrpo_cfg}")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)  # actor 同步给 vLLM

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):  # 调试固定 rollout 用的
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):  # 不阻塞上一 step 末尾的异步调用
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # ── 正常 rollout ─────────────────────────────────────
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(
                            gen_batch_output
                        )
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # ── REMAX 双 rollout 分支(verl 原版,我们不动)──────────
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                gen_baseline_batch
                            )
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    # ── 正常 reward ──────────────────────────────────────
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # =====================================================
                    # ★ T-GRPO 双 rollout 注入点 ★
                    # =====================================================
                    if tgrpo_cfg["enable"] and _has_video_samples(batch):
                        with marked_timer("tgrpo_shuffled_gen", timing_raw, color="purple"):
                            # 1. 跑 shuffled rollout(同 prompt,VideoR1ShuffledAgentLoop 内部打乱帧)
                            shuffled_gen_output = self._run_tgrpo_shuffled_rollout(gen_batch)
                        with marked_timer("tgrpo_shuffled_reward", timing_raw, color="yellow"):
                            # 2. 对 shuffled rollout 算 reward
                            shuffled_reward_batch = self._compute_tgrpo_shuffled_reward(
                                batch, shuffled_gen_output
                            )
                        # 3. 对比 + 触发 bonus(就地改 reward_tensor)
                        reward_tensor, tgrpo_metrics = self._apply_tgrpo_bonus(
                            batch,
                            reward_tensor,
                            reward_extra_infos_dict,
                            shuffled_reward_batch,
                            tgrpo_cfg,
                        )
                        metrics.update(tgrpo_metrics)
                        # 把更新后的 rm_scores 写回 batch(覆盖原值,后面 advantage 用它)
                        batch.batch["rm_scores"] = reward_tensor
                        del shuffled_gen_output, shuffled_reward_batch
                    # =====================================================
                    # ★ T-GRPO 注入结束 ★
                    # =====================================================

                    # ── Operating Mode Selection(verl 原版逻辑,不动)──────
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                        "bypass_mode", False
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup > self.global_steps:
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(
                    compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm)
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
