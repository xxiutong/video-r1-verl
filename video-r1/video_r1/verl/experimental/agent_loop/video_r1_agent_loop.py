# Copyright 2025 Video-R1 implementation on verl 0.8
# Licensed under the Apache License, Version 2.0
"""Video-R1 自定义 AgentLoop —— 打乱视频帧用,服务 T-GRPO 第二次 rollout。

继承 SingleTurnAgentLoop,只改一处:在 process_vision_info 加载完视频帧之后、
apply_chat_template 之前,沿时间维度 randperm 打乱每条视频的帧序。

使用方式:
    在 fit() 里构造 T-GRPO 第二次 rollout 的 batch 时,把
    batch.non_tensor_batch["agent_name"] 设成 "video_r1_shuffled_agent",
    其余流程跟正常 rollout 一致(同样的 async_rollout_manager.generate_sequences)。

模块导入:
    必须在程序启动早期 import 这个模块,让 @register 装饰器执行注册。
    在 video_r1.verl.trainer.main_ppo 里加 `from video_r1.verl.experimental.agent_loop import video_r1_agent_loop  # noqa`。

实现要点(对照 verl SingleTurnAgentLoop):
    1. process_vision_info 返回 multi_modal_data,其中 videos 是 list[(Tensor, dict)],
       每个 Tensor 形状 (T, C, H, W)
    2. randperm 第 0 维 → 打乱帧序,空间内容、metadata 不变
    3. 打乱后的 videos 同时进 apply_chat_template 和 server_manager.generate
"""
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput


def _shuffle_video_frames(video):
    """沿时间维度 randperm 打乱视频帧。

    video 是 (Tensor, dict) tuple,Tensor 形状 (T, C, H, W)。
    返回 (shuffled_Tensor, 原 dict)。

    若上游返回的是裸 Tensor(无 metadata),也兼容处理。
    """
    if isinstance(video, tuple) and len(video) == 2:
        tensor, meta = video
        idx = torch.randperm(tensor.shape[0])
        return (tensor[idx], meta)
    if torch.is_tensor(video):
        idx = torch.randperm(video.shape[0])
        return video[idx]
    raise TypeError(f"Unexpected video element type: {type(video)}")


@register("video_r1_shuffled_agent")
class VideoR1ShuffledAgentLoop(SingleTurnAgentLoop):
    """Video-R1 T-GRPO 第二次 rollout 专用 —— 把视频帧打乱后再生成回答。

    其他逻辑完全沿用 verl 的 SingleTurnAgentLoop,只覆盖 run() 中
    `process_vision_info` 与 `apply_chat_template` 之间的一段。
    """

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. 加载图像 / 视频帧(同 SingleTurnAgentLoop)
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. T-GRPO 唯一改动:randperm 打乱视频帧
        if videos:
            videos = [_shuffle_video_frames(v) for v in videos]
            multi_modal_data["videos"] = videos

        # 3. tokenize prompt(同 SingleTurnAgentLoop)
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 4. 生成(同 SingleTurnAgentLoop)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = (
                output.num_preempted if output.num_preempted is not None else -1
            )
        response_mask = [1] * len(output.token_ids)

        out: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        out.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return out
