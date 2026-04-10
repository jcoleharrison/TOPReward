from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from topreward.clients.base import BaseModelClient
from topreward.metrics.instruction_reward import InstructionRewardResult
from topreward.utils.aliases import Event, ImageEvent, ImageNumpy, ImageT, TextEvent
from topreward.utils.constants import MAX_TOKENS_TO_GENERATE
from topreward.utils.images import to_pil


class QwenClient(BaseModelClient):
    ALIGNED_VIDEO_SAMPLE_FPS = 2.0
    PREFIX_CACHE_SUPPORTED = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        rpm: float = 0.0,
        max_input_length: int = 32768,
        prefix_cache_enabled: bool = True,
    ):
        super().__init__(
            rpm=rpm,
            max_input_length=max_input_length,
            prefix_cache_enabled=prefix_cache_enabled,
        )
        self.model_name = model_name
        self.model_config = AutoConfig.from_pretrained(model_name)
        model_cls = self._resolve_model_class(self.model_config)
        if model_cls is None:
            raise ValueError(
                "Unsupported Qwen model architecture "
                f"('{self.model_config.model_type}'). "
                "Supported model types are: qwen3_vl, qwen3_vl_moe, qwen3_5, and qwen3_5_moe."
            )
        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        if not hasattr(self.processor, "video_processor"):
            raise ValueError(
                f"Model '{model_name}' does not expose a video processor through AutoProcessor; "
                "TOPReward Qwen client requires a vision-capable Qwen model for instruction scoring."
            )
        logger.info(type(self.processor))

    @staticmethod
    def _resolve_model_class(config) -> type:
        """Return the best Qwen model class for the detected model type/architecture."""
        qwen_model_type = getattr(config, "model_type", "").lower()
        qwen_model_archs = [arch.lower() for arch in getattr(config, "architectures", [])]

        if qwen_model_type == "qwen3_vl":
            return Qwen3VLForConditionalGeneration
        if qwen_model_type == "qwen3_vl_moe":
            return Qwen3VLMoeForConditionalGeneration
        if qwen_model_type == "qwen3_5":
            return Qwen3_5ForConditionalGeneration
        if qwen_model_type == "qwen3_5_moe":
            return Qwen3_5MoeForConditionalGeneration

        if "qwen3vlforconditionalgeneration" in qwen_model_archs:
            return Qwen3VLForConditionalGeneration
        if "qwen3vlmoeforconditionalgeneration" in qwen_model_archs:
            return Qwen3VLMoeForConditionalGeneration
        if "qwen3_5forconditionalgeneration" in qwen_model_archs:
            return Qwen3_5ForConditionalGeneration
        if "qwen3_5moeforconditionalgeneration" in qwen_model_archs:
            return Qwen3_5MoeForConditionalGeneration

        return None

    @staticmethod
    def _aligned_video_indices(
        total_frames: int,
        raw_fps: float,
        sample_fps: float = ALIGNED_VIDEO_SAMPLE_FPS,
    ) -> list[int]:
        """Return a prefix-stable frame schedule for Qwen video prompts."""
        if total_frames <= 0:
            return []
        if total_frames == 1:
            return [0]
        if raw_fps <= 0:
            raise ValueError(f"raw_fps must be positive, got {raw_fps}")
        if sample_fps <= 0:
            raise ValueError(f"sample_fps must be positive, got {sample_fps}")

        step = max(raw_fps / sample_fps, 1.0)
        positions = np.arange(0.0, float(total_frames), step, dtype=np.float64)
        indices = np.round(positions).astype(int)
        indices = np.clip(indices, 0, total_frames - 1)
        return list(dict.fromkeys(indices.tolist()))

    @classmethod
    def _aligned_video_content(
        cls,
        pil_frames: list,
        raw_fps: float,
        raw_total_frames: int | None = None,
        raw_frame_indices: list[int] | None = None,
    ) -> tuple[dict, dict]:
        """Build a Qwen video input with explicit prefix-stable metadata."""
        video_tensor, video_metadata = cls._aligned_video_tensor_and_metadata(
            pil_frames=pil_frames,
            raw_fps=raw_fps,
            raw_total_frames=raw_total_frames,
            raw_frame_indices=raw_frame_indices,
        )
        video_element = {"type": "video", "video": "aligned_prefix_video"}
        vision_kwargs = {
            "videos": [video_tensor],
            "video_metadata": [video_metadata],
            "do_sample_frames": False,
        }
        return video_element, vision_kwargs

    @classmethod
    def _aligned_video_tensor_and_metadata(
        cls,
        pil_frames: list,
        raw_fps: float,
        raw_total_frames: int | None = None,
        raw_frame_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Return the sampled video tensor and its metadata for aligned prefixes."""
        if raw_total_frames is None:
            raw_total_frames = len(pil_frames)
        if raw_frame_indices is None:
            raw_frame_indices = cls._aligned_video_indices(total_frames=raw_total_frames, raw_fps=raw_fps)
        if not raw_frame_indices:
            raw_frame_indices = [0]
        padded_raw_frame_indices = list(raw_frame_indices)
        if len(padded_raw_frame_indices) == 1:
            # Qwen3-VL video preprocessing requires at least two temporal steps.
            padded_raw_frame_indices.append(padded_raw_frame_indices[0])

        sampled_frames = [pil_frames[idx] for idx in padded_raw_frame_indices]
        video_tensor = torch.stack(
            [torch.from_numpy(np.array(frame.convert("RGB")).transpose(2, 0, 1)) for frame in sampled_frames]
        )
        video_metadata = {
            "fps": raw_fps,
            "frames_indices": padded_raw_frame_indices,
            "total_num_frames": raw_total_frames,
        }
        return video_tensor, video_metadata

    def _instruction_reward_prompt_parts(
        self,
        instruction: str,
        trajectory_description: str | None,
        add_chat_template: bool,
        video_element: dict,
    ) -> tuple[str, list[dict], str]:
        """Build the instruction-reward prompt text and tokenized text payload."""
        if trajectory_description is not None:
            prompt_text = (
                f"{trajectory_description} Therefore given the above "
                f"description and the video, the video shows a robot "
                f"manipulation trajectory that **completes** the following "
                f"instruction: "
            )
        else:
            prompt_text = "The above video shows a robot manipulation trajectory that completes the following task: "

        user_messages = [{"role": "user", "content": [video_element, {"type": "text", "text": prompt_text}]}]
        eos_token = self.processor.tokenizer.eos_token

        if add_chat_template:
            instruction_suffix = f"{instruction} Decide whether the above statement is True or not. The answer is:"
            templated_messages = [
                {
                    "role": "user",
                    "content": [
                        video_element,
                        {"type": "text", "text": f"{prompt_text}{instruction_suffix}"},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                templated_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = f"{prompt_chat}True"
        else:
            instruction_suffix = f"{instruction} Decide whether the above statement is True or not. The answer is: True"
            prompt_chat = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if eos_token is not None:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"

        return prompt_text, user_messages, full_text

    def _find_video_prefix_end(self, input_ids: torch.Tensor) -> int:
        """Return the exclusive token index of the video block within a prompt."""
        vision_end_positions = (input_ids == self.processor.vision_end_token_id).nonzero(as_tuple=False).flatten()
        if vision_end_positions.numel() == 0:
            raise ValueError("Unable to locate the end of the Qwen video block in the prompt")
        return int(vision_end_positions[-1].item()) + 1

    def _prepare_instruction_reward_cache_inputs(
        self,
        frames: list[ImageNumpy],
        instruction: str,
        fps: float,
        add_chat_template: bool,
        raw_frame_indices: list[int],
        raw_total_frames: int,
    ) -> dict:
        """Tokenize the full aligned prompt once and split reusable/tail segments."""
        pil_frames = [to_pil(cast(ImageT, f)) for f in frames]
        video_element, vision_kwargs = self._aligned_video_content(
            pil_frames=pil_frames,
            raw_fps=fps,
            raw_total_frames=raw_total_frames,
            raw_frame_indices=raw_frame_indices,
        )
        _, _, full_text = self._instruction_reward_prompt_parts(
            instruction=instruction,
            trajectory_description=None,
            add_chat_template=add_chat_template,
            video_element=video_element,
        )
        inputs = self.processor(text=[full_text], padding=True, return_tensors="pt", **vision_kwargs)
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        video_grid_thw = inputs["video_grid_thw"].to("cuda")
        pixel_values_videos = inputs["pixel_values_videos"].to("cuda")
        position_ids, _ = self.model.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

        split_after_video = self._find_video_prefix_end(input_ids[0])
        false_token_id = self.processor.tokenizer.encode(" False", add_special_tokens=False)[0]
        return {
            "target_token_id": input_ids[:, -1],
            "false_token_id": false_token_id,
            "base_input_ids": input_ids[:, :split_after_video],
            "base_position_ids": position_ids[:, :, :split_after_video],
            "tail_input_ids": input_ids[:, split_after_video:-1],
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    @staticmethod
    def _build_text_position_ids(
        start_position: int,
        batch_size: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build 3-channel sequential text position ids at a specific absolute index."""
        return (
            torch.arange(start_position, start_position + length, device=device, dtype=dtype)
            .view(1, 1, length)
            .expand(3, batch_size, length)
        )

    def _build_partial_video_block(
        self,
        frames: list[ImageT],
        raw_frame_indices: list[int],
        raw_total_frames: int,
        fps: float,
        base_start_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one temporary padded video block for prefixes with an incomplete merge group."""
        merge_size = self.processor.video_processor.merge_size
        padded_raw_frame_indices = list(raw_frame_indices)
        if len(padded_raw_frame_indices) % merge_size != 0:
            padded_raw_frame_indices.extend([padded_raw_frame_indices[-1]] * (merge_size - len(padded_raw_frame_indices) % merge_size))
        partial_frames = [to_pil(cast(ImageT, frames[idx])) for idx in padded_raw_frame_indices]
        video_tensor = torch.stack(
            [torch.from_numpy(np.array(frame.convert("RGB")).transpose(2, 0, 1)) for frame in partial_frames]
        )
        video_metadata = {
            "fps": fps,
            "frames_indices": raw_frame_indices,
            "total_num_frames": raw_total_frames,
        }
        processed = self.processor.video_processor(
            videos=[video_tensor],
            do_sample_frames=False,
            video_metadata=[video_metadata],
            return_metadata=True,
            return_tensors="pt",
        )
        pixel_values_videos = processed["pixel_values_videos"].to("cuda")
        video_grid_thw = processed["video_grid_thw"].to("cuda")
        merge_length = merge_size**2
        frame_seqlen = int(video_grid_thw[0][1:].prod().item() // merge_length)
        timestamps = self.processor._calculate_timestamps(raw_frame_indices, fps, merge_size)
        block_text = ""
        for timestamp in timestamps:
            block_text += f"<{timestamp:.1f} seconds>"
            block_text += self.processor.vision_start_token + (self.processor.video_token * frame_seqlen) + self.processor.vision_end_token
        block_input_ids = self.processor.tokenizer(
            block_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to("cuda")
        block_attention_mask = torch.ones_like(block_input_ids)
        block_position_ids, _ = self.model.model.get_rope_index(
            input_ids=block_input_ids,
            video_grid_thw=video_grid_thw,
            attention_mask=block_attention_mask,
        )
        block_position_ids = block_position_ids + base_start_position
        return block_input_ids, block_position_ids, pixel_values_videos, video_grid_thw

    def _build_multimodal_segment(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Inject aligned video features into token embeddings for a prompt segment."""
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        visual_pos_masks = None
        deepstack_visual_embeds = None

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.model.model.get_video_features(
                pixel_values_videos,
                video_grid_thw,
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask_1d = input_ids == self.model.config.video_token_id
            video_mask = video_mask_1d.unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[video_mask].numel() != video_embeds.numel():
                raise ValueError(
                    "Aligned video features and placeholder tokens do not match: "
                    f"{int(video_mask_1d.sum())} tokens vs {video_embeds.shape[0]} features"
                )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            visual_pos_masks = video_mask_1d.to(inputs_embeds.device)
            deepstack_visual_embeds = [embed.to(inputs_embeds.device, inputs_embeds.dtype) for embed in deepstack_video_embeds]

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    @staticmethod
    def _with_flash_packed_text_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
        """Prepend zero-based text positions so FA2 can pack cached query chunks safely."""
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            return position_ids
        _, batch_size, query_len = position_ids.shape
        text_position_ids = torch.arange(query_len, device=position_ids.device, dtype=position_ids.dtype).view(1, 1, query_len)
        text_position_ids = text_position_ids.expand(1, batch_size, query_len)
        return torch.cat((text_position_ids, position_ids), dim=0)

    def _forward_language_model(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        prefer_eager: bool = False,
    ):
        original_attn_impl = self.model.language_model.config._attn_implementation
        packed_position_ids = position_ids
        if original_attn_impl == "flash_attention_2":
            packed_position_ids = self._with_flash_packed_text_position_ids(position_ids)
        self.model.eval()

        def _run():
            with torch.no_grad():
                return self.model.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    position_ids=packed_position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )

        def _is_oom(exc: BaseException) -> bool:
            return "out of memory" in str(exc).lower()

        if prefer_eager:
            self.model.language_model.config._attn_implementation = "eager"
            try:
                return _run()
            finally:
                self.model.language_model.config._attn_implementation = original_attn_impl

        try:
            self.model.language_model.config._attn_implementation = original_attn_impl
            return _run()
        except (RuntimeError, ValueError) as exc:
            if _is_oom(exc) or original_attn_impl == "eager":
                raise
            logger.debug(f"Falling back to eager attention for cached Qwen LM forward: {exc}")
            self.model.language_model.config._attn_implementation = "eager"
            try:
                return _run()
            finally:
                self.model.language_model.config._attn_implementation = original_attn_impl

    def _compute_instruction_reward_with_cached_append(
        self,
        cached_prefix: dict | None,
        prepared_inputs: dict,
        append_input_ids: torch.Tensor,
        append_position_ids: torch.Tensor,
        reusable_prefix_length: int,
        temporal_blocks: int,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[InstructionRewardResult, dict]:
        """Append a segment plus tail in one forward, then crop the cache back to the reusable video prefix."""
        if append_input_ids.shape[1] == 0:
            raise ValueError("Combined cached append is empty; cannot score target token")

        append_inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._build_multimodal_segment(
            append_input_ids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        if cached_prefix is None:
            prefix_cache = None
            prefix_base_length = 0
            cache_position = torch.arange(
                append_input_ids.shape[1],
                device=append_input_ids.device,
            )
            attention_mask = torch.ones(
                append_input_ids.shape,
                dtype=torch.long,
                device=append_input_ids.device,
            )
        else:
            prefix_cache = cached_prefix["past_key_values"]
            prefix_base_length = cached_prefix["base_length"]
            cache_position = torch.arange(
                prefix_base_length,
                prefix_base_length + append_input_ids.shape[1],
                device=append_input_ids.device,
            )
            attention_mask = torch.ones(
                (append_input_ids.shape[0], prefix_base_length + append_input_ids.shape[1]),
                dtype=torch.long,
                device=append_input_ids.device,
            )

        outputs = self._forward_language_model(
            inputs_embeds=append_inputs_embeds,
            position_ids=append_position_ids,
            attention_mask=attention_mask,
            past_key_values=prefix_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
        log_probs = F.log_softmax(logits, dim=-1)
        target_token = prepared_inputs["target_token_id"][0]
        reward = log_probs[0, target_token].item()
        false_reward = log_probs[0, prepared_inputs["false_token_id"]].item()

        new_cache = outputs.past_key_values
        if hasattr(new_cache, "crop"):
            new_cache.crop(reusable_prefix_length)

        updated_cache = {
            "past_key_values": new_cache,
            "base_length": reusable_prefix_length,
            "temporal_blocks": temporal_blocks,
        }
        return (
            InstructionRewardResult(
                reward=reward,
                reduction="mean",
                token_count=1,
                per_token_log_probs=[reward],
                token_ids=[int(target_token.item())],
                false_reward=false_reward,
            ),
            updated_cache,
        )

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore[misc]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        input_len = inputs["input_ids"].shape[-1]
        if input_len > self.max_input_length:
            raise ValueError(f"Input length {input_len} exceeds maximum of {self.max_input_length} tokens")
        logger.info(f"Input length: {input_len}")

        # Inference: Generation of the output
        if temperature == 0.0:
            generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, do_sample=False)
        else:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS_TO_GENERATE,
                do_sample=True,
                temperature=temperature,
            )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def generate_object_state_reasoning(
        self,
        frames: Sequence[ImageT],
        fps: float = 2.0,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a description of the robot manipulation trajectory.

        This generates an instruction-agnostic description to avoid circular dependencies
        where mentioning instruction objects would artificially inflate likelihood scores.

        Args:
            frames: List of images representing the video.
            fps: Frames per second for video input (default: 2.0).
            max_new_tokens: Maximum tokens to generate for reasoning.

        Returns:
            Generated text describing the trajectory.
        """
        # Convert frames to PIL images
        pil_frames = [to_pil(cast(ImageT, f)) for f in frames]

        content = [
            {"type": "video", "video": pil_frames, "fps": fps},
            {
                "type": "text",
                "text": "Describe the robot manipulation trajectory in this video:",
            },
        ]

        user_messages = [{"role": "user", "content": content}]

        # Apply chat template
        prompt_chat = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)

        # Process vision info
        image_inputs, video_inputs = process_vision_info(user_messages)  # type: ignore[misc]

        # Prepare inputs
        inputs = self.processor(
            text=[prompt_chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Generate description
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        # Decode response
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Extract only the generated part (after the prompt)
        # The response includes the prompt, so we need to extract just the
        # generated text

        prompt_text = self.processor.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        description = response[len(prompt_text) :].strip() if response.startswith(prompt_text) else response.strip()

        return description

    def compute_instruction_reward(
        self,
        frames: list[ImageNumpy],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
        use_video_description: bool = False,
        add_chat_template: bool = False,
        raw_frame_indices: list[int] | None = None,
        raw_total_frames: int | None = None,
    ) -> InstructionRewardResult:
        """Compute a log-likelihood reward for an instruction conditioned on a trajectory of frames.

        This implements the instruction reward approach from "Vision Language Models are
        In-Context Value Learners", measuring how well the trajectory matches the given
        instruction by computing the log-probability of generating the instruction text.

        Args:
            frames: List of images representing the trajectory (at least 2 frames).
            instruction: Instruction text to evaluate.
            reduction: Reduction to apply to token log probabilities ("mean" or "sum").
            fps: Frames per second for video input (default: 2.0).
            use_video_description: If True, generate instruction-agnostic description of
                                  the robot manipulation trajectory, then prepend it as context
                                  before evaluating instruction likelihood. This avoids circular
                                  dependencies that would artificially inflate scores.
            add_chat_template: If True, wrap the full prompt (including instruction) with
                               the chat template before tokenization.

        Returns:
            InstructionRewardResult with the computed reward and metadata.
        """
        pil_frames = [to_pil(cast(ImageT, f)) for f in frames]
        trajectory_description = None
        if use_video_description:
            logger.info("Generating trajectory description...")
            trajectory_description = self.generate_object_state_reasoning(frames, fps=fps)
            logger.info("Generated trajectory description.")
            logger.info(f"Trajectory description: {trajectory_description!r}")

        video_element, vision_kwargs = self._aligned_video_content(
            pil_frames=pil_frames,
            raw_fps=fps,
            raw_total_frames=raw_total_frames,
            raw_frame_indices=raw_frame_indices,
        )
        _, _, full_text = self._instruction_reward_prompt_parts(
            instruction=instruction,
            trajectory_description=trajectory_description,
            add_chat_template=add_chat_template,
            video_element=video_element,
        )

        inputs = self.processor(text=[full_text], padding=True, return_tensors="pt", **vision_kwargs)

        inputs = inputs.to("cuda")
        labels = inputs["input_ids"].clone()

        # Mask the prompt so we only compute loss on the instruction + "True" part
        prompt_length = inputs["input_ids"].shape[1] - 1
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        # Compute per-token log probabilities
        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]

        # Apply reduction
        reward = masked_log_probs.sum().item() if reduction == "sum" else masked_log_probs.mean().item()

        # Extract per-token info for metadata
        per_token_log_probs_list = masked_log_probs.detach().cpu().tolist()
        token_ids_list = target_labels[mask].detach().cpu().tolist()

        false_token_id = self.processor.tokenizer.encode(" False", add_special_tokens=False)[0]
        last_logit_pos = mask.nonzero(as_tuple=True)[1][-1]
        false_log_prob = log_probs[0, last_logit_pos, false_token_id].item()

        return InstructionRewardResult(
            reward=reward,
            reduction=reduction,
            token_count=len(per_token_log_probs_list),
            per_token_log_probs=per_token_log_probs_list,
            token_ids=token_ids_list,
            trajectory_description=trajectory_description,
            false_reward=false_log_prob,
        )

    @staticmethod
    def normalize_rewards(
        rewards: Sequence[float],
        method: str = "minmax",
    ) -> np.ndarray:
        """Normalize a sequence of instruction rewards to a 0-1 range.

        This is useful for comparing rewards across different trajectories or
        trajectory prefixes, as raw log-likelihood values can be hard to interpret.

        Args:
            rewards: Sequence of raw reward values (log-likelihoods).
            method: Normalization method. Options:
                - "minmax": Scale to [0, 1] using min-max normalization.

        Returns:
            Normalized rewards as a numpy array in [0, 1] range.
        """
        rewards_arr = np.array(rewards, dtype=np.float64)

        if len(rewards_arr) == 0:
            return rewards_arr

        if len(rewards_arr) == 1:
            return np.array([1.0])

        if method == "minmax":
            r_min, r_max = rewards_arr.min(), rewards_arr.max()
            if r_max == r_min:
                # All rewards are identical
                return np.ones_like(rewards_arr)
            return (rewards_arr - r_min) / (r_max - r_min)

        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'softmax'.")

    def compute_instruction_rewards_for_prefixes(  # type: ignore[override]
        self,
        frames: list[ImageNumpy],
        instruction: str,
        num_samples: int = 15,
        reduction: str = "mean",
        fps: float = 2.0,
        use_video_description: bool = False,
        add_chat_template: bool = False,
        predict_last_n_prefixes: int | None = None,
    ) -> InstructionRewardResult:
        """Compute instruction rewards for trajectory prefixes of varying lengths.

        Args:
            frames: Full list of trajectory frames.
            instruction: Instruction text to evaluate.
            num_samples: Number of prefix lengths to sample (uniformly spaced).
            reduction: Reduction method ("mean" or "sum").
            fps: Frames per second for video input.
            add_chat_template: Whether to wrap the instruction prompt with the chat template.
            predict_last_n_prefixes: If set, only predict for the last N prefix lengths
                (e.g., 3 means only run on the 3 longest prefixes). All prefix lengths
                are still computed for metadata but only the last N get model calls.

        Returns:
            InstructionRewardResult with prefix_lengths, prefix_rewards, and normalized_prefix_rewards.
            The main reward is the full trajectory reward (last prefix).
        """

        num_frames = len(frames)

        if num_samples <= 0:
            # Use every frame as a prefix length (1, 2, ..., num_frames)
            prefix_lengths = list(range(1, num_frames + 1))
        elif num_frames > 2:
            num_samples = min(num_samples, num_frames)
            # Generate uniformly spaced prefix lengths from 1 to full trajectory
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted({int(x) for x in prefix_lengths})
        else:
            prefix_lengths = [num_frames]

        # Optionally limit to only the last N prefix lengths
        if predict_last_n_prefixes is not None:
            prefix_lengths = prefix_lengths[-predict_last_n_prefixes:]

        rewards: list[float] = []
        false_rewards: list[float] = []
        token_counts: list[int] = []
        aligned_full_indices = self._aligned_video_indices(total_frames=num_frames, raw_fps=fps)

        def _compute_uncached_for_length(length: int) -> InstructionRewardResult:
            prefix_raw_indices = [idx for idx in aligned_full_indices if idx < length]
            if not prefix_raw_indices:
                prefix_raw_indices = [0]
            return self.compute_instruction_reward(
                frames=frames[:length],
                instruction=instruction,
                reduction=reduction,
                fps=fps,
                use_video_description=use_video_description,
                add_chat_template=add_chat_template,
                raw_frame_indices=prefix_raw_indices,
                raw_total_frames=length,
            )

        if not self.prefix_cache_enabled or use_video_description or add_chat_template:
            for length in prefix_lengths:
                result = _compute_uncached_for_length(length)
                rewards.append(result.reward)
                false_rewards.append(result.false_reward if result.false_reward is not None else float("nan"))
                token_counts.append(result.token_count)
                prefix_raw_indices = [idx for idx in aligned_full_indices if idx < length] or [0]
                logger.info(
                    f"Prefix {length:3d} raw frames -> {len(prefix_raw_indices)} aligned video frames: "
                    f"reward={result.reward:.4f} false={result.false_reward:.4f} ({result.token_count} tokens)"
                )
        else:
            full_prepared_inputs = self._prepare_instruction_reward_cache_inputs(
                frames=frames,
                instruction=instruction,
                fps=fps,
                add_chat_template=add_chat_template,
                raw_frame_indices=aligned_full_indices,
                raw_total_frames=num_frames,
            )
            full_base_input_ids = full_prepared_inputs["base_input_ids"]
            full_base_position_ids = full_prepared_inputs["base_position_ids"]
            full_tail_input_ids = full_prepared_inputs["tail_input_ids"]
            merge_size = self.processor.video_processor.merge_size
            full_video_grid = full_prepared_inputs["video_grid_thw"][0]
            full_temporal_blocks = int(full_video_grid[0].item())
            block_patch_count = int(full_video_grid[1].item() * full_video_grid[2].item())
            base_vision_end_positions = (full_base_input_ids[0] == self.processor.vision_end_token_id).nonzero(as_tuple=False).flatten()
            if base_vision_end_positions.numel() != full_temporal_blocks:
                raise ValueError(
                    "Aligned full prompt vision block count mismatch: "
                    f"{int(base_vision_end_positions.numel())} vision ends vs {full_temporal_blocks} temporal blocks"
                )

            cached_prefix = None
            try:
                for length in prefix_lengths:
                    prefix_raw_indices = [idx for idx in aligned_full_indices if idx < length]
                    if not prefix_raw_indices:
                        prefix_raw_indices = [0]
                    complete_blocks = len(prefix_raw_indices) // merge_size

                    if complete_blocks == 0:
                        result = _compute_uncached_for_length(length)
                        rewards.append(result.reward)
                        false_rewards.append(result.false_reward if result.false_reward is not None else float("nan"))
                        token_counts.append(result.token_count)
                        logger.info(
                            f"Prefix {length:3d} raw frames -> {len(prefix_raw_indices)} aligned video frames: "
                            f"reward={result.reward:.4f} false={result.false_reward:.4f} ({result.token_count} tokens, cache=warmup)"
                        )
                        continue

                    has_partial_block = len(prefix_raw_indices) % merge_size != 0
                    reusable_prefix_length = int(base_vision_end_positions[complete_blocks - 1].item()) + 1
                    tail_base_start_position = int(full_base_position_ids[:, :, reusable_prefix_length - 1].max().item()) + 1
                    partial_input_ids = None
                    partial_position_ids = None
                    partial_pixel_values_videos = None
                    partial_video_grid_thw = None
                    partial_temporal_blocks = 0

                    if has_partial_block:
                        partial_input_ids, partial_position_ids, partial_pixel_values_videos, partial_video_grid_thw = self._build_partial_video_block(
                            frames=frames,
                            raw_frame_indices=prefix_raw_indices[complete_blocks * merge_size :],
                            raw_total_frames=length,
                            fps=fps,
                            base_start_position=tail_base_start_position,
                        )
                        tail_base_start_position = int(partial_position_ids[:, :, -1].max().item()) + 1
                        partial_temporal_blocks = int(partial_video_grid_thw[0, 0].item())

                    tail_position_ids = self._build_text_position_ids(
                        start_position=tail_base_start_position,
                        batch_size=full_tail_input_ids.shape[0],
                        length=full_tail_input_ids.shape[1],
                        device=full_tail_input_ids.device,
                        dtype=full_base_position_ids.dtype,
                    )
                    prepared_inputs = {
                        "tail_input_ids": full_tail_input_ids,
                        "tail_position_ids": tail_position_ids,
                        "target_token_id": full_prepared_inputs["target_token_id"],
                        "false_token_id": full_prepared_inputs["false_token_id"],
                    }

                    if cached_prefix is None:
                        first_base_input_ids = full_base_input_ids[:, :reusable_prefix_length]
                        first_base_position_ids = full_base_position_ids[:, :, :reusable_prefix_length]
                        first_append_segments = [first_base_input_ids]
                        first_append_position_segments = [first_base_position_ids]
                        first_pixel_segments = []
                        first_temporal_blocks = complete_blocks
                        if complete_blocks > 0:
                            first_pixel_segments.append(full_prepared_inputs["pixel_values_videos"][: complete_blocks * block_patch_count])
                        if partial_input_ids is not None:
                            first_append_segments.append(partial_input_ids)
                            first_append_position_segments.append(partial_position_ids)
                            first_pixel_segments.append(partial_pixel_values_videos)
                            first_temporal_blocks += partial_temporal_blocks
                        first_append_segments.append(prepared_inputs["tail_input_ids"])
                        first_append_position_segments.append(prepared_inputs["tail_position_ids"])
                        first_append_input_ids = torch.cat(first_append_segments, dim=1)
                        first_append_position_ids = torch.cat(first_append_position_segments, dim=2)
                        first_pixel_values_videos = torch.cat(first_pixel_segments, dim=0) if first_pixel_segments else None
                        first_video_grid_thw = None
                        if first_pixel_values_videos is not None:
                            first_video_grid_thw = full_video_grid.new_tensor(
                                [[first_temporal_blocks, int(full_video_grid[1].item()), int(full_video_grid[2].item())]]
                            )
                        result, cached_prefix = self._compute_instruction_reward_with_cached_append(
                            cached_prefix=None,
                            prepared_inputs=prepared_inputs,
                            append_input_ids=first_append_input_ids,
                            append_position_ids=first_append_position_ids,
                            reusable_prefix_length=reusable_prefix_length,
                            temporal_blocks=complete_blocks,
                            pixel_values_videos=first_pixel_values_videos,
                            video_grid_thw=first_video_grid_thw,
                        )
                    else:
                        delta_input_ids = full_base_input_ids[:, cached_prefix["base_length"] : reusable_prefix_length]
                        delta_position_ids = full_base_position_ids[
                            :,
                            :,
                            cached_prefix["base_length"] : reusable_prefix_length,
                        ]
                        previous_temporal_blocks = int(cached_prefix["temporal_blocks"])
                        if complete_blocks < previous_temporal_blocks:
                            raise ValueError("Aligned cache append found fewer complete temporal blocks than the cached prefix")
                        delta_temporal_blocks = complete_blocks - previous_temporal_blocks
                        delta_patch_start = previous_temporal_blocks * block_patch_count
                        delta_patch_end = complete_blocks * block_patch_count
                        append_input_segments = []
                        append_position_segments = []
                        append_pixel_segments = []
                        append_temporal_blocks = 0

                        if delta_input_ids.shape[1] > 0:
                            append_input_segments.append(delta_input_ids)
                            append_position_segments.append(delta_position_ids)
                        if delta_temporal_blocks > 0:
                            append_pixel_segments.append(full_prepared_inputs["pixel_values_videos"][delta_patch_start:delta_patch_end])
                            append_temporal_blocks += delta_temporal_blocks
                        if partial_input_ids is not None:
                            append_input_segments.append(partial_input_ids)
                            append_position_segments.append(partial_position_ids)
                            append_pixel_segments.append(partial_pixel_values_videos)
                            append_temporal_blocks += partial_temporal_blocks
                        append_input_segments.append(prepared_inputs["tail_input_ids"])
                        append_position_segments.append(prepared_inputs["tail_position_ids"])
                        combined_input_ids = torch.cat(append_input_segments, dim=1)
                        combined_position_ids = torch.cat(append_position_segments, dim=2)
                        delta_pixel_values_videos = torch.cat(append_pixel_segments, dim=0) if append_pixel_segments else None
                        delta_video_grid_thw = None
                        if delta_pixel_values_videos is not None:
                            delta_video_grid_thw = full_video_grid.new_tensor(
                                [[append_temporal_blocks, int(full_video_grid[1].item()), int(full_video_grid[2].item())]]
                            )
                        result, cached_prefix = self._compute_instruction_reward_with_cached_append(
                            cached_prefix=cached_prefix,
                            prepared_inputs=prepared_inputs,
                            append_input_ids=combined_input_ids,
                            append_position_ids=combined_position_ids,
                            reusable_prefix_length=reusable_prefix_length,
                            temporal_blocks=complete_blocks,
                            pixel_values_videos=delta_pixel_values_videos,
                            video_grid_thw=delta_video_grid_thw,
                        )

                    rewards.append(result.reward)
                    false_rewards.append(result.false_reward if result.false_reward is not None else float("nan"))
                    token_counts.append(result.token_count)
                    logger.info(
                        f"Prefix {length:3d} raw frames -> {len(prefix_raw_indices)} aligned video frames: "
                        f"reward={result.reward:.4f} false={result.false_reward:.4f} ({result.token_count} tokens, cache=yes)"
                    )
            except (RuntimeError, ValueError) as exc:
                logger.warning(f"Aligned KV-cache reuse failed ({exc}); falling back to aligned full recompute for this example")
                torch.cuda.empty_cache()
                rewards.clear()
                false_rewards.clear()
                token_counts.clear()
                for length in prefix_lengths:
                    result = _compute_uncached_for_length(length)
                    rewards.append(result.reward)
                    false_rewards.append(result.false_reward if result.false_reward is not None else float("nan"))
                    token_counts.append(result.token_count)
                    prefix_raw_indices = [idx for idx in aligned_full_indices if idx < length] or [0]
                    logger.info(
                        f"Prefix {length:3d} raw frames -> {len(prefix_raw_indices)} aligned video frames: "
                        f"reward={result.reward:.4f} false={result.false_reward:.4f} ({result.token_count} tokens, cache=fallback)"
                    )

        normalized_rewards = self.normalize_rewards(rewards).tolist()

        # Full trajectory is the last prefix
        return InstructionRewardResult(
            reward=rewards[-1],
            reduction=reduction,
            token_count=token_counts[-1],
            prefix_lengths=list(prefix_lengths),
            prefix_rewards=rewards,
            normalized_prefix_rewards=normalized_rewards,
            false_reward=false_rewards[-1],
            prefix_false_rewards=false_rewards,
        )
