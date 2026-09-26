from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from coconut import CoconutForCausalLM, CurriculumStage, curriculum_stage_for_epoch


@dataclass
class SimCoTCausalLMOutput(CausalLMOutputWithPast):
    language_model_loss: torch.Tensor | None = None
    region_loss: torch.Tensor | None = None
    region_positive_loss: torch.Tensor | None = None
    region_negative_loss: torch.Tensor | None = None
    decoder_loss: torch.Tensor | None = None
    terminal_scores: torch.Tensor | None = None
    decoded_block_count: torch.Tensor | None = None


@dataclass
class SimCoTDataCollator:
    """Apply a step-wise curriculum and build auxiliary decoder targets."""

    pad_token_id: int
    latent_id: int
    think_start_ids: Sequence[int]
    think_end_ids: Sequence[int]
    eos_token_id: int
    c_thought: int = 2
    epochs_per_stage: int = 3
    max_latent_stage: int = 10
    start_stage: int = 1
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8
    align_first_latent: bool = True

    def __post_init__(self) -> None:
        self.think_start_ids = tuple(int(value) for value in self.think_start_ids)
        self.think_end_ids = tuple(int(value) for value in self.think_end_ids)
        if self.c_thought <= 0:
            raise ValueError("c_thought must be positive")
        if self.epochs_per_stage <= 0 or self.max_latent_stage <= 0:
            raise ValueError("SIM-CoT curriculum sizes must be positive")
        if self.start_stage <= 0:
            raise ValueError("SIM-CoT must start with at least one latent block")
        if not self.think_start_ids or not self.think_end_ids:
            raise ValueError("Reasoning boundary token sequences cannot be empty")
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> CurriculumStage:
        self.stage = curriculum_stage_for_epoch(
            epoch,
            start_stage=self.start_stage,
            epochs_per_stage=self.epochs_per_stage,
            num_latent_blocks=self.max_latent_stage,
        )
        return self.stage

    def _serialize(self, feature: dict[str, Any]) -> dict[str, Any]:
        question_ids = [int(value) for value in feature["question_ids"]]
        steps_ids = [
            [int(value) for value in step] for step in feature["steps_ids"]
        ]
        if not steps_ids:
            raise ValueError("SIM-CoT requires at least one explicit reasoning step")
        answer_ids = [int(value) for value in feature["answer_ids"]]
        block_count = min(self.stage.latent_blocks, len(steps_ids))
        sample_is_fully_latent = (
            self.stage.fully_implicit or block_count == len(steps_ids)
        )
        explicit_steps = [] if sample_is_fully_latent else steps_ids[block_count:]
        explicit_body = [token for step in explicit_steps for token in step]
        latent_body = [self.latent_id] * (block_count * self.c_thought)

        if self.stage.fully_implicit and len(steps_ids) > block_count:
            decoder_targets = [
                step + [self.eos_token_id] for step in steps_ids[: block_count - 1]
            ]
            remaining = [
                token for step in steps_ids[block_count - 1 :] for token in step
            ]
            decoder_targets.append(remaining + [self.eos_token_id])
        else:
            decoder_targets = [
                step + [self.eos_token_id] for step in steps_ids[:block_count]
            ]

        input_ids = (
            question_ids
            + list(self.think_start_ids)
            + latent_body
            + explicit_body
            + list(self.think_end_ids)
            + answer_ids
        )
        labels = (
            [self.label_pad_token_id] * len(question_ids)
            + list(self.think_start_ids)
            + [self.label_pad_token_id] * len(latent_body)
            + explicit_body
            + [self.label_pad_token_id] * len(self.think_end_ids)
            + answer_ids
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "position_ids": list(range(len(input_ids))),
            "decoder_targets": decoder_targets,
            "decoder_terminal_blocks": [False] * (block_count - 1)
            + [sample_is_fully_latent],
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty feature list")
        serialized = [self._serialize(feature) for feature in features]
        first_latent_positions = [
            feature["input_ids"].index(self.latent_id)
            for feature in serialized
            if self.latent_id in feature["input_ids"]
        ]
        aligned_first_latent = (
            max(first_latent_positions)
            if self.align_first_latent and first_latent_positions
            else None
        )
        left_padding = []
        for feature in serialized:
            if aligned_first_latent is None or self.latent_id not in feature["input_ids"]:
                left_padding.append(0)
            else:
                left_padding.append(
                    aligned_first_latent - feature["input_ids"].index(self.latent_id)
                )

        max_length = max(
            len(feature["input_ids"]) + pad
            for feature, pad in zip(serialized, left_padding, strict=True)
        )
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "position_ids": [],
        }
        for feature, left_pad in zip(serialized, left_padding, strict=True):
            right_pad = max_length - left_pad - len(feature["input_ids"])
            batch["input_ids"].append(
                [self.pad_token_id] * left_pad
                + feature["input_ids"]
                + [self.pad_token_id] * right_pad
            )
            batch["attention_mask"].append(
                [0] * left_pad + feature["attention_mask"] + [0] * right_pad
            )
            batch["labels"].append(
                [self.label_pad_token_id] * left_pad
                + feature["labels"]
                + [self.label_pad_token_id] * right_pad
            )
            batch["position_ids"].append(
                [0] * left_pad + feature["position_ids"] + [0] * right_pad
            )

        max_blocks = max(len(feature["decoder_targets"]) for feature in serialized)
        max_target_length = max(
            (
                len(target)
                for feature in serialized
                for target in feature["decoder_targets"]
            ),
            default=0,
        )
        decoder_target_ids = torch.full(
            (len(serialized), max_blocks, max_target_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        decoder_target_labels = torch.full(
            (len(serialized), max_blocks, max_target_length),
            self.label_pad_token_id,
            dtype=torch.long,
        )
        decoder_target_attention_mask = torch.zeros(
            (len(serialized), max_blocks, max_target_length), dtype=torch.long
        )
        decoder_block_mask = torch.zeros(
            (len(serialized), max_blocks), dtype=torch.bool
        )
        decoder_terminal_block_mask = torch.zeros(
            (len(serialized), max_blocks), dtype=torch.bool
        )
        for batch_index, feature in enumerate(serialized):
            for block_index, target in enumerate(feature["decoder_targets"]):
                target_length = len(target)
                decoder_target_ids[
                    batch_index, block_index, :target_length
                ] = torch.tensor(target, dtype=torch.long)
                decoder_target_labels[
                    batch_index, block_index, :target_length
                ] = torch.tensor(target, dtype=torch.long)
                decoder_target_attention_mask[
                    batch_index, block_index, :target_length
                ] = 1
                decoder_block_mask[batch_index, block_index] = True
                decoder_terminal_block_mask[batch_index, block_index] = feature[
                    "decoder_terminal_blocks"
                ][block_index]

        result = {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in batch.items()
        }
        result.update(
            {
                "decoder_target_ids": decoder_target_ids,
                "decoder_target_labels": decoder_target_labels,
                "decoder_target_attention_mask": decoder_target_attention_mask,
                "decoder_block_mask": decoder_block_mask,
                "decoder_terminal_block_mask": decoder_terminal_block_mask,
            }
        )
        return result


class SimCoTLayoutBucketSampler(Sampler[int]):
    """Build curriculum-aware batches with similar lengths and latent layouts."""

    required_columns = (
        "question_length",
        "step_lengths",
        "answer_length",
    )

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        seed: int,
        think_start_length: int,
        think_end_length: int,
        c_thought: int,
        epochs_per_stage: int,
        max_latent_stage: int,
        start_stage: int,
        length_bucket_width: int = 32,
        first_latent_bucket_width: int = 16,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if length_bucket_width <= 0 or first_latent_bucket_width <= 0:
            raise ValueError("Bucket widths must be positive")
        missing = [
            name
            for name in self.required_columns
            if name not in getattr(dataset, "column_names", ())
        ]
        if missing:
            raise ValueError(f"SIM-CoT layout metadata is missing: {missing}")

        self.batch_size = batch_size
        self.seed = seed
        self.think_start_length = think_start_length
        self.think_end_length = think_end_length
        self.c_thought = c_thought
        self.epochs_per_stage = epochs_per_stage
        self.max_latent_stage = max_latent_stage
        self.start_stage = start_stage
        self.length_bucket_width = length_bucket_width
        self.first_latent_bucket_width = first_latent_bucket_width
        self.shuffle = shuffle
        self.epoch = 0
        self.question_lengths = [int(value) for value in dataset["question_length"]]
        self.step_lengths = [
            tuple(int(value) for value in lengths) for lengths in dataset["step_lengths"]
        ]
        self.answer_lengths = [int(value) for value in dataset["answer_length"]]
        if not (
            len(self.question_lengths)
            == len(self.step_lengths)
            == len(self.answer_lengths)
        ):
            raise ValueError("SIM-CoT layout metadata columns have different lengths")

    def __len__(self) -> int:
        return len(self.question_lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _layouts(self) -> list[tuple[int, int, int]]:
        stage = curriculum_stage_for_epoch(
            self.epoch,
            start_stage=self.start_stage,
            epochs_per_stage=self.epochs_per_stage,
            num_latent_blocks=self.max_latent_stage,
        )
        layouts = []
        for question_length, step_lengths, answer_length in zip(
            self.question_lengths,
            self.step_lengths,
            self.answer_lengths,
            strict=True,
        ):
            block_count = min(stage.latent_blocks, len(step_lengths))
            sample_is_fully_latent = stage.fully_implicit or block_count == len(
                step_lengths
            )
            explicit_length = (
                0 if sample_is_fully_latent else sum(step_lengths[block_count:])
            )
            serialized_length = (
                question_length
                + self.think_start_length
                + block_count * self.c_thought
                + explicit_length
                + self.think_end_length
                + answer_length
            )
            first_latent_position = question_length + self.think_start_length
            layouts.append(
                (serialized_length, first_latent_position, block_count * self.c_thought)
            )
        return layouts

    def __iter__(self):
        layouts = self._layouts()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        tie_breakers = torch.randperm(len(self), generator=generator).tolist()
        tie_rank = {index: rank for rank, index in enumerate(tie_breakers)}

        def bucket_key(index: int) -> tuple[int, int, int]:
            serialized_length, first_latent_position, latent_count = layouts[index]
            return (
                serialized_length // self.length_bucket_width,
                first_latent_position // self.first_latent_bucket_width,
                latent_count,
            )

        def sort_key(index: int) -> tuple[int, int, int, int, int, int]:
            serialized_length, first_latent_position, _ = layouts[index]
            return (
                *bucket_key(index),
                serialized_length,
                first_latent_position,
                tie_rank[index],
            )

        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index in range(len(self)):
            buckets.setdefault(bucket_key(index), []).append(index)

        batches: list[list[int]] = []
        leftovers: list[int] = []
        for key in sorted(buckets):
            bucket = sorted(buckets[key], key=sort_key)
            full_length = len(bucket) - len(bucket) % self.batch_size
            batches.extend(
                bucket[start : start + self.batch_size]
                for start in range(0, full_length, self.batch_size)
            )
            leftovers.extend(bucket[full_length:])
        leftovers.sort(key=sort_key)
        batches.extend(
            leftovers[start : start + self.batch_size]
            for start in range(0, len(leftovers), self.batch_size)
        )
        if self.shuffle:
            for batch_index, batch in enumerate(batches):
                order = torch.randperm(len(batch), generator=generator).tolist()
                batches[batch_index] = [batch[index] for index in order]
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in batch_order]
        return iter(index for batch in batches for index in batch)


class SimCoTForCausalLM(CoconutForCausalLM):
    """Continue a trained Coconut model with step-level auxiliary decoding."""

    _tied_weights_keys = [
        "base_causallm.lm_head.weight",
        "auxiliary_decoder.lm_head.weight",
    ]
    _keys_to_ignore_on_save = [
        "base_causallm.lm_head.weight",
        "auxiliary_decoder.lm_head.weight",
    ]
    _keys_to_ignore_on_load_missing = [
        r"base_causallm\.lm_head\.weight",
        r"auxiliary_decoder\.lm_head\.weight",
    ]

    def __init__(
        self,
        base_causallm: nn.Module,
        auxiliary_decoder: nn.Module | None,
        latent_token_id: int,
        think_end_token_id: int,
        think_region_center: torch.Tensor,
        think_region_cosine_threshold: float,
        nonterminal_region_cosine_threshold: float,
        region_loss_weight: float,
        region_negative_loss_weight: float = 1.0,
        region_positive_loss_type: str = "linear",
        decoder_loss_weight: float = 1.0,
        c_thought: int = 2,
        decoder_loss_normalization: str = "block",
    ) -> None:
        super().__init__(
            base_causallm=base_causallm,
            latent_token_id=latent_token_id,
            think_end_token_id=think_end_token_id,
            think_region_center=think_region_center,
            think_region_cosine_threshold=think_region_cosine_threshold,
            region_loss_weight=region_loss_weight,
        )
        if decoder_loss_weight < 0:
            raise ValueError("decoder_loss_weight must be non-negative")
        if region_negative_loss_weight < 0:
            raise ValueError("region_negative_loss_weight must be non-negative")
        if region_positive_loss_type not in {"linear", "squared"}:
            raise ValueError(
                "region_positive_loss_type must be 'linear' or 'squared'"
            )
        if not -1.0 <= nonterminal_region_cosine_threshold <= 1.0:
            raise ValueError(
                "nonterminal_region_cosine_threshold must lie in [-1, 1]"
            )
        if nonterminal_region_cosine_threshold >= think_region_cosine_threshold:
            raise ValueError(
                "The nonterminal threshold must be below the terminal threshold"
            )
        if c_thought <= 0:
            raise ValueError("c_thought must be positive")
        if decoder_loss_normalization not in {"block", "token"}:
            raise ValueError("decoder_loss_normalization must be 'block' or 'token'")
        if auxiliary_decoder is not None:
            if auxiliary_decoder.config.hidden_size != base_causallm.config.hidden_size:
                raise ValueError("Auxiliary decoder and Coconut hidden sizes differ")
            if auxiliary_decoder.config.vocab_size != base_causallm.config.vocab_size:
                raise ValueError("Auxiliary decoder and Coconut vocabulary sizes differ")

        self.auxiliary_decoder = auxiliary_decoder
        self.use_auxiliary_decoder = auxiliary_decoder is not None
        self.decoder_loss_weight = decoder_loss_weight
        self.region_negative_loss_weight = region_negative_loss_weight
        self.region_positive_loss_type = region_positive_loss_type
        self.nonterminal_region_cosine_threshold = (
            nonterminal_region_cosine_threshold
        )
        self.c_thought = c_thought
        self.decoder_loss_normalization = decoder_loss_normalization
        ignored_at_inference = set(
            getattr(self.config, "keys_to_ignore_at_inference", [])
        )
        ignored_at_inference.update(
            {
                "past_key_values",
                "language_model_loss",
                "region_loss",
                "region_positive_loss",
                "region_negative_loss",
                "decoder_loss",
                "terminal_scores",
                "decoded_block_count",
            }
        )
        self.config.keys_to_ignore_at_inference = sorted(ignored_at_inference)

    def block_region_loss(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_block_mask: torch.Tensor,
        decoder_terminal_block_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Keep nonterminal block ends outside q95 and the final end inside q90."""
        if (
            decoder_block_mask.ndim != 2
            or decoder_block_mask.shape[0] != input_ids.shape[0]
        ):
            raise ValueError("decoder_block_mask has an invalid shape")
        if decoder_terminal_block_mask.shape != decoder_block_mask.shape:
            raise ValueError("decoder_terminal_block_mask has an invalid shape")
        if bool((decoder_terminal_block_mask & ~decoder_block_mask).any()):
            raise ValueError("Only active decoder blocks can be terminal")

        latent_mask = input_ids.eq(self.latent_token_id) & attention_mask.bool()
        boundary_mask = input_ids.eq(self.think_end_token_id) & attention_mask.bool()
        if not bool(boundary_mask.sum(dim=1).eq(1).all()):
            raise ValueError(
                "Every sample must contain exactly one active </think> token"
            )

        zero = last_hidden_state.sum() * 0.0
        positive_losses: list[torch.Tensor] = []
        negative_losses: list[torch.Tensor] = []
        terminal_scores: list[torch.Tensor] = []
        for row in range(input_ids.shape[0]):
            block_count = int(decoder_block_mask[row].sum().item())
            expected_block_mask = torch.arange(
                decoder_block_mask.shape[1], device=decoder_block_mask.device
            ) < block_count
            if not torch.equal(decoder_block_mask[row], expected_block_mask):
                raise ValueError("Active decoder blocks must form a contiguous prefix")
            if block_count <= 0:
                raise ValueError(
                    "SIM-CoT region supervision requires at least one block"
                )

            latent_positions = latent_mask[row].nonzero(as_tuple=True)[0]
            if latent_positions.numel() != block_count * self.c_thought:
                raise ValueError("Latent positions do not match decoder block targets")
            block_end_positions = latent_positions[self.c_thought - 1 :: self.c_thought]
            boundary_position = int(boundary_mask[row].nonzero(as_tuple=True)[0].item())
            terminal_mask = decoder_terminal_block_mask[row, :block_count]
            terminal_count = int(terminal_mask.sum().item())
            if terminal_count > 1 or (
                terminal_count == 1 and not bool(terminal_mask[-1])
            ):
                raise ValueError("Only the final active block may be terminal")
            terminal_precedes_boundary = (
                int(block_end_positions[-1].item()) == boundary_position - 1
            )
            if terminal_count == 1 and not terminal_precedes_boundary:
                raise ValueError(
                    "A terminal latent block must immediately precede </think>"
                )

            block_scores = self.think_region_scores(
                last_hidden_state[row, block_end_positions]
            )
            final_score = block_scores[-1]
            terminal_scores.append(final_score)
            if terminal_count == 1:
                positive_violation = F.relu(
                    self.think_region_cosine_threshold - final_score
                )
                if self.region_positive_loss_type == "linear":
                    positive_losses.append(positive_violation)
                else:
                    positive_losses.append(positive_violation.square())
                nonterminal_scores = block_scores[:-1]
            else:
                nonterminal_scores = block_scores
            if nonterminal_scores.numel() > 0:
                negative_losses.append(
                    F.relu(
                        nonterminal_scores
                        - self.nonterminal_region_cosine_threshold
                    )
                    .square()
                    .mean()
                )

        positive_loss = (
            torch.stack(positive_losses).mean() if positive_losses else zero
        )
        negative_loss = (
            torch.stack(negative_losses).mean() if negative_losses else zero
        )
        region_loss = positive_loss + negative_loss
        return (
            region_loss,
            positive_loss,
            negative_loss,
            torch.stack(terminal_scores),
        )

    def auxiliary_step_loss(
        self,
        latent_inputs_by_sample: list[list[torch.Tensor]],
        decoder_target_ids: torch.Tensor,
        decoder_target_labels: torch.Tensor,
        decoder_target_attention_mask: torch.Tensor,
        decoder_block_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch_size = len(latent_inputs_by_sample)
        expected_shape = decoder_block_mask.shape
        if decoder_block_mask.ndim != 2 or expected_shape[0] != batch_size:
            raise ValueError("decoder_block_mask has an invalid shape")
        if (
            decoder_target_ids.shape != decoder_target_labels.shape
            or decoder_target_ids.shape != decoder_target_attention_mask.shape
            or decoder_target_ids.ndim != 3
            or decoder_target_ids.shape[:2] != expected_shape
        ):
            raise ValueError("Decoder target tensors have incompatible shapes")

        sequences: list[torch.Tensor] = []
        sequence_labels: list[torch.Tensor] = []
        for batch_index, latent_inputs in enumerate(latent_inputs_by_sample):
            block_count = int(decoder_block_mask[batch_index].sum().item())
            expected_block_mask = torch.arange(
                decoder_block_mask.shape[1], device=decoder_block_mask.device
            ) < block_count
            if not torch.equal(decoder_block_mask[batch_index], expected_block_mask):
                raise ValueError("Active decoder blocks must form a contiguous prefix")
            expected_latent_count = block_count * self.c_thought
            if len(latent_inputs) != expected_latent_count:
                raise ValueError(
                    f"Sample {batch_index} has {len(latent_inputs)} injected latent inputs; "
                    f"expected {expected_latent_count}"
                )
            for block_index in range(block_count):
                target_mask = decoder_target_attention_mask[
                    batch_index, block_index
                ].bool()
                if not bool(target_mask.any()):
                    raise ValueError("Every active decoder block needs a target")
                target_ids = decoder_target_ids[
                    batch_index, block_index, target_mask
                ]
                target_labels = decoder_target_labels[
                    batch_index, block_index, target_mask
                ]
                if bool(target_labels.eq(-100).any()):
                    raise ValueError("Active decoder targets cannot contain masked labels")

                start = block_index * self.c_thought
                latent_prefix = torch.stack(
                    latent_inputs[start : start + self.c_thought], dim=0
                )
                target_embeddings = self.get_input_embeddings()(target_ids)
                sequences.append(
                    torch.cat((latent_prefix, target_embeddings), dim=0)
                )
                prefix_labels = target_labels.new_full((self.c_thought,), -100)
                sequence_labels.append(
                    torch.cat((prefix_labels, target_labels), dim=0)
                )

        if not sequences:
            zero = self.get_input_embeddings().weight.sum() * 0.0
            return zero, 0

        padded_embeds = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        padded_labels = pad_sequence(
            sequence_labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.zeros(
            padded_labels.shape, dtype=torch.long, device=padded_labels.device
        )
        for row, sequence in enumerate(sequences):
            attention_mask[row, : sequence.shape[0]] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.auxiliary_decoder(
            inputs_embeds=padded_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = padded_labels[:, 1:].contiguous()
        loss_sum = CrossEntropyLoss(reduction="sum")(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        if self.decoder_loss_normalization == "token":
            denominator = int(shift_labels.ne(-100).sum().item())
        else:
            denominator = len(sequences)
        if denominator <= 0:
            raise ValueError("The auxiliary decoder has no supervised targets")
        return loss_sum / denominator, len(sequences)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        decoder_target_ids: torch.Tensor | None = None,
        decoder_target_labels: torch.Tensor | None = None,
        decoder_target_attention_mask: torch.Tensor | None = None,
        decoder_block_mask: torch.Tensor | None = None,
        decoder_terminal_block_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> SimCoTCausalLMOutput:
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        latent_mask = input_ids.eq(self.latent_token_id) & attention_mask.bool()
        if not bool(latent_mask.any()):
            if (
                labels is not None
                and decoder_block_mask is not None
                and bool(decoder_block_mask.any())
            ):
                raise ValueError("Decoder targets were provided without latent inputs")
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=labels is not None,
                return_dict=True,
            )
            loss = outputs.loss
            region_loss = None
            region_positive_loss = None
            region_negative_loss = None
            terminal_scores = None
            decoder_loss = None
            if labels is not None:
                region_loss, terminal_scores = self.terminal_region_loss(
                    outputs.hidden_states[-1], input_ids, attention_mask
                )
                region_positive_loss = region_loss
                region_negative_loss = region_loss * 0.0
                decoder_loss = outputs.hidden_states[-1].sum() * 0.0
                loss = loss + self.region_loss_weight * region_loss
            return SimCoTCausalLMOutput(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                language_model_loss=outputs.loss,
                region_loss=region_loss,
                region_positive_loss=region_positive_loss,
                region_negative_loss=region_negative_loss,
                decoder_loss=decoder_loss,
                terminal_scores=terminal_scores,
                decoded_block_count=input_ids.new_tensor(0),
            )

        if labels is not None and (
            decoder_block_mask is None or decoder_terminal_block_mask is None
        ):
            raise ValueError("SIM-CoT training requires region block masks")
        if labels is not None and self.use_auxiliary_decoder:
            required_decoder_tensors = (
                decoder_target_ids,
                decoder_target_labels,
                decoder_target_attention_mask,
            )
            if any(value is None for value in required_decoder_tensors):
                raise ValueError("SIM-CoT training requires all decoder target tensors")

        embedding = self.get_input_embeddings()
        token_embeddings = embedding(input_ids)
        sequence_length = input_ids.shape[1]
        split_positions = latent_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()
        cursor = 0
        past_key_values = None
        previous_hidden = None
        loss_sum = None
        valid_label_count = 0
        last_outputs = None
        hidden_chunks: list[torch.Tensor] = []
        latent_inputs_by_sample: list[list[torch.Tensor]] = [
            [] for _ in range(input_ids.shape[0])
        ]

        def run_chunk(start: int, end: int, chunk_embeds: torch.Tensor) -> None:
            nonlocal past_key_values, previous_hidden, loss_sum
            nonlocal valid_label_count, last_outputs
            outputs = self.base_causallm(
                inputs_embeds=chunk_embeds,
                attention_mask=attention_mask[:, :end],
                position_ids=position_ids[:, start:end],
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            if not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            previous_hidden = outputs.hidden_states[-1][:, -1:, :]
            hidden_chunks.append(outputs.hidden_states[-1])
            last_outputs = outputs

            if labels is None or start >= sequence_length - 1:
                return
            target_count = min(end, sequence_length - 1) - start
            if target_count <= 0:
                return
            target_labels = labels[:, start + 1 : start + 1 + target_count]
            chunk_logits = outputs.logits[:, :target_count, :]
            valid_count = int(target_labels.ne(-100).sum().item())
            if valid_count == 0:
                return
            chunk_loss = CrossEntropyLoss(reduction="sum")(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),
                target_labels.reshape(-1),
            )
            loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
            valid_label_count += valid_count

        for latent_position in split_positions:
            if latent_position > cursor:
                run_chunk(
                    cursor,
                    latent_position,
                    token_embeddings[:, cursor:latent_position],
                )
                cursor = latent_position
            if previous_hidden is None:
                raise ValueError("A latent token cannot be the first sequence position")
            current_embeds = token_embeddings[:, cursor : cursor + 1]
            row_mask = latent_mask[:, cursor].view(-1, 1, 1)
            current_embeds = torch.where(row_mask, previous_hidden, current_embeds)
            for row in row_mask[:, 0, 0].nonzero(as_tuple=True)[0].tolist():
                latent_inputs_by_sample[row].append(current_embeds[row, 0])
            run_chunk(cursor, cursor + 1, current_embeds)
            cursor += 1

        if cursor < sequence_length:
            run_chunk(cursor, sequence_length, token_embeddings[:, cursor:])
        if last_outputs is None:
            raise RuntimeError("SIM-CoT forward produced no model outputs")
        last_hidden_state = torch.cat(hidden_chunks, dim=1)
        if last_hidden_state.shape[1] != sequence_length:
            raise RuntimeError("SIM-CoT chunks did not cover the complete sequence")

        loss = None
        language_model_loss = None
        region_loss = None
        region_positive_loss = None
        region_negative_loss = None
        decoder_loss = None
        terminal_scores = None
        decoded_block_count = 0
        if labels is not None:
            if loss_sum is None or valid_label_count == 0:
                raise ValueError("The batch contains no supervised main-model targets")
            language_model_loss = loss_sum / valid_label_count
            assert decoder_block_mask is not None
            assert decoder_terminal_block_mask is not None
            (
                region_loss,
                region_positive_loss,
                region_negative_loss,
                terminal_scores,
            ) = self.block_region_loss(
                last_hidden_state,
                input_ids,
                attention_mask,
                decoder_block_mask,
                decoder_terminal_block_mask,
            )
            if self.use_auxiliary_decoder:
                assert decoder_target_ids is not None
                assert decoder_target_labels is not None
                assert decoder_target_attention_mask is not None
                decoder_loss, decoded_block_count = self.auxiliary_step_loss(
                    latent_inputs_by_sample,
                    decoder_target_ids,
                    decoder_target_labels,
                    decoder_target_attention_mask,
                    decoder_block_mask,
                )
            else:
                decoder_loss = language_model_loss * 0.0
                decoded_block_count = 0
            loss = (
                language_model_loss
                + self.region_loss_weight * region_positive_loss
                + self.region_negative_loss_weight * region_negative_loss
                + self.decoder_loss_weight * decoder_loss
            )

        return SimCoTCausalLMOutput(
            loss=loss,
            logits=last_outputs.logits,
            past_key_values=last_outputs.past_key_values,
            language_model_loss=language_model_loss,
            region_loss=region_loss,
            region_positive_loss=region_positive_loss,
            region_negative_loss=region_negative_loss,
            decoder_loss=decoder_loss,
            terminal_scores=terminal_scores,
            decoded_block_count=input_ids.new_tensor(decoded_block_count),
        )
