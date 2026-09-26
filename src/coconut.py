from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import Sampler
from safetensors import safe_open
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


LATENT_TOKEN = "<|latent|>"
COCONUT_SPECIAL_TOKENS = (LATENT_TOKEN,)


@dataclass
class CoconutCausalLMOutput(CausalLMOutputWithPast):
    language_model_loss: torch.Tensor | None = None
    region_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class ThinkRegion:
    center: torch.Tensor
    q90_cosine_threshold: float
    q95_cosine_threshold: float
    q90_angular_radius_degrees: float
    q95_angular_radius_degrees: float
    source_model: str
    think_end_token_id: int
    state_count: int


def load_think_region(path: Path) -> ThinkRegion:
    if not path.is_file():
        raise FileNotFoundError(
            f"Think-region artifact does not exist: {path}. Re-run "
            "Analysis/run_llama1b_think_geometry.sh to export it."
        )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"center"}:
            raise ValueError("Think-region artifact must contain exactly one 'center' tensor")
        center = handle.get_tensor("center").float().flatten()
        metadata = handle.metadata() or {}

    required_metadata = {
        "schema_version",
        "source_model",
        "think_end_token_id",
        "hidden_size",
        "state_count",
        "space",
        "target",
        "q90_angular_radius_degrees",
        "q95_angular_radius_degrees",
        "q90_cosine_threshold",
        "q95_cosine_threshold",
    }
    missing = sorted(required_metadata - metadata.keys())
    if missing:
        raise ValueError(f"Think-region artifact is missing metadata: {missing}")
    if metadata["schema_version"] != "1":
        raise ValueError(
            f"Unsupported think-region schema: {metadata['schema_version']}"
        )
    if metadata["space"] != "final_layer_hidden_after_per_vector_l2_normalization":
        raise ValueError(f"Unexpected think-region space: {metadata['space']}")
    if metadata["target"] != "predictor_hidden_whose_lm_head_argmax_is_think_end":
        raise ValueError(f"Unexpected think-region target: {metadata['target']}")
    hidden_size = int(metadata["hidden_size"])
    if center.numel() != hidden_size:
        raise ValueError(
            f"Think-region center has {center.numel()} values; expected {hidden_size}"
        )
    if not bool(torch.isfinite(center).all()):
        raise ValueError("Think-region center contains NaN or infinity")
    center_norm = center.norm()
    if not bool(center_norm > 0):
        raise ValueError("Think-region center has zero norm")
    center = center / center_norm

    q90_threshold = float(metadata["q90_cosine_threshold"])
    q95_threshold = float(metadata["q95_cosine_threshold"])
    if not -1.0 <= q95_threshold <= q90_threshold <= 1.0:
        raise ValueError("Think-region cosine thresholds are invalid or not nested")
    return ThinkRegion(
        center=center,
        q90_cosine_threshold=q90_threshold,
        q95_cosine_threshold=q95_threshold,
        q90_angular_radius_degrees=float(
            metadata["q90_angular_radius_degrees"]
        ),
        q95_angular_radius_degrees=float(
            metadata["q95_angular_radius_degrees"]
        ),
        source_model=metadata["source_model"],
        think_end_token_id=int(metadata["think_end_token_id"]),
        state_count=int(metadata["state_count"]),
    )


def add_coconut_special_tokens(tokenizer: Any) -> int:
    existing = list(tokenizer.additional_special_tokens)
    combined = existing + [
        token for token in COCONUT_SPECIAL_TOKENS if token not in existing
    ]
    return tokenizer.add_special_tokens(
        {"additional_special_tokens": combined}
    )


def initialize_coconut_token_embeddings(
    model: nn.Module,
    tokenizer: Any,
    *,
    previous_vocab_size: int,
    initialization_token: str = "<<",
) -> None:
    """Resize the model and initialize newly-added Coconut tokens identically."""
    source_ids = tokenizer.encode(initialization_token, add_special_tokens=False)
    if len(source_ids) != 1:
        raise ValueError(
            f"Initialization token {initialization_token!r} must map to one token; "
            f"got ids {source_ids}"
        )

    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    new_token_ids = tokenizer.convert_tokens_to_ids(list(COCONUT_SPECIAL_TOKENS))
    if any(token_id is None or token_id < 0 for token_id in new_token_ids):
        raise ValueError("Coconut special tokens were not registered in the tokenizer")

    source_id = source_ids[0]
    input_embeddings = model.get_input_embeddings().weight
    output_layer = model.get_output_embeddings()
    with torch.no_grad():
        for token_id in new_token_ids:
            if token_id < previous_vocab_size:
                continue
            input_embeddings[token_id].copy_(input_embeddings[source_id])
            if output_layer is not None and output_layer.weight.data_ptr() != input_embeddings.data_ptr():
                output_layer.weight[token_id].copy_(output_layer.weight[source_id])
    if hasattr(model, "tie_weights"):
        model.tie_weights()


@dataclass(frozen=True)
class CurriculumStage:
    epoch: int
    scheduled_stage: int
    latent_blocks: int
    fully_implicit: bool


def curriculum_stage_for_epoch(
    epoch: int,
    *,
    start_stage: int,
    epochs_per_stage: int,
    num_latent_blocks: int,
) -> CurriculumStage:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if start_stage < 0:
        raise ValueError("start_stage must be non-negative")
    if epochs_per_stage <= 0:
        raise ValueError("epochs_per_stage must be positive")
    if num_latent_blocks <= 0:
        raise ValueError("num_latent_blocks must be positive")

    scheduled_stage = start_stage + epoch // epochs_per_stage
    return CurriculumStage(
        epoch=epoch,
        scheduled_stage=scheduled_stage,
        latent_blocks=min(scheduled_stage, num_latent_blocks),
        fully_implicit=scheduled_stage > num_latent_blocks,
    )


def _validated_step_ranks(
    step_count: int, step_difficulty_ranks: Sequence[int]
) -> tuple[int, ...]:
    ranks = tuple(int(rank) for rank in step_difficulty_ranks)
    if not ranks:
        ranks = tuple(range(1, step_count + 1))
    if len(ranks) != step_count or sorted(ranks) != list(range(1, step_count + 1)):
        raise ValueError(
            "step_difficulty_ranks must be a permutation of 1..len(steps)"
        )
    return ranks


@dataclass(frozen=True)
class CoconutExampleLayout:
    serialized_length: int
    first_latent_position: int
    latent_count: int
    relative_latent_block_positions: tuple[int, ...]


def coconut_example_layout(
    *,
    question_length: int,
    step_lengths: Sequence[int],
    answer_length: int,
    step_difficulty_ranks: Sequence[int],
    think_start_length: int,
    think_end_length: int,
    c_thought: int,
    latent_blocks: int,
) -> CoconutExampleLayout:
    """Describe one example after applying a particular curriculum stage."""
    if min(question_length, answer_length, think_start_length, think_end_length) < 0:
        raise ValueError("Component lengths must be non-negative")
    if c_thought <= 0 or latent_blocks < 0:
        raise ValueError("c_thought must be positive and latent_blocks non-negative")
    normalized_step_lengths = tuple(int(length) for length in step_lengths)
    if any(length < 0 for length in normalized_step_lengths):
        raise ValueError("Step lengths must be non-negative")
    if step_difficulty_ranks:
        _validated_step_ranks(len(normalized_step_lengths), step_difficulty_ranks)

    cursor = question_length + think_start_length
    latent_block_positions: list[int] = []
    for step_index, step_length in enumerate(normalized_step_lengths):
        if step_index < latent_blocks:
            latent_block_positions.append(cursor)
            cursor += c_thought
        else:
            cursor += step_length
    serialized_length = cursor + think_end_length + answer_length
    if latent_block_positions:
        first_latent_position = latent_block_positions[0]
        relative_positions = tuple(
            position - first_latent_position for position in latent_block_positions
        )
    else:
        first_latent_position = -1
        relative_positions = ()
    return CoconutExampleLayout(
        serialized_length=serialized_length,
        first_latent_position=first_latent_position,
        latent_count=len(latent_block_positions) * c_thought,
        relative_latent_block_positions=relative_positions,
    )


class CoconutLayoutBucketSampler(Sampler[int]):
    """Shuffle batches within curriculum-aware length and latent-layout buckets."""

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
        num_latent_blocks: int,
        start_stage: int,
        length_bucket_width: int = 32,
        first_latent_bucket_width: int = 16,
        latent_layout_bucket_width: int = 16,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if min(
            length_bucket_width,
            first_latent_bucket_width,
            latent_layout_bucket_width,
        ) <= 0:
            raise ValueError("Bucket widths must be positive")
        missing = [
            name
            for name in self.required_columns
            if name not in getattr(dataset, "column_names", ())
        ]
        if missing:
            raise ValueError(f"Coconut layout metadata is missing: {missing}")

        self.batch_size = batch_size
        self.seed = seed
        self.think_start_length = think_start_length
        self.think_end_length = think_end_length
        self.c_thought = c_thought
        self.epochs_per_stage = epochs_per_stage
        self.num_latent_blocks = num_latent_blocks
        self.start_stage = start_stage
        self.length_bucket_width = length_bucket_width
        self.first_latent_bucket_width = first_latent_bucket_width
        self.latent_layout_bucket_width = latent_layout_bucket_width
        self.shuffle = shuffle
        self.epoch = 0
        self.question_lengths = [int(value) for value in dataset["question_length"]]
        self.step_lengths = [
            tuple(int(value) for value in lengths)
            for lengths in dataset["step_lengths"]
        ]
        self.answer_lengths = [int(value) for value in dataset["answer_length"]]
        if "step_difficulty_ranks" in getattr(dataset, "column_names", ()):
            self.step_difficulty_ranks = [
                _validated_step_ranks(len(lengths), ranks)
                for lengths, ranks in zip(
                    self.step_lengths,
                    dataset["step_difficulty_ranks"],
                    strict=True,
                )
            ]
        else:
            self.step_difficulty_ranks = [() for _ in self.step_lengths]
        row_count = len(self.question_lengths)
        if not (
            len(self.step_lengths)
            == len(self.answer_lengths)
            == len(self.step_difficulty_ranks)
            == row_count
        ):
            raise ValueError("Coconut layout metadata columns have different lengths")

    def __len__(self) -> int:
        return len(self.question_lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _layouts(self) -> list[CoconutExampleLayout]:
        stage = curriculum_stage_for_epoch(
            self.epoch,
            start_stage=self.start_stage,
            epochs_per_stage=self.epochs_per_stage,
            num_latent_blocks=self.num_latent_blocks,
        )
        return [
            coconut_example_layout(
                question_length=question_length,
                step_lengths=step_lengths,
                answer_length=answer_length,
                step_difficulty_ranks=ranks,
                think_start_length=self.think_start_length,
                think_end_length=self.think_end_length,
                c_thought=self.c_thought,
                latent_blocks=stage.latent_blocks,
            )
            for question_length, step_lengths, answer_length, ranks in zip(
                self.question_lengths,
                self.step_lengths,
                self.answer_lengths,
                self.step_difficulty_ranks,
                strict=True,
            )
        ]

    def __iter__(self):
        layouts = self._layouts()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            source_order = torch.randperm(len(self), generator=generator).tolist()
        else:
            source_order = list(range(len(self)))

        def bucket_key(index: int) -> tuple[int, int, int, tuple[int, ...]]:
            layout = layouts[index]
            first_latent_bucket = (
                layout.first_latent_position // self.first_latent_bucket_width
                if layout.first_latent_position >= 0
                else -1
            )
            relative_layout = tuple(
                position // self.latent_layout_bucket_width
                for position in layout.relative_latent_block_positions
            )
            return (
                layout.serialized_length // self.length_bucket_width,
                first_latent_bucket,
                layout.latent_count,
                relative_layout,
            )

        def sort_key(index: int) -> tuple[Any, ...]:
            layout = layouts[index]
            return (
                *bucket_key(index),
                layout.serialized_length,
                layout.first_latent_position,
                layout.relative_latent_block_positions,
            )

        buckets: dict[tuple[int, int, int, tuple[int, ...]], list[int]] = {}
        for index in source_order:
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


@dataclass
class CoconutDataCollator:
    pad_token_id: int
    latent_id: int
    think_start_ids: Sequence[int]
    think_end_ids: Sequence[int]
    c_thought: int = 2
    epochs_per_stage: int = 3
    num_latent_blocks: int = 10
    start_stage: int = 1
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8
    align_first_latent: bool = True

    def __post_init__(self) -> None:
        if self.c_thought <= 0:
            raise ValueError("c_thought must be positive")
        self.think_start_ids = tuple(self.think_start_ids)
        self.think_end_ids = tuple(self.think_end_ids)
        if not self.think_start_ids or not self.think_end_ids:
            raise ValueError("Reasoning boundary token sequences cannot be empty")
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> CurriculumStage:
        self.stage = curriculum_stage_for_epoch(
            epoch,
            start_stage=self.start_stage,
            epochs_per_stage=self.epochs_per_stage,
            num_latent_blocks=self.num_latent_blocks,
        )
        return self.stage

    def _serialize(self, feature: dict[str, Any]) -> dict[str, list[int]]:
        question_ids = list(feature["question_ids"])
        steps_ids = [list(step) for step in feature["steps_ids"]]
        answer_ids = list(feature["answer_ids"])

        body_ids: list[int] = []
        body_labels: list[int] = []
        for step_index, step_ids in enumerate(steps_ids):
            if step_index < self.stage.latent_blocks:
                block = [self.latent_id] * self.c_thought
                body_ids.extend(block)
                body_labels.extend([self.label_pad_token_id] * len(block))
            else:
                body_ids.extend(step_ids)
                body_labels.extend(step_ids)

        input_ids = (
            question_ids
            + list(self.think_start_ids)
            + body_ids
            + list(self.think_end_ids)
            + answer_ids
        )
        labels = (
            [self.label_pad_token_id] * len(question_ids)
            + list(self.think_start_ids)
            + body_labels
            + [self.label_pad_token_id] * len(self.think_end_ids)
            + answer_ids
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "position_ids": list(range(len(input_ids))),
        }

    def __call__(
        self, features: Sequence[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty feature list")
        serialized = [self._serialize(dict(feature)) for feature in features]
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
        left_padding: list[int] = []
        for feature in serialized:
            if aligned_first_latent is None or self.latent_id not in feature["input_ids"]:
                left_padding.append(0)
            else:
                left_padding.append(
                    aligned_first_latent - feature["input_ids"].index(self.latent_id)
                )

        max_length = max(
            len(feature["input_ids"]) + left_pad
            for feature, left_pad in zip(serialized, left_padding, strict=True)
        )
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = math.ceil(max_length / multiple) * multiple

        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "position_ids": [],
        }
        for feature, left_pad in zip(serialized, left_padding, strict=True):
            right_padding = max_length - left_pad - len(feature["input_ids"])
            batch["input_ids"].append(
                [self.pad_token_id] * left_pad
                + feature["input_ids"]
                + [self.pad_token_id] * right_padding
            )
            batch["attention_mask"].append(
                [0] * left_pad
                + feature["attention_mask"]
                + [0] * right_padding
            )
            batch["labels"].append(
                [self.label_pad_token_id] * left_pad
                + feature["labels"]
                + [self.label_pad_token_id] * right_padding
            )
            batch["position_ids"].append(
                [0] * left_pad
                + feature["position_ids"]
                + [0] * right_padding
            )

        return {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in batch.items()
        }


class CoconutForCausalLM(PreTrainedModel):
    """Coconut wrapper that feeds each last hidden state into the next slot."""

    base_model_prefix = "base_causallm"
    _tied_weights_keys = ["base_causallm.lm_head.weight"]
    _keys_to_ignore_on_save = ["base_causallm.lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"base_causallm\.lm_head\.weight"]

    def __init__(
        self,
        base_causallm: nn.Module,
        latent_token_id: int,
        think_end_token_id: int,
        think_region_center: torch.Tensor,
        think_region_cosine_threshold: float,
        region_loss_weight: float,
    ) -> None:
        super().__init__(base_causallm.config)
        center = think_region_center.float().flatten()
        if center.numel() != base_causallm.config.hidden_size:
            raise ValueError("Think-region center does not match the model hidden size")
        if not bool(torch.isfinite(center).all()) or not bool(center.norm() > 0):
            raise ValueError("Think-region center must be finite and nonzero")
        if not -1.0 <= think_region_cosine_threshold <= 1.0:
            raise ValueError("Think-region cosine threshold must lie in [-1, 1]")
        if region_loss_weight < 0:
            raise ValueError("region_loss_weight must be non-negative")
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.think_end_token_id = think_end_token_id
        self.think_region_cosine_threshold = think_region_cosine_threshold
        self.region_loss_weight = region_loss_weight
        self.register_buffer(
            "think_region_center", F.normalize(center, dim=0), persistent=True
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.base_causallm.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.base_causallm.get_output_embeddings()

    def think_region_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        unit_states = F.normalize(hidden_states.float(), dim=-1)
        return unit_states @ self.think_region_center.float()

    def terminal_region_loss(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundary_mask = input_ids.eq(self.think_end_token_id) & attention_mask.bool()
        boundary_counts = boundary_mask.sum(dim=1)
        if not bool(boundary_counts.eq(1).all()):
            raise ValueError("Every sample must contain exactly one active </think> token")
        boundary_positions = boundary_mask.long().argmax(dim=1)
        terminal_positions = boundary_positions - 1
        if not bool(terminal_positions.ge(0).all()):
            raise ValueError("</think> cannot be the first active token")
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        terminal_hidden = last_hidden_state[rows, terminal_positions]
        scores = self.think_region_scores(terminal_hidden)
        violations = F.relu(self.think_region_cosine_threshold - scores)
        return violations.square().mean(), scores

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> CoconutCausalLMOutput:
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        latent_mask = input_ids.eq(self.latent_token_id)
        if not bool(latent_mask.any()):
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=labels is not None,
                return_dict=True,
            )
            language_model_loss = outputs.loss
            region_loss = None
            loss = language_model_loss
            if labels is not None:
                region_loss, _ = self.terminal_region_loss(
                    outputs.hidden_states[-1], input_ids, attention_mask
                )
                loss = loss + self.region_loss_weight * region_loss
            return CoconutCausalLMOutput(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                language_model_loss=language_model_loss,
                region_loss=region_loss,
            )

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
            run_chunk(cursor, cursor + 1, current_embeds)
            cursor += 1

        if cursor < sequence_length:
            run_chunk(cursor, sequence_length, token_embeddings[:, cursor:])
        if last_outputs is None:
            raise RuntimeError("Coconut forward produced no model outputs")
        last_hidden_state = torch.cat(hidden_chunks, dim=1)
        if last_hidden_state.shape[1] != sequence_length:
            raise RuntimeError("Coconut chunks did not cover the complete sequence")

        loss = None
        language_model_loss = None
        region_loss = None
        if labels is not None:
            if loss_sum is None or valid_label_count == 0:
                raise ValueError("The batch contains no supervised target tokens")
            language_model_loss = loss_sum / valid_label_count
            region_loss, _ = self.terminal_region_loss(
                last_hidden_state, input_ids, attention_mask
            )
            loss = language_model_loss + self.region_loss_weight * region_loss

        return CoconutCausalLMOutput(
            loss=loss,
            logits=last_outputs.logits,
            past_key_values=last_outputs.past_key_values,
            language_model_loss=language_model_loss,
            region_loss=region_loss,
        )
