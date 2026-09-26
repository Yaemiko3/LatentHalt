#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from coconut import (
    COCONUT_SPECIAL_TOKENS,
    LATENT_TOKEN,
    CoconutDataCollator,
    CoconutForCausalLM,
    CoconutLayoutBucketSampler,
    ThinkRegion,
    add_coconut_special_tokens,
    initialize_coconut_token_embeddings,
    load_think_region,
)
from data import THINK_END_TOKEN, THINK_START_TOKEN, normalize_sft_example
from experiment_log import (
    ExperimentTracker,
    collect_accelerator_memory,
    local_accelerator_memory,
)
from train_sft_cot import configure_pad_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "datasets" / "gsm8k-aug" / "data"
_ACTIVE_TRACKER: ExperimentTracker | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coconut curriculum training from the local explicit-CoT SFT model."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sft-cot-llama1b",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=DEFAULT_DATA_ROOT / "train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--validation_file",
        type=Path,
        default=DEFAULT_DATA_ROOT / "validation-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "coconut-llama1b",
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument("--log_dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument(
        "--think_region_file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "analysis"
            / "think_hidden_geometry"
            / "think_region.safetensors"
        ),
    )
    parser.add_argument(
        "--platform_type",
        choices=["internal_cluster", "local_server", "cloud"],
        default=os.environ.get("RUN_PLATFORM_TYPE", "local_server"),
    )
    parser.add_argument(
        "--platform_name", default=os.environ.get("RUN_PLATFORM_NAME", "local")
    )
    parser.add_argument(
        "--formal_experiment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--c_thought", type=int, default=2)
    parser.add_argument("--region_loss_weight", type=float, default=1.0)
    parser.add_argument("--epochs_per_stage", type=int, default=3)
    parser.add_argument(
        "--num_latent_blocks",
        type=int,
        default=10,
        help="Final latent-block budget; the scored training data currently needs 10.",
    )
    parser.add_argument(
        "--curriculum_start_stage",
        type=int,
        default=1,
        help="Start at one latent block because the input model is already CoT-SFT.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument(
        "--align_first_latent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align each batch's first latent using masked left padding.",
    )
    parser.add_argument(
        "--bucket_by_layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group each curriculum epoch by serialized length and latent layout.",
    )
    parser.add_argument("--length_bucket_width", type=int, default=32)
    parser.add_argument("--first_latent_bucket_width", type=int, default=16)
    parser.add_argument("--latent_layout_bucket_width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="Checkpoint directory, 'auto' to use the latest checkpoint, or 'none'.",
    )
    parser.add_argument("--report_to", default="tensorboard", choices=["none", "tensorboard"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--preview_samples", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument(
        "--require_step_scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate step_difficulty_ranks if you explicitly want the scored split contract.",
    )
    parser.add_argument(
        "--reset_optimizer_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear AdamW moments each epoch, matching the released Coconut config.",
    )
    parser.add_argument(
        "--save_final_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--select_best_full_latent_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the lowest-eval-loss model from fully latent epochs only.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local_files_only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _validate_step_ranks(steps: list[str], ranks: Any, require_scores: bool) -> list[int]:
    if ranks is None:
        ranks = []
    normalized = [int(rank) for rank in ranks]
    expected = list(range(1, len(steps) + 1))
    if require_scores and not normalized:
        raise ValueError(
            "step_difficulty_ranks are required for Coconut training and validation"
        )
    if normalized and (len(normalized) != len(steps) or sorted(normalized) != expected):
        raise ValueError(
            "step_difficulty_ranks must be a permutation of 1..len(steps)"
        )
    return normalized


def _maximum_serialized_length(
    question_ids: list[int],
    steps_ids: list[list[int]],
    answer_ids: list[int],
    step_difficulty_ranks: list[int],
    *,
    c_thought: int,
    num_latent_blocks: int,
    reasoning_boundary_length: int,
) -> int:
    if step_difficulty_ranks:
        _validate_step_ranks(
            [""] * len(steps_ids), step_difficulty_ranks, require_scores=False
        )
    lengths: list[int] = []
    for latent_blocks in range(num_latent_blocks + 1):
        body_length = 0
        for step_index, step_ids in enumerate(steps_ids):
            if step_index < latent_blocks:
                body_length += c_thought
            else:
                body_length += len(step_ids)
        lengths.append(
            len(question_ids)
            + reasoning_boundary_length
            + body_length
            + len(answer_ids)
        )
    return max(lengths)


def tokenize_coconut_batch(
    batch: Mapping[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int,
    c_thought: int,
    num_latent_blocks: int,
    require_step_scores: bool,
) -> dict[str, list[Any]]:
    encoded: dict[str, list[Any]] = {
        "question_ids": [],
        "steps_ids": [],
        "answer_ids": [],
        "step_difficulty_ranks": [],
        "question_length": [],
        "step_lengths": [],
        "answer_length": [],
    }
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id")
    think_start_ids = tokenizer.encode(
        THINK_START_TOKEN + "\n", add_special_tokens=False
    )
    think_end_ids = tokenizer.encode(
        THINK_END_TOKEN + "\n", add_special_tokens=False
    )
    score_rows = batch.get("step_difficulty_ranks")

    for row_index, (question, raw_steps, answer) in enumerate(
        zip(batch["question"], batch["steps"], batch["answer"], strict=True)
    ):
        question, steps, answer = normalize_sft_example(
            {"question": question, "steps": raw_steps, "answer": answer}
        )
        # The former scored parquet excluded the ten raw GSM8K-Aug examples
        # whose reasoning chains exceed the configured ten-block budget.
        if len(steps) > num_latent_blocks:
            continue
        ranks = _validate_step_ranks(
            steps,
            None if score_rows is None else score_rows[row_index],
            require_step_scores,
        )
        question_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
        steps_ids = [
            tokenizer.encode(step + "\n", add_special_tokens=False) for step in steps
        ]
        answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [eos_token_id]
        if (
            _maximum_serialized_length(
                question_ids,
                steps_ids,
                answer_ids,
                ranks,
                c_thought=c_thought,
                num_latent_blocks=num_latent_blocks,
                reasoning_boundary_length=len(think_start_ids) + len(think_end_ids),
            )
            > max_length
        ):
            continue

        encoded["question_ids"].append(question_ids)
        encoded["steps_ids"].append(steps_ids)
        encoded["answer_ids"].append(answer_ids)
        encoded["step_difficulty_ranks"].append(ranks)
        encoded["question_length"].append(len(question_ids))
        encoded["step_lengths"].append([len(step_ids) for step_ids in steps_ids])
        encoded["answer_length"].append(len(answer_ids))
    return encoded


def prepare_dataset(
    dataset: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    max_samples: int | None,
    require_step_scores: bool,
) -> tuple[Any, int]:
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    input_count = len(dataset)
    tokenized = dataset.map(
        tokenize_coconut_batch,
        batched=True,
        batch_size=1_000,
        num_proc=max(1, args.preprocessing_num_workers),
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
            "c_thought": args.c_thought,
            "num_latent_blocks": args.num_latent_blocks,
            "require_step_scores": require_step_scores,
        },
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing Coconut components",
    )
    if len(tokenized) == 0:
        raise ValueError(f"All {input_count} samples exceeded max_length={args.max_length}")
    return tokenized, input_count - len(tokenized)


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    requested = args.resume_from_checkpoint.strip()
    if requested.lower() in {"", "none", "false"}:
        return None
    if requested.lower() != "auto":
        checkpoint = Path(requested).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
        return str(checkpoint)
    if not args.output_dir.is_dir():
        return None
    return get_last_checkpoint(str(args.output_dir))


class CoconutCurriculumCallback(TrainerCallback):
    def __init__(
        self,
        collator: CoconutDataCollator,
        *,
        reset_optimizer_each_epoch: bool,
    ) -> None:
        self.collator = collator
        self.reset_optimizer_each_epoch = reset_optimizer_each_epoch
        self.last_epoch: int | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        checkpoint_save_steps = state.save_steps
        state.save_steps = args.save_steps
        checkpoint_train_batch_size = state.train_batch_size
        state.train_batch_size = args.train_batch_size
        if (
            state.is_world_process_zero
            and checkpoint_save_steps != state.save_steps
        ):
            print(
                "Coconut save interval overridden by current command: "
                f"checkpoint={checkpoint_save_steps}, current={state.save_steps}"
            )
        if (
            state.is_world_process_zero
            and checkpoint_train_batch_size != state.train_batch_size
        ):
            print(
                "Coconut train batch size overridden by current command: "
                f"checkpoint={checkpoint_train_batch_size}, "
                f"current={state.train_batch_size}"
            )
        return control

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        train_dataloader = kwargs.get("train_dataloader")
        if train_dataloader is None:
            epoch = int(math.floor(state.epoch or 0.0))
        else:
            updates_per_epoch = max(
                len(train_dataloader) // args.gradient_accumulation_steps,
                1,
            )
            epoch = state.global_step // updates_per_epoch
        stage = self.collator.set_epoch(epoch)
        if (
            self.reset_optimizer_each_epoch
            and epoch > 0
            and epoch != self.last_epoch
            and kwargs.get("optimizer") is not None
        ):
            kwargs["optimizer"].state.clear()
        self.last_epoch = epoch
        if state.is_world_process_zero:
            mode = "fully implicit" if stage.fully_implicit else "hybrid"
            print(
                f"Coconut epoch {epoch}: scheduled_stage={stage.scheduled_stage}, "
                f"latent_blocks={stage.latent_blocks}, mode={mode}"
            )
        return control


class CoconutTrainer(Trainer):
    """Use layout buckets and add component losses to Trainer logging."""

    def __init__(
        self,
        *args: Any,
        bucket_by_layout: bool = True,
        length_bucket_width: int = 32,
        first_latent_bucket_width: int = 16,
        latent_layout_bucket_width: int = 16,
        select_best_full_latent_model: bool = False,
        **kwargs: Any,
    ) -> None:
        self.bucket_by_layout = bucket_by_layout
        self.length_bucket_width = length_bucket_width
        self.first_latent_bucket_width = first_latent_bucket_width
        self.latent_layout_bucket_width = latent_layout_bucket_width
        self.select_best_full_latent_model = select_best_full_latent_model
        super().__init__(*args, **kwargs)
        self._component_sums = {
            "language_model_loss": 0.0,
            "region_loss": 0.0,
        }
        self._component_count = 0
        self._best_full_latent = self._load_best_full_latent_state()

    @property
    def _best_full_latent_root(self) -> Path:
        return Path(self.args.output_dir) / "best-full-latent"

    def _load_best_full_latent_state(self) -> dict[str, Any] | None:
        state_path = self._best_full_latent_root / "state.json"
        if not state_path.is_file():
            return None
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        checkpoint = Path(state["checkpoint"])
        if checkpoint.parent != self._best_full_latent_root or not checkpoint.is_dir():
            raise ValueError(f"Invalid best full-latent checkpoint: {checkpoint}")
        if state.get("metric_name") != "eval_loss":
            raise ValueError(f"Invalid best full-latent metric in {state_path}")
        return state

    def _maybe_save_best_full_latent(self, metrics: Mapping[str, float]) -> bool:
        if not self.select_best_full_latent_model:
            return False
        collator = self.data_collator
        if not isinstance(collator, CoconutDataCollator):
            raise TypeError("Full-latent model selection requires CoconutDataCollator")
        if collator.stage.latent_blocks < collator.num_latent_blocks:
            return False

        metric = float(metrics["eval_loss"])
        if not math.isfinite(metric):
            raise ValueError(f"Non-finite full-latent eval_loss: {metric}")
        if (
            self._best_full_latent is not None
            and metric >= float(self._best_full_latent["metric_value"])
        ):
            return False

        checkpoint = self._best_full_latent_root / f"checkpoint-{self.state.global_step}"
        self.save_model(str(checkpoint), _internal_call=True)
        self.accelerator.wait_for_everyone()

        previous_checkpoint = (
            Path(self._best_full_latent["checkpoint"])
            if self._best_full_latent is not None
            else None
        )
        candidate = {
            "checkpoint": str(checkpoint),
            "epoch": collator.stage.epoch + 1,
            "global_step": self.state.global_step,
            "metric_name": "eval_loss",
            "metric_value": metric,
        }
        if self.is_world_process_zero():
            self._best_full_latent_root.mkdir(parents=True, exist_ok=True)
            state_path = self._best_full_latent_root / "state.json"
            temporary = state_path.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(candidate, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, state_path)
            if previous_checkpoint is not None and previous_checkpoint != checkpoint:
                shutil.rmtree(previous_checkpoint)
            print(
                "New best fully latent model: "
                f"epoch={candidate['epoch']}, step={self.state.global_step}, "
                f"eval_loss={metric:.8f}"
            )
        self.accelerator.wait_for_everyone()
        self._best_full_latent = candidate
        self.state.best_metric = metric
        self.state.best_model_checkpoint = str(checkpoint)
        return True

    def load_best_full_latent_model(self) -> dict[str, Any]:
        if self._best_full_latent is None:
            raise RuntimeError(
                "No fully latent model was evaluated; refusing to export an earlier stage"
            )
        self.state.best_metric = float(self._best_full_latent["metric_value"])
        self.state.best_model_checkpoint = self._best_full_latent["checkpoint"]
        self._load_best_model()
        self.select_best_full_latent_model = False
        if self.is_world_process_zero():
            print(
                "Loaded best fully latent model for final export: "
                f"epoch={self._best_full_latent['epoch']}, "
                f"step={self._best_full_latent['global_step']}, "
                f"eval_loss={self._best_full_latent['metric_value']:.8f}"
            )
        return dict(self._best_full_latent)

    def _inner_training_loop(
        self,
        batch_size: int | None = None,
        args: TrainingArguments | None = None,
        resume_from_checkpoint: str | None = None,
        trial: Any = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> Any:
        current_batch_size = self.args.train_batch_size
        if batch_size != current_batch_size and self.args.should_log:
            print(
                "Coconut runtime batch size overridden by current command: "
                f"checkpoint={batch_size}, current={current_batch_size}"
            )
        return super()._inner_training_loop(
            batch_size=current_batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _get_train_sampler(self) -> Any:
        if not self.bucket_by_layout:
            return super()._get_train_sampler()
        if self.train_dataset is None:
            return None
        collator = self.data_collator
        if not isinstance(collator, CoconutDataCollator):
            raise TypeError(
                "Coconut layout bucketing requires CoconutDataCollator"
            )
        return CoconutLayoutBucketSampler(
            self.train_dataset,
            batch_size=self._train_batch_size,
            seed=self.args.data_seed,
            think_start_length=len(collator.think_start_ids),
            think_end_length=len(collator.think_end_ids),
            c_thought=collator.c_thought,
            epochs_per_stage=collator.epochs_per_stage,
            num_latent_blocks=collator.num_latent_blocks,
            start_stage=collator.start_stage,
            length_bucket_width=self.length_bucket_width,
            first_latent_bucket_width=self.first_latent_bucket_width,
            latent_layout_bucket_width=self.latent_layout_bucket_width,
            shuffle=True,
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Coconut model did not return a training loss")
        if model.training:
            for name in self._component_sums:
                value = getattr(outputs, name, None)
                if value is not None:
                    self._component_sums[name] += float(
                        value.detach().float().item()
                    )
            self._component_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float]) -> None:
        if "loss" in logs and self._component_count:
            for name, total in self._component_sums.items():
                logs[name] = total / self._component_count
                self._component_sums[name] = 0.0
            self._component_count = 0
        super().log(logs)

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if metric_key_prefix == "eval":
            self._maybe_save_best_full_latent(metrics)
        return metrics


def write_coconut_config(
    output_dir: Path,
    args: argparse.Namespace,
    tokenizer: Any,
    think_region: ThinkRegion,
) -> None:
    payload = {
        "schema_version": 5,
        "source_model": str(args.model_path),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "special_tokens": {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in COCONUT_SPECIAL_TOKENS
        },
        "initialization_token": "<<",
        "c_thought": args.c_thought,
        "forced_boundary": {
            "tokens": THINK_END_TOKEN + "\n",
            "token_loss": False,
        },
        "think_region": {
            "artifact": str(args.think_region_file),
            "source_model": think_region.source_model,
            "state_count": think_region.state_count,
            "training_radius": "q90",
            "q90_angular_radius_degrees": think_region.q90_angular_radius_degrees,
            "q90_cosine_threshold": think_region.q90_cosine_threshold,
            "q95_angular_radius_degrees": think_region.q95_angular_radius_degrees,
            "q95_cosine_threshold": think_region.q95_cosine_threshold,
            "region_loss_weight": args.region_loss_weight,
            "inside_region_loss": 0.0,
        },
        "epochs_per_stage": args.epochs_per_stage,
        "num_latent_blocks": args.num_latent_blocks,
        "curriculum_start_stage": args.curriculum_start_stage,
        "model_selection": {
            "enabled": args.select_best_full_latent_model,
            "eligible_stage": "all_cot_steps_replaced_by_latent",
            "metric": "eval_loss",
            "policy": "lowest",
        },
        "step_selection": "prefix_original_order",
        "step_difficulty_ranks_used": False,
        "short_example_padding": "none",
        "batching": {
            "align_first_latent_with_masked_left_padding": args.align_first_latent,
            "preserve_original_position_ids": True,
            "bucket_by_layout": args.bucket_by_layout,
            "length_bucket_width": args.length_bucket_width,
            "first_latent_bucket_width": args.first_latent_bucket_width,
            "latent_layout_bucket_width": args.latent_layout_bucket_width,
            "layout_recomputed_for_each_curriculum_epoch": True,
        },
        "reasoning_boundaries": {
            "start": THINK_START_TOKEN,
            "end": THINK_END_TOKEN,
        },
    }
    path = output_dir / "coconut_config.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError(
                f"Existing Coconut config differs: {path}. Use a new output directory."
            )
        return
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    global _ACTIVE_TRACKER
    args = parse_args()
    for name in (
        "model_path",
        "train_file",
        "validation_file",
        "output_dir",
        "cache_dir",
        "log_dir",
        "think_region_file",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    if args.c_thought <= 0 or args.num_latent_blocks <= 0:
        raise ValueError("--c_thought and --num_latent_blocks must be positive")
    if args.region_loss_weight < 0:
        raise ValueError("--region_loss_weight must be non-negative")
    if args.epochs_per_stage <= 0 or args.curriculum_start_stage < 0:
        raise ValueError("Invalid curriculum schedule")
    if args.select_best_full_latent_model and args.eval_strategy != "epoch":
        raise ValueError(
            "--select_best_full_latent_model requires --eval_strategy epoch"
        )
    if min(
        args.length_bucket_width,
        args.first_latent_bucket_width,
        args.latent_layout_bucket_width,
    ) <= 0:
        raise ValueError("Bucket widths must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    if not args.train_file.is_file():
        raise FileNotFoundError(
            f"GSM8K-Aug training parquet does not exist: {args.train_file}."
        )
    if not args.validation_file.is_file():
        raise FileNotFoundError(
            f"GSM8K-Aug validation parquet does not exist: {args.validation_file}."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tracker = ExperimentTracker(
        experiment_type="train_coconut_dry_run" if args.dry_run else "train_coconut",
        formal_experiment=args.formal_experiment and not args.dry_run,
        platform_type=args.platform_type,
        platform_name=args.platform_name,
        log_root=args.log_dir,
        model_path=args.model_path,
        data_root=args.train_file,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    _ACTIVE_TRACKER = tracker

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        use_fast=True,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
    )
    configure_pad_token(tokenizer)
    previous_vocab_size = len(tokenizer)
    added_token_count = add_coconut_special_tokens(tokenizer)
    token_ids = tokenizer.convert_tokens_to_ids(list(COCONUT_SPECIAL_TOKENS))
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Coconut special tokens: "
            f"{dict(zip(COCONUT_SPECIAL_TOKENS, token_ids, strict=True))}; "
            f"new vocabulary entries: {added_token_count}"
        )
    think_region = load_think_region(args.think_region_file)
    think_end_token_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
    if think_region.think_end_token_id != think_end_token_id:
        raise ValueError(
            "Think-region </think> token ID differs from the training tokenizer: "
            f"{think_region.think_end_token_id} != {think_end_token_id}"
        )
    write_coconut_config(args.output_dir, args, tokenizer, think_region)

    # Raw GSM8K-Aug is the default; scored files remain accepted when supplied.
    raw_datasets = {
        "train": load_dataset(
            "parquet",
            data_files=str(args.train_file),
            split="train",
            cache_dir=str(args.cache_dir),
        ),
        "validation": load_dataset(
            "parquet",
            data_files=str(args.validation_file),
            split="train",
            cache_dir=str(args.cache_dir),
        ),
    }
    if args.require_step_scores:
        for split_name, dataset in raw_datasets.items():
            if "step_difficulty_ranks" not in dataset.column_names:
                raise ValueError(
                    f"The {split_name} parquet has no step_difficulty_ranks column; "
                    "use the merged outputs of score_llama1b_sft_cot_threefold.sh"
                )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        logging_dir=str(tracker.run_dir / "tensorboard"),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=args.eval_strategy != "no",
        eval_strategy=args.eval_strategy,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        max_grad_norm=2.0,
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.output_dir.name,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    with training_args.main_process_first(desc="preprocess Coconut data"):
        train_dataset, dropped_train = prepare_dataset(
            raw_datasets["train"],
            tokenizer,
            args,
            max_samples=args.max_train_samples,
            require_step_scores=args.require_step_scores,
        )
        eval_dataset, dropped_eval = prepare_dataset(
            raw_datasets["validation"],
            tokenizer,
            args,
            max_samples=args.max_eval_samples,
            require_step_scores=args.require_step_scores,
        )

    collator = CoconutDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        latent_id=tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        think_start_ids=tokenizer.encode(
            THINK_START_TOKEN + "\n", add_special_tokens=False
        ),
        think_end_ids=tokenizer.encode(
            THINK_END_TOKEN + "\n", add_special_tokens=False
        ),
        c_thought=args.c_thought,
        epochs_per_stage=args.epochs_per_stage,
        num_latent_blocks=args.num_latent_blocks,
        start_stage=args.curriculum_start_stage,
        align_first_latent=args.align_first_latent,
    )
    if training_args.should_log:
        print(
            f"Train samples: {len(train_dataset)} (dropped {dropped_train}); "
            f"validation samples: {len(eval_dataset)} (dropped {dropped_eval})"
        )
        print(
            "Curriculum replaces a prefix of the original reasoning chain, "
            "avoiding latent/explicit interleaving."
        )
        print(
            "Batching: "
            f"align_first_latent={args.align_first_latent}, "
            f"bucket_by_layout={args.bucket_by_layout}, "
            f"bucket_widths=({args.length_bucket_width}, "
            f"{args.first_latent_bucket_width}, "
            f"{args.latent_layout_bucket_width})"
        )
        for index in range(min(args.preview_samples, len(train_dataset))):
            serialized = collator._serialize(train_dataset[index])
            rendered = tokenizer.decode(serialized["input_ids"], skip_special_tokens=False)
            print(f"\n--- Coconut stage {collator.stage.scheduled_stage} sample {index} ---\n{rendered}\n")

    if args.dry_run:
        print("Dry run complete; model weights were not loaded.")
        tracker.finish(
            status="completed",
            metrics={
                "dry_run": True,
                "train_samples": len(train_dataset),
                "validation_samples": len(eval_dataset),
                "dropped_train_samples": dropped_train,
                "dropped_validation_samples": dropped_eval,
            },
            accelerator_memory=collect_accelerator_memory(),
        )
        _ACTIVE_TRACKER = None
        return

    if args.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --no-bf16 --fp16")
    dtype = (
        torch.bfloat16
        if args.bf16
        else torch.float16
        if args.fp16
        else torch.float32
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
        low_cpu_mem_usage=True,
    )
    initialize_coconut_token_embeddings(
        base_model,
        tokenizer,
        previous_vocab_size=previous_vocab_size,
    )
    base_model.config.use_cache = True
    model = CoconutForCausalLM(
        base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        think_end_token_id=think_end_token_id,
        think_region_center=think_region.center,
        think_region_cosine_threshold=think_region.q90_cosine_threshold,
        region_loss_weight=args.region_loss_weight,
    )

    trainer = CoconutTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_strategy != "no" else None,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[
            CoconutCurriculumCallback(
                collator,
                reset_optimizer_each_epoch=args.reset_optimizer_each_epoch,
            )
        ],
        bucket_by_layout=args.bucket_by_layout,
        length_bucket_width=args.length_bucket_width,
        first_latent_bucket_width=args.first_latent_bucket_width,
        latent_layout_bucket_width=args.latent_layout_bucket_width,
        select_best_full_latent_model=args.select_best_full_latent_model,
    )
    resume_checkpoint = resolve_resume_checkpoint(args)
    if training_args.should_log and resume_checkpoint:
        print(f"Resuming from {resume_checkpoint}")
    if resume_checkpoint:
        # Transformers 4.46 saves NumPy RNG state in rng_state.pth, while
        # PyTorch >=2.6 defaults torch.load to weights_only=True. These local
        # checkpoints are produced by this script and require the legacy load.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    best_full_latent: dict[str, Any] = {}
    if args.select_best_full_latent_model:
        best_full_latent = trainer.load_best_full_latent_model()
    if args.save_final_model:
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            base_output_dir = args.output_dir / "base_model"
            unwrapped.base_causallm.save_pretrained(
                str(base_output_dir), safe_serialization=True
            )
            tokenizer.save_pretrained(str(base_output_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics: dict[str, Any] = {}
    if args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    tracker.finish(
        status="completed",
        metrics={
            "train": train_result.metrics,
            "eval": eval_metrics,
            "coconut": {
                "c_thought": args.c_thought,
                "region_loss_weight": args.region_loss_weight,
                "region_q90_cosine_threshold": think_region.q90_cosine_threshold,
                "region_q95_cosine_threshold": think_region.q95_cosine_threshold,
                "epochs_per_stage": args.epochs_per_stage,
                "num_latent_blocks": args.num_latent_blocks,
                "curriculum_start_stage": args.curriculum_start_stage,
                "best_full_latent_model": best_full_latent,
            },
        },
        accelerator_memory=collect_accelerator_memory(),
    )
    _ACTIVE_TRACKER = None


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        main()
    except BaseException as exc:
        if _ACTIVE_TRACKER is not None:
            local_memory = local_accelerator_memory()
            _ACTIVE_TRACKER.finish(
                status="failed",
                accelerator_memory=[local_memory] if local_memory else [],
                error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
