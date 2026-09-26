#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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
from simcot import (
    SimCoTDataCollator,
    SimCoTForCausalLM,
    SimCoTLayoutBucketSampler,
)
from train_sft_cot import configure_pad_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DURABLE_OUTPUT_ROOT = Path(os.environ.get("LATENTHALT_STORAGE_ROOT", PROJECT_ROOT)) / "outputs"
DEFAULT_COCONUT_MODEL_PATH = PROJECT_ROOT / "outputs" / "coconut-llama1b" / "base_model"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "outputs" / "coconut-llama1b"
DEFAULT_SFT_MODEL_PATH = PROJECT_ROOT / "outputs" / "sft-cot-llama1b"
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "datasets" / "gsm8k-aug" / "data"
MAX_DATASET_REASONING_STEPS = 10
_ACTIVE_TRACKER: ExperimentTracker | None = None
FSDP_TRANSFORMER_LAYER_CLASS = "LlamaDecoderLayer"


def fsdp_training_kwargs(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {}
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": [FSDP_TRANSFORMER_LAYER_CLASS],
            "use_orig_params": True,
            "backward_prefetch": "no_prefetch",
            "forward_prefetch": False,
        },
    }


def configure_auxiliary_gradient_checkpointing(
    auxiliary_decoder: torch.nn.Module | None,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    if auxiliary_decoder is None:
        raise ValueError(
            "Auxiliary gradient checkpointing requires the auxiliary decoder"
        )
    auxiliary_decoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def component_state_dict(
    state_dict: Mapping[str, torch.Tensor], component: str
) -> dict[str, torch.Tensor]:
    prefix = f"{component}."
    selected = {
        name.removeprefix(prefix): tensor
        for name, tensor in state_dict.items()
        if name.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"Full model state has no {component!r} parameters")
    return selected


def stage_lr_multiplier(
    step: int,
    *,
    updates_per_epoch: int,
    epochs_per_stage: int,
    start_stage: int,
    max_latent_stage: int,
    minimum_ratio: float,
    ramp_ratio: float,
) -> float:
    """Return a stage-local cosine multiplier with one terminal six-epoch phase."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if updates_per_epoch <= 0 or epochs_per_stage <= 0:
        raise ValueError("stage schedule lengths must be positive")
    if start_stage <= 0 or max_latent_stage <= 0:
        raise ValueError("curriculum stages must be positive")
    if not 0.0 <= minimum_ratio < 1.0:
        raise ValueError("minimum_ratio must lie in [0, 1)")
    if not 0.0 <= ramp_ratio < 1.0:
        raise ValueError("ramp_ratio must lie in [0, 1)")

    stage_steps = updates_per_epoch * epochs_per_stage
    regular_stage_count = max(max_latent_stage - start_stage, 0)
    terminal_start = regular_stage_count * stage_steps
    terminal_steps = 2 * stage_steps

    if step < terminal_start:
        phase_index, local_step = divmod(step, stage_steps)
        phase_steps = stage_steps
        initial_phase = phase_index == 0
    else:
        local_step = step - terminal_start
        phase_steps = terminal_steps
        initial_phase = regular_stage_count == 0
        if local_step >= phase_steps:
            return minimum_ratio

    ramp_steps = int(round(phase_steps * ramp_ratio))
    if ramp_ratio > 0.0:
        ramp_steps = max(ramp_steps, 2)
    ramp_steps = min(ramp_steps, phase_steps - 1)
    if local_step < ramp_steps:
        ramp_progress = local_step / max(ramp_steps - 1, 1)
        ramp_start = 0.0 if initial_phase else minimum_ratio
        return ramp_start + (1.0 - ramp_start) * ramp_progress

    decay_steps = phase_steps - ramp_steps
    decay_progress = (local_step - ramp_steps) / max(decay_steps - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SIM-CoT either from a fully trained Coconut model or jointly "
            "from the explicit CoT SFT model."
        )
    )
    parser.add_argument(
        "--coconut_model_path",
        type=Path,
        default=DEFAULT_COCONUT_MODEL_PATH,
        help="Final base_model directory exported by train_coconut.py.",
    )
    parser.add_argument(
        "--init_from_sft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start joint Coconut+SIM-CoT training directly from the explicit CoT SFT model.",
    )
    parser.add_argument(
        "--sft_model_path",
        type=Path,
        default=DEFAULT_SFT_MODEL_PATH,
        help="Explicit CoT SFT model used when --init_from_sft is enabled.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="Coconut output root containing its tokenizer and config.",
    )
    parser.add_argument(
        "--auxiliary_model_path",
        type=Path,
        default=WORKSPACE_ROOT / "models" / "Llama-3.2-1B-Instruct",
        help="Model used to initialize the train-only auxiliary decoder.",
    )
    parser.add_argument(
        "--use_auxiliary_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the auxiliary step decoder alongside the base model.",
    )
    parser.add_argument(
        "--auxiliary_gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Checkpoint only auxiliary-decoder transformer activations; the "
            "cache-dependent base-model path remains unchanged."
        ),
    )
    parser.add_argument(
        "--fsdp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable FULL_SHARD FSDP with Llama decoder-layer auto wrapping. "
            "Disabled by default."
        ),
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
        "--output_dir",
        type=Path,
        default=DURABLE_OUTPUT_ROOT / "simcot-llama1b",
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument("--log_dir", type=Path, default=Path(os.environ.get("LATENTHALT_DURABLE_LOG_ROOT", DURABLE_OUTPUT_ROOT.parent / "logs")))
    parser.add_argument("--c_thought", type=int, default=2)
    parser.add_argument("--epochs_per_stage", type=int, default=3)
    parser.add_argument("--max_latent_stage", type=int, default=7, help="Maximum latent-block stage; training may stop before the curriculum completes.")
    parser.add_argument("--curriculum_start_stage", type=int, default=1)
    parser.add_argument("--region_loss_weight", type=float, default=5.0)
    parser.add_argument("--region_negative_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--region_positive_loss_type",
        choices=["linear", "squared"],
        default="linear",
        help="Penalty for terminal q90 violations; nonterminal violations remain squared.",
    )
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--decoder_loss_normalization", choices=["block", "token"], default="block"
    )
    parser.add_argument("--num_train_epochs", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--decoder_max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--base_learning_rate", type=float, default=1e-5)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--stage_lr_decay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a fixed-peak cosine schedule inside each curriculum stage; "
            "the max-latent and fully-latent stages share one two-stage window."
        ),
    )
    parser.add_argument("--stage_lr_min_ratio", type=float, default=0.1)
    parser.add_argument("--stage_lr_ramp_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument(
        "--bucket_by_layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group batches by main length, first latent position, and latent count.",
    )
    parser.add_argument("--length_bucket_width", type=int, default=32)
    parser.add_argument("--first_latent_bucket_width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="A SIM-CoT checkpoint, 'auto', or 'none'; never the Coconut checkpoint.",
    )
    parser.add_argument(
        "--reset_optimizer_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--report_to", default="tensorboard", choices=["none", "tensorboard"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--preview_samples", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument(
        "--save_final_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local_files_only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--formal_experiment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--platform_type",
        choices=["internal_cluster", "local_server", "cloud"],
        default=os.environ.get("RUN_PLATFORM_TYPE", "local_server"),
    )
    parser.add_argument(
        "--platform_name", default=os.environ.get("RUN_PLATFORM_NAME", "local")
    )
    return parser.parse_args()


def tokenize_simcot_batch(
    batch: Mapping[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int,
    decoder_max_length: int,
    c_thought: int,
    max_latent_stage: int,
    use_auxiliary_decoder: bool = True,
) -> dict[str, list[Any]]:
    encoded: dict[str, list[Any]] = {
        "question_ids": [],
        "steps_ids": [],
        "answer_ids": [],
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

    for question, raw_steps, answer in zip(
        batch["question"], batch["steps"], batch["answer"], strict=True
    ):
        question, steps, answer = normalize_sft_example(
            {"question": question, "steps": raw_steps, "answer": answer}
        )
        # The former scored parquet excluded these ten GSM8K-Aug outliers.
        # Preserve that effective training set when reading the raw parquet.
        if len(steps) > MAX_DATASET_REASONING_STEPS:
            continue
        question_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
        steps_ids = [
            tokenizer.encode(step + "\n", add_special_tokens=False)
            for step in steps
        ]
        answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [eos_token_id]
        block_budget = min(max_latent_stage, len(steps_ids))
        curriculum_lengths = []
        for block_count in range(1, block_budget + 1):
            curriculum_lengths.append(
                len(question_ids)
                + len(think_start_ids)
                + block_count * c_thought
                + sum(len(step) for step in steps_ids[block_count:])
                + len(think_end_ids)
                + len(answer_ids)
            )
        curriculum_lengths.append(
            len(question_ids)
            + len(think_start_ids)
            + block_budget * c_thought
            + len(think_end_ids)
            + len(answer_ids)
        )
        decoder_target_lengths = [len(step_ids) + 1 for step_ids in steps_ids]
        if len(steps_ids) > block_budget:
            decoder_target_lengths.append(
                sum(len(step) for step in steps_ids[block_budget - 1 :]) + 1
            )
        if max(curriculum_lengths, default=0) > max_length:
            continue
        if (
            use_auxiliary_decoder
            and c_thought + max(decoder_target_lengths, default=0) > decoder_max_length
        ):
            continue
        encoded["question_ids"].append(question_ids)
        encoded["steps_ids"].append(steps_ids)
        encoded["answer_ids"].append(answer_ids)
        encoded["question_length"].append(len(question_ids))
        encoded["step_lengths"].append([len(step) for step in steps_ids])
        encoded["answer_length"].append(len(answer_ids))
    return encoded


def prepare_dataset(
    dataset: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    max_samples: int | None,
) -> tuple[Any, int]:
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    input_count = len(dataset)
    tokenized = dataset.map(
        tokenize_simcot_batch,
        batched=True,
        batch_size=1_000,
        num_proc=max(1, args.preprocessing_num_workers),
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
            "decoder_max_length": args.decoder_max_length,
            "c_thought": args.c_thought,
            "max_latent_stage": args.max_latent_stage,
            "use_auxiliary_decoder": args.use_auxiliary_decoder,
        },
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing curriculum SIM-CoT components",
    )
    if len(tokenized) == 0:
        raise ValueError("All samples exceeded the main or auxiliary length limit")
    return tokenized, input_count - len(tokenized)


def validate_coconut_config(
    tokenizer_path: Path, *, c_thought: int
) -> None:
    config_path = tokenizer_path / "coconut_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Trained Coconut config is missing: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("c_thought", -1)) != c_thought:
        raise ValueError("SIM-CoT c_thought differs from the trained Coconut model")
    forced_boundary = config.get("forced_boundary", {})
    if forced_boundary.get("tokens") != THINK_END_TOKEN + "\n":
        raise ValueError("Trained Coconut uses a different forced reasoning boundary")
    if forced_boundary.get("token_loss") is not False:
        raise ValueError("Trained Coconut unexpectedly supervises the boundary token")


def validate_auxiliary_tokenizer(
    tokenizer: Any, auxiliary_model_path: Path, cache_dir: Path, local_files_only: bool
) -> None:
    auxiliary_tokenizer = AutoTokenizer.from_pretrained(
        str(auxiliary_model_path),
        use_fast=True,
        local_files_only=local_files_only,
        cache_dir=str(cache_dir),
    )
    auxiliary_vocab = auxiliary_tokenizer.get_vocab()
    coconut_vocab = tokenizer.get_vocab()
    mismatches = [
        token
        for token, token_id in auxiliary_vocab.items()
        if coconut_vocab.get(token) != token_id
    ]
    if mismatches:
        preview = mismatches[:5]
        raise ValueError(
            f"Auxiliary and Coconut tokenizers use different token IDs: {preview}"
        )
    extra_tokens = set(coconut_vocab) - set(auxiliary_vocab)
    allowed_extra_tokens = {LATENT_TOKEN, THINK_START_TOKEN, THINK_END_TOKEN}
    if not extra_tokens.issubset(allowed_extra_tokens):
        raise ValueError(
            "The Coconut tokenizer has unsupported additions relative to the "
            f"auxiliary tokenizer: {sorted(extra_tokens - allowed_extra_tokens)}"
        )
    if auxiliary_tokenizer.eos_token_id != tokenizer.eos_token_id:
        raise ValueError("Auxiliary and Coconut tokenizers use different EOS tokens")


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    requested = args.resume_from_checkpoint.strip()
    if requested.lower() in {"", "none", "false"}:
        return None
    if requested.lower() != "auto":
        checkpoint = Path(requested).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"SIM-CoT checkpoint does not exist: {checkpoint}")
        return str(checkpoint)
    if not args.output_dir.is_dir():
        return None
    return get_last_checkpoint(str(args.output_dir))


class SimCoTCurriculumCallback(TrainerCallback):
    def __init__(
        self,
        collator: SimCoTDataCollator,
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
        state.save_steps = args.save_steps
        state.train_batch_size = args.train_batch_size
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
            mode = "fully latent" if stage.fully_implicit else "hybrid"
            print(
                f"SIM-CoT epoch {epoch}: scheduled_stage={stage.scheduled_stage}, "
                f"latent_blocks={stage.latent_blocks}, mode={mode}, "
                "optimizer_reset="
                f"{self.reset_optimizer_each_epoch and epoch > 0}"
            )
        return control


class SimCoTTrainer(Trainer):
    """Add component losses to the normal Trainer logging interval."""

    def __init__(
        self,
        *args: Any,
        bucket_by_layout: bool = True,
        length_bucket_width: int = 32,
        first_latent_bucket_width: int = 16,
        base_learning_rate: float = 1e-5,
        decoder_learning_rate: float = 1e-5,
        stage_lr_decay: bool = False,
        stage_lr_min_ratio: float = 0.1,
        stage_lr_ramp_ratio: float = 0.03,
        epochs_per_stage: int = 3,
        curriculum_start_stage: int = 1,
        max_latent_stage: int = 10,
        **kwargs: Any,
    ) -> None:
        self.bucket_by_layout = bucket_by_layout
        self.length_bucket_width = length_bucket_width
        self.first_latent_bucket_width = first_latent_bucket_width
        self.base_learning_rate = base_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        self.stage_lr_decay = stage_lr_decay
        self.stage_lr_min_ratio = stage_lr_min_ratio
        self.stage_lr_ramp_ratio = stage_lr_ramp_ratio
        self.epochs_per_stage = epochs_per_stage
        self.curriculum_start_stage = curriculum_start_stage
        self.max_latent_stage = max_latent_stage
        self._updates_per_epoch: int | None = None
        super().__init__(*args, **kwargs)
        self._component_sums = {
            "language_model_loss": 0.0,
            "region_loss": 0.0,
            "region_positive_loss": 0.0,
            "region_negative_loss": 0.0,
            "decoder_loss": 0.0,
        }
        self._component_count = 0

    def get_train_dataloader(self) -> Any:
        dataloader = super().get_train_dataloader()
        self._updates_per_epoch = max(
            len(dataloader) // self.args.gradient_accumulation_steps,
            1,
        )
        return dataloader

    def _get_train_sampler(self) -> Any:
        if not self.bucket_by_layout:
            return super()._get_train_sampler()
        if self.train_dataset is None:
            return None
        if not isinstance(self.data_collator, SimCoTDataCollator):
            raise TypeError("Curriculum batching requires SimCoTDataCollator")
        collator = self.data_collator
        return SimCoTLayoutBucketSampler(
            self.train_dataset,
            batch_size=self._train_batch_size,
            seed=self.args.data_seed,
            think_start_length=len(collator.think_start_ids),
            think_end_length=len(collator.think_end_ids),
            c_thought=collator.c_thought,
            epochs_per_stage=collator.epochs_per_stage,
            max_latent_stage=collator.max_latent_stage,
            start_stage=collator.start_stage,
            length_bucket_width=self.length_bucket_width,
            first_latent_bucket_width=self.first_latent_bucket_width,
            shuffle=True,
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = set(self.get_decay_parameter_names(self.model))
        grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
            ("base", True): [],
            ("base", False): [],
            ("decoder", True): [],
            ("decoder", False): [],
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("base_causallm.") or ".base_causallm." in name:
                component = "base"
            elif name.startswith("auxiliary_decoder.") or ".auxiliary_decoder." in name:
                component = "decoder"
            else:
                raise ValueError(f"Unassigned trainable SIM-CoT parameter: {name}")
            grouped[(component, name in decay_parameters)].append(parameter)

        optimizer_groups = []
        for component, learning_rate in (
            ("base", self.base_learning_rate),
            ("decoder", self.decoder_learning_rate),
        ):
            for use_decay in (True, False):
                parameters = grouped[(component, use_decay)]
                if parameters:
                    optimizer_groups.append(
                        {
                            "params": parameters,
                            "lr": learning_rate,
                            "component": component,
                            "weight_decay": (
                                self.args.weight_decay if use_decay else 0.0
                            ),
                        }
                    )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        if "optimizer_dict" in optimizer_kwargs:
            raise ValueError(
                "SIM-CoT component learning rates do not support optimizer_dict"
            )
        self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Any:
        if not self.stage_lr_decay:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        if self._updates_per_epoch is None:
            raise RuntimeError(
                "Training dataloader must be created before the stage LR scheduler"
            )

        def lr_lambda(current_step: int) -> float:
            return stage_lr_multiplier(
                current_step,
                updates_per_epoch=self._updates_per_epoch or 1,
                epochs_per_stage=self.epochs_per_stage,
                start_stage=self.curriculum_start_stage,
                max_latent_stage=self.max_latent_stage,
                minimum_ratio=self.stage_lr_min_ratio,
                ramp_ratio=self.stage_lr_ramp_ratio,
            )

        scheduler_optimizer = self.optimizer if optimizer is None else optimizer
        if scheduler_optimizer is None:
            raise RuntimeError("Optimizer must be created before the LR scheduler")
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            scheduler_optimizer,
            lr_lambda,
        )
        self._created_lr_scheduler = True
        return self.lr_scheduler

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
            raise ValueError("SIM-CoT model did not return a training loss")
        if model.training:
            for name in self._component_sums:
                value = getattr(outputs, name, None)
                if value is not None:
                    self._component_sums[name] += float(value.detach().float().item())
            self._component_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float]) -> None:
        if "loss" in logs and self._component_count:
            for name, total in self._component_sums.items():
                logs[f"simcot/{name}"] = total / self._component_count
                self._component_sums[name] = 0.0
            self._component_count = 0
            if self.optimizer is not None:
                for group in self.optimizer.param_groups:
                    component = group.get("component")
                    if component in {"base", "decoder"}:
                        logs[f"simcot/{component}_learning_rate"] = float(group["lr"])
        super().log(logs)


def write_simcot_config(
    output_dir: Path,
    args: argparse.Namespace,
    tokenizer: Any,
    think_region: Any,
) -> None:
    payload = {
        "schema_version": 8,
        "initialization": (
            {
                "mode": "sft_joint",
                "source_model": str(args.sft_model_path),
                "added_special_tokens": {
                    token: tokenizer.convert_tokens_to_ids(token)
                    for token in COCONUT_SPECIAL_TOKENS
                },
                "latent_initialization_token": "<<",
            }
            if args.init_from_sft
            else {
                "mode": "trained_coconut",
                "source_model": str(args.coconut_model_path),
                "tokenizer_path": str(args.tokenizer_path),
            }
        ),
        "coconut_model_path": str(args.coconut_model_path),
        "tokenizer_path": str(args.tokenizer_path),
        "sft_model_path": str(args.sft_model_path),
        "auxiliary_model_path": str(args.auxiliary_model_path),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "fully_latent": True,
        "training_mode": (
            "joint_curriculum_from_sft"
            if args.init_from_sft
            else "curriculum_to_fully_latent"
        ),
        "latent_token": LATENT_TOKEN,
        "latent_token_id": tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        "c_thought": args.c_thought,
        "curriculum": {
            "epochs_per_stage": args.epochs_per_stage,
            "start_stage": args.curriculum_start_stage,
            "max_latent_stage": args.max_latent_stage,
            "stage_rule": "replace one additional explicit step at each stage",
            "fully_latent_rule": "scheduled_stage exceeds max_latent_stage",
        },
        "latent_blocks": (
            "one block per replaced step, capped at max_latent_stage"
        ),
        "decoder_target": (
            "corresponding step plus EOS; at the fully-latent cap, the final "
            "block reconstructs all remaining steps plus EOS"
        ),
        "decoder_prefix": "actual injected continuous embeddings",
        "decoder_loss_weight": args.decoder_loss_weight,
        "decoder_loss_normalization": args.decoder_loss_normalization,
        "region_loss_weight": args.region_loss_weight,
        "region_negative_loss_weight": args.region_negative_loss_weight,
        "region_positive_loss_type": args.region_positive_loss_type,
        "region_loss": {
            "positive": (
                f"{args.region_positive_loss_type} hinge until a sample's "
                "fully-latent terminal block enters q90"
            ),
            "negative": (
                "squared hinge when any nonterminal block end enters q95; hybrid "
                "stage block ends are nonterminal"
            ),
            "sample_balanced": True,
        },
        "think_region_artifact": str(args.think_region_file),
        "region_q90_cosine_threshold": think_region.q90_cosine_threshold,
        "region_q95_cosine_threshold": think_region.q95_cosine_threshold,
        "optimizer": {
            "base_learning_rate": args.base_learning_rate,
            "decoder_learning_rate": (
                args.decoder_learning_rate if args.use_auxiliary_decoder else None
            ),
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": {
                "type": (
                    "stage_cosine_with_terminal_group"
                    if args.stage_lr_decay
                    else "constant_with_warmup"
                ),
                "stage_lr_min_ratio": args.stage_lr_min_ratio,
                "stage_lr_ramp_ratio": args.stage_lr_ramp_ratio,
                "fixed_peak_across_stages": True,
                "terminal_group": "max_latent_stage_plus_fully_latent",
                "terminal_group_epochs": 2 * args.epochs_per_stage,
            },
        },
        "forced_boundary": {"tokens": THINK_END_TOKEN + "\n", "token_loss": False},
        "batching": {
            "align_first_latent_with_masked_left_padding": True,
            "preserve_original_position_ids": True,
            "bucket_by_layout": args.bucket_by_layout,
            "length_bucket_width": args.length_bucket_width,
            "first_latent_bucket_width": args.first_latent_bucket_width,
            "latent_count_is_exact_sort_key": True,
            "layout_recomputed_for_each_curriculum_epoch": True,
        },
        "inference": "discard auxiliary_decoder and use base_model",
    }
    if not args.use_auxiliary_decoder:
        payload.update(
            {
                "use_auxiliary_decoder": False,
                "auxiliary_model_path": None,
                "decoder_target": None,
                "decoder_prefix": None,
                "inference": "use base_model (no auxiliary decoder was trained)",
            }
        )
    path = output_dir / "simcot_config.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError(f"Existing SIM-CoT config differs: {path}")
        return
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    global _ACTIVE_TRACKER
    args = parse_args()
    if args.auxiliary_gradient_checkpointing and not args.use_auxiliary_decoder:
        raise ValueError(
            "--auxiliary_gradient_checkpointing requires --use_auxiliary_decoder"
        )
    if not args.use_auxiliary_decoder:
        # Keep the ablation objective unambiguous even when invoked directly
        # instead of through the dedicated wrapper.
        args.decoder_loss_weight = 0.0
    path_fields = (
        "coconut_model_path",
        "sft_model_path",
        "tokenizer_path",
        "auxiliary_model_path",
        "train_file",
        "validation_file",
        "think_region_file",
        "output_dir",
        "cache_dir",
        "log_dir",
    )
    for name in path_fields:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.log_dir = Path(os.environ.get("LATENTHALT_LOG_ROOT", args.log_dir)).expanduser().resolve()

    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    if args.c_thought <= 0:
        raise ValueError("c_thought must be positive")
    if args.save_total_limit <= 0:
        raise ValueError("save_total_limit must be a positive integer")
    if args.epochs_per_stage <= 0 or args.max_latent_stage <= 0:
        raise ValueError("SIM-CoT curriculum sizes must be positive")
    if args.curriculum_start_stage <= 0:
        raise ValueError("SIM-CoT curriculum must start at stage 1 or later")
    # Allow training to stop before every curriculum stage has completed.
    if args.max_steps < 0 and args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")
    if (
        args.region_loss_weight < 0
        or args.region_negative_loss_weight < 0
        or args.decoder_loss_weight < 0
    ):
        raise ValueError("Loss weights must be non-negative")
    if args.base_learning_rate <= 0 or args.decoder_learning_rate <= 0:
        raise ValueError("Component learning rates must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must lie in [0, 1)")
    if not 0.0 <= args.stage_lr_min_ratio < 1.0:
        raise ValueError("stage_lr_min_ratio must lie in [0, 1)")
    if not 0.0 <= args.stage_lr_ramp_ratio < 1.0:
        raise ValueError("stage_lr_ramp_ratio must lie in [0, 1)")
    if args.stage_lr_decay:
        if args.max_steps > 0:
            raise ValueError("Stage LR decay requires epoch-based training")
        expected_epochs = (
            max(args.max_latent_stage - args.curriculum_start_stage, 0) + 2
        ) * args.epochs_per_stage
        if not math.isclose(args.num_train_epochs, expected_epochs):
            raise ValueError(
                "Stage LR decay requires one window per pre-terminal stage and "
                "one shared max/full-latent window; expected "
                f"num_train_epochs={expected_epochs}, got {args.num_train_epochs}"
            )
    if args.length_bucket_width <= 0 or args.first_latent_bucket_width <= 0:
        raise ValueError("Bucket widths must be positive")
    if args.init_from_sft and args.tokenizer_path == DEFAULT_TOKENIZER_PATH:
        args.tokenizer_path = args.sft_model_path

    source_model_path = args.sft_model_path if args.init_from_sft else args.coconut_model_path
    tokenizer_source_path = args.tokenizer_path

    required_model_paths = [
        ("sft_model_path" if args.init_from_sft else "coconut_model_path"),
        "tokenizer_path",
    ]
    if args.use_auxiliary_decoder:
        required_model_paths.append("auxiliary_model_path")
    for name in required_model_paths:
        if not getattr(args, name).is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {getattr(args, name)}")
    if not args.train_file.is_file() or not args.validation_file.is_file():
        raise FileNotFoundError("SIM-CoT train or validation parquet is missing")
    if not args.think_region_file.is_file():
        raise FileNotFoundError(f"Think-region artifact is missing: {args.think_region_file}")
    if not args.init_from_sft:
        validate_coconut_config(
            args.tokenizer_path,
            c_thought=args.c_thought,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tracker = ExperimentTracker(
        experiment_type="train_simcot_dry_run" if args.dry_run else "train_simcot",
        formal_experiment=args.formal_experiment and not args.dry_run,
        platform_type=args.platform_type,
        platform_name=args.platform_name,
        log_root=args.log_dir,
        model_path=source_model_path,
        data_root=args.train_file,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    _ACTIVE_TRACKER = tracker

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source_path),
        use_fast=True,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
    )
    configure_pad_token(tokenizer)
    previous_vocab_size = len(tokenizer)
    if args.init_from_sft:
        added_token_count = add_coconut_special_tokens(tokenizer)
        if int(os.environ.get("RANK", "0")) == 0:
            token_ids = tokenizer.convert_tokens_to_ids(list(COCONUT_SPECIAL_TOKENS))
            print(
                f"Joint SIM-CoT special tokens: "
                f"{dict(zip(COCONUT_SPECIAL_TOKENS, token_ids, strict=True))}; "
                f"new vocabulary entries: {added_token_count}"
            )
    if LATENT_TOKEN not in tokenizer.get_vocab():
        raise ValueError(
            f"The trained Coconut tokenizer does not contain {LATENT_TOKEN!r}"
        )
    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    if tokenizer.eos_token_id is None:
        raise ValueError("The Coconut tokenizer does not define eos_token_id")
    if args.use_auxiliary_decoder:
        validate_auxiliary_tokenizer(
            tokenizer,
            args.auxiliary_model_path,
            args.cache_dir,
            args.local_files_only,
        )
    think_end_token_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
    think_region = load_think_region(args.think_region_file)
    if think_region.think_end_token_id != think_end_token_id:
        raise ValueError("Think-region and Coconut tokenizer use different </think> IDs")

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
        learning_rate=(
            args.decoder_learning_rate
            if args.use_auxiliary_decoder
            else args.base_learning_rate
        ),
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="constant_with_warmup",
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
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        **fsdp_training_kwargs(args.fsdp),
    )

    with training_args.main_process_first(desc="preprocess SIM-CoT data"):
        train_dataset, dropped_train = prepare_dataset(
            raw_datasets["train"],
            tokenizer,
            args,
            max_samples=args.max_train_samples,
        )
        eval_dataset, dropped_eval = prepare_dataset(
            raw_datasets["validation"],
            tokenizer,
            args,
            max_samples=args.max_eval_samples,
        )

    collator = SimCoTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        latent_id=latent_token_id,
        think_start_ids=tokenizer.encode(
            THINK_START_TOKEN + "\n", add_special_tokens=False
        ),
        think_end_ids=tokenizer.encode(
            THINK_END_TOKEN + "\n", add_special_tokens=False
        ),
        eos_token_id=tokenizer.eos_token_id,
        c_thought=args.c_thought,
        epochs_per_stage=args.epochs_per_stage,
        max_latent_stage=args.max_latent_stage,
        start_stage=args.curriculum_start_stage,
    )
    write_simcot_config(args.output_dir, args, tokenizer, think_region)

    if training_args.should_log:
        print(
            f"SIM-CoT train samples: {len(train_dataset)} (dropped {dropped_train}); "
            f"validation samples: {len(eval_dataset)} (dropped {dropped_eval})"
        )
        for index in range(min(args.preview_samples, len(train_dataset))):
            serialized = collator._serialize(train_dataset[index])
            rendered = tokenizer.decode(
                serialized["input_ids"], skip_special_tokens=False
            )
            target_text = [
                tokenizer.decode(target, skip_special_tokens=False)
                for target in serialized["decoder_targets"]
            ]
            print(f"\n--- SIM-CoT sample {index} ---\n{rendered}\nDecoder targets: {target_text}\n")

    if args.dry_run:
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
        str(source_model_path),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
        low_cpu_mem_usage=True,
    )
    if args.init_from_sft:
        initialize_coconut_token_embeddings(
            base_model,
            tokenizer,
            previous_vocab_size=previous_vocab_size,
        )
    elif base_model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError(
            "Trained Coconut base vocabulary differs from its tokenizer; refusing to resize"
        )
    base_model.config.use_cache = True
    auxiliary_decoder = None
    if args.use_auxiliary_decoder:
        auxiliary_decoder = AutoModelForCausalLM.from_pretrained(
            str(args.auxiliary_model_path),
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
            cache_dir=str(args.cache_dir),
            low_cpu_mem_usage=True,
        )
        if auxiliary_decoder.get_input_embeddings().num_embeddings != len(tokenizer):
            auxiliary_decoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        auxiliary_decoder.config.use_cache = False
        configure_auxiliary_gradient_checkpointing(
            auxiliary_decoder,
            enabled=args.auxiliary_gradient_checkpointing,
        )

    model = SimCoTForCausalLM(
        base_causallm=base_model,
        auxiliary_decoder=auxiliary_decoder,
        latent_token_id=latent_token_id,
        think_end_token_id=think_end_token_id,
        think_region_center=think_region.center,
        think_region_cosine_threshold=think_region.q90_cosine_threshold,
        nonterminal_region_cosine_threshold=think_region.q95_cosine_threshold,
        region_loss_weight=args.region_loss_weight,
        region_negative_loss_weight=args.region_negative_loss_weight,
        region_positive_loss_type=args.region_positive_loss_type,
        decoder_loss_weight=args.decoder_loss_weight,
        c_thought=args.c_thought,
        decoder_loss_normalization=args.decoder_loss_normalization,
    )
    trainer = SimCoTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_strategy != "no" else None,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[
            SimCoTCurriculumCallback(
                collator,
                reset_optimizer_each_epoch=args.reset_optimizer_each_epoch,
            )
        ],
        bucket_by_layout=args.bucket_by_layout,
        length_bucket_width=args.length_bucket_width,
        first_latent_bucket_width=args.first_latent_bucket_width,
        base_learning_rate=args.base_learning_rate,
        decoder_learning_rate=args.decoder_learning_rate,
        stage_lr_decay=args.stage_lr_decay,
        stage_lr_min_ratio=args.stage_lr_min_ratio,
        stage_lr_ramp_ratio=args.stage_lr_ramp_ratio,
        epochs_per_stage=args.epochs_per_stage,
        curriculum_start_stage=args.curriculum_start_stage,
        max_latent_stage=args.max_latent_stage,
    )

    resume_checkpoint = resolve_resume_checkpoint(args)
    if training_args.should_log and resume_checkpoint:
        print(f"Resuming SIM-CoT from {resume_checkpoint}")
    if resume_checkpoint:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    if args.save_final_model:
        if trainer.is_fsdp_enabled:
            # Gathering is collective; only rank zero receives the full CPU state.
            full_state_dict = trainer.accelerator.get_state_dict(trainer.model)
            if trainer.is_world_process_zero():
                base_state_dict = component_state_dict(
                    full_state_dict, "base_causallm"
                )
                decoder_state_dict = (
                    component_state_dict(full_state_dict, "auxiliary_decoder")
                    if args.use_auxiliary_decoder
                    else None
                )
                trainer._save(str(args.output_dir), state_dict=full_state_dict)
                tokenizer.save_pretrained(str(args.output_dir))
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                base_output_dir = args.output_dir / "base_model"
                unwrapped.base_causallm.save_pretrained(
                    str(base_output_dir),
                    state_dict=base_state_dict,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(str(base_output_dir))
                if args.use_auxiliary_decoder:
                    decoder_output_dir = args.output_dir / "auxiliary_decoder"
                    if unwrapped.auxiliary_decoder is None:
                        raise RuntimeError(
                            "Auxiliary decoder is enabled but was not attached to the model"
                        )
                    unwrapped.auxiliary_decoder.save_pretrained(
                        str(decoder_output_dir),
                        state_dict=decoder_state_dict,
                        safe_serialization=True,
                    )
                    tokenizer.save_pretrained(str(decoder_output_dir))
                del base_state_dict, decoder_state_dict
            del full_state_dict
            trainer.accelerator.wait_for_everyone()
        else:
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
                if args.use_auxiliary_decoder:
                    decoder_output_dir = args.output_dir / "auxiliary_decoder"
                    if unwrapped.auxiliary_decoder is None:
                        raise RuntimeError(
                            "Auxiliary decoder is enabled but was not attached to the model"
                        )
                    unwrapped.auxiliary_decoder.save_pretrained(
                        str(decoder_output_dir), safe_serialization=True
                    )
                    tokenizer.save_pretrained(str(decoder_output_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics: dict[str, Any] = {}
    if args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    trainer.accelerator.wait_for_everyone()

    tracker.finish(
        status="completed",
        metrics={
            "train": train_result.metrics,
            "eval": eval_metrics,
            "simcot": {
                "c_thought": args.c_thought,
                "use_auxiliary_decoder": args.use_auxiliary_decoder,
                "fsdp": args.fsdp,
                "auxiliary_gradient_checkpointing": (
                    args.auxiliary_gradient_checkpointing
                ),
                "decoder_loss_weight": args.decoder_loss_weight,
                "decoder_loss_normalization": args.decoder_loss_normalization,
                "region_loss_weight": args.region_loss_weight,
                "region_negative_loss_weight": args.region_negative_loss_weight,
                "region_positive_loss_type": args.region_positive_loss_type,
                "base_learning_rate": args.base_learning_rate,
                "decoder_learning_rate": (
                    args.decoder_learning_rate if args.use_auxiliary_decoder else None
                ),
                "warmup_ratio": args.warmup_ratio,
                "stage_lr_decay": args.stage_lr_decay,
                "stage_lr_min_ratio": args.stage_lr_min_ratio,
                "stage_lr_ramp_ratio": args.stage_lr_ramp_ratio,
                "terminal_lr_group_epochs": 2 * args.epochs_per_stage,
                "reset_optimizer_each_epoch": args.reset_optimizer_each_epoch,
                "epochs_per_stage": args.epochs_per_stage,
                "max_latent_stage": args.max_latent_stage,
                "curriculum_start_stage": args.curriculum_start_stage,
                "fully_latent_when_scheduled_stage_exceeds": args.max_latent_stage,
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
