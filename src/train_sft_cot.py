#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from data import (
    THINK_SPECIAL_TOKENS,
    format_sft_example,
    load_gsm8k_aug,
    select_random_dataset_parts,
    tokenize_sft_batch,
)
from experiment_log import (
    ExperimentTracker,
    collect_accelerator_memory,
    local_accelerator_memory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DURABLE_OUTPUT_ROOT = Path(os.environ.get("LATENTHALT_STORAGE_ROOT", PROJECT_ROOT)) / "outputs"
_ACTIVE_TRACKER: ExperimentTracker | None = None



@dataclass
class CompletionOnlyCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [self.label_pad_token_id] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicit chain-of-thought SFT for local Llama-3.2-1B."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=WORKSPACE_ROOT / "models" / "Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--data_root", type=Path, default=WORKSPACE_ROOT / "datasets"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DURABLE_OUTPUT_ROOT / "sft-cot-llama1b"
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument("--log_dir", type=Path, default=Path(os.environ.get("LATENTHALT_DURABLE_LOG_ROOT", DURABLE_OUTPUT_ROOT.parent / "logs")))
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
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save a checkpoint every N optimizer steps when save_strategy=steps.",
    )
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--num_train_parts",
        type=int,
        default=3,
        help="Number of random train partitions used with --train_parts.",
    )
    parser.add_argument(
        "--train_parts",
        type=int,
        nargs="+",
        help="One-based random train partitions to include, for example: 2 3.",
    )
    parser.add_argument(
        "--partition_seed",
        type=int,
        help="Random partition seed; defaults to --seed.",
    )
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
        "--save_final_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local_files_only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def configure_pad_token(tokenizer: Any) -> None:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is not None:
        return
    finetune_pad_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    if finetune_pad_id != tokenizer.unk_token_id:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        tokenizer.pad_token = tokenizer.eos_token


def add_think_special_tokens(tokenizer: Any) -> int:
    """Register atomic reasoning-boundary tokens and return the added count."""
    return tokenizer.add_special_tokens(
        {"additional_special_tokens": list(THINK_SPECIAL_TOKENS)}
    )


def prepare_dataset(
    dataset: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    max_samples: int | None,
) -> tuple[Any, int]:
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    input_count = len(dataset)
    tokenized = dataset.map(
        tokenize_sft_batch,
        batched=True,
        batch_size=1_000,
        num_proc=max(1, args.preprocessing_num_workers),
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing completion-only CoT samples",
    )
    if len(tokenized) == 0:
        raise ValueError(
            f"All {input_count} samples exceeded max_length={args.max_length}"
        )
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


def main() -> None:
    global _ACTIVE_TRACKER
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.log_dir = Path(os.environ.get("LATENTHALT_LOG_ROOT", args.log_dir)).expanduser().resolve()
    if args.save_strategy == "steps" and args.save_steps <= 0:
        raise ValueError("--save_steps must be greater than zero")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tracker = ExperimentTracker(
        experiment_type="train_sft_cot_dry_run" if args.dry_run else "train_sft_cot",
        formal_experiment=args.formal_experiment and not args.dry_run,
        platform_type=args.platform_type,
        platform_name=args.platform_name,
        log_root=args.log_dir,
        model_path=args.model_path,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    _ACTIVE_TRACKER = tracker

    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    if args.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --no-bf16 --fp16")
    if args.local_files_only and not args.model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        use_fast=True,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
    )
    configure_pad_token(tokenizer)
    added_token_count = add_think_special_tokens(tokenizer)
    if int(os.environ.get("RANK", "0")) == 0:
        think_ids = tokenizer.convert_tokens_to_ids(list(THINK_SPECIAL_TOKENS))
        print(
            f"Reasoning special tokens: {dict(zip(THINK_SPECIAL_TOKENS, think_ids, strict=True))}; "
            f"new vocabulary entries: {added_token_count}"
        )

    raw_datasets = load_gsm8k_aug(args.data_root, args.cache_dir)
    partition_manifest: dict[str, Any] | None = None
    if args.train_parts is not None:
        partition_seed = args.seed if args.partition_seed is None else args.partition_seed
        source_train_samples = len(raw_datasets["train"])
        selected_train, part_sizes = select_random_dataset_parts(
            raw_datasets["train"],
            num_parts=args.num_train_parts,
            selected_parts=args.train_parts,
            seed=partition_seed,
        )
        raw_datasets["train"] = selected_train
        selected_parts = list(args.train_parts)
        held_out_parts = [
            part
            for part in range(1, args.num_train_parts + 1)
            if part not in selected_parts
        ]
        partition_manifest = {
            "algorithm": "python_random_v1",
            "source": str(args.data_root / "gsm8k-aug" / "data"),
            "source_train_samples": source_train_samples,
            "num_parts": args.num_train_parts,
            "partition_seed": partition_seed,
            "part_sizes": {
                f"part{index}": size
                for index, size in enumerate(part_sizes, start=1)
            },
            "selected_parts": selected_parts,
            "held_out_parts": held_out_parts,
            "selected_train_samples": len(selected_train),
        }
        if int(os.environ.get("RANK", "0")) == 0:
            manifest_path = args.output_dir / "data_partition.json"
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    existing_manifest = json.load(handle)
                if existing_manifest != partition_manifest:
                    raise ValueError(
                        f"Existing partition manifest differs: {manifest_path}. "
                        "Use a different --output_dir for a different partition."
                    )
            else:
                with manifest_path.open("w", encoding="utf-8") as handle:
                    json.dump(partition_manifest, handle, indent=2, sort_keys=True)
                    handle.write("\n")

    load_best_model_at_end = (
        args.eval_strategy != "no" and args.save_strategy == args.eval_strategy
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        logging_dir=str(tracker.run_dir / "tensorboard"),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=args.eval_strategy != "no",
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=2.0,
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.output_dir.name,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    with training_args.main_process_first(desc="preprocess train and validation data"):
        train_dataset, dropped_train = prepare_dataset(
            raw_datasets["train"], tokenizer, args, args.max_train_samples
        )
        eval_dataset, dropped_eval = prepare_dataset(
            raw_datasets["validation"], tokenizer, args, args.max_eval_samples
        )

    if training_args.should_log:
        if partition_manifest is not None:
            print(
                "Train partition selection: "
                f"parts {partition_manifest['selected_parts']} of "
                f"{partition_manifest['num_parts']} with seed "
                f"{partition_manifest['partition_seed']} "
                f"({partition_manifest['selected_train_samples']} samples); "
                f"held out {partition_manifest['held_out_parts']}"
            )
        print(
            f"Train samples: {len(train_dataset)} (dropped {dropped_train} over length); "
            f"validation samples: {len(eval_dataset)} (dropped {dropped_eval} over length)"
        )
        for index in range(min(args.preview_samples, len(raw_datasets["train"]))):
            prompt, completion = format_sft_example(raw_datasets["train"][index])
            print(f"\n--- sample {index} / masked prompt ---\n{prompt}")
            print(f"--- sample {index} / supervised completion ---\n{completion}\n")

    if args.dry_run:
        print("Dry run complete; model weights were not loaded.")
        tracker.finish(
            status="completed",
            metrics={
                "dry_run": True,
                "data_partition": partition_manifest,
                "train_samples": len(train_dataset),
                "validation_samples": len(eval_dataset),
                "dropped_train_samples": dropped_train,
                "dropped_validation_samples": dropped_eval,
            },
            accelerator_memory=collect_accelerator_memory(),
        )
        _ACTIVE_TRACKER = None
        return

    dtype = (
        torch.bfloat16
        if args.bf16
        else torch.float16
        if args.fp16
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
        low_cpu_mem_usage=True,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_strategy != "no" else None,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    resume_checkpoint = resolve_resume_checkpoint(args)
    if training_args.should_log and resume_checkpoint:
        print(f"Resuming from {resume_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    if args.save_final_model:
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
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
            "data_partition": partition_manifest,
        },
        accelerator_memory=collect_accelerator_memory(),
    )
    _ACTIVE_TRACKER = None


if __name__ == "__main__":
    # Avoid oversubscribing CPU workers on each torchrun process.
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
