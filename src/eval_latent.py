#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import LATENT_TOKEN, load_think_region
from data import (
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    batched,
    extract_answer_after_think,
    extract_numeric_answer,
    load_math_examples,
    numeric_answers_equal,
)
from eval_math import (
    count_dense_parameters,
    estimate_latent_continuation_flops,
    normalize_token_ids,
)
from experiment_log import (
    ExperimentTracker,
    collect_accelerator_memory,
    local_accelerator_memory,
)
from latent_inference import generate_latent_batch
from train_sft_cot import configure_pad_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SUPPORTED_DATASETS = ("gsm8k", "gsm-hard", "multi-arith", "svamp")
_ACTIVE_TRACKER: ExperimentTracker | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fully latent Coconut/SIM-CoT Llama model."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "latent-halt-llama1b"
            / "base_model"
        ),
        help="Standard Hugging Face base_model export, not a wrapped Trainer checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Defaults to model_path; must contain the trained <|latent|> token.",
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
        "--data_root", type=Path, default=WORKSPACE_ROOT / "datasets"
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "latent-llama1b_v1",
    )
    parser.add_argument("--log_dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument(
        "--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS)
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--c_thought", type=int)
    parser.add_argument("--min_latent_blocks", type=int, default=1)
    parser.add_argument("--max_latent_blocks", type=int)
    parser.add_argument(
        "--halt_threshold",
        type=float,
        help="Cosine threshold override; defaults to the artifact's q95 boundary.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of answer tokens after the forced </think> boundary.",
    )
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="sdpa")
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_training_protocol(
    model_path: Path, tokenizer_path: Path
) -> tuple[int, int, list[str]]:
    """Resolve c_thought and the trained final block budget when available."""
    candidate_roots = []
    for root in (model_path, model_path.parent, tokenizer_path, tokenizer_path.parent):
        if root not in candidate_roots:
            candidate_roots.append(root)

    sources: list[str] = []
    c_thought: int | None = None
    max_blocks: int | None = None
    for root in candidate_roots:
        coconut_path = root / "coconut_config.json"
        if coconut_path.is_file():
            config = _read_json(coconut_path)
            sources.append(str(coconut_path))
            configured_c_thought = int(config["c_thought"])
            if c_thought is not None and c_thought != configured_c_thought:
                raise ValueError("Conflicting c_thought values in training configs")
            c_thought = configured_c_thought
            max_blocks = int(config["num_latent_blocks"])

        simcot_path = root / "simcot_config.json"
        if simcot_path.is_file():
            config = _read_json(simcot_path)
            sources.append(str(simcot_path))
            configured_c_thought = int(config["c_thought"])
            if c_thought is not None and c_thought != configured_c_thought:
                raise ValueError("Conflicting c_thought values in training configs")
            c_thought = configured_c_thought

    return (
        2 if c_thought is None else c_thought,
        10 if max_blocks is None else max_blocks,
        sources,
    )


def left_pad_prompts(
    prompt_ids: Sequence[Sequence[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prompt_ids or any(not sequence for sequence in prompt_ids):
        raise ValueError("Every evaluation prompt must contain at least one token")
    max_length = max(len(sequence) for sequence in prompt_ids)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for sequence in prompt_ids:
        padding = max_length - len(sequence)
        input_rows.append([pad_token_id] * padding + list(sequence))
        mask_rows.append([0] * padding + [1] * len(sequence))
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.long),
    )


def write_result(handle: Any, result: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(result), ensure_ascii=True) + "\n")
    handle.flush()


def _dataset_summary(
    *,
    correct: int,
    total: int,
    region_halts: int,
    latent_blocks: list[int],
    answer_tokens: int,
    eos_answers: int,
    logical_positions: int,
    flops_excluding_prefill: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    histogram = Counter(latent_blocks)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "region_halts": region_halts,
        "region_halt_rate": region_halts / total if total else 0.0,
        "max_budget_fallbacks": total - region_halts,
        "latent_blocks_total": sum(latent_blocks),
        "average_latent_blocks": sum(latent_blocks) / total if total else 0.0,
        "min_latent_blocks": min(latent_blocks, default=0),
        "max_latent_blocks": max(latent_blocks, default=0),
        "latent_block_histogram": {
            str(blocks): histogram[blocks] for blocks in sorted(histogram)
        },
        "answer_tokens_total": answer_tokens,
        "average_answer_tokens": answer_tokens / total if total else 0.0,
        "answers_with_eos": eos_answers,
        "answer_eos_rate": eos_answers / total if total else 0.0,
        "logical_sequence_positions": logical_positions,
        "average_logical_sequence_positions": (
            logical_positions / total if total else 0.0
        ),
        "total_flops_excluding_prefill": flops_excluding_prefill,
        "total_tera_flops_excluding_prefill": flops_excluding_prefill / 1e12,
        "average_flops_excluding_prefill": (
            flops_excluding_prefill / total if total else 0.0
        ),
        "average_flops_per_question_excluding_prefill": (
            flops_excluding_prefill / total if total else 0.0
        ),
        "runtime_seconds": runtime_seconds,
    }


def main() -> None:
    global _ACTIVE_TRACKER
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.tokenizer_path = (
        args.tokenizer_path.expanduser().resolve()
        if args.tokenizer_path is not None
        else args.model_path
    )
    for name in ("think_region_file", "data_root", "cache_dir", "output_dir", "log_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())

    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Latent model directory does not exist: {args.model_path}")
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {args.tokenizer_path}")
    if not args.think_region_file.is_file():
        raise FileNotFoundError(f"Think-region artifact is missing: {args.think_region_file}")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")

    configured_c_thought, configured_max_blocks, protocol_sources = (
        resolve_training_protocol(args.model_path, args.tokenizer_path)
    )
    c_thought = (
        configured_c_thought if args.c_thought is None else args.c_thought
    )
    max_latent_blocks = (
        configured_max_blocks
        if args.max_latent_blocks is None
        else args.max_latent_blocks
    )
    if c_thought != configured_c_thought and protocol_sources:
        raise ValueError(
            f"Requested c_thought={c_thought}, but training used {configured_c_thought}"
        )
    if args.min_latent_blocks <= 0 or args.min_latent_blocks > max_latent_blocks:
        raise ValueError("min_latent_blocks must lie in [1, max_latent_blocks]")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(
        experiment_type="eval_latent",
        formal_experiment=args.formal_experiment,
        platform_type=args.platform_type,
        platform_name=args.platform_name,
        log_root=args.log_dir,
        model_path=args.model_path,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    _ACTIVE_TRACKER = tracker

    if not torch.cuda.is_available():
        raise RuntimeError("Latent evaluation requires a CUDA device")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --no-bf16")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        use_fast=True,
        padding_side="left",
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
    )
    configure_pad_token(tokenizer)
    tokenizer.padding_side = "left"
    if LATENT_TOKEN not in tokenizer.get_vocab():
        raise ValueError(f"Tokenizer does not contain the trained {LATENT_TOKEN!r} token")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not define a pad token")

    think_start_ids = tokenizer.encode(
        THINK_START_TOKEN + "\n", add_special_tokens=False
    )
    think_end_ids = tokenizer.encode(THINK_END_TOKEN + "\n", add_special_tokens=False)
    if not think_start_ids or not think_end_ids:
        raise ValueError("Reasoning boundary token sequences cannot be empty")
    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

    think_region = load_think_region(args.think_region_file)
    if think_region.think_end_token_id != tokenizer.convert_tokens_to_ids(THINK_END_TOKEN):
        raise ValueError("Think-region and evaluation tokenizer use different </think> IDs")
    halt_threshold = (
        think_region.q95_cosine_threshold
        if args.halt_threshold is None
        else args.halt_threshold
    )
    if not -1.0 <= halt_threshold <= 1.0:
        raise ValueError("halt_threshold must lie in [-1, 1]")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
        low_cpu_mem_usage=True,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError("Latent model vocabulary differs from its trained tokenizer")
    if int(model.config.hidden_size) != think_region.center.numel():
        raise ValueError("Think-region hidden size differs from the latent model")
    model = model.to("cuda")
    model.eval()
    dense_parameter_count = count_dense_parameters(model)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    hidden_size = int(model.config.hidden_size)
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    num_hidden_layers = int(model.config.num_hidden_layers)
    num_attention_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_attention_heads))
    query_projection_size = num_attention_heads * head_dim
    eos_token_ids = normalize_token_ids(model.config.eos_token_id)
    if not eos_token_ids and tokenizer.eos_token_id is not None:
        eos_token_ids = {int(tokenizer.eos_token_id)}
    if not eos_token_ids:
        raise ValueError("Model and tokenizer do not define an EOS token")

    summary: dict[str, Any] = {
        "protocol": {
            "model_path": str(args.model_path),
            "tokenizer_path": str(args.tokenizer_path),
            "training_config_sources": protocol_sources,
            "sequence": "question + <think> + continuous latent blocks + </think> + answer",
            "c_thought": c_thought,
            "min_latent_blocks": args.min_latent_blocks,
            "max_latent_blocks": max_latent_blocks,
            "halt_boundary": "q95" if args.halt_threshold is None else "override",
            "halt_cosine_threshold": halt_threshold,
            "think_region_file": str(args.think_region_file),
            "answer_decoding": "greedy",
            "max_answer_tokens": args.max_new_tokens,
        },
        "model_compute": {
            "total_parameters": total_parameter_count,
            "dense_parameters_excluding_embeddings_and_lm_head": dense_parameter_count,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "head_dim": head_dim,
            "flops_scope": (
                "Per-question continuation FLOPs only: latent rollout, forced </think> "
                "boundary, and answer decoding. The initial question + <think> prefill "
                "is excluded, but remains in the KV-cache context for attention."
            ),
            "flops_estimate": (
                "Dense transformer matrix operations, cached attention score/value "
                "matmuls, and answer LM-head projections; excludes padding, "
                "normalization, activations, sampling, and hardware/kernel overhead."
            ),
        },
        "datasets": {},
    }
    overall_started_at = time.perf_counter()
    overall_correct = 0
    overall_total = 0
    overall_region_halts = 0
    overall_latent_blocks: list[int] = []
    overall_answer_tokens = 0
    overall_eos_answers = 0
    overall_logical_positions = 0
    overall_flops_excluding_prefill = 0

    for dataset_name in args.datasets:
        dataset_started_at = time.perf_counter()
        examples = load_math_examples(dataset_name, args.data_root, args.cache_dir)
        if args.max_samples is not None:
            examples = examples[: args.max_samples]

        output_path = args.output_dir / f"{dataset_name}.jsonl"
        correct = 0
        seen = 0
        region_halts = 0
        dataset_blocks: list[int] = []
        answer_token_count = 0
        eos_answer_count = 0
        logical_positions = 0
        dataset_flops_excluding_prefill = 0

        with output_path.open("w", encoding="utf-8") as handle:
            batches = batched(examples, args.batch_size)
            total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
            for batch in tqdm(batches, total=total_batches, desc=f"latent:{dataset_name}"):
                question_ids = [
                    tokenizer.encode(
                        example.question.strip() + "\n", add_special_tokens=True
                    )
                    for example in batch
                ]
                prompt_ids = [
                    ids + think_start_ids for ids in question_ids
                ]
                input_ids, attention_mask = left_pad_prompts(
                    prompt_ids, tokenizer.pad_token_id
                )
                generated = generate_latent_batch(
                    model,
                    input_ids=input_ids.to("cuda"),
                    attention_mask=attention_mask.to("cuda"),
                    think_end_ids=think_end_ids,
                    think_region_center=think_region.center,
                    halt_cosine_threshold=halt_threshold,
                    c_thought=c_thought,
                    min_latent_blocks=args.min_latent_blocks,
                    max_latent_blocks=max_latent_blocks,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )

                for row, example in enumerate(batch):
                    blocks = generated.latent_blocks[row]
                    latent_positions = blocks * c_thought
                    answer_ids = generated.answer_token_ids[row]
                    virtual_completion_ids = (
                        think_start_ids
                        + [latent_token_id] * latent_positions
                        + think_end_ids
                        + answer_ids
                    )
                    completion = tokenizer.decode(
                        virtual_completion_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    answer_text = extract_answer_after_think(completion)
                    prediction = (
                        extract_numeric_answer(answer_text)
                        if answer_text is not None
                        else None
                    )
                    is_correct = numeric_answers_equal(prediction, example.answer)
                    ended_with_eos = bool(answer_ids and answer_ids[-1] in eos_token_ids)
                    row_logical_positions = (
                        len(prompt_ids[row])
                        + latent_positions
                        + len(think_end_ids)
                        + max(0, len(answer_ids) - 1)
                    )
                    flops_excluding_prefill = estimate_latent_continuation_flops(
                        prompt_tokens=len(prompt_ids[row]),
                        latent_tokens=latent_positions,
                        boundary_tokens=len(think_end_ids),
                        answer_tokens=len(answer_ids),
                        dense_parameter_count=dense_parameter_count,
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        num_hidden_layers=num_hidden_layers,
                        query_projection_size=query_projection_size,
                    )

                    correct += int(is_correct)
                    seen += 1
                    region_halts += int(generated.halted_by_region[row])
                    dataset_blocks.append(blocks)
                    answer_token_count += len(answer_ids)
                    eos_answer_count += int(ended_with_eos)
                    logical_positions += row_logical_positions
                    dataset_flops_excluding_prefill += flops_excluding_prefill
                    write_result(
                        handle,
                        {
                            "question": example.question,
                            "gold": example.answer,
                            "prediction": None if prediction is None else str(prediction),
                            "correct": is_correct,
                            "answer_text": answer_text,
                            "completion": completion,
                            "latent_blocks": blocks,
                            "latent_positions": latent_positions,
                            "halted_by_region": generated.halted_by_region[row],
                            "stop_reason": (
                                "q95_region"
                                if generated.halted_by_region[row]
                                else "max_latent_blocks"
                            ),
                            "halt_score_trajectory": generated.halt_score_trajectories[row],
                            "terminal_halt_score": generated.halt_score_trajectories[row][-1],
                            "answer_tokens": len(answer_ids),
                            "answer_ended_with_eos": ended_with_eos,
                            "prefill_tokens": len(prompt_ids[row]),
                            "logical_sequence_positions": row_logical_positions,
                            "flops_excluding_prefill": flops_excluding_prefill,
                        },
                    )

        runtime_seconds = time.perf_counter() - dataset_started_at
        dataset_summary = _dataset_summary(
            correct=correct,
            total=seen,
            region_halts=region_halts,
            latent_blocks=dataset_blocks,
            answer_tokens=answer_token_count,
            eos_answers=eos_answer_count,
            logical_positions=logical_positions,
            flops_excluding_prefill=dataset_flops_excluding_prefill,
            runtime_seconds=runtime_seconds,
        )
        summary["datasets"][dataset_name] = dataset_summary
        overall_correct += correct
        overall_total += seen
        overall_region_halts += region_halts
        overall_latent_blocks.extend(dataset_blocks)
        overall_answer_tokens += answer_token_count
        overall_eos_answers += eos_answer_count
        overall_logical_positions += logical_positions
        overall_flops_excluding_prefill += dataset_flops_excluding_prefill
        print(
            f"{dataset_name}: accuracy {correct}/{seen} = {dataset_summary['accuracy']:.4%}; "
            f"region halt {region_halts}/{seen} = {dataset_summary['region_halt_rate']:.4%}; "
            f"latent blocks avg={dataset_summary['average_latent_blocks']:.2f}, "
            f"range=[{dataset_summary['min_latent_blocks']}, "
            f"{dataset_summary['max_latent_blocks']}], "
            f"post-prefill FLOPs/question="
            f"{dataset_summary['average_flops_per_question_excluding_prefill']:.0f}"
        )

    overall_runtime_seconds = time.perf_counter() - overall_started_at
    summary["overall"] = _dataset_summary(
        correct=overall_correct,
        total=overall_total,
        region_halts=overall_region_halts,
        latent_blocks=overall_latent_blocks,
        answer_tokens=overall_answer_tokens,
        eos_answers=overall_eos_answers,
        logical_positions=overall_logical_positions,
        flops_excluding_prefill=overall_flops_excluding_prefill,
        runtime_seconds=overall_runtime_seconds,
    )
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(f"Latent evaluation results written to {args.output_dir}")
    tracker.finish(
        status="completed",
        metrics={"evaluation": summary},
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
