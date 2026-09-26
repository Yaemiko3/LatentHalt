#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import (
    batched,
    extract_answer_after_think,
    extract_numeric_answer,
    load_math_examples,
    numeric_answers_equal,
)
from experiment_log import (
    ExperimentTracker,
    collect_accelerator_memory,
    local_accelerator_memory,
)
from train_sft_cot import add_think_special_tokens, configure_pad_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SUPPORTED_DATASETS = ("gsm8k", "gsm-hard", "multi-arith", "svamp")
_ACTIVE_TRACKER: ExperimentTracker | None = None


def normalize_token_ids(token_ids: int | Iterable[int] | None) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(token_id) for token_id in token_ids}


def count_generated_tokens(token_ids: Iterable[int], eos_token_ids: set[int]) -> int:
    """Count generated tokens through EOS, excluding padding added after EOS."""
    count = 0
    for token_id in token_ids:
        count += 1
        if int(token_id) in eos_token_ids:
            break
    return count


def estimate_generation_flops(
    *,
    prompt_tokens: int,
    output_tokens: int,
    dense_parameter_count: int,
    hidden_size: int,
    vocab_size: int,
    num_hidden_layers: int,
    query_projection_size: int,
) -> int:
    """Count generation continuation FLOPs after excluding prompt prefill.

    ``output_tokens`` includes the first token predicted from the prompt
    prefill.  That prediction and its prompt forward are excluded; only the
    remaining cached decode forwards and their LM-head projections are counted.
    """
    if min(prompt_tokens, output_tokens) < 0:
        raise ValueError("Token counts cannot be negative")
    decode_tokens = max(output_tokens - 1, 0)
    if decode_tokens == 0:
        return 0

    dense_flops = 2 * dense_parameter_count * decode_tokens
    decode_context = decode_tokens * prompt_tokens
    decode_context += decode_tokens * (decode_tokens + 1) // 2
    attention_flops = (
        4
        * num_hidden_layers
        * query_projection_size
        * decode_context
    )
    lm_head_flops = 2 * hidden_size * vocab_size * decode_tokens
    return dense_flops + attention_flops + lm_head_flops


def estimate_latent_continuation_flops(
    *,
    prompt_tokens: int,
    latent_tokens: int,
    boundary_tokens: int,
    answer_tokens: int,
    dense_parameter_count: int,
    hidden_size: int,
    vocab_size: int,
    num_hidden_layers: int,
    query_projection_size: int,
) -> int:
    """Count continuation FLOPs after prompt prefill for latent generation.

    The initial ``question + <think>`` forward is deliberately excluded.  Its
    length is still used as the KV-cache context for every subsequent latent,
    boundary, and answer forward.  ``answer_tokens`` includes the first token
    sampled from the boundary logits, so only ``answer_tokens - 1`` answer
    tokens require an additional backbone forward.
    """
    if min(prompt_tokens, latent_tokens, boundary_tokens, answer_tokens) < 0:
        raise ValueError("Token counts cannot be negative")
    # Every latent/boundary token and every answer token after the first uses a
    # cached single-token (or short boundary) transformer forward.
    backbone_tokens = latent_tokens + boundary_tokens + max(answer_tokens - 1, 0)
    dense_flops = 2 * dense_parameter_count * backbone_tokens

    # Attention score/value matmuls for cached forwards.  Each query attends to
    # the prompt, all prior continuation tokens, and itself.  The question
    # prefill's own prompt x prompt attention is not charged here.
    latent_context = (
        latent_tokens * prompt_tokens
        + latent_tokens * (latent_tokens + 1) // 2
    )
    # The forced boundary is passed as one short sequence, so its SDPA kernel
    # materializes the full query-by-key rectangle, matching the existing
    # prefill estimate's square-matrix convention.
    boundary_context = boundary_tokens * (
        prompt_tokens + latent_tokens + boundary_tokens
    )
    answer_decode_tokens = max(answer_tokens - 1, 0)
    answer_context = answer_decode_tokens * (
        prompt_tokens + latent_tokens + boundary_tokens
    )
    answer_context += answer_decode_tokens * (answer_decode_tokens + 1) // 2
    attention_flops = (
        4
        * num_hidden_layers
        * query_projection_size
        * (latent_context + boundary_context + answer_context)
    )

    # The output projection is evaluated once for each sampled answer token;
    # latent and boundary hidden states are not projected to vocabulary logits.
    lm_head_flops = 2 * hidden_size * vocab_size * answer_tokens
    return dense_flops + attention_flops + lm_head_flops


def count_dense_parameters(model: Any) -> int:
    """Count parameters used in dense transformer compute, excluding embeddings/LM head."""
    excluded_parameter_ids: set[int] = set()
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None:
            excluded_parameter_ids.update(id(parameter) for parameter in module.parameters())
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in excluded_parameter_ids
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the explicit-CoT SFT model.")
    parser.add_argument(
        "--model_path", type=Path, default=PROJECT_ROOT / "outputs" / "sft-cot-llama1b"
    )
    parser.add_argument(
        "--data_root", type=Path, default=WORKSPACE_ROOT / "datasets"
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--log_dir", type=Path, default=PROJECT_ROOT / "logs")
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
    parser.add_argument(
        "--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS)
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument(
        "--local_files_only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def write_result(handle: Any, result: dict[str, Any]) -> None:
    handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    handle.flush()


def main() -> None:
    global _ACTIVE_TRACKER
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(
        experiment_type="eval_sft_cot",
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
        raise RuntimeError("Evaluation requires a CUDA device")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --no-bf16")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        use_fast=True,
        padding_side="left",
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
    )
    configure_pad_token(tokenizer)
    add_think_special_tokens(tokenizer)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir),
        low_cpu_mem_usage=True,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
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

    summary: dict[str, Any] = {
        "model_compute": {
            "total_parameters": total_parameter_count,
            "dense_parameters_excluding_embeddings_and_lm_head": dense_parameter_count,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "head_dim": head_dim,
            "flops_scope": (
                "Per-question continuation FLOPs only: cached decode after the question "
                "prefill. The question prefill and first token prediction are excluded."
            ),
            "flops_estimate": (
                "Dense transformer matrix operations, cached attention score/value matmuls, "
                "and LM-head projections after prefill; excludes padding, normalization, "
                "activations, sampling, and hardware/kernel overhead."
            ),
            "output_token_count": (
                "Excludes the prompt, includes a terminating EOS, and excludes padding after EOS; "
                "the first token predicted by prompt prefill is excluded from FLOPs."
            ),
        },
        "datasets": {},
    }
    overall_started_at = time.perf_counter()
    overall_correct = 0
    overall_seen = 0
    overall_think_end_found = 0
    overall_prompt_tokens = 0
    overall_output_tokens = 0
    overall_flops_excluding_prefill = 0

    for dataset_name in args.datasets:
        dataset_started_at = time.perf_counter()
        examples = load_math_examples(dataset_name, args.data_root, args.cache_dir)
        if args.max_samples is not None:
            examples = examples[: args.max_samples]
        output_path = args.output_dir / f"{dataset_name}.jsonl"
        correct = 0
        think_end_found = 0
        seen = 0
        total_prompt_tokens = 0
        total_output_tokens = 0
        min_output_tokens: int | None = None
        max_output_tokens = 0
        dataset_flops_excluding_prefill = 0

        with output_path.open("w", encoding="utf-8") as handle:
            batches = batched(examples, args.batch_size)
            total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
            for batch in tqdm(batches, total=total_batches, desc=dataset_name):
                prompts = [example.question.strip() + "\n" for example in batch]
                tokenized = tokenizer(
                    prompts,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(
                        **tokenized,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=model.config.eos_token_id,
                        use_cache=True,
                    )
                completions = tokenizer.batch_decode(
                    generated[:, tokenized.input_ids.shape[1] :],
                    # </think> is an added special token and must remain visible
                    # so answer extraction cannot accidentally use rationale numbers.
                    skip_special_tokens=False,
                )
                completion_token_ids = generated[:, tokenized.input_ids.shape[1] :]
                prompt_token_counts = tokenized.attention_mask.sum(dim=1).tolist()

                for example, completion, output_ids, prompt_tokens in zip(
                    batch,
                    completions,
                    completion_token_ids,
                    prompt_token_counts,
                    strict=True,
                ):
                    output_tokens = count_generated_tokens(output_ids, eos_token_ids)
                    flops_excluding_prefill = estimate_generation_flops(
                        prompt_tokens=int(prompt_tokens),
                        output_tokens=output_tokens,
                        dense_parameter_count=dense_parameter_count,
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        num_hidden_layers=num_hidden_layers,
                        query_projection_size=query_projection_size,
                    )
                    answer_text = extract_answer_after_think(completion)
                    has_think_end = answer_text is not None
                    prediction = (
                        extract_numeric_answer(answer_text) if answer_text is not None else None
                    )
                    is_correct = numeric_answers_equal(prediction, example.answer)
                    correct += int(is_correct)
                    think_end_found += int(has_think_end)
                    seen += 1
                    total_prompt_tokens += int(prompt_tokens)
                    total_output_tokens += output_tokens
                    min_output_tokens = (
                        output_tokens
                        if min_output_tokens is None
                        else min(min_output_tokens, output_tokens)
                    )
                    max_output_tokens = max(max_output_tokens, output_tokens)
                    dataset_flops_excluding_prefill += flops_excluding_prefill
                    write_result(
                        handle,
                        {
                            "question": example.question,
                            "gold": example.answer,
                            "prediction": None if prediction is None else str(prediction),
                            "correct": is_correct,
                            "has_think_end": has_think_end,
                            "answer_text": answer_text,
                            "prompt_tokens": int(prompt_tokens),
                            "output_tokens": output_tokens,
                            "flops_excluding_prefill": flops_excluding_prefill,
                            "completion": completion,
                        },
                    )

        accuracy = correct / seen if seen else 0.0
        format_rate = think_end_found / seen if seen else 0.0
        runtime_seconds = time.perf_counter() - dataset_started_at
        dataset_summary = {
            "correct": correct,
            "total": seen,
            "accuracy": accuracy,
            "think_end_found": think_end_found,
            "format_rate": format_rate,
            "total_prompt_tokens": total_prompt_tokens,
            "average_prompt_tokens": total_prompt_tokens / seen if seen else 0.0,
            "total_output_tokens": total_output_tokens,
            "average_output_tokens": total_output_tokens / seen if seen else 0.0,
            "min_output_tokens": min_output_tokens or 0,
            "max_output_tokens": max_output_tokens,
            "total_flops_excluding_prefill": dataset_flops_excluding_prefill,
            "total_tera_flops_excluding_prefill": dataset_flops_excluding_prefill / 1e12,
            "average_flops_per_question_excluding_prefill": (
                dataset_flops_excluding_prefill / seen if seen else 0.0
            ),
            "runtime_seconds": runtime_seconds,
            "tera_flops_per_second_excluding_prefill": (
                dataset_flops_excluding_prefill / runtime_seconds / 1e12
                if runtime_seconds
                else 0.0
            ),
        }
        summary["datasets"][dataset_name] = dataset_summary
        overall_correct += correct
        overall_seen += seen
        overall_think_end_found += think_end_found
        overall_prompt_tokens += total_prompt_tokens
        overall_output_tokens += total_output_tokens
        overall_flops_excluding_prefill += dataset_flops_excluding_prefill
        print(
            f"{dataset_name}: accuracy {correct}/{seen} = {accuracy:.4%}; "
            f"found </think> {think_end_found}/{seen} = {format_rate:.4%}; "
            f"output tokens {total_output_tokens} (avg {dataset_summary['average_output_tokens']:.2f}); "
            f"continuation compute "
            f"{dataset_summary['total_tera_flops_excluding_prefill']:.3f} TFLOPs"
        )

    overall_runtime_seconds = time.perf_counter() - overall_started_at
    summary["overall"] = {
        "correct": overall_correct,
        "total": overall_seen,
        "accuracy": overall_correct / overall_seen if overall_seen else 0.0,
        "think_end_found": overall_think_end_found,
        "format_rate": (
            overall_think_end_found / overall_seen if overall_seen else 0.0
        ),
        "total_prompt_tokens": overall_prompt_tokens,
        "average_prompt_tokens": (
            overall_prompt_tokens / overall_seen if overall_seen else 0.0
        ),
        "total_output_tokens": overall_output_tokens,
        "average_output_tokens": (
            overall_output_tokens / overall_seen if overall_seen else 0.0
        ),
        "total_flops_excluding_prefill": overall_flops_excluding_prefill,
        "total_tera_flops_excluding_prefill": overall_flops_excluding_prefill / 1e12,
        "average_flops_per_question_excluding_prefill": (
            overall_flops_excluding_prefill / overall_seen if overall_seen else 0.0
        ),
        "runtime_seconds": overall_runtime_seconds,
        "tera_flops_per_second_excluding_prefill": (
            overall_flops_excluding_prefill / overall_runtime_seconds / 1e12
            if overall_runtime_seconds
            else 0.0
        ),
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(f"Results written to {args.output_dir}")
    tracker.finish(
        status="completed",
        metrics={"evaluation": summary},
        accelerator_memory=collect_accelerator_memory(),
    )
    _ACTIVE_TRACKER = None


if __name__ == "__main__":
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
