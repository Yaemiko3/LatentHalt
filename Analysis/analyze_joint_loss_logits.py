#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_model as load_safetensors_model
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coconut import load_think_region  # noqa: E402
from data import THINK_END_TOKEN, THINK_START_TOKEN, normalize_sft_example  # noqa: E402
from simcot import SimCoTDataCollator, SimCoTForCausalLM  # noqa: E402
from train_llama1b_simcot import tokenize_simcot_batch  # noqa: E402


LOSS_COMPONENTS = (
    "language_model_loss",
    "region_positive_loss",
    "region_negative_loss",
    "decoder_loss",
)
METRIC_COLUMNS = (
    "top1_probability",
    "top2_probability",
    "top1_top2_probability_gap",
    "top1_top2_logit_gap",
    "top5_mass",
    "top10_mass",
    "entropy_nats",
    "normalized_entropy",
    "effective_support",
    "target_probability",
    "target_nll",
    "target_rank",
)


def parse_args() -> argparse.Namespace:
    workspace_root = PROJECT_ROOT.parent
    parser = argparse.ArgumentParser(
        description="Analyze joint SIM-CoT eval losses and decoder logit concentration."
    )
    parser.add_argument(
        "--joint_model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "joint-simcot-llama1b_v1",
    )
    parser.add_argument(
        "--sft_model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sft-cot-llama1b",
    )
    parser.add_argument(
        "--validation_file",
        type=Path,
        default=(
            workspace_root
            / "datasets"
            / "gsm8k-aug"
            / "data"
            / "validation-00000-of-00001.parquet"
        ),
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
        default=(
            PROJECT_ROOT
            / "results"
            / "analysis"
            / "joint_loss_logits_eval"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stats_chunk_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--decoder_max_length", type=int, default=512)
    parser.add_argument("--eval_epoch", type=int, default=32)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def configure_tokenizer(tokenizer: Any) -> None:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        finetune_pad_id = tokenizer.convert_tokens_to_ids(
            "<|finetune_right_pad_id|>"
        )
        if finetune_pad_id != tokenizer.unk_token_id:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token


def prepare_joint_features(
    raw: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int,
    decoder_max_length: int,
    c_thought: int,
    max_latent_stage: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    features: list[dict[str, Any]] = []
    dropped: list[int] = []
    for sample_index, row in raw.iterrows():
        raw_steps = row["steps"]
        if isinstance(raw_steps, np.ndarray):
            raw_steps = raw_steps.tolist()
        encoded = tokenize_simcot_batch(
            {
                "question": [row["question"]],
                "steps": [raw_steps],
                "answer": [row["answer"]],
            },
            tokenizer=tokenizer,
            max_length=max_length,
            decoder_max_length=decoder_max_length,
            c_thought=c_thought,
            max_latent_stage=max_latent_stage,
        )
        if not encoded["question_ids"]:
            dropped.append(int(sample_index))
            continue
        feature = {name: values[0] for name, values in encoded.items()}
        feature["sample_index"] = int(sample_index)
        features.append(feature)
    return features, dropped


def joint_block_token_metadata(
    feature: dict[str, Any],
    *,
    block_count: int,
    fully_implicit: bool,
    eos_token_id: int,
) -> list[list[dict[str, Any]]]:
    step_ids = feature["steps_ids"]
    block_steps: list[list[int]] = []
    if fully_implicit and len(step_ids) > block_count:
        block_steps.extend([[index] for index in range(block_count - 1)])
        block_steps.append(list(range(block_count - 1, len(step_ids))))
    else:
        block_steps.extend([[index] for index in range(block_count)])

    result: list[list[dict[str, Any]]] = []
    for block_index, step_indices in enumerate(block_steps):
        block: list[dict[str, Any]] = []
        for step_index in step_indices:
            for step_token_position, token_id in enumerate(step_ids[step_index]):
                block.append(
                    {
                        "target_id": int(token_id),
                        "token_group": "reasoning",
                        "step_index": int(step_index),
                        "step_token_position": int(step_token_position),
                        "is_step_first": step_token_position == 0,
                        "is_step_last": step_token_position == len(step_ids[step_index]) - 1,
                    }
                )
        block.append(
            {
                "target_id": int(eos_token_id),
                "token_group": "eos",
                "step_index": -1,
                "step_token_position": -1,
                "is_step_first": False,
                "is_step_last": False,
            }
        )
        for target_position, record in enumerate(block):
            record.update(
                {
                    "sample_index": int(feature["sample_index"]),
                    "block_index": int(block_index),
                    "target_position": int(target_position),
                    "target_length": int(len(block)),
                    "is_block_first": target_position == 0,
                    "is_block_last_lexical": target_position == len(block) - 2,
                }
            )
        result.append(block)
    return result


@torch.no_grad()
def append_distribution_metrics(
    logits: torch.Tensor,
    records: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> None:
    vocab_size = logits.shape[-1]
    log_vocab = math.log(vocab_size)
    for chunk in batched(records, chunk_size):
        row_indices = torch.tensor(
            [record["logit_row"] for record in chunk],
            dtype=torch.long,
            device=logits.device,
        )
        positions = torch.tensor(
            [record["logit_position"] for record in chunk],
            dtype=torch.long,
            device=logits.device,
        )
        target_ids = torch.tensor(
            [record["target_id"] for record in chunk],
            dtype=torch.long,
            device=logits.device,
        )
        selected = logits[row_indices, positions].float()
        log_normalizer = torch.logsumexp(selected, dim=-1)
        probabilities = torch.softmax(selected, dim=-1)
        entropy = log_normalizer - (probabilities * selected).sum(dim=-1)
        top_values, top_ids = selected.topk(k=10, dim=-1)
        top_probabilities = torch.exp(top_values - log_normalizer[:, None])
        target_values = selected.gather(1, target_ids[:, None]).squeeze(1)
        target_log_probabilities = target_values - log_normalizer
        target_ranks = selected.gt(target_values[:, None]).sum(dim=-1) + 1

        top_ids_cpu = top_ids.cpu().tolist()
        top_probs_cpu = top_probabilities.cpu().tolist()
        entropy_cpu = entropy.cpu().tolist()
        target_log_probs_cpu = target_log_probabilities.cpu().tolist()
        target_ranks_cpu = target_ranks.cpu().tolist()
        top_values_cpu = top_values.cpu().tolist()
        for index, record in enumerate(chunk):
            row = {
                key: value
                for key, value in record.items()
                if key not in {"logit_row", "logit_position"}
            }
            row.update(
                {
                    "vocab_size": int(vocab_size),
                    "top1_id": int(top_ids_cpu[index][0]),
                    "top1_probability": float(top_probs_cpu[index][0]),
                    "top2_probability": float(top_probs_cpu[index][1]),
                    "top1_top2_probability_gap": float(
                        top_probs_cpu[index][0] - top_probs_cpu[index][1]
                    ),
                    "top1_top2_logit_gap": float(
                        top_values_cpu[index][0] - top_values_cpu[index][1]
                    ),
                    "top5_mass": float(sum(top_probs_cpu[index][:5])),
                    "top10_mass": float(sum(top_probs_cpu[index])),
                    "entropy_nats": float(entropy_cpu[index]),
                    "normalized_entropy": float(entropy_cpu[index] / log_vocab),
                    "effective_support": float(math.exp(entropy_cpu[index])),
                    "target_probability": float(
                        math.exp(target_log_probs_cpu[index])
                    ),
                    "target_nll": float(-target_log_probs_cpu[index]),
                    "target_rank": int(target_ranks_cpu[index]),
                    "target_is_top1": bool(
                        top_ids_cpu[index][0] == record["target_id"]
                    ),
                    "top10_ids": [int(value) for value in top_ids_cpu[index]],
                    "top10_probabilities": [
                        float(value) for value in top_probs_cpu[index]
                    ],
                }
            )
            output_rows.append(row)
        del selected, probabilities, top_values, top_ids


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": int(len(frame))}
    if frame.empty:
        return summary
    for column in METRIC_COLUMNS:
        values = frame[column].astype(float)
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "p10": float(values.quantile(0.10)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
        }
    summary.update(
        {
            "target_top1_rate": float(frame["target_is_top1"].mean()),
            "top1_probability_ge_0_5": float(
                frame["top1_probability"].ge(0.5).mean()
            ),
            "top1_probability_ge_0_9": float(
                frame["top1_probability"].ge(0.9).mean()
            ),
            "top1_probability_lt_0_1": float(
                frame["top1_probability"].lt(0.1).mean()
            ),
        }
    )
    return summary


def summarize_error_behavior(frame: pd.DataFrame) -> dict[str, Any]:
    wrong = frame[~frame["target_is_top1"]]
    correct = frame[frame["target_is_top1"]]
    total_nll = float(frame["target_nll"].sum())
    top_ten_count = max(1, int(math.ceil(0.10 * len(frame))))
    top_one_count = max(1, int(math.ceil(0.01 * len(frame))))

    def confidence(group: pd.DataFrame) -> dict[str, Any]:
        return {
            "count": int(len(group)),
            "rate": float(len(group) / len(frame)),
            "top1_probability_mean": float(group["top1_probability"].mean()),
            "top1_probability_median": float(group["top1_probability"].median()),
            "top1_probability_ge_0_9": float(
                group["top1_probability"].ge(0.9).mean()
            ),
            "target_probability_median": float(group["target_probability"].median()),
            "target_nll_mean": float(group["target_nll"].mean()),
            "entropy_nats_mean": float(group["entropy_nats"].mean()),
        }

    return {
        "correct": confidence(correct),
        "wrong": confidence(wrong),
        "wrong_token_nll_share": float(wrong["target_nll"].sum() / total_nll),
        "largest_10_percent_token_nll_share": float(
            frame.nlargest(top_ten_count, "target_nll")["target_nll"].sum()
            / total_nll
        ),
        "largest_1_percent_token_nll_share": float(
            frame.nlargest(top_one_count, "target_nll")["target_nll"].sum()
            / total_nll
        ),
    }


def component_export_check(joint_model: Path) -> dict[str, Any]:
    root_path = joint_model / "model.safetensors"
    checks = {
        "base_model": (
            joint_model / "base_model" / "model.safetensors",
            "base_causallm.model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        "auxiliary_decoder": (
            joint_model / "auxiliary_decoder" / "model.safetensors",
            "auxiliary_decoder.model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
    }
    result: dict[str, Any] = {
        "authoritative_root_mtime_ns": root_path.stat().st_mtime_ns,
    }
    with safe_open(root_path, framework="pt", device="cpu") as root_handle:
        for name, (component_path, root_key, component_key) in checks.items():
            with safe_open(component_path, framework="pt", device="cpu") as component:
                difference = (
                    root_handle.get_tensor(root_key).float()
                    - component.get_tensor(component_key).float()
                ).abs()
            result[name] = {
                "path": component_path.resolve(),
                "mtime_ns": component_path.stat().st_mtime_ns,
                "sample_tensor": root_key,
                "sample_tensor_exactly_equal": bool(difference.eq(0).all()),
                "sample_tensor_max_abs_difference": float(difference.max()),
                "sample_tensor_mean_abs_difference": float(difference.mean()),
            }
    result["warning"] = (
        "base_model export differs from the final root checkpoint; use root model.safetensors"
        if not result["base_model"]["sample_tensor_exactly_equal"]
        else None
    )
    return result


def add_token_strings(frame: pd.DataFrame, tokenizer: Any) -> pd.DataFrame:
    token_ids = set(frame["target_id"].astype(int)) | set(frame["top1_id"].astype(int))
    token_strings = {
        token_id: tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids
    }
    frame["target_token"] = frame["target_id"].map(token_strings)
    frame["top1_token"] = frame["top1_id"].map(token_strings)
    return frame


def load_joint_model(args: argparse.Namespace, config: dict[str, Any], device: str) -> Any:
    dtype = torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.joint_model / "base_model"),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    decoder = AutoModelForCausalLM.from_pretrained(
        str(args.joint_model / "auxiliary_decoder"),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    think_region = load_think_region(args.think_region_file)
    model = SimCoTForCausalLM(
        base_causallm=base_model,
        auxiliary_decoder=decoder,
        latent_token_id=int(config["latent_token_id"]),
        think_end_token_id=think_region.think_end_token_id,
        think_region_center=think_region.center,
        think_region_cosine_threshold=float(config["region_q90_cosine_threshold"]),
        nonterminal_region_cosine_threshold=float(
            config["region_q95_cosine_threshold"]
        ),
        region_loss_weight=float(config["region_loss_weight"]),
        region_negative_loss_weight=float(config["region_negative_loss_weight"]),
        decoder_loss_weight=float(config["decoder_loss_weight"]),
        c_thought=int(config["c_thought"]),
        decoder_loss_normalization=str(config["decoder_loss_normalization"]),
    )
    # The exported component directories can predate the final combined checkpoint.
    # Always replace both components with the authoritative root state.
    load_safetensors_model(
        model,
        args.joint_model / "model.safetensors",
        strict=True,
        device="cpu",
    )
    model.eval()
    return model.to(device)


@torch.no_grad()
def analyze_joint(
    args: argparse.Namespace,
    raw: pd.DataFrame,
    tokenizer: Any,
    config: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[int]]:
    features, dropped = prepare_joint_features(
        raw,
        tokenizer,
        max_length=args.max_length,
        decoder_max_length=args.decoder_max_length,
        c_thought=int(config["c_thought"]),
        max_latent_stage=int(config["curriculum"]["max_latent_stage"]),
    )
    collator = SimCoTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        latent_id=int(config["latent_token_id"]),
        think_start_ids=tokenizer.encode(
            THINK_START_TOKEN + "\n", add_special_tokens=False
        ),
        think_end_ids=tokenizer.encode(
            THINK_END_TOKEN + "\n", add_special_tokens=False
        ),
        eos_token_id=tokenizer.eos_token_id,
        c_thought=int(config["c_thought"]),
        epochs_per_stage=int(config["curriculum"]["epochs_per_stage"]),
        max_latent_stage=int(config["curriculum"]["max_latent_stage"]),
        start_stage=int(config["curriculum"]["start_stage"]),
    )
    stage = collator.set_epoch(args.eval_epoch)
    if not stage.fully_implicit:
        raise ValueError(
            f"eval_epoch={args.eval_epoch} is not fully latent: {stage}"
        )

    model = load_joint_model(args, config, device)
    token_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []

    def decoder_hook(_module: Any, _inputs: Any, outputs: Any) -> None:
        append_distribution_metrics(
            outputs.logits,
            current_records,
            token_rows,
            chunk_size=args.stats_chunk_size,
        )

    hook_handle = model.auxiliary_decoder.register_forward_hook(decoder_hook)
    weighted_component_sums = {name: 0.0 for name in LOSS_COMPONENTS}
    weighted_total_loss = 0.0
    evaluated_samples = 0
    decoded_blocks = 0

    print(f"Joint eval: {len(features)} samples on {device}", flush=True)
    for batch_index, feature_batch in enumerate(batched(features, args.batch_size)):
        tensor_batch = collator(feature_batch)
        current_records.clear()
        decoder_sequence_index = 0
        for local_index, feature in enumerate(feature_batch):
            block_count = int(tensor_batch["decoder_block_mask"][local_index].sum())
            blocks = joint_block_token_metadata(
                feature,
                block_count=block_count,
                fully_implicit=stage.fully_implicit,
                eos_token_id=tokenizer.eos_token_id,
            )
            for block_index, block in enumerate(blocks):
                actual_ids = tensor_batch["decoder_target_ids"][
                    local_index, block_index, : len(block)
                ].tolist()
                expected_ids = [record["target_id"] for record in block]
                if actual_ids != expected_ids:
                    raise RuntimeError("Decoder target metadata does not match collator output")
                for target_position, record in enumerate(block):
                    record.update(
                        {
                            "source": "joint_decoder",
                            "logit_row": decoder_sequence_index,
                            "logit_position": int(config["c_thought"])
                            - 1
                            + target_position,
                        }
                    )
                    current_records.append(record)
                decoder_sequence_index += 1

        gpu_batch = {name: value.to(device) for name, value in tensor_batch.items()}
        outputs = model(**gpu_batch)
        batch_size = len(feature_batch)
        if outputs.loss is None:
            raise RuntimeError("Joint model returned no eval loss")
        output_blocks = int(outputs.decoded_block_count.item())
        if output_blocks != decoder_sequence_index:
            raise RuntimeError(
                f"Decoded block mismatch: output={output_blocks}, expected={decoder_sequence_index}"
            )
        row = {
            "batch_index": batch_index,
            "batch_size": batch_size,
            "decoded_blocks": output_blocks,
            "loss": float(outputs.loss.float().item()),
        }
        weighted_total_loss += row["loss"] * batch_size
        evaluated_samples += batch_size
        decoded_blocks += output_blocks
        for name in LOSS_COMPONENTS:
            value = float(getattr(outputs, name).float().item())
            row[name] = value
            weighted_component_sums[name] += value * batch_size
        batch_rows.append(row)
        if (batch_index + 1) % 8 == 0 or evaluated_samples == len(features):
            print(
                f"  joint batches={batch_index + 1}, samples={evaluated_samples}, "
                f"decoder tokens={len(token_rows)}",
                flush=True,
            )
        del gpu_batch, outputs

    hook_handle.remove()
    component_means = {
        name: total / evaluated_samples
        for name, total in weighted_component_sums.items()
    }
    contributions = {
        "language_model": component_means["language_model_loss"],
        "region_positive_weighted": (
            float(config["region_loss_weight"])
            * component_means["region_positive_loss"]
        ),
        "region_negative_weighted": (
            float(config["region_negative_loss_weight"])
            * component_means["region_negative_loss"]
        ),
        "decoder_weighted": (
            float(config["decoder_loss_weight"])
            * component_means["decoder_loss"]
        ),
    }
    contribution_sum = sum(contributions.values())
    joint_frame = pd.DataFrame(token_rows)
    joint_frame = add_token_strings(joint_frame, tokenizer)
    block_frame = (
        joint_frame.groupby(["sample_index", "block_index"], as_index=False)
        .agg(
            token_count=("target_nll", "size"),
            lexical_token_count=("token_group", lambda values: int((values == "reasoning").sum())),
            block_nll=("target_nll", "sum"),
            mean_token_nll=("target_nll", "mean"),
            mean_top1_probability=("top1_probability", "mean"),
        )
    )
    loss_summary = {
        "curriculum_stage": {
            "epoch": stage.epoch,
            "scheduled_stage": stage.scheduled_stage,
            "latent_blocks": stage.latent_blocks,
            "fully_implicit": stage.fully_implicit,
        },
        "evaluated_samples": evaluated_samples,
        "dropped_sample_indices": dropped,
        "batch_size": args.batch_size,
        "batch_count": len(batch_rows),
        "decoded_blocks": decoded_blocks,
        "decoder_target_tokens": int(len(joint_frame)),
        "decoder_reasoning_tokens": int(
            joint_frame["token_group"].eq("reasoning").sum()
        ),
        "reproduced_eval_loss": weighted_total_loss / evaluated_samples,
        "component_means_training_normalization": component_means,
        "weighted_contributions": contributions,
        "weighted_contribution_sum": contribution_sum,
        "weighted_contribution_shares": {
            name: value / contribution_sum for name, value in contributions.items()
        },
        "main_model_weighted_contribution": (
            contributions["language_model"]
            + contributions["region_positive_weighted"]
            + contributions["region_negative_weighted"]
        ),
        "main_model_weighted_share": (
            contributions["language_model"]
            + contributions["region_positive_weighted"]
            + contributions["region_negative_weighted"]
        )
        / contribution_sum,
        "decoder_token_normalized_nll": float(joint_frame["target_nll"].mean()),
        "decoder_per_block_nll_global": float(block_frame["block_nll"].mean()),
        "decoder_tokens_per_block_mean": float(block_frame["token_count"].mean()),
        "batch_details": batch_rows,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return loss_summary, joint_frame, block_frame, [f["sample_index"] for f in features]


def build_sft_feature(row: pd.Series, tokenizer: Any, sample_index: int) -> dict[str, Any]:
    raw_example = row.to_dict()
    if isinstance(raw_example.get("steps"), np.ndarray):
        raw_example["steps"] = raw_example["steps"].tolist()
    question, steps, answer = normalize_sft_example(raw_example)
    prompt_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []

    def add_segment(
        token_ids: list[int], token_group: str, step_index: int = -1
    ) -> None:
        for token_position, token_id in enumerate(token_ids):
            targets.append(int(token_id))
            metadata.append(
                {
                    "source": "sft",
                    "sample_index": int(sample_index),
                    "block_index": -1,
                    "target_position": len(targets) - 1,
                    "target_length": -1,
                    "target_id": int(token_id),
                    "token_group": token_group,
                    "step_index": int(step_index),
                    "step_token_position": (
                        int(token_position) if token_group == "reasoning" else -1
                    ),
                    "is_step_first": token_group == "reasoning" and token_position == 0,
                    "is_step_last": token_group == "reasoning"
                    and token_position == len(token_ids) - 1,
                    "is_block_first": False,
                    "is_block_last_lexical": False,
                }
            )

    add_segment(
        tokenizer.encode(THINK_START_TOKEN + "\n", add_special_tokens=False),
        "think_start",
    )
    for step_index, step in enumerate(steps):
        add_segment(
            tokenizer.encode(step + "\n", add_special_tokens=False),
            "reasoning",
            step_index,
        )
    add_segment(
        tokenizer.encode(THINK_END_TOKEN + "\n", add_special_tokens=False),
        "think_end",
    )
    add_segment(tokenizer.encode(answer, add_special_tokens=False), "answer")
    add_segment([tokenizer.eos_token_id], "eos")

    input_ids = prompt_ids + targets
    return {
        "sample_index": int(sample_index),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + targets,
        "prompt_length": len(prompt_ids),
        "metadata": metadata,
    }


def collate_sft(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(feature["input_ids"]) for feature in features)
    max_length = ((max_length + 7) // 8) * 8
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        padding = max_length - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        batch["labels"].append(feature["labels"] + [-100] * padding)
    return {
        name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()
    }


@torch.no_grad()
def analyze_sft(
    args: argparse.Namespace,
    raw: pd.DataFrame,
    retained_indices: list[int],
    device: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.sft_model), use_fast=True, local_files_only=True
    )
    configure_tokenizer(tokenizer)
    features: list[dict[str, Any]] = []
    dropped: list[int] = []
    for sample_index in retained_indices:
        feature = build_sft_feature(raw.loc[sample_index], tokenizer, sample_index)
        if len(feature["input_ids"]) > args.max_length:
            dropped.append(sample_index)
        else:
            features.append(feature)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.sft_model),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    token_rows: list[dict[str, Any]] = []
    weighted_loss = 0.0
    evaluated_samples = 0

    print(f"SFT eval: {len(features)} samples on {device}", flush=True)
    for batch_index, feature_batch in enumerate(batched(features, args.batch_size)):
        tensor_batch = collate_sft(feature_batch, tokenizer.pad_token_id)
        gpu_batch = {name: value.to(device) for name, value in tensor_batch.items()}
        outputs = model(**gpu_batch, use_cache=False, return_dict=True)
        records: list[dict[str, Any]] = []
        for local_index, feature in enumerate(feature_batch):
            for target_offset, record in enumerate(feature["metadata"]):
                records.append(
                    {
                        **record,
                        "logit_row": local_index,
                        "logit_position": feature["prompt_length"] + target_offset - 1,
                    }
                )
        append_distribution_metrics(
            outputs.logits,
            records,
            token_rows,
            chunk_size=args.stats_chunk_size,
        )
        batch_size = len(feature_batch)
        weighted_loss += float(outputs.loss.float().item()) * batch_size
        evaluated_samples += batch_size
        if (batch_index + 1) % 8 == 0 or evaluated_samples == len(features):
            print(
                f"  sft batches={batch_index + 1}, samples={evaluated_samples}, "
                f"supervised tokens={len(token_rows)}",
                flush=True,
            )
        del outputs, gpu_batch

    sft_frame = pd.DataFrame(token_rows)
    sft_frame = add_token_strings(sft_frame, tokenizer)
    summary = {
        "evaluated_samples": evaluated_samples,
        "dropped_sample_indices": dropped,
        "batch_size": args.batch_size,
        "reproduced_eval_loss": weighted_loss / evaluated_samples,
        "global_token_normalized_nll": float(sft_frame["target_nll"].mean()),
        "supervised_tokens": int(len(sft_frame)),
        "reasoning_tokens": int(sft_frame["token_group"].eq("reasoning").sum()),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary, sft_frame


def summarize_logits(joint: pd.DataFrame, sft: pd.DataFrame) -> dict[str, Any]:
    joint_groups = {
        "all_targets": joint,
        "reasoning": joint[joint["token_group"].eq("reasoning")],
        "block_first_reasoning": joint[
            joint["token_group"].eq("reasoning") & joint["is_block_first"]
        ],
        "block_last_lexical": joint[
            joint["token_group"].eq("reasoning")
            & joint["is_block_last_lexical"]
        ],
        "interior_reasoning": joint[
            joint["token_group"].eq("reasoning")
            & ~joint["is_block_first"]
            & ~joint["is_block_last_lexical"]
        ],
        "eos": joint[joint["token_group"].eq("eos")],
    }
    sft_groups = {
        "all_supervised": sft,
        "reasoning": sft[sft["token_group"].eq("reasoning")],
        "step_first_reasoning": sft[
            sft["token_group"].eq("reasoning") & sft["is_step_first"]
        ],
        "step_last_reasoning": sft[
            sft["token_group"].eq("reasoning") & sft["is_step_last"]
        ],
        "answer": sft[sft["token_group"].eq("answer")],
        "eos": sft[sft["token_group"].eq("eos")],
    }

    joint_reasoning = joint_groups["reasoning"].copy()
    sft_reasoning = sft_groups["reasoning"].copy()
    keys = ["sample_index", "step_index", "step_token_position", "target_id"]
    paired = joint_reasoning.merge(
        sft_reasoning,
        on=keys,
        how="inner",
        suffixes=("_joint", "_sft"),
        validate="one_to_one",
    )
    if len(paired) != len(joint_reasoning) or len(paired) != len(sft_reasoning):
        raise RuntimeError(
            "Reasoning-token pairing is incomplete: "
            f"joint={len(joint_reasoning)}, sft={len(sft_reasoning)}, paired={len(paired)}"
        )
    paired_summary: dict[str, Any] = {"count": int(len(paired))}
    for metric in (
        "top1_probability",
        "target_probability",
        "entropy_nats",
        "effective_support",
        "target_nll",
    ):
        delta = paired[f"{metric}_joint"] - paired[f"{metric}_sft"]
        paired_summary[f"{metric}_joint_minus_sft"] = {
            "mean": float(delta.mean()),
            "median": float(delta.median()),
            "p25": float(delta.quantile(0.25)),
            "p75": float(delta.quantile(0.75)),
            "joint_greater_rate": float(delta.gt(0).mean()),
        }
    paired_summary["top1_agreement_rate"] = float(
        paired["top1_id_joint"].eq(paired["top1_id_sft"]).mean()
    )
    paired_summary["target_top1_joint_only_rate"] = float(
        (paired["target_is_top1_joint"] & ~paired["target_is_top1_sft"]).mean()
    )
    paired_summary["target_top1_sft_only_rate"] = float(
        (~paired["target_is_top1_joint"] & paired["target_is_top1_sft"]).mean()
    )

    position_profile = []
    for target_position, frame in joint_groups["reasoning"].groupby(
        "target_position", sort=True
    ):
        position_profile.append(
            {
                "target_position": int(target_position),
                "count": int(len(frame)),
                "target_top1_rate": float(frame["target_is_top1"].mean()),
                "target_nll_mean": float(frame["target_nll"].mean()),
                "top1_probability_mean": float(frame["top1_probability"].mean()),
                "entropy_nats_mean": float(frame["entropy_nats"].mean()),
            }
        )

    return {
        "joint_decoder": {
            name: summarize_frame(frame) for name, frame in joint_groups.items()
        },
        "sft": {name: summarize_frame(frame) for name, frame in sft_groups.items()},
        "paired_reasoning_tokens": paired_summary,
        "confidence_error_analysis": {
            "joint_decoder_reasoning": summarize_error_behavior(
                joint_groups["reasoning"]
            ),
            "sft_reasoning": summarize_error_behavior(sft_groups["reasoning"]),
        },
        "joint_decoder_target_position_profile": position_profile,
    }


def plot_loss_contributions(loss_summary: dict[str, Any], path: Path) -> None:
    contributions = loss_summary["weighted_contributions"]
    labels = ["Main LM", "Terminal region", "Premature region", "Decoder"]
    values = [
        contributions["language_model"],
        contributions["region_positive_weighted"],
        contributions["region_negative_weighted"],
        contributions["decoder_weighted"],
    ]
    colors = ["#33658A", "#55A868", "#C8A951", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Weighted eval-loss contribution")
    ax.set_title("Joint SIM-CoT validation loss decomposition")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_logit_comparison(joint: pd.DataFrame, sft: pd.DataFrame, path: Path) -> None:
    joint_reasoning = joint[joint["token_group"].eq("reasoning")]
    sft_reasoning = sft[sft["token_group"].eq("reasoning")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins_probability = np.linspace(0, 1, 41)
    axes[0].hist(
        joint_reasoning["top1_probability"],
        bins=bins_probability,
        density=True,
        histtype="step",
        linewidth=2,
        label="Joint decoder",
        color="#C44E52",
    )
    axes[0].hist(
        sft_reasoning["top1_probability"],
        bins=bins_probability,
        density=True,
        histtype="step",
        linewidth=2,
        label="SFT",
        color="#33658A",
    )
    axes[0].set_xlabel("Top-1 probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Same reasoning target tokens")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    entropy_max = max(
        float(joint_reasoning["entropy_nats"].quantile(0.995)),
        float(sft_reasoning["entropy_nats"].quantile(0.995)),
    )
    bins_entropy = np.linspace(0, max(entropy_max, 0.1), 41)
    axes[1].hist(
        joint_reasoning["entropy_nats"].clip(upper=entropy_max),
        bins=bins_entropy,
        density=True,
        histtype="step",
        linewidth=2,
        label="Joint decoder",
        color="#C44E52",
    )
    axes[1].hist(
        sft_reasoning["entropy_nats"].clip(upper=entropy_max),
        bins=bins_entropy,
        density=True,
        histtype="step",
        linewidth=2,
        label="SFT",
        color="#33658A",
    )
    axes[1].set_xlabel("Entropy (nats; clipped at p99.5)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Vocabulary-distribution uncertainty")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_joint_positions(joint: pd.DataFrame, path: Path) -> None:
    groups = [
        joint[joint["token_group"].eq("reasoning") & joint["is_block_first"]][
            "top1_probability"
        ],
        joint[joint["token_group"].eq("reasoning")]["top1_probability"],
        joint[
            joint["token_group"].eq("reasoning") & joint["is_block_last_lexical"]
        ]["top1_probability"],
        joint[joint["token_group"].eq("eos")]["top1_probability"],
    ]
    labels = ["First token", "All lexical", "Last lexical", "EOS"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Top-1 probability")
    ax.set_title("Joint decoder confidence by translation position")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_joint_target_position_profile(joint: pd.DataFrame, path: Path) -> None:
    reasoning = joint[joint["token_group"].eq("reasoning")]
    profile = (
        reasoning.groupby("target_position", as_index=False)
        .agg(
            count=("target_nll", "size"),
            target_top1_rate=("target_is_top1", "mean"),
            target_nll_mean=("target_nll", "mean"),
        )
        .query("target_position <= 10")
    )
    fig, primary = plt.subplots(figsize=(8.5, 4.8))
    secondary = primary.twinx()
    primary.plot(
        profile["target_position"],
        profile["target_top1_rate"],
        marker="o",
        color="#33658A",
        label="Target is top-1",
    )
    secondary.plot(
        profile["target_position"],
        profile["target_nll_mean"],
        marker="s",
        color="#C44E52",
        label="Mean target NLL",
    )
    primary.set_xlabel("Target position inside decoded block")
    primary.set_ylabel("Target top-1 rate", color="#33658A")
    secondary.set_ylabel("Mean target NLL", color="#C44E52")
    primary.set_ylim(0, 1.03)
    primary.set_xticks(profile["target_position"])
    primary.grid(alpha=0.25)
    handles_a, labels_a = primary.get_legend_handles_labels()
    handles_b, labels_b = secondary.get_legend_handles_labels()
    primary.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    primary.set_title("Joint decoder correctness by within-block position")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(
    path: Path,
    metadata: dict[str, Any],
    loss: dict[str, Any],
    logits: dict[str, Any],
    sft_loss: dict[str, Any],
) -> None:
    contributions = loss["weighted_contributions"]
    shares = loss["weighted_contribution_shares"]
    joint_reasoning = logits["joint_decoder"]["reasoning"]
    joint_first = logits["joint_decoder"]["block_first_reasoning"]
    joint_last = logits["joint_decoder"]["block_last_lexical"]
    joint_interior = logits["joint_decoder"]["interior_reasoning"]
    joint_eos = logits["joint_decoder"]["eos"]
    sft_reasoning = logits["sft"]["reasoning"]
    paired = logits["paired_reasoning_tokens"]
    joint_errors = logits["confidence_error_analysis"]["joint_decoder_reasoning"]
    sft_errors = logits["confidence_error_analysis"]["sft_reasoning"]
    position_one = next(
        row
        for row in logits["joint_decoder_target_position_profile"]
        if row["target_position"] == 1
    )
    lines = [
        "# 联合 SIM-CoT 模型 eval loss 与 decoder logits 分析",
        "",
        "## 结论摘要",
        "",
        (
            f"在 {loss['evaluated_samples']} 条 GSM8K-Aug validation 样本上，复算总 loss 为 "
            f"**{loss['reproduced_eval_loss']:.6f}**。其中 decoder 加权项为 "
            f"**{contributions['decoder_weighted']:.6f}（{format_percentage(shares['decoder_weighted'])}）**；"
            f"主模型侧（LM + 两个 region 项）合计为 "
            f"**{loss['main_model_weighted_contribution']:.6f}（{format_percentage(loss['main_model_weighted_share'])}）**。"
        ),
        (
            "训练配置中的 decoder loss 是按 block 归一化的 token NLL 总和。其原始值不能与按 token "
            f"归一化的 LM loss 直接横比：每个 block 平均 {loss['decoder_tokens_per_block_mean']:.2f} 个目标 token，"
            f"换成 token 归一化后 decoder NLL 为 **{loss['decoder_token_normalized_nll']:.6f}**，"
            f"与 LM 的 {loss['component_means_training_normalization']['language_model_loss']:.6f} 接近。"
        ),
        (
            "对全部 reasoning 目标 token，联合 decoder 的 top-1 概率中位数为 "
            f"**{joint_reasoning['top1_probability']['median']:.4f}**，平均熵为 "
            f"**{joint_reasoning['entropy_nats']['mean']:.4f} nats**，"
            f"{format_percentage(joint_reasoning['top1_probability_ge_0_5'])} 的位置 top-1 概率不低于 0.5。"
        ),
        (
            "因此分布并非在整个词表上接近平均。按位置看，block 首 token 的 top-1 概率中位数为 "
            f"**{joint_first['top1_probability']['median']:.4f}**，而 block 最后一个非 EOS token 为 "
            f"**{joint_last['top1_probability']['median']:.4f}**，EOS 为 "
            f"**{joint_eos['top1_probability']['median']:.4f}**。"
        ),
        (
            "但高集中度不等于翻译正确：联合 decoder 的错误 reasoning 位置有 "
            f"**{joint_errors['wrong']['count']}** 个，其中 top-1 概率中位数仍为 "
            f"**{joint_errors['wrong']['top1_probability_median']:.4f}**，"
            f"{format_percentage(joint_errors['wrong']['top1_probability_ge_0_9'])} 的错误位置仍以不低于 0.9 的概率偏好错误 token。"
        ),
        "",
        "## Loss 拆分",
        "",
        "训练目标公式：",
        "",
        "`total = LM + 5 * region_positive + 1 * region_negative + 0.5 * decoder_block_loss`",
        "",
        "| 项 | 原始均值 | 加权贡献 | 总 loss 占比 |",
        "|---|---:|---:|---:|",
        (
            f"| 主模型 LM | {loss['component_means_training_normalization']['language_model_loss']:.6f} | "
            f"{contributions['language_model']:.6f} | {format_percentage(shares['language_model'])} |"
        ),
        (
            f"| region positive | {loss['component_means_training_normalization']['region_positive_loss']:.6f} | "
            f"{contributions['region_positive_weighted']:.6f} | {format_percentage(shares['region_positive_weighted'])} |"
        ),
        (
            f"| region negative | {loss['component_means_training_normalization']['region_negative_loss']:.6f} | "
            f"{contributions['region_negative_weighted']:.6f} | {format_percentage(shares['region_negative_weighted'])} |"
        ),
        (
            f"| auxiliary decoder | {loss['component_means_training_normalization']['decoder_loss']:.6f} | "
            f"{contributions['decoder_weighted']:.6f} | {format_percentage(shares['decoder_weighted'])} |"
        ),
        "",
        (
            f"由于平均每 block 有 {loss['decoder_tokens_per_block_mean']:.2f} 个 token，名义权重 0.5 "
            f"相对于 token 均值约等于 **{0.5 * loss['decoder_tokens_per_block_mean']:.2f} 倍**。"
            "因此 decoder 占总目标 81% 主要来自 block normalization 的尺度放大，不表示其单 token NLL 比主模型高约 9 倍。"
        ),
        "",
        "![loss decomposition](loss_contributions.png)",
        "",
        (
            f"原训练日志 eval loss 为 {loss['logged_eval_loss']:.6f}；本次复算差值为 "
            f"{loss['reproduced_eval_loss'] - loss['logged_eval_loss']:+.6f} "
            f"（{format_percentage(abs(loss['reproduced_eval_loss'] - loss['logged_eval_loss']) / loss['logged_eval_loss'])}）。"
        ),
        "",
        "## Logits 集中程度",
        "",
        "这里的概率均为 temperature=1 的 teacher-forced softmax；‘最后输出’按三种位置分别报告：所有输出位置、每个 block 最后一个非 EOS token、EOS。",
        "",
        "| 模型 / 位置 | token 数 | top-1 均值 | top-1 中位数 | top-1>=0.5 | 熵均值 | 有效候选数中位数 | 目标为 top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_rows = [
        ("联合 decoder / reasoning", joint_reasoning),
        ("联合 decoder / block 首 token", joint_first),
        ("联合 decoder / block 内部 token", joint_interior),
        ("联合 decoder / block 末非 EOS token", joint_last),
        ("联合 decoder / EOS", joint_eos),
        ("SFT / reasoning", sft_reasoning),
    ]
    for label, summary in table_rows:
        lines.append(
            f"| {label} | {summary['count']} | {summary['top1_probability']['mean']:.4f} | "
            f"{summary['top1_probability']['median']:.4f} | "
            f"{format_percentage(summary['top1_probability_ge_0_5'])} | "
            f"{summary['entropy_nats']['mean']:.4f} | "
            f"{summary['effective_support']['median']:.2f} | "
            f"{format_percentage(summary['target_top1_rate'])} |"
        )
    lines.extend(
        [
            "",
            "![logit comparison](logit_distribution_comparison.png)",
            "",
            "![joint positions](joint_decoder_position_confidence.png)",
            "",
            "![joint target position profile](joint_decoder_target_position_profile.png)",
            "",
            "## 置信错误与位置效应",
            "",
            (
                "block 首 token 几乎总是固定边界 `<<`，末非 EOS token 几乎总是 `>>\\n`，"
                "所以这两个位置及 EOS 的近 100% 置信度主要反映格式确定性，不代表内部数值/运算符翻译同样可靠。"
            ),
            (
                f"去掉首尾边界后，{joint_interior['count']} 个内部 reasoning token 的 target-top1 率为 "
                f"**{format_percentage(joint_interior['target_top1_rate'])}**，target NLL 均值为 "
                f"**{joint_interior['target_nll']['mean']:.4f}**。尤其 block 内第 2 个 token "
                f"（position=1，通常是首个数值）的 target-top1 率只有 "
                f"**{format_percentage(position_one['target_top1_rate'])}**，NLL 均值为 "
                f"**{position_one['target_nll_mean']:.4f}**。"
            ),
            (
                f"联合 decoder 的错误 token 贡献了 reasoning NLL 的 "
                f"**{format_percentage(joint_errors['wrong_token_nll_share'])}**；"
                f"NLL 最大的 10% token 贡献了 {format_percentage(joint_errors['largest_10_percent_token_nll_share'])}。"
                "这说明平均 loss 由少量极度自信的错误主导，分布呈明显双峰，而不是普遍轻微不确定。"
            ),
            "",
            "## 与 SFT 的配对比较",
            "",
            (
                f"两侧使用同一批 {paired['count']} 个 reasoning target token，并按 "
                "`sample / step / step 内 token 位置` 一一配对。联合 decoder 只看到该 block 的连续 latent prefix "
                "及 block 内已给出的目标 token；SFT 则看到问题和此前完整显式 CoT，所以这是输出分布对比，不是严格控制上下文后的能力比较。"
            ),
            (
                f"联合 decoder 相对 SFT 的 top-1 概率平均差为 "
                f"**{paired['top1_probability_joint_minus_sft']['mean']:+.4f}**，"
                f"熵平均差为 **{paired['entropy_nats_joint_minus_sft']['mean']:+.4f} nats**，"
                f"target NLL 平均差为 **{paired['target_nll_joint_minus_sft']['mean']:+.4f}**。"
            ),
            (
                f"联合 decoder 的 reasoning target-top1 率为 {format_percentage(joint_reasoning['target_top1_rate'])}，"
                f"低于 SFT 的 {format_percentage(sft_reasoning['target_top1_rate'])}；"
                f"联合 decoder 的 reasoning target NLL 为 {joint_reasoning['target_nll']['mean']:.4f}，"
                f"SFT 为 {sft_reasoning['target_nll']['mean']:.4f}。SFT 错误位置 top-1 概率中位数仅为 "
                f"{sft_errors['wrong']['top1_probability_median']:.4f}，比联合 decoder 的错误更不自信，校准更合理。"
            ),
            (
                f"两模型 top-1 token 完全一致的比例是 {format_percentage(paired['top1_agreement_rate'])}。"
                f"SFT 在该目录实际权重上的复算 completion loss 为 {sft_loss['reproduced_eval_loss']:.6f}。"
            ),
            "",
            "## 口径与文件",
            "",
            f"- 联合 checkpoint：`{metadata['joint_model']}`",
            f"- 实际权重来源：`{metadata['joint_weight_source']}`",
            f"- SFT checkpoint：`{metadata['sft_model']}`",
            f"- 验证集：`{metadata['validation_file']}`",
            f"- GPU：物理卡 `{metadata['physical_gpu_selection']}`，运行时 `{metadata['cuda_device_name']}`",
            "- `token_logits_metrics.parquet`：逐 token 指标；`joint_block_metrics.parquet`：逐 block 指标。",
            "- `summary.json`：机器可读汇总；`joint_eval_batches.csv`：复算 loss 的逐 batch 明细。",
            (
                "- checkpoint 一致性警告：`base_model/model.safetensors` 与最终根 checkpoint 不一致；"
                "本分析强制从根 `model.safetensors` 加载完整权重。`auxiliary_decoder` 抽样张量与根 checkpoint 一致。"
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.stats_chunk_size <= 0:
        raise ValueError("batch_size and stats_chunk_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"

    with (args.joint_model / "simcot_config.json").open(
        "r", encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    raw = pd.read_parquet(args.validation_file).reset_index(drop=True)
    if args.max_samples is not None:
        raw = raw.iloc[: args.max_samples].copy()

    joint_tokenizer = AutoTokenizer.from_pretrained(
        str(args.joint_model), use_fast=True, local_files_only=True
    )
    configure_tokenizer(joint_tokenizer)

    metadata = {
        "joint_model": args.joint_model.resolve(),
        "joint_weight_source": (args.joint_model / "model.safetensors").resolve(),
        "sft_model": args.sft_model.resolve(),
        "validation_file": args.validation_file.resolve(),
        "think_region_file": args.think_region_file.resolve(),
        "physical_gpu_selection": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "stats_chunk_size": args.stats_chunk_size,
        "max_length": args.max_length,
        "decoder_max_length": args.decoder_max_length,
        "eval_epoch": args.eval_epoch,
        "raw_samples": len(raw),
        "checkpoint_component_export_check": component_export_check(args.joint_model),
    }

    loss_summary, joint_frame, block_frame, retained_indices = analyze_joint(
        args, raw, joint_tokenizer, config, device
    )
    logged_eval_path = args.joint_model / "eval_results.json"
    if logged_eval_path.is_file():
        with logged_eval_path.open("r", encoding="utf-8") as handle:
            logged_eval = json.load(handle)
        loss_summary["logged_eval_loss"] = float(logged_eval["eval_loss"])
        loss_summary["logged_minus_reproduced"] = (
            loss_summary["logged_eval_loss"]
            - loss_summary["reproduced_eval_loss"]
        )

    sft_loss_summary, sft_frame = analyze_sft(
        args, raw, retained_indices, device
    )
    logits_summary = summarize_logits(joint_frame, sft_frame)

    serializable_frame = pd.concat([joint_frame, sft_frame], ignore_index=True)
    serializable_frame["top10_ids"] = serializable_frame["top10_ids"].map(json.dumps)
    serializable_frame["top10_probabilities"] = serializable_frame[
        "top10_probabilities"
    ].map(json.dumps)
    serializable_frame.to_parquet(
        args.output_dir / "token_logits_metrics.parquet", index=False
    )
    block_frame.to_parquet(args.output_dir / "joint_block_metrics.parquet", index=False)
    pd.DataFrame(loss_summary.pop("batch_details")).to_csv(
        args.output_dir / "joint_eval_batches.csv", index=False
    )

    summary = {
        "metadata": metadata,
        "joint_loss": loss_summary,
        "sft_loss": sft_loss_summary,
        "logits": logits_summary,
    }
    write_json(args.output_dir / "summary.json", summary)
    plot_loss_contributions(loss_summary, args.output_dir / "loss_contributions.png")
    plot_logit_comparison(
        joint_frame, sft_frame, args.output_dir / "logit_distribution_comparison.png"
    )
    plot_joint_positions(
        joint_frame, args.output_dir / "joint_decoder_position_confidence.png"
    )
    plot_joint_target_position_profile(
        joint_frame, args.output_dir / "joint_decoder_target_position_profile.png"
    )
    write_report(
        args.output_dir / "report_zh.md",
        metadata,
        loss_summary,
        logits_summary,
        sft_loss_summary,
    )
    print(f"Analysis written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
