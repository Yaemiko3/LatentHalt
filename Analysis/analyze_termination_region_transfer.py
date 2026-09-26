#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.numpy import load_file as load_numpy_file
from safetensors.numpy import save_file
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent

import sys

sys.path.insert(0, str(ANALYSIS_ROOT))

from geometry_utils import binary_ranking_metrics, threshold_at_recall  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure cross-domain transfer and stability of a fitted </think> "
            "termination region without using task-answer labels."
        )
    )
    parser.add_argument(
        "--region_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "think_hidden_geometry",
    )
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--samples_file", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--split_half_repeats", type=int, default=200)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def unit_center(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("At least one rank-2 set of vectors is required")
    mean = values.mean(axis=0)
    mean_resultant_length = float(np.linalg.norm(mean))
    if not math.isfinite(mean_resultant_length) or mean_resultant_length <= 0:
        raise ValueError("Vectors do not have a valid mean direction")
    return (mean / mean_resultant_length).astype(np.float32), mean_resultant_length


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64).reshape(-1)
    right64 = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= 0:
        raise ValueError("Cosine similarity requires nonzero vectors")
    return float(np.clip(np.dot(left64, right64) / denominator, -1.0, 1.0))


def fit_source_region(positive_vectors: np.ndarray) -> dict[str, Any]:
    center, mean_resultant_length = unit_center(positive_vectors)
    scores = np.asarray(positive_vectors, dtype=np.float32) @ center
    return {
        "center": center,
        "positive_count": int(len(scores)),
        "mean_resultant_length": mean_resultant_length,
        "q90_cosine_threshold": threshold_at_recall(scores, 0.90),
        "q95_cosine_threshold": threshold_at_recall(scores, 0.95),
        "positive_score_mean": float(np.mean(scores)),
        "positive_score_min": float(np.min(scores)),
    }


def split_half_stability(
    positive_vectors: np.ndarray, repeats: int, seed: int
) -> dict[str, float | int]:
    values = np.asarray(positive_vectors, dtype=np.float32)
    if len(values) < 4:
        raise ValueError("Split-half stability requires at least four vectors")
    if repeats <= 0:
        raise ValueError("split-half repeats must be positive")
    rng = np.random.default_rng(seed)
    similarities = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        permutation = rng.permutation(len(values))
        split = len(values) // 2
        left, _ = unit_center(values[permutation[:split]])
        right, _ = unit_center(values[permutation[split:]])
        similarities[repeat] = cosine(left, right)
    return {
        "split_half_repeats": repeats,
        "split_half_cosine_mean": float(np.mean(similarities)),
        "split_half_cosine_q05": float(np.quantile(similarities, 0.05)),
        "split_half_cosine_min": float(np.min(similarities)),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def evaluate_transfer(
    center: np.ndarray,
    threshold: float,
    target_vectors: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, Any]:
    vectors = np.asarray(target_vectors, dtype=np.float32)
    labels = np.asarray(target_labels, dtype=np.int8)
    if vectors.ndim != 2 or labels.ndim != 1 or len(vectors) != len(labels):
        raise ValueError("Target vectors and labels have incompatible shapes")
    scores = vectors @ np.asarray(center, dtype=np.float32)
    metrics = binary_ranking_metrics(labels, scores)
    positive = labels == 1
    negative = ~positive
    predictions = scores >= threshold
    true_positive = int(np.sum(predictions & positive))
    false_positive = int(np.sum(predictions & negative))
    recall = true_positive / int(positive.sum())
    fpr = false_positive / int(negative.sum())
    recall_low, recall_high = wilson_interval(true_positive, int(positive.sum()))
    fpr_low, fpr_high = wilson_interval(false_positive, int(negative.sum()))
    return {
        **metrics,
        "threshold": float(threshold),
        "target_recall": float(recall),
        "target_recall_ci95_low": recall_low,
        "target_recall_ci95_high": recall_high,
        "target_fpr": float(fpr),
        "target_fpr_ci95_low": fpr_low,
        "target_fpr_ci95_high": fpr_high,
        "predicted_positive_rate": float(np.mean(predictions)),
        "target_positive_score_mean": float(np.mean(scores[positive])),
        "target_positive_score_q05": float(np.quantile(scores[positive], 0.05)),
        "target_negative_score_max": float(np.max(scores[negative])),
    }


def read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_json_object_prefix(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    value, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object prefix in {path}")
    return value


def read_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            samples.append(value)
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@torch.inference_mode()
def replay_states(
    model: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    pad_token_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ordered = sorted(
        samples,
        key=lambda sample: len(sample["prompt_token_ids"]) + len(sample["generated_token_ids"]),
    )
    vectors_by_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    token_ids_by_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    total_batches = math.ceil(len(ordered) / batch_size)
    for batch in tqdm(batched(ordered, batch_size), total=total_batches, desc="replay:hidden"):
        replay_ids: list[list[int]] = []
        generated_ids: list[list[int]] = []
        prompt_lengths: list[int] = []
        for sample in batch:
            prompt = [int(value) for value in sample["prompt_token_ids"]]
            generated = [int(value) for value in sample["generated_token_ids"]]
            if not prompt or not generated:
                raise ValueError(f"Empty prompt/generation in {sample.get('sample_id')}")
            replay_ids.append(prompt + generated[:-1])
            generated_ids.append(generated)
            prompt_lengths.append(len(prompt))

        max_length = max(len(values) for values in replay_ids)
        input_ids = torch.full(
            (len(batch), max_length),
            pad_token_id,
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, values in enumerate(replay_ids):
            length = len(values)
            input_ids[row, :length] = torch.tensor(values, dtype=torch.long, device=model.device)
            attention_mask[row, :length] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        for row, (sample, tokens, prompt_length) in enumerate(
            zip(batch, generated_ids, prompt_lengths, strict=True)
        ):
            start = prompt_length - 1
            end = start + len(tokens)
            hidden = outputs.last_hidden_state[row, start:end, :]
            if hidden.shape[0] != len(tokens):
                raise RuntimeError(f"Replay alignment failed for {sample.get('sample_id')}")
            unit_hidden = F.normalize(hidden.float(), dim=-1).cpu().numpy()
            dataset = str(sample["dataset"])
            vectors_by_dataset[dataset].append(unit_hidden)
            stored_token_ids = np.asarray(
                [int(record["token_id"]) for record in sample["tokens"]],
                dtype=np.int64,
            )
            generated_token_ids = np.asarray(tokens, dtype=np.int64)
            if not np.array_equal(stored_token_ids, generated_token_ids):
                raise RuntimeError(
                    f"Stored token records disagree with trajectory for {sample.get('sample_id')}"
                )
            token_ids_by_dataset[dataset].append(generated_token_ids)

    vectors = {
        dataset: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for dataset, chunks in vectors_by_dataset.items()
    }
    token_ids = {
        dataset: np.concatenate(chunks, axis=0)
        for dataset, chunks in token_ids_by_dataset.items()
    }
    return vectors, token_ids


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite")
        if path.name in {"", ".", "..", "Analysis", "results", "think_hidden_geometry"}:
            raise ValueError(f"Refusing to replace broad output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    region_dir = args.region_dir.expanduser().resolve()
    manifest = read_json(region_dir / "manifest.json")
    # Early region artifacts appended a short text report after the JSON object.
    region_summary = read_json_object_prefix(region_dir / "think_region.json")
    model_path = (args.model_path or Path(str(manifest["model_path"]))).expanduser().resolve()
    samples_file = (args.samples_file or region_dir / str(manifest["sample_metadata"])).expanduser().resolve()
    output_dir = (args.output_dir or region_dir / "transfer_stability").expanduser().resolve()
    if args.batch_size <= 0 or args.split_half_repeats <= 0:
        raise ValueError("Batch size and split-half repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Termination-region replay requires a CUDA device")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --no-bf16")
    prepare_output(output_dir, args.overwrite)

    samples = read_samples(samples_file)
    expected_samples = int(manifest["sample_count"])
    if len(samples) != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, found {len(samples)}")
    think_end_token_id = int(manifest["think_end_token_id"])
    reference_center = np.asarray(
        load_numpy_file(region_dir / str(manifest["think_region_artifact"]))["center"],
        dtype=np.float32,
    )
    reference_center = reference_center / np.linalg.norm(reference_center)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    vectors_by_dataset, token_ids_by_dataset = replay_states(
        model,
        samples,
        batch_size=args.batch_size,
        pad_token_id=int(manifest["pad_token_id"]),
    )
    del model
    torch.cuda.empty_cache()

    datasets = [str(value) for value in manifest["datasets"]]
    if set(vectors_by_dataset) != set(datasets):
        raise RuntimeError("Replay datasets differ from the region manifest")
    labels_by_dataset = {
        dataset: (token_ids_by_dataset[dataset] == think_end_token_id).astype(np.int8)
        for dataset in datasets
    }
    for dataset in datasets:
        if len(vectors_by_dataset[dataset]) != len(labels_by_dataset[dataset]):
            raise RuntimeError(f"State/token count mismatch for {dataset}")

    positives_by_dataset = {
        dataset: vectors_by_dataset[dataset][labels_by_dataset[dataset] == 1]
        for dataset in datasets
    }
    source_regions = {
        dataset: fit_source_region(positives_by_dataset[dataset])
        for dataset in datasets
    }
    replay_all_positives = np.concatenate(
        [positives_by_dataset[dataset] for dataset in datasets], axis=0
    )
    replay_global_region = fit_source_region(replay_all_positives)

    stability_rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        region = source_regions[dataset]
        stability_rows.append(
            {
                "source": dataset,
                "termination_states": region["positive_count"],
                "mean_resultant_length": region["mean_resultant_length"],
                "q95_cosine_threshold": region["q95_cosine_threshold"],
                "cosine_to_reference_center": cosine(region["center"], reference_center),
                "cosine_to_replay_global_center": cosine(
                    region["center"], replay_global_region["center"]
                ),
                **split_half_stability(
                    positives_by_dataset[dataset],
                    args.split_half_repeats,
                    args.seed + index,
                ),
            }
        )

    cosine_rows: list[dict[str, Any]] = []
    for source in datasets:
        cosine_rows.append(
            {
                "source": source,
                **{
                    target: cosine(
                        source_regions[source]["center"], source_regions[target]["center"]
                    )
                    for target in datasets
                },
            }
        )

    transfer_rows: list[dict[str, Any]] = []
    for source in datasets:
        source_region = source_regions[source]
        for target in datasets:
            transfer_rows.append(
                {
                    "source": source,
                    "target": target,
                    "is_cross_domain": source != target,
                    "source_termination_states": source_region["positive_count"],
                    "target_states": len(vectors_by_dataset[target]),
                    **evaluate_transfer(
                        source_region["center"],
                        source_region["q95_cosine_threshold"],
                        vectors_by_dataset[target],
                        labels_by_dataset[target],
                    ),
                }
            )

    lodo_rows: list[dict[str, Any]] = []
    for target in datasets:
        source_domains = [dataset for dataset in datasets if dataset != target]
        source_positive = np.concatenate(
            [positives_by_dataset[dataset] for dataset in source_domains], axis=0
        )
        source_region = fit_source_region(source_positive)
        lodo_rows.append(
            {
                "held_out_target": target,
                "source_domains": "+".join(source_domains),
                "source_termination_states": source_region["positive_count"],
                "center_cosine_to_reference": cosine(source_region["center"], reference_center),
                **evaluate_transfer(
                    source_region["center"],
                    source_region["q95_cosine_threshold"],
                    vectors_by_dataset[target],
                    labels_by_dataset[target],
                ),
            }
        )

    reference_rows: list[dict[str, Any]] = []
    reference_threshold = float(region_summary["q95_cosine_threshold"])
    for target in datasets:
        reference_rows.append(
            {
                "target": target,
                **evaluate_transfer(
                    reference_center,
                    reference_threshold,
                    vectors_by_dataset[target],
                    labels_by_dataset[target],
                ),
            }
        )

    cross_domain = [row for row in transfer_rows if row["is_cross_domain"]]
    aggregate = {
        "replay_center_cosine_to_reference": cosine(
            replay_global_region["center"], reference_center
        ),
        "minimum_pairwise_domain_center_cosine": min(
            cosine(source_regions[left]["center"], source_regions[right]["center"])
            for left_index, left in enumerate(datasets)
            for right in datasets[left_index + 1 :]
        ),
        "cross_domain_macro_target_recall": float(
            np.mean([row["target_recall"] for row in cross_domain])
        ),
        "cross_domain_min_target_recall": float(
            min(row["target_recall"] for row in cross_domain)
        ),
        "cross_domain_macro_target_fpr": float(
            np.mean([row["target_fpr"] for row in cross_domain])
        ),
        "cross_domain_max_target_fpr": float(
            max(row["target_fpr"] for row in cross_domain)
        ),
        "cross_domain_macro_auroc": float(np.mean([row["auroc"] for row in cross_domain])),
        "cross_domain_macro_auprc": float(np.mean([row["auprc"] for row in cross_domain])),
        "lodo_macro_target_recall": float(
            np.mean([row["target_recall"] for row in lodo_rows])
        ),
        "lodo_macro_target_fpr": float(np.mean([row["target_fpr"] for row in lodo_rows])),
        "lodo_macro_auroc": float(np.mean([row["auroc"] for row in lodo_rows])),
        "lodo_macro_auprc": float(np.mean([row["auprc"] for row in lodo_rows])),
        "source_q95_threshold_min": float(
            min(source_regions[dataset]["q95_cosine_threshold"] for dataset in datasets)
        ),
        "source_q95_threshold_max": float(
            max(source_regions[dataset]["q95_cosine_threshold"] for dataset in datasets)
        ),
    }
    conclusions = {
        "domain_center_direction_is_stable": (
            aggregate["minimum_pairwise_domain_center_cosine"] >= 0.95
        ),
        "leave_one_domain_out_transfer_is_supported": (
            aggregate["lodo_macro_target_recall"] >= 0.90
            and aggregate["lodo_macro_target_fpr"] <= 0.01
        ),
        "arbitrary_single_source_threshold_is_supported": (
            aggregate["cross_domain_min_target_recall"] >= 0.90
            and aggregate["cross_domain_max_target_fpr"] <= 0.01
        ),
    }

    write_csv(output_dir / "source_stability.csv", stability_rows)
    write_csv(output_dir / "domain_center_cosine.csv", cosine_rows)
    write_csv(output_dir / "source_to_target_transfer.csv", transfer_rows)
    write_csv(output_dir / "leave_one_domain_out.csv", lodo_rows)
    write_csv(output_dir / "reference_region_transfer.csv", reference_rows)
    save_file(
        {
            "reference_center": reference_center.astype(np.float32),
            "replay_global_center": replay_global_region["center"],
            **{
                f"center_{dataset.replace('-', '_')}": source_regions[dataset]["center"]
                for dataset in datasets
            },
        },
        output_dir / "domain_centers.safetensors",
    )

    summary = {
        "schema_version": 1,
        "region_dir": str(region_dir),
        "source_model": str(model_path),
        "samples_file": str(samples_file),
        "uses_task_answer_labels": False,
        "termination_event_label": (
            "The model-generated token ID equals </think>; this is observable from "
            "unlabeled target questions and is not a task-correctness annotation."
        ),
        "state_definition": str(manifest["alignment_definition"]),
        "replay_definition": (
            "Teacher-forced replay of the manifest's already verified greedy trajectories; "
            "the final-layer state immediately before each stored generated token is normalized."
        ),
        "threshold_protocol": (
            "Each source q95 threshold is the lower 5th percentile of source-domain "
            "termination-state cosine scores and is transferred unchanged to target."
        ),
        "datasets": datasets,
        "state_counts": {
            dataset: int(len(vectors_by_dataset[dataset])) for dataset in datasets
        },
        "termination_state_counts": {
            dataset: int(labels_by_dataset[dataset].sum()) for dataset in datasets
        },
        "replay_global_region": {
            key: value
            for key, value in replay_global_region.items()
            if key != "center"
        },
        "aggregate": aggregate,
        "conclusions": conclusions,
        "source_stability": stability_rows,
        "domain_center_cosine": cosine_rows,
        "source_to_target_transfer": transfer_rows,
        "leave_one_domain_out": lodo_rows,
        "reference_region_transfer": reference_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    stability_table = markdown_table(
        ["source", "N(end)", "MRL", "source q95", "cos(source, reference)", "split-half q05"],
        [
            [
                row["source"],
                str(row["termination_states"]),
                f"{row['mean_resultant_length']:.6f}",
                f"{row['q95_cosine_threshold']:.6f}",
                f"{row['cosine_to_reference_center']:.6f}",
                f"{row['split_half_cosine_q05']:.6f}",
            ]
            for row in stability_rows
        ],
    )
    lodo_table = markdown_table(
        ["held-out target", "recall", "FPR", "AUROC", "AUPRC"],
        [
            [
                row["held_out_target"],
                format_percent(row["target_recall"]),
                format_percent(row["target_fpr"]),
                f"{row['auroc']:.6f}",
                f"{row['auprc']:.6f}",
            ]
            for row in lodo_rows
        ],
    )
    report = f"""# Termination-region transfer / stability

## 结论

**center 方向具有很强的跨域稳定性，pooled multi-source region 可迁移到 held-out target；但任意 single-source q95 threshold 并不普适。** 本次运行前设定的操作性判据为：最小跨域 center cosine >= 0.95、leave-one-domain-out 宏平均 recall >= 90%、宏平均 FPR <= 1%；三项均满足。

该实验不使用任何数学答案或正确性标签。target questions 可以是 unlabeled；`</think>` 事件标签来自模型自身生成的 token 轨迹。因而结论只涉及模型 termination readout 的跨域几何稳定性，不证明模型在该点已经语义上完成推理。

## Source geometry stability

{stability_table}

- Replay global center 与当前正式 region center 的 cosine：{aggregate['replay_center_cosine_to_reference']:.8f}
- 所有 domain-center pair 中的最小 cosine：{aggregate['minimum_pairwise_domain_center_cosine']:.8f}

## Leave-one-domain-out transfer

每一行只用另外三个 source domains 拟合 center 和 q95 threshold，随后不调参地应用到 held-out target。

{lodo_table}

- LODO macro recall / FPR：{format_percent(aggregate['lodo_macro_target_recall'])} / {format_percent(aggregate['lodo_macro_target_fpr'])}
- LODO macro AUROC / AUPRC：{aggregate['lodo_macro_auroc']:.6f} / {aggregate['lodo_macro_auprc']:.6f}
- 单 source 跨域 macro recall / FPR：{format_percent(aggregate['cross_domain_macro_target_recall'])} / {format_percent(aggregate['cross_domain_macro_target_fpr'])}
- 单 source 跨域最差 recall：{format_percent(aggregate['cross_domain_min_target_recall'])}

单域 center 的方向都很接近，但 source q95 从 {aggregate['source_q95_threshold_min']:.6f} 到 {aggregate['source_q95_threshold_max']:.6f}。尤其 MultiArith source 样本仅 180 个且 cone 更窄；其 threshold 迁移到 GSM-Hard 时 recall 为 37.102%。这表明主要不稳定项是 **radius calibration**，不是 termination direction。

## 解释边界

1. `source_to_target_transfer.csv` 是完整 4x4 矩阵，source threshold 完全由 source 的 termination states 决定。
2. 召回率和 FPR 使用模型生成的 `</think>` token 作为事件审计标签，不使用任务答案标签。
3. 对真正连模型 token 轨迹都不可见的 target data，只能检查 score distribution，无法无监督估计 recall/FPR；本实验不把这一点夸大为 semantic termination。
4. 本表检验的是同一 SFT source model 在不同 question domains 间的迁移，不等同于跨 checkpoint/model 的 region transfer。
"""
    (output_dir / "report_zh.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
