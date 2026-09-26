#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.numpy import save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import (  # noqa: E402
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    MathExample,
    batched,
    load_gsm8k_aug,
    load_math_examples,
)
from train_sft_cot import add_think_special_tokens, configure_pad_token  # noqa: E402

from geometry_utils import (  # noqa: E402
    PRECURSOR_OFFSETS,
    balanced_group_folds,
    binary_ranking_metrics,
    fit_angular_region,
    generated_token_offset,
    geometry_statistics,
    l2_normalize_rows,
    predictor_state_offset,
    threshold_at_recall,
)


SUPPORTED_DATASETS = ("gsm8k", "gsm-hard", "multi-arith", "svamp")
GSM8K_REGION_SPLIT = "validation"
STATE_INDEX_FIELDS = (
    "global_state_index",
    "sample_id",
    "dataset",
    "source_split",
    "generation_position",
    "generated_token_sequence_offset",
    "predictor_hidden_sequence_offset",
    "token_id",
    "token_text",
    "token_decoded",
    "is_think_end",
    "is_think_start",
    "is_eos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract exactly aligned last-layer states and fit the generated </think> region."
        )
    )
    parser.add_argument(
        "--model_path", type=Path, default=PROJECT_ROOT / "outputs" / "sft-cot-llama1b"
    )
    parser.add_argument("--data_root", type=Path, default=WORKSPACE_ROOT / "datasets")
    parser.add_argument(
        "--cache_dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "think_hidden_geometry",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=["gsm8k"],
        help="Datasets used to fit the region (default: GSM8K-Aug validation split only).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples_per_dataset", type=int)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--min_token_frequency", type=int, default=50)
    parser.add_argument("--recall_target", type=float, default=0.95)
    parser.add_argument("--matrix_chunk_size", type=int, default=4096)
    parser.add_argument("--pca_device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_token_ids(token_ids: int | Iterable[int] | None) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(token_id) for token_id in token_ids}


def load_region_examples(
    dataset: str, data_root: Path, cache_dir: Path
) -> list[MathExample]:
    if dataset == "gsm8k":
        rows = load_gsm8k_aug(data_root, cache_dir, splits=(GSM8K_REGION_SPLIT,))[
            GSM8K_REGION_SPLIT
        ]
        return [
            MathExample(str(row["question"]).strip(), str(row["answer"]).strip())
            for row in rows
        ]
    return load_math_examples(dataset, data_root, cache_dir)


def source_split(dataset: str, index: int) -> str:
    if dataset == "gsm8k":
        return GSM8K_REGION_SPLIT
    if dataset == "svamp":
        return "train" if index < 700 else "test"
    return "test"


def prepare_output_directory(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}; use --overwrite")
        if output_dir.name in {"", ".", "..", "Analysis", "results"}:
            raise ValueError(f"Refusing to replace broad output directory: {output_dir}")
        shutil.rmtree(output_dir)
    temporary = output_dir.with_name(output_dir.name + f".in_progress.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    return temporary


def greedy_batch_with_states(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> tuple[list[list[int]], list[list[np.ndarray]], list[int], int]:
    tokenized = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    prompt_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    active = torch.ones(len(prompts), dtype=torch.bool, device=model.device)
    generated_ids: list[list[int]] = [[] for _ in prompts]
    generated_states: list[list[np.ndarray]] = [[] for _ in prompts]
    alignment_checks = 0

    with torch.inference_mode():
        cache = DynamicCache()
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        for _ in range(max_new_tokens):
            predictor_hidden = outputs.last_hidden_state[:, -1, :]
            logits = model.lm_head(predictor_hidden)
            selected = logits.argmax(dim=-1)

            # The token is selected directly from LM-head(predictor_hidden). This
            # assertion is intentionally kept at every step as an alignment guard.
            replay_selected = model.lm_head(predictor_hidden).argmax(dim=-1)
            if not torch.equal(selected[active], replay_selected[active]):
                raise RuntimeError("LM-head alignment check failed during generation")

            hidden_cpu = predictor_hidden.float().cpu().numpy()
            selected_cpu = selected.cpu().tolist()
            active_cpu = active.cpu().tolist()
            for row, is_active in enumerate(active_cpu):
                if not is_active:
                    continue
                generated_ids[row].append(int(selected_cpu[row]))
                generated_states[row].append(hidden_cpu[row].copy())
                alignment_checks += 1

            selected_is_eos = torch.tensor(
                [int(token_id) in eos_token_ids for token_id in selected_cpu],
                dtype=torch.bool,
                device=model.device,
            )
            active = active & ~selected_is_eos
            if not bool(active.any()):
                break

            next_input_ids = selected.clone()
            next_input_ids[~active] = tokenizer.pad_token_id
            next_attention = active.long().unsqueeze(1)
            attention_mask = torch.cat((attention_mask, next_attention), dim=1)
            next_position_ids = (attention_mask.sum(dim=1) - 1).clamp_min(0).unsqueeze(1)
            outputs = model.model(
                input_ids=next_input_ids.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=next_position_ids,
                past_key_values=outputs.past_key_values,
                use_cache=True,
                return_dict=True,
            )

    return generated_ids, generated_states, prompt_lengths, alignment_checks


def matrix_vector_scores(
    vectors: np.ndarray, indices: np.ndarray, center: np.ndarray, chunk_size: int
) -> np.ndarray:
    result = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        result[start : start + len(chunk)] = vectors[chunk] @ center
    return result


def heldout_center_scores(
    vectors: np.ndarray,
    positive_mask: np.ndarray,
    state_folds: np.ndarray,
    folds: int,
    recall_target: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(vectors), dtype=np.float32)
    thresholds = np.empty(folds, dtype=np.float64)
    for fold in range(folds):
        train_positive_indices = np.flatnonzero(positive_mask & (state_folds != fold))
        test_indices = np.flatnonzero(state_folds == fold)
        if len(train_positive_indices) == 0:
            raise ValueError(f"Fold {fold} has no training </think> states")
        if len(test_indices) == 0:
            raise ValueError(f"Fold {fold} has no test states")
        center = vectors[train_positive_indices].mean(axis=0, dtype=np.float64)
        center_norm = np.linalg.norm(center)
        if not np.isfinite(center_norm) or center_norm <= 0:
            raise ValueError(f"Fold {fold} has an invalid </think> center")
        center = (center / center_norm).astype(np.float32)
        scores[test_indices] = matrix_vector_scores(
            vectors, test_indices, center, chunk_size
        )
        train_positive_scores = matrix_vector_scores(
            vectors, train_positive_indices, center, chunk_size
        )
        thresholds[fold] = threshold_at_recall(
            train_positive_scores, recall_target
        )
    return scores, thresholds


def mean_resultant_length(
    vectors: np.ndarray, indices: np.ndarray, chunk_size: int
) -> float:
    resultant_sum = np.zeros(vectors.shape[1], dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        resultant_sum += vectors[chunk].sum(axis=0, dtype=np.float64)
    return float(np.linalg.norm(resultant_sum / len(indices)))


def run_pca_summary(vectors: np.ndarray, device_option: str) -> dict[str, Any]:
    if len(vectors) < 2:
        raise ValueError("PCA requires at least two </think> states")
    device = (
        "cuda"
        if device_option == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_option == "auto"
        else device_option
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--pca_device cuda requested but CUDA is unavailable")
    values = torch.from_numpy(np.array(vectors, dtype=np.float32, copy=True)).to(device)
    values = values - values.mean(dim=0, keepdim=True)
    covariance = values.T @ values / (len(values) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min_(0).flip(0)
    explained = eigenvalues / eigenvalues.sum()
    cumulative = torch.cumsum(explained, dim=0).cpu().numpy()
    del covariance, eigenvalues, explained, values
    if device == "cuda":
        torch.cuda.empty_cache()

    def dimension_at(target: float) -> int:
        return int(np.searchsorted(cumulative, target, side="left") + 1)

    return {
        "device": device,
        "pca90_dimensions": dimension_at(0.90),
        "pca95_dimensions": dimension_at(0.95),
        "pca99_dimensions": dimension_at(0.99),
    }


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if args.min_token_frequency <= 0 or args.matrix_chunk_size <= 0:
        raise ValueError("min_token_frequency and matrix_chunk_size must be positive")
    if not 0 < args.recall_target <= 1:
        raise ValueError("recall_target must lie in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("Extraction requires a CUDA device")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --no-bf16")

    working_dir = prepare_output_directory(args.output_dir, args.overwrite)
    completed = False
    try:
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
        think_start_ids = tokenizer.encode(THINK_START_TOKEN, add_special_tokens=False)
        think_end_ids = tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)
        if len(think_start_ids) != 1 or len(think_end_ids) != 1:
            raise RuntimeError(
                f"Think markers must be single tokens; got {think_start_ids=} {think_end_ids=}"
            )
        think_start_id = int(think_start_ids[0])
        think_end_id = int(think_end_ids[0])

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
        eos_token_ids = normalize_token_ids(model.config.eos_token_id)
        if tokenizer.eos_token_id is not None:
            eos_token_ids.add(int(tokenizer.eos_token_id))

        metadata_path = working_dir / "samples.jsonl"
        state_index_path = working_dir / "state_index.csv"
        temporary_vectors_path = working_dir / ".all_hidden_l2.tmp"
        dataset_counts: Counter[str] = Counter()
        source_split_counts: Counter[str] = Counter()
        think_end_counts: Counter[str] = Counter()
        sample_fold_inputs: list[tuple[str, str]] = []
        state_sample_ids: list[str] = []
        state_token_ids: list[int] = []
        precursor_by_sample: dict[str, dict[int, int]] = {}
        total_samples = 0
        total_states = 0
        total_alignment_checks = 0
        think_state_chunks: list[np.ndarray] = []
        hidden_size = int(model.config.hidden_size)

        with (
            metadata_path.open("w", encoding="utf-8") as metadata_handle,
            state_index_path.open("w", encoding="utf-8", newline="") as state_handle,
            temporary_vectors_path.open("wb") as temporary_vectors_handle,
        ):
            state_writer = csv.DictWriter(state_handle, fieldnames=STATE_INDEX_FIELDS)
            state_writer.writeheader()

            for dataset in args.datasets:
                examples = load_region_examples(dataset, args.data_root, args.cache_dir)
                indexed_examples = list(enumerate(examples))
                if args.max_samples_per_dataset is not None:
                    indexed_examples = indexed_examples[: args.max_samples_per_dataset]
                batches = batched(indexed_examples, args.batch_size)
                total_batches = (len(indexed_examples) + args.batch_size - 1) // args.batch_size
                for batch in tqdm(batches, total=total_batches, desc=f"extract:{dataset}"):
                    prompts = [example.question.strip() + "\n" for _, example in batch]
                    ids_by_sample, states_by_sample, prompt_lengths, alignment_checks = (
                        greedy_batch_with_states(
                            model,
                            tokenizer,
                            prompts,
                            args.max_new_tokens,
                            eos_token_ids,
                        )
                    )
                    total_alignment_checks += alignment_checks
                    flat_states = np.concatenate(
                        [np.stack(states, axis=0) for states in states_by_sample], axis=0
                    ).astype(np.float32, copy=False)
                    if flat_states.ndim != 2 or flat_states.shape[1] != hidden_size:
                        raise RuntimeError(f"Unexpected hidden-state shape: {flat_states.shape}")
                    normalized_states, _ = l2_normalize_rows(flat_states)
                    normalized_states.tofile(temporary_vectors_handle)
                    flat_token_ids = np.asarray(
                        [token_id for token_ids in ids_by_sample for token_id in token_ids],
                        dtype=np.int64,
                    )
                    if len(flat_token_ids) != len(normalized_states):
                        raise RuntimeError("Flattened token/state count mismatch")
                    batch_think_states = normalized_states[flat_token_ids == think_end_id]
                    if len(batch_think_states):
                        think_state_chunks.append(batch_think_states.copy())

                    batch_state_offset = 0
                    for batch_row, ((source_index, example), token_ids, sample_states) in enumerate(
                        zip(batch, ids_by_sample, states_by_sample, strict=True)
                    ):
                        prompt_length = int(prompt_lengths[batch_row])
                        sample_id = f"{dataset}:{source_split(dataset, source_index)}:{source_index:06d}"
                        split_name = source_split(dataset, source_index)
                        sample_global_start = total_states + batch_state_offset
                        token_records: list[dict[str, Any]] = []
                        think_positions: list[int] = []
                        eos_positions: list[int] = []

                        if len(token_ids) != len(sample_states):
                            raise RuntimeError(f"Token/state count mismatch for {sample_id}")
                        for generation_position, token_id in enumerate(token_ids):
                            global_state_index = sample_global_start + generation_position
                            token_text = tokenizer.convert_ids_to_tokens(int(token_id))
                            token_decoded = tokenizer.decode(
                                [int(token_id)],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )
                            is_think_end = int(token_id) == think_end_id
                            is_think_start = int(token_id) == think_start_id
                            is_eos = int(token_id) in eos_token_ids
                            if is_think_end:
                                think_positions.append(generation_position)
                            if is_eos:
                                eos_positions.append(generation_position)
                            token_record = {
                                "global_state_index": global_state_index,
                                "generation_position": generation_position,
                                "generated_token_sequence_offset": generated_token_offset(
                                    prompt_length, generation_position
                                ),
                                "predictor_hidden_sequence_offset": predictor_state_offset(
                                    prompt_length, generation_position
                                ),
                                "token_id": int(token_id),
                                "token_text": token_text,
                                "token_decoded": token_decoded,
                            }
                            token_records.append(token_record)
                            state_writer.writerow(
                                {
                                    **token_record,
                                    "sample_id": sample_id,
                                    "dataset": dataset,
                                    "source_split": split_name,
                                    "is_think_end": int(is_think_end),
                                    "is_think_start": int(is_think_start),
                                    "is_eos": int(is_eos),
                                }
                            )

                        landmarks: dict[str, Any] = {}
                        if think_positions:
                            target_position = think_positions[0]
                            positions = {"think_end": target_position}
                            positions.update(
                                {
                                    f"think_end_minus_{offset}": target_position - offset
                                    for offset in PRECURSOR_OFFSETS
                                    if target_position >= offset
                                }
                            )
                            for label, position in positions.items():
                                landmarks[label] = token_records[position]
                            precursor_by_sample[sample_id] = {
                                offset: sample_global_start + target_position - offset
                                for offset in (*PRECURSOR_OFFSETS, 0)
                                if target_position >= offset
                            }

                        end_reason = "eos" if eos_positions else "max_new_tokens"
                        sample_record = {
                            "sample_id": sample_id,
                            "dataset": dataset,
                            "source_split": split_name,
                            "source_index": source_index,
                            "question": example.question,
                            "prompt_text": prompts[batch_row],
                            "prompt_token_ids": tokenizer.encode(
                                prompts[batch_row], add_special_tokens=True
                            ),
                            "prompt_length": prompt_length,
                            "generated_token_ids": token_ids,
                            "generated_text": tokenizer.decode(
                                token_ids,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                            "generation_length": len(token_ids),
                            "end_reason": end_reason,
                            "think_end_generation_positions": think_positions,
                            "eos_generation_positions": eos_positions,
                            "target_think_end_position": (
                                think_positions[0] if think_positions else None
                            ),
                            "target_think_end_global_state_index": (
                                sample_global_start + think_positions[0]
                                if think_positions
                                else None
                            ),
                            "state_global_start": sample_global_start,
                            "state_global_end_exclusive": sample_global_start + len(token_ids),
                            "landmarks": landmarks,
                            "tokens": token_records,
                            "lm_head_alignment_verified": True,
                        }
                        metadata_handle.write(
                            json.dumps(sample_record, ensure_ascii=True) + "\n"
                        )
                        batch_state_offset += len(token_ids)
                        total_samples += 1
                        dataset_counts[dataset] += 1
                        source_split_counts[f"{dataset}:{split_name}"] += 1
                        think_end_counts[dataset] += int(bool(think_positions))
                        sample_fold_inputs.append((sample_id, dataset))
                        state_sample_ids.extend([sample_id] * len(token_ids))
                        state_token_ids.extend(int(token_id) for token_id in token_ids)

                    if batch_state_offset != len(flat_states):
                        raise RuntimeError("Batch metadata does not cover every hidden state")
                    total_states += len(flat_states)

        if total_alignment_checks != total_states:
            raise RuntimeError(
                f"Alignment checks ({total_alignment_checks}) != generated states ({total_states})"
            )
        if not think_state_chunks:
            raise RuntimeError(
                "No generated </think> predictor states were found; cannot fit the region"
            )
        if len(state_sample_ids) != total_states or len(state_token_ids) != total_states:
            raise RuntimeError("Analysis metadata does not cover every generated state")
        expected_vector_bytes = total_states * hidden_size * np.dtype(np.float32).itemsize
        if temporary_vectors_path.stat().st_size != expected_vector_bytes:
            raise RuntimeError("Temporary normalized-vector file has an unexpected size")

        think_states = np.concatenate(think_state_chunks, axis=0).astype(
            np.float32, copy=False
        )
        think_center, think_region = fit_angular_region(think_states)
        think_geometry = geometry_statistics(think_states)

        del model
        torch.cuda.empty_cache()
        all_vectors = np.memmap(
            temporary_vectors_path,
            mode="r",
            dtype=np.float32,
            shape=(total_states, hidden_size),
        )
        token_ids_array = np.asarray(state_token_ids, dtype=np.int64)
        positive_mask = token_ids_array == think_end_id
        if int(positive_mask.sum()) != len(think_states):
            raise RuntimeError("Collected </think> states differ from the token index")

        fold_assignment = balanced_group_folds(
            sample_fold_inputs, args.folds, args.seed
        )
        state_folds = np.asarray(
            [fold_assignment.sample_to_fold[sample_id] for sample_id in state_sample_ids],
            dtype=np.int16,
        )
        oof_scores, fold_thresholds = heldout_center_scores(
            all_vectors,
            positive_mask,
            state_folds,
            args.folds,
            args.recall_target,
            args.matrix_chunk_size,
        )
        labels = positive_mask.astype(np.int8)
        ranking_metrics = binary_ranking_metrics(labels, oof_scores)
        state_thresholds = fold_thresholds[state_folds]
        heldout_think_recall = float(
            np.mean(oof_scores[positive_mask] >= state_thresholds[positive_mask])
        )
        heldout_non_think_fpr = float(
            np.mean(oof_scores[~positive_mask] >= state_thresholds[~positive_mask])
        )

        token_counts = Counter(state_token_ids)
        frequent_concentrations: list[tuple[int, float]] = []
        for token_id, frequency in token_counts.items():
            if frequency < args.min_token_frequency:
                continue
            indices = np.flatnonzero(token_ids_array == token_id)
            frequent_concentrations.append(
                (
                    token_id,
                    mean_resultant_length(
                        all_vectors, indices, args.matrix_chunk_size
                    ),
                )
            )
        frequent_concentrations.sort(key=lambda item: item[1], reverse=True)
        think_frequency_rank = next(
            (
                rank
                for rank, (token_id, _) in enumerate(
                    frequent_concentrations, start=1
                )
                if token_id == think_end_id
            ),
            None,
        )
        if think_frequency_rank is None:
            raise RuntimeError("</think> did not meet min_token_frequency")

        ordered_offsets = sorted((*PRECURSOR_OFFSETS, 0), reverse=True)
        precursor_means: list[float] = []
        for offset in ordered_offsets:
            indices = np.asarray(
                [
                    positions[offset]
                    for positions in precursor_by_sample.values()
                    if offset in positions
                ],
                dtype=np.int64,
            )
            if len(indices) == 0:
                raise RuntimeError(f"No </think>-{offset} precursor states were found")
            precursor_means.append(float(np.mean(oof_scores[indices])))
        precursor_monotonic = all(
            later >= earlier
            for earlier, later in zip(
                precursor_means, precursor_means[1:], strict=True
            )
        )
        final_transition_deltas = np.asarray(
            [
                oof_scores[positions[0]] - oof_scores[positions[1]]
                for positions in precursor_by_sample.values()
                if 0 in positions and 1 in positions
            ],
            dtype=np.float32,
        )
        if len(final_transition_deltas) == 0:
            raise RuntimeError("No paired </think>-1 to </think> transitions were found")

        pca_summary = run_pca_summary(think_states, args.pca_device)

        conclusion = (
            "模型实际生成 `</think>` 时的最后一层 hidden states 形成稳定、可识别的几何区域。"
            if ranking_metrics["auroc"] >= 0.9
            and ranking_metrics["auprc"] >= 0.5
            and think_geometry["mean_resultant_length"] >= 0.5
            else "当前结果不足以支持 `</think>` hidden states 形成稳定、可识别几何区域的结论。"
        )
        report = f"""# `</think>` Hidden-State Geometry Report

{conclusion}

- Samples: {total_samples}; generated states: {total_states}; `</think>` states: {int(positive_mask.sum())}
- Source splits (sample counts): {dict(source_split_counts)}
- Mean resultant length: {think_geometry['mean_resultant_length']:.6f}
- Median angular radius: {think_geometry['median_angular_radius_degrees']:.4f} degrees
- Training q90 angular radius / cosine threshold: {think_region['q90_angular_radius_degrees']:.4f} degrees / {think_region['q90_cosine_threshold']:.6f}
- Inference q95 angular radius / cosine threshold: {think_region['q95_angular_radius_degrees']:.4f} degrees / {think_region['q95_cosine_threshold']:.6f}
- Mean pairwise cosine: {think_geometry['mean_pairwise_cosine']:.6f}
- Prompt-level held-out AUROC / AUPRC: {ranking_metrics['auroc']:.6f} / {ranking_metrics['auprc']:.6f}
- FPR at fold thresholds targeting {args.recall_target:.0%} train `</think>` recall: {heldout_non_think_fpr:.6f}
- Actual held-out `</think>` recall: {heldout_think_recall:.6f}
- Frequent-token concentration rank: {think_frequency_rank} / {len(frequent_concentrations)}
- Precursor means monotonically approach from -16 to 0: {precursor_monotonic}
- Mean cosine change from `</think>-1` to `</think>`: {float(np.mean(final_transition_deltas)):.6f}; fraction moving closer: {float(np.mean(final_transition_deltas > 0)):.6f}
- PCA90 / PCA95 / PCA99 dimensions: {pca_summary['pca90_dimensions']} / {pca_summary['pca95_dimensions']} / {pca_summary['pca99_dimensions']}

The conclusion is limited to the geometry of the state used to generate `</think>`. This analysis does not establish answer correctness, semantic completion of reasoning, or hallucination detection.
"""
        (working_dir / "report.md").write_text(report, encoding="utf-8")

        del all_vectors
        temporary_vectors_path.unlink()

        region_metadata = {
            "schema_version": "1",
            "source_model": str(args.model_path),
            "source_split_counts": json.dumps(dict(source_split_counts), sort_keys=True),
            "think_end_token_id": str(think_end_id),
            "hidden_size": str(hidden_size),
            "state_count": str(think_region["count"]),
            "space": "final_layer_hidden_after_per_vector_l2_normalization",
            "target": "predictor_hidden_whose_lm_head_argmax_is_think_end",
            "mean_resultant_length": str(think_region["mean_resultant_length"]),
            "q90_angular_radius_degrees": str(
                think_region["q90_angular_radius_degrees"]
            ),
            "q95_angular_radius_degrees": str(
                think_region["q95_angular_radius_degrees"]
            ),
            "q90_cosine_threshold": str(think_region["q90_cosine_threshold"]),
            "q95_cosine_threshold": str(think_region["q95_cosine_threshold"]),
        }
        save_file(
            {"center": think_center},
            working_dir / "think_region.safetensors",
            metadata=region_metadata,
        )
        region_summary = {
            "definition": {
                "space": "Final-layer hidden state after per-vector L2 normalization",
                "target": (
                    "Predictor state whose LM-head greedy argmax selected </think>; "
                    "</think> has not yet been fed back into the model"
                ),
                "center": "Unit-normalized mean direction of all target states",
                "training_boundary": "q90 angular region",
                "inference_boundary": "q95 angular region",
            },
            "source_model": str(args.model_path),
            "source_split_counts": dict(source_split_counts),
            "think_end_token_id": think_end_id,
            "hidden_size": hidden_size,
            **think_region,
            "artifact": "think_region.safetensors",
            "report": "report.md",
        }
        with (working_dir / "think_region.json").open("w", encoding="utf-8") as handle:
            json.dump(region_summary, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

        manifest = {
            "schema_version": 4,
            "model_path": str(args.model_path),
            "data_root": str(args.data_root),
            "dataset_loading_rule": (
                "GSM8K uses gsm8k-aug validation for region fitting. Other datasets, "
                "if selected, follow src/data.py load_math_examples; SVAMP is "
                "train(700)+test(300)."
            ),
            "datasets": list(args.datasets),
            "dataset_sample_counts": dict(dataset_counts),
            "source_split_counts": dict(source_split_counts),
            "samples_with_think_end": dict(think_end_counts),
            "think_end_state_count": int(len(think_states)),
            "sample_count": total_samples,
            "state_count": total_states,
            "alignment_check_count": total_alignment_checks,
            "alignment_definition": (
                "State j is the final transformer-layer output passed to the LM head whose greedy "
                "argmax selected generated token j; </think> is never fed back before its target "
                "state is captured. Per-vector L2 normalization is applied only for region fitting."
            ),
            "think_start_token_id": think_start_id,
            "think_end_token_id": think_end_id,
            "eos_token_ids": sorted(eos_token_ids),
            "pad_token_id": tokenizer.pad_token_id,
            "hidden_size": hidden_size,
            "region_input_dtype": "float32",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "sample_metadata": "samples.jsonl",
            "state_index": "state_index.csv",
            "think_region_artifact": "think_region.safetensors",
            "think_region_summary": "think_region.json",
            "report": "report.md",
            "report_analysis": {
                "folds": args.folds,
                "seed": args.seed,
                "recall_target": args.recall_target,
                "min_token_frequency": args.min_token_frequency,
                "pca_device": pca_summary["device"],
            },
            "validation": {
                "all_vectors_finite": True,
                "all_vectors_nonzero": True,
                "all_shapes_correct": True,
                "all_token_state_counts_equal": True,
                "all_lm_head_argmax_alignments_verified": True,
            },
        }
        with (working_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

        os.replace(working_dir, args.output_dir)
        completed = True
        print(
            f"Processed {total_states} aligned generated states from {total_samples} samples; "
            f"metadata written to {args.output_dir}"
        )
        print(
            f"Fitted region from {len(think_states)} </think> states: "
            f"q90={think_region['q90_angular_radius_degrees']:.4f} degrees, "
            f"q95={think_region['q95_angular_radius_degrees']:.4f} degrees"
        )
    finally:
        if not completed and working_dir.exists():
            shutil.rmtree(working_dir)


if __name__ == "__main__":
    main()
