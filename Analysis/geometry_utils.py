from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PRECURSOR_OFFSETS = (1, 2, 4, 8, 16)


def l2_normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a rank-2 array, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Hidden states contain NaN or infinity")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Hidden states contain a zero-norm vector")
    normalized = values / norms[:, None]
    return normalized.astype(np.float32, copy=False), norms.astype(np.float32, copy=False)


def generated_token_offset(prompt_length: int, generation_position: int) -> int:
    return prompt_length + generation_position


def predictor_state_offset(prompt_length: int, generation_position: int) -> int:
    return prompt_length + generation_position - 1


def precursor_positions(target_position: int) -> dict[int, int]:
    return {
        offset: target_position - offset
        for offset in PRECURSOR_OFFSETS
        if target_position >= offset
    }


def geometry_statistics(unit_vectors: np.ndarray) -> dict[str, float | int]:
    vectors = np.asarray(unit_vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("geometry_statistics requires at least one rank-2 vector")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError("geometry_statistics expects L2-normalized vectors")

    resultant = vectors.mean(axis=0)
    mean_resultant_length = float(np.linalg.norm(resultant))
    if mean_resultant_length > 0:
        center = resultant / mean_resultant_length
        center_cosines = np.clip(vectors @ center, -1.0, 1.0)
    else:
        center_cosines = np.zeros(len(vectors), dtype=np.float64)
    angles = np.degrees(np.arccos(center_cosines))

    if len(vectors) == 1:
        mean_pairwise_cosine = 1.0
    else:
        resultant_sum_squared = float(np.square(vectors.sum(axis=0)).sum())
        mean_pairwise_cosine = (resultant_sum_squared - len(vectors)) / (
            len(vectors) * (len(vectors) - 1)
        )

    return {
        "count": int(len(vectors)),
        "mean_resultant_length": mean_resultant_length,
        "median_angular_radius_degrees": float(np.median(angles)),
        "q90_angular_radius_degrees": float(np.quantile(angles, 0.9)),
        "mean_pairwise_cosine": float(mean_pairwise_cosine),
    }


def fit_angular_region(
    unit_vectors: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    vectors = np.asarray(unit_vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("fit_angular_region requires at least one rank-2 vector")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError("fit_angular_region expects L2-normalized vectors")

    resultant = vectors.mean(axis=0)
    resultant_length = float(np.linalg.norm(resultant))
    if not np.isfinite(resultant_length) or resultant_length <= 0:
        raise ValueError("The angular region has no valid mean direction")
    center = resultant / resultant_length
    center_cosines = np.clip(vectors @ center, -1.0, 1.0)
    angles = np.degrees(np.arccos(center_cosines))
    q90_angle = float(np.quantile(angles, 0.90))
    q95_angle = float(np.quantile(angles, 0.95))
    return center.astype(np.float32), {
        "count": int(len(vectors)),
        "mean_resultant_length": resultant_length,
        "q90_angular_radius_degrees": q90_angle,
        "q95_angular_radius_degrees": q95_angle,
        "q90_cosine_threshold": float(np.cos(np.radians(q90_angle))),
        "q95_cosine_threshold": float(np.cos(np.radians(q95_angle))),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional arrays of equal size")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    positives = labels == 1
    negatives = labels == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("Both positive and negative examples are required")

    ranks = _average_ranks(scores)
    auroc = (
        float(ranks[positives].sum()) - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)

    descending = np.argsort(-scores, kind="mergesort")
    sorted_positive = positives[descending]
    cumulative_positive = np.cumsum(sorted_positive)
    positive_ranks = np.flatnonzero(sorted_positive) + 1
    average_precision = float(
        np.mean(cumulative_positive[positive_ranks - 1] / positive_ranks)
    )
    return {
        "positive_count": n_positive,
        "negative_count": n_negative,
        "auroc": float(auroc),
        "auprc": average_precision,
    }


def threshold_at_recall(positive_scores: np.ndarray, recall: float = 0.95) -> float:
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    if len(positive_scores) == 0:
        raise ValueError("At least one positive score is required")
    if not 0 < recall <= 1:
        raise ValueError("recall must lie in (0, 1]")
    return float(np.quantile(positive_scores, 1.0 - recall, method="lower"))


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_labels = labels[order]
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    true_positive = np.cumsum(sorted_labels == 1) / positives
    false_positive = np.cumsum(sorted_labels == 0) / negatives
    return np.r_[0.0, false_positive], np.r_[0.0, true_positive]


def precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_positive = labels[order] == 1
    cumulative_positive = np.cumsum(sorted_positive)
    ranks = np.arange(1, len(labels) + 1)
    precision = cumulative_positive / ranks
    recall = cumulative_positive / max(int(sorted_positive.sum()), 1)
    return np.r_[0.0, recall], np.r_[1.0, precision]


@dataclass(frozen=True)
class FoldAssignment:
    sample_to_fold: dict[str, int]
    folds: int


def balanced_group_folds(
    sample_dataset_pairs: Iterable[tuple[str, str]], folds: int, seed: int
) -> FoldAssignment:
    if folds < 2:
        raise ValueError("At least two folds are required")
    by_dataset: dict[str, list[str]] = {}
    for sample_id, dataset in sample_dataset_pairs:
        by_dataset.setdefault(dataset, []).append(sample_id)
    rng = np.random.default_rng(seed)
    sample_to_fold: dict[str, int] = {}
    for dataset in sorted(by_dataset):
        sample_ids = sorted(set(by_dataset[dataset]))
        rng.shuffle(sample_ids)
        for index, sample_id in enumerate(sample_ids):
            sample_to_fold[sample_id] = index % folds
    return FoldAssignment(sample_to_fold=sample_to_fold, folds=folds)
