#!/usr/bin/env python3
"""Run a GSM8K inference-threshold sweep for the LatentHalt v1 model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "joint-simcot-llama1b_v1" / "base_model"
DEFAULT_REGION = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "think_hidden_geometry"
    / "think_region.safetensors"
)
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "datasets"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
DEFAULT_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_llama1b_latent.sh"


@dataclass(frozen=True)
class ThresholdSpec:
    label: str
    percent: float
    cosine: float


REPORT_FIELDS = (
    "label",
    "threshold_percent",
    "halt_threshold_cosine",
    "correct",
    "total",
    "accuracy",
    "region_halts",
    "region_halt_rate",
    "max_budget_fallbacks",
    "average_latent_blocks",
    "min_latent_blocks",
    "max_latent_blocks",
    "average_answer_tokens",
    "average_flops_per_question_excluding_prefill",
    "result_dir",
    "summary_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep LatentHalt inference cosine thresholds on GSM8K. "
            "A threshold percentage p means cosine=p/100."
        )
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer_path", type=Path)
    parser.add_argument("--think_region_file", type=Path, default=DEFAULT_REGION)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--eval_script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "ablations" / "threshold_sweep",
        help="Directory containing one evaluation subdirectory per threshold.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "ablations" / "threshold_sweep",
        help="Directory containing the per-threshold evaluator logs.",
    )
    parser.add_argument("--start_percent", type=int, default=0)
    parser.add_argument("--end_percent", type=int, default=100)
    parser.add_argument("--step_percent", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--cuda_visible_devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--include_region_thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate the artifact's exact q90 and q95 thresholds.",
    )
    parser.add_argument(
        "--formal_experiment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Register every sweep point as a formal experiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned evaluations without running them.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def read_region_thresholds(region_file: Path) -> list[ThresholdSpec]:
    """Read exact q90/q95 values from the sidecar produced by geometry fitting."""
    sidecar = region_file.with_suffix(".json")
    if not sidecar.is_file():
        return []
    # Some existing geometry summaries append human-readable metrics after the
    # JSON object, so decode the first object instead of requiring EOF after it.
    with sidecar.open("r", encoding="utf-8") as handle:
        text = handle.read()
    payload, _ = json.JSONDecoder().raw_decode(text.lstrip())
    if not isinstance(payload, dict):
        raise ValueError(f"Region summary must start with a JSON object: {sidecar}")
    specs: list[ThresholdSpec] = []
    for label, key in (("q90", "q90_cosine_threshold"), ("q95", "q95_cosine_threshold")):
        value = payload.get(key)
        if value is None:
            continue
        cosine = float(value)
        if not 0.0 <= cosine <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1], got {cosine}")
        specs.append(ThresholdSpec(label, cosine * 100.0, cosine))
    return specs


def build_thresholds(args: argparse.Namespace) -> list[ThresholdSpec]:
    if not 0 <= args.start_percent <= 100:
        raise ValueError("start_percent must lie in [0, 100]")
    if not 0 <= args.end_percent <= 100:
        raise ValueError("end_percent must lie in [0, 100]")
    if args.start_percent > args.end_percent:
        raise ValueError("start_percent must not exceed end_percent")
    if args.step_percent <= 0:
        raise ValueError("step_percent must be positive")

    specs = [
        ThresholdSpec(f"p{percent:03d}", float(percent), percent / 100.0)
        for percent in range(args.start_percent, args.end_percent + 1, args.step_percent)
    ]
    if not specs or specs[-1].percent != float(args.end_percent):
        specs.append(
            ThresholdSpec(
                f"p{args.end_percent:03d}",
                float(args.end_percent),
                args.end_percent / 100.0,
            )
        )

    if args.include_region_thresholds:
        specs.extend(read_region_thresholds(resolve(args.think_region_file)))

    deduplicated: dict[tuple[str, float], ThresholdSpec] = {}
    for spec in specs:
        if not 0.0 <= spec.cosine <= 1.0:
            raise ValueError(f"Threshold must lie in [0, 1], got {spec.cosine}")
        deduplicated[(spec.label, round(spec.cosine, 12))] = spec
    return sorted(deduplicated.values(), key=lambda spec: (spec.percent, spec.label))


def make_eval_command(args: argparse.Namespace, spec: ThresholdSpec) -> list[str]:
    command = [
        "bash",
        str(resolve(args.eval_script)),
        "--datasets",
        "gsm8k",
        "--batch_size",
        str(args.batch_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--halt_threshold",
        f"{spec.cosine:.12f}",
    ]
    if args.max_samples is not None:
        command.extend(("--max_samples", str(args.max_samples)))
    if args.formal_experiment:
        command.append("--formal_experiment")
    else:
        command.append("--no-formal_experiment")
    return command


def read_record(spec: ThresholdSpec, result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Evaluation summary was not produced: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    dataset = summary.get("datasets", {}).get("gsm8k")
    if not isinstance(dataset, dict):
        raise ValueError(f"GSM8K summary is missing from {summary_path}")
    return {
        "label": spec.label,
        "threshold_percent": spec.percent,
        "halt_threshold_cosine": spec.cosine,
        "correct": dataset["correct"],
        "total": dataset["total"],
        "accuracy": dataset["accuracy"],
        "region_halts": dataset["region_halts"],
        "region_halt_rate": dataset["region_halt_rate"],
        "max_budget_fallbacks": dataset["max_budget_fallbacks"],
        "average_latent_blocks": dataset["average_latent_blocks"],
        "min_latent_blocks": dataset["min_latent_blocks"],
        "max_latent_blocks": dataset["max_latent_blocks"],
        "average_answer_tokens": dataset["average_answer_tokens"],
        "average_flops_per_question_excluding_prefill": dataset[
            "average_flops_per_question_excluding_prefill"
        ],
        "result_dir": str(result_dir),
        "summary_path": str(summary_path),
    }


def write_reports(output_dir: Path, records: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "threshold_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    json_path = output_dir / "threshold_report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    markdown_path = output_dir / "threshold_report.md"
    columns = (
        "label",
        "threshold_percent",
        "halt_threshold_cosine",
        "accuracy",
        "region_halt_rate",
        "max_budget_fallbacks",
        "average_latent_blocks",
        "average_flops_per_question_excluding_prefill",
    )
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# LatentHalt inference threshold sweep\n\n")
        handle.write("GSM8K evaluation using the v1 base model. Percent is 100 times cosine.\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for record in records:
            values = []
            for column in columns:
                value = record[column]
                if column in {"accuracy", "region_halt_rate"}:
                    values.append(f"{float(value):.6f}")
                elif isinstance(value, float):
                    values.append(f"{value:.8f}")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    args.model_path = resolve(args.model_path)
    args.tokenizer_path = resolve(args.tokenizer_path or args.model_path)
    args.think_region_file = resolve(args.think_region_file)
    args.data_root = resolve(args.data_root)
    args.cache_dir = resolve(args.cache_dir)
    args.eval_script = resolve(args.eval_script)
    args.output_dir = resolve(args.output_dir)
    args.log_dir = resolve(args.log_dir)

    for path, label in (
        (args.model_path, "model_path"),
        (args.tokenizer_path, "tokenizer_path"),
        (args.think_region_file, "think_region_file"),
        (args.eval_script, "eval_script"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")

    thresholds = build_thresholds(args)
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory is non-empty: {args.output_dir}; use --overwrite"
                )
            if not args.dry_run:
                shutil.rmtree(args.output_dir)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    print(f"Model: {args.model_path}")
    print("Dataset: gsm8k")
    print(f"Threshold points: {len(thresholds)}")
    print(f"Output: {args.output_dir}")

    for index, spec in enumerate(thresholds, start=1):
        result_dir = args.output_dir / f"threshold_{spec.label}"
        run_id = f"threshold-sweep-{spec.label}-{timestamp}"
        environment = os.environ.copy()
        environment.update(
            {
                "MODEL_PATH": str(args.model_path),
                "TOKENIZER_PATH": str(args.tokenizer_path),
                "THINK_REGION_FILE": str(args.think_region_file),
                "DATA_ROOT": str(args.data_root),
                "CACHE_DIR": str(args.cache_dir),
                "RESULTS_DIR": str(result_dir),
                "LOG_DIR": str(args.log_dir),
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                "RUN_ID": run_id,
            }
        )
        command = make_eval_command(args, spec)
        print(
            f"[{index}/{len(thresholds)}] {spec.label}: "
            f"cosine={spec.cosine:.12f}"
        )
        if args.dry_run:
            print("  " + " ".join(command))
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        record = read_record(spec, result_dir)
        records.append(record)
        write_reports(args.output_dir, records)
        print(
            f"  accuracy={record['accuracy']:.4%}; "
            f"halt_rate={record['region_halt_rate']:.4%}; "
            f"avg_blocks={record['average_latent_blocks']:.3f}"
        )

    if args.dry_run:
        print("Dry run complete; no evaluation was started.")
        return
    write_reports(args.output_dir, records)
    print(f"Sweep report written to {args.output_dir / 'threshold_report.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
