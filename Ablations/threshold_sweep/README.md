# Inference threshold sweep

This report evaluates the LatentHalt v1 model on GSM8K while changing only the
inference halt threshold. The integer sweep value `p` is converted to cosine
threshold `p / 100`, so the default run covers `0.00` through `1.00` in steps of
`0.01`. The exact q90 and q95 thresholds from the region artifact are also
included by default, making 103 points in total. Pass
`--no-include_region_thresholds` for exactly the 101 integer percentage points.

Run from the repository root:

```bash
cd /path/to/LatentHalt
bash Ablations/threshold_sweep/run.sh
```

The default model is
`outputs/joint-simcot-llama1b_v1/base_model`, the dataset is GSM8K, and the
default CUDA device is `0`. Override paths or the device with environment
variables only when needed, for example:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash Ablations/threshold_sweep/run.sh --overwrite
```

Each threshold has its own raw evaluator output under `results/ablations/threshold_sweep/threshold_*`.
Logs are saved under `logs/ablations/threshold_sweep/`. The aggregated reports are:

```text
results/ablations/threshold_sweep/threshold_report.csv
results/ablations/threshold_sweep/threshold_report.json
results/ablations/threshold_sweep/threshold_report.md
```

They contain accuracy, region halt rate, max-budget fallbacks, average latent
blocks, answer length, and continuation FLOPs per question. Use
`--no-include_region_thresholds` to restrict the report to integer percentage
points, `--max_samples N` for a smoke test, or `--dry_run` to inspect the
planned commands without evaluating.

The sweep launches one existing `scripts/eval_llama1b_latent.sh` process per
threshold. This keeps each point identical to the standard evaluation protocol
but requires loading and evaluating the model once for every point.
