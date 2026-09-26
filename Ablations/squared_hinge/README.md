# Squared terminal hinge ablation

The formal LatentHalt joint recipe uses a linear terminal hinge. This ablation
keeps that recipe fixed and changes only the terminal positive region term to
the natural squared hinge:

```text
L_terminal = mean(max(tau90 - final_score, 0)^2)
```

The nonterminal region term remains the original squared hinge,
`mean(max(nonterminal_score - tau95, 0)^2)`. SFT initialization, curriculum,
region weights, decoder loss, optimizer settings, and the q95 inference halt
rule are unchanged. The wrapper fixes the mode after any user arguments so it
cannot silently run the linear objective.

Checkpoints are saved directly to `outputs/ablations/squared_hinge/checkpoint-<step>/` on disk.
After a new checkpoint is saved, Trainer removes older checkpoints according
to `SAVE_TOTAL_LIMIT` (default 2). Evaluation and resume use this same
model output directory; no end-of-run migration is needed.

From the repository root:

```bash
cd /path/to/LatentHalt
bash Ablations/squared_hinge/train.sh
bash Ablations/squared_hinge/eval.sh
```

Artifacts are isolated under this directory:

```text
outputs/ablations/squared_hinge/   # checkpoints and final base_model/
results/ablations/squared_hinge/   # per-dataset predictions and summary
logs/ablations/squared_hinge/      # training and evaluation logs
```

Use `--max_train_samples`, `--max_steps`, `--max_samples`, or
`--eval_strategy no` after `train.sh` for a smoke test. Set `RESUME_FROM_CHECKPOINT`
explicitly to continue this ablation; it defaults to `none` so an old run is
never resumed accidentally.

The selected objective is recorded as
`region_positive_loss_type: "squared"` in `outputs/ablations/squared_hinge/simcot_config.json` and the
training run summary. Compare its `results/ablations/squared_hinge/summary.json` with the formal linear
run's evaluation summary using the same datasets and checkpoint-selection
rule.
