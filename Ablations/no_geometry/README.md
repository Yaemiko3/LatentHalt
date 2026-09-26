# No-geometry ablation

This ablation removes both sides of the termination-region training loss:

```text
region_loss_weight = 0
region_negative_loss_weight = 0
```

The language-model loss, auxiliary step-decoder loss, curriculum, optimizer,
and training budget remain those of the formal 1B joint SIM-CoT recipe. In
particular, the default per-device batch size is 32, gradient accumulation is 2,
and stage-wise learning-rate decay is disabled. The frozen
`think_region.safetensors` artifact is still loaded so training can log the
unweighted region losses and terminal scores, but both terms have zero weight
and therefore provide no geometry gradient. Evaluation uses the same artifact
to measure whether the trained model enters the region without being explicitly
trained to do so.

Checkpoints are saved directly to `outputs/ablations/no_geometry/checkpoint-<step>/` on disk.
After a new checkpoint is saved, Trainer removes older checkpoints according
to `SAVE_TOTAL_LIMIT` (default 2). Evaluation and resume use this same
model output directory; no end-of-run migration is needed.

Train from the SFT checkpoint with no automatic checkpoint resume:

```bash
cd /path/to/LatentHalt
bash Ablations/no_geometry/train.sh
```

Default artifact locations, relative to the LatentHalt repository root:

```text
outputs/ablations/no_geometry/   # checkpoints and final base_model/
results/ablations/no_geometry/   # per-dataset predictions and summary
logs/ablations/no_geometry/      # training and evaluation logs
```

Evaluate the exported model after training:

```bash
bash Ablations/no_geometry/eval.sh
```

The key readouts are in `results/ablations/no_geometry/summary.json`:

- `overall.region_halt_rate`: fraction of problems that naturally enter the
  frozen q95 region;
- `overall.max_budget_fallbacks`: problems that never enter it before the
  maximum latent-block budget;
- `*.jsonl`: per-problem `halt_score_trajectory` and `terminal_halt_score`.

The training entrypoint appends the two zero-valued region weights after any
extra command-line arguments, so this ablation cannot silently run with a
nonzero terminal or nonterminal geometry term. Set `RESUME_FROM_CHECKPOINT`
explicitly only when intentionally continuing this same ablation.
