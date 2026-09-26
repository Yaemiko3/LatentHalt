# No auxiliary decoder ablation

This ablation removes the train-only auxiliary step decoder entirely. The
base model, curriculum, language-model loss, termination-region losses,
optimizer settings, and training budget match the formal 1B joint SIM-CoT
recipe. In particular:

```text
use_auxiliary_decoder = false
decoder_loss_weight = 0
```

The default single-card launcher uses per-device batch size 32 with gradient
accumulation 2 (effective batch size 64), matching the other local ablations.

When disabled, the training entrypoint does not load or validate an auxiliary
model, does not create decoder optimizer parameters, and does not compute the
decoder loss. The decoder target metadata remains available to identify latent
blocks and terminal blocks for the region objective; decoder-only length
filtering is skipped. The final export contains only `base_model` and no
`auxiliary_decoder` directory.

Checkpoints are saved directly to `outputs/ablations/no_auxiliary_decoder/checkpoint-<step>/` on disk.
After a new checkpoint is saved, Trainer removes older checkpoints according
to `SAVE_TOTAL_LIMIT` (default 2). Evaluation and resume use this same
model output directory; no end-of-run migration is needed.

Run from the repository root:

```bash
cd /path/to/LatentHalt
bash Ablations/no_auxiliary_decoder/train.sh
bash Ablations/no_auxiliary_decoder/eval.sh
```

Artifacts are grouped by experiment under the repository root:

```text
outputs/ablations/no_auxiliary_decoder/   # checkpoints and final base_model/
results/ablations/no_auxiliary_decoder/   # per-dataset predictions and summary
logs/ablations/no_auxiliary_decoder/      # training and evaluation logs
```

Use `--max_train_samples`, `--max_steps`, `--max_eval_samples`, or
`--eval_strategy no` after `train.sh` for a smoke test. The wrapper defaults to
`RESUME_FROM_CHECKPOINT=none`; set it explicitly only when continuing this
same ablation.
