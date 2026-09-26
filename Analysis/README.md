# `</think>` Hidden-State Region

`extract_think_hidden.py` performs greedy generation, captures the final-layer predictor hidden
state for every generated token, and directly fits the `</think>` angular region. There is no
separate analysis step.

By default, extraction uses only the GSM8K-Aug validation split, loaded from
`$DATA_ROOT/gsm8k-aug/data/validation-*.parquet` via `src/data.py::load_gsm8k_aug`.
This applies to direct Python calls and both the 1B and 8B launchers. To run an explicit
multi-dataset analysis, override `--datasets`; GSM8K still uses validation in that case.
The regular math evaluators continue to use the GSM8K test split.

The target state is the hidden vector passed to the LM head when greedy argmax selects `</think>`.
At that point, `</think>` has not been fed back into the model. All region calculations use
per-vector L2-normalized float32 states.

Run the complete extraction and region fit with:

```bash
cd /path/to/LatentHalt
bash Analysis/run_llama1b_think_geometry.sh
```

Replace an existing output deliberately with `OVERWRITE=1`. The extractor can also be called
directly:

```bash
python Analysis/extract_think_hidden.py \
  --output_dir results/analysis/think_hidden_geometry
```

The output directory contains:

```text
think_hidden_geometry/
├── think_region.safetensors
├── think_region.json
├── report.md
├── manifest.json
├── samples.jsonl
└── state_index.csv
```

`think_region.safetensors` contains the global unit mean direction under the tensor name `center`.
Its metadata records the source model, source split sample counts, hidden size, `</think>` token
ID, state count, mean resultant length, and q90/q95 angular and cosine thresholds. Coconut training uses q90 for
the region loss; halt inference uses q95. `think_region.json` provides the same geometry in a
human-readable form. Full raw and normalized hidden states are not persisted; normalization and
region fitting happen in memory during extraction. The full normalized state matrix exists only as
a temporary file while the held-out metrics, token controls, precursor trajectory, and PCA summary
for `report.md` are calculated; it is deleted before the output directory is committed.
