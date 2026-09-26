# LatentHalt

LatentHalt is a research codebase for adaptive reasoning in Llama-style causal
language models. It contains an explicit chain-of-thought SFT baseline,
Coconut/SIM-CoT latent-reasoning training, termination-region analysis,
and the corresponding math evaluation and ablation launchers.

This release contains the training, evaluation, analysis, and ablation code.
Model weights, datasets, caches, and runtime outputs are stored externally.
Set their paths through the environment variables described below.

## Repository layout

```text
LatentHalt/
|-- src/                 # data loading, models, training and evaluation
|-- scripts/             # main training/evaluation launchers
|-- Analysis/            # hidden-state geometry extraction and analysis
|-- Ablations/           # controlled ablation launchers
|-- requirements.txt
`-- README.md
```

Generated directories such as `outputs/`, `results/`, `logs/`, `output/`,
`eval/`, `.cache/`, and `.simcot_checkpoint_monitor/` are ignored by Git.
Large model/data files (`*.safetensors`, `*.bin`, `*.parquet`, `*.arrow`, etc.)
are also ignored. Keep them in a model hub or external storage and pass their
locations through the documented variables.

Analysis artifacts default to `results/analysis/`; ablation runs use
`outputs/ablations/<name>/`, `results/ablations/<name>/`, and
`logs/ablations/<name>/`. Training checkpoints are saved under `outputs/`.

## Requirements

- Linux with Bash and an NVIDIA H200 GPU visible to `nvidia-smi`
- Python 3.12 in a Conda environment
- PyTorch 2.5.1 with CUDA 12.4
- A local copy of the base model and the datasets described below

Create the environment once and install dependencies from the repository root:

```bash
git clone https://github.com/Yaemiko3/LatentHalt.git
cd LatentHalt
conda create -n latent-halt python=3.12 pip -y
conda activate latent-halt
conda install -c conda-forge jq -y
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported()"
```

The CUDA wheel follows the [official PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251).
The host NVIDIA driver must support the H200 and CUDA 12.4. `jq` is used by the
resume launcher to update checkpoint batch-size metadata.

For gated Hugging Face models, authenticate with `huggingface-cli login` or
the standard `HF_TOKEN` environment variable. Never place a token in a tracked
file. `.env.example` lists the path variables without any credentials.

## External data layout

The default launchers expect the following layout outside the repository. The
defaults are convenient for a sibling `models/` and `datasets/` directory, but
all paths can be overridden.

```text
<workspace>/
├── models/Llama-3.2-1B-Instruct/
└── datasets/
    ├── gsm8k-aug/data/{train,validation,test}-*.parquet
    ├── gsm-hard/gsmhardv2.jsonl
    ├── MultiArith/test.json
    └── SVAMP/{train,test}.json
```

Set paths explicitly on another machine:

```bash
export MODEL_PATH=/data/models/Llama-3.2-1B-Instruct
export DATA_ROOT=/data/datasets
export CACHE_DIR=/data/cache/latenthalt
```

`DATA_ROOT` is used by the SFT and evaluation launchers. Coconut and
SIM-CoT training additionally accept explicit `TRAIN_FILE` and
`VALIDATION_FILE` paths. `OUTPUT_DIR`, `RESULTS_DIR`, and `LOG_ROOT` can be set
to any writable location.

## Quick checks

The preview command checks data formatting and loss masks without performing a
training run. It needs the external model tokenizer and a small local dataset:

```bash
MODEL_PATH="$MODEL_PATH" DATA_ROOT="$DATA_ROOT" \
  bash scripts/preview_sft_cot_data.sh
```

Run a syntax check before submitting changes:

```bash
python -m compileall -q src Analysis Ablations
```

## Main workflows

### Explicit CoT SFT

The baseline trains on the sequence
`question + <think> + explicit steps + </think> + answer + EOS`. Only the
completion contributes to the language-model loss; over-length examples are
dropped instead of truncating the answer.

```bash
MODEL_PATH="$MODEL_PATH" DATA_ROOT="$DATA_ROOT" \
  bash scripts/train_llama1b_sft_cot.sh

MODEL_PATH="outputs/sft-cot-llama1b" \
DATA_ROOT="$DATA_ROOT" \
  bash scripts/eval_llama1b_sft_cot.sh --datasets gsm8k --max_samples 100
```

The first command writes checkpoints below `outputs/` and logs below `logs/`.
Each checkpoint is saved directly to `OUTPUT_DIR/checkpoint-<step>/` on disk.
Older checkpoints are removed after saving according to `SAVE_TOTAL_LIMIT`.
Resume and checkpoint monitoring use `OUTPUT_DIR` during training; final models
and logs stay in their disk directories without an end-of-run migration.
Use `PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `NUM_EPOCHS`,
`CUDA_VISIBLE_DEVICES`, `OUTPUT_DIR`, and the other variables defined at the
top of the launcher to adapt the run to available hardware.

### Hidden-state termination region

Fit the angular region used by latent halting from the trained explicit-CoT
model using only the GSM8K-Aug `validation` split by default:

```bash
MODEL_PATH="outputs/sft-cot-llama1b" \
DATA_ROOT="$DATA_ROOT" \
  bash Analysis/run_llama1b_think_geometry.sh
```

This creates an ignored artifact under
`results/analysis/think_hidden_geometry/`, including
`think_region.safetensors`. Use `OVERWRITE=1` to intentionally replace an
existing result.

### Coconut and SIM-CoT

Coconut replaces prefixes of explicit reasoning with continuous latent
positions and uses the fitted termination region during training. SIM-CoT adds
the auxiliary step decoder and curriculum supervision.

```bash
MODEL_PATH="outputs/sft-cot-llama1b" \
TRAIN_FILE="$DATA_ROOT/gsm8k-aug/data/train-00000-of-00001.parquet" \
VALIDATION_FILE="$DATA_ROOT/gsm8k-aug/data/validation-00000-of-00001.parquet" \
  bash scripts/train_llama1b_joint_simcot.sh
```

The joint launcher initializes from the SFT model. Evaluate an exported latent
model with:

```bash
MODEL_PATH=/path/to/exported/base_model \
TOKENIZER_PATH=/path/to/exported/base_model \
THINK_REGION_FILE=/path/to/think_region.safetensors \
DATA_ROOT="$DATA_ROOT" \
  bash scripts/eval_llama1b_latent.sh --datasets gsm8k --max_samples 100
```

The evaluator checks the q95 cosine boundary once per complete latent block,
forces `</think>` after halting, and extracts numeric answers only from the
text after the first closing think token. Per-example JSONL and summary files
are written to the ignored results directory.

### 8B joint SIM-CoT with FSDP (H200)

After training `outputs/sft-cot-llama8b`, fit the region with that same 8B model:

```bash
bash Analysis/run_llama8b_think_geometry.sh

CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_llama8b_joint_simcot.sh
```

The 8B analysis wrapper reads `outputs/sft-cot-llama8b` and uses only the GSM8K-Aug
`validation` split by default, using the same extraction as the 1B script. It writes
`results/analysis/think_hidden_geometry_llama8b/think_region.safetensors`,
with console logs under `logs/analysis/`. Run it before latent-halt training.

The 8B launcher reuses the 1B joint curriculum logic, loss weights, learning
rates, checkpoint rotation, and resume handling. It defaults to `C_THOUGHT=1`,
matching `../coconut/train_llama8b_gsm8k_aug.sh`. Its defaults are two H200 GPUs,
per-device train/eval batch size 4, and gradient accumulation 8 (global training
batch 64). Override these variables for a different GPU count or memory budget.

Training defaults to `NUM_EPOCHS=22` and `EPOCHS_PER_STAGE=3`, with
`MAX_LATENT_STAGE=10`. Stages 1–7 run for three epochs each; epoch 22 runs
stage 8, then training ends. This matches Coconut 8B's 22 executed training
epochs while preserving latent-halt's three-epoch stage schedule.

FSDP uses `full_shard auto_wrap`, wraps `LlamaDecoderLayer`, and keeps
`use_orig_params=True` with forward/backward prefetch disabled. BF16 and TF32
are enabled. Auxiliary-decoder gradient checkpointing is enabled by default;
set `AUXILIARY_GRADIENT_CHECKPOINTING=0` to disable it. The base-model path
retains its KV cache for latent reasoning.

Checkpoints are written directly to `outputs/latent-halt-llama8b/`, with logs
under `logs/`. The launcher selects `FULL_STATE_DICT` to keep checkpoint files
readable by the evaluator; final exports include `base_model/`
and `auxiliary_decoder/`. See the [Transformers FSDP documentation](https://huggingface.co/docs/transformers/v4.46.2/en/fsdp)
for sharding and state-dictionary configuration.

The auxiliary decoder initializes from `../models/Llama-3.1-8B-Instruct`. Model and region
paths can be overridden with `LLAMA8B_SFT_MODEL_PATH`,
`LLAMA8B_TOKENIZER_PATH`, `LLAMA8B_AUXILIARY_MODEL_PATH`, and
`LLAMA8B_THINK_REGION_FILE` (or the generic variables used by the 1B launcher).

### Ablations

Controlled variants are available under `Ablations/`:

```bash
bash Ablations/no_geometry/train.sh
bash Ablations/no_geometry/eval.sh
bash Ablations/no_auxiliary_decoder/train.sh
bash Ablations/squared_hinge/train.sh
bash Ablations/threshold_sweep/run.sh --dry_run
```

Inspect each launcher before a long run and override `CUDA_VISIBLE_DEVICES`,
data paths, output paths, and batch sizes for your environment.

## Reproducibility and provenance

Launchers record run metadata, hardware snapshots, and estimated continuation
FLOPs under the ignored `logs/` directory. Set `RUN_PLATFORM_TYPE` to one of
`internal_cluster`, `local_server`, or `cloud`, and set `RUN_PLATFORM_NAME` to a
non-sensitive label if you need platform metadata in experiment summaries.

The repository intentionally does not bundle Llama weights or benchmark data.
Follow the licenses and terms of the base model and each dataset separately.
See the README files under `Analysis/` and `Ablations/`, and the implementations
in `src/`, for details.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
