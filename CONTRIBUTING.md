# Contributing

## Development setup

Use the pre-provisioned micromamba environment `latent-halt`; install the pinned
project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch must match the CUDA version on the machine used for training. The
dependency file intentionally leaves the PyTorch build selector to the user.

## Checks

Run a syntax check before opening a pull request:

```bash
python -m compileall -q src Analysis Ablations
```

Training and evaluation require external model weights and datasets; they are
never committed to this repository. Keep generated files under the ignored
`outputs/`, `results/`, `logs/`, and cache directories.

## Changes

Keep changes focused, document new command-line options, and avoid committing
machine-specific paths, credentials, checkpoints, or generated reports.
