# 复现配置与运行步骤

本文说明当前发布代码的配置、数据用途和运行流程，不提供实验结果表。
所有命令均在解压后的仓库根目录运行；环境安装和外部数据布局见
[README](../README.md)。

## 训练配置

| LatentHalt 模型 | 训练入口 | 默认总轮数 | 每阶段轮数 |
| --- | --- | ---: | ---: |
| 1B | `scripts/train_llama1b_joint_simcot.sh` | 20 | 3 |
| 3B | `scripts/train_llama3b_joint_simcot.sh` | 20 | 3 |
| 8B | `scripts/train_llama8b_joint_simcot.sh` | 20 | 3 |

上述入口从 stage 1 开始，最大 latent stage 为 10；前 18 轮完成 stage 1–6，
第 19–20 轮执行 stage 7，然后结束训练。`NUM_EPOCHS` 可显式覆盖总轮数。
SFT 默认训练 3 轮，独立 Coconut 入口默认训练 30 轮；这两个配置与
LatentHalt 的 20 轮配置分别管理。

首次运行使用新的 `OUTPUT_DIR`，并设置 `RESUME_FROM_CHECKPOINT=none`。
恢复训练时，20 轮表示包含已完成轮数的总训练预算。

## 数据划分与模型导出

- 训练数据：`gsm8k-aug/data/train-*.parquet`。
- 训练期间的验证损失：`gsm8k-aug/data/validation-*.parquet`，默认每轮计算一次。
- 终止区域拟合：仅使用 GSM8K-Aug `validation` 划分，来源记录在 region 产物中。
- GSM8K 最终评测：`gsm8k-aug/data/test-*.parquet`，由单独的评测入口执行。

当前发布的 LatentHalt 训练入口按步数保存 checkpoint，默认每 500 步保存一次。
正常完成训练后，导出当时的模型到 `OUTPUT_DIR/base_model/`；启用辅助 decoder
时，还导出 `OUTPUT_DIR/auxiliary_decoder/`。当前入口没有启用按验证损失恢复
最佳模型，也不包含按 GSM8K 测试集分数选择 checkpoint 的 watcher。
因此，下述评测流程使用正常结束训练时导出的模型。

## 运行流程

先设置外部路径，并按模型尺寸选择相应的 SFT 入口：

```bash
export DATA_ROOT=/path/to/datasets
export CACHE_DIR=/path/to/cache/huggingface
export TRAIN_FILE="$DATA_ROOT/gsm8k-aug/data/train-00000-of-00001.parquet"
export VALIDATION_FILE="$DATA_ROOT/gsm8k-aug/data/validation-00000-of-00001.parquet"

# 1B 示例；3B / 8B 使用相应尺寸的基础模型和 SFT 启动脚本。
MODEL_PATH=/path/to/models/Llama-3.2-1B-Instruct \
  bash scripts/train_llama1b_sft_cot.sh
```

在 LatentHalt 训练前，用对应尺寸的 SFT 模型拟合 region。1B / 8B 分别运行
`Analysis/run_llama1b_think_geometry.sh` / `Analysis/run_llama8b_think_geometry.sh`。
3B 通过通用入口显式指定模型和输出路径：

```bash
MODEL_PATH=outputs/sft-cot-llama3b \
OUTPUT_DIR=results/analysis/think_hidden_geometry_llama3b \
  bash Analysis/run_llama1b_think_geometry.sh
```

选择一个尺寸训练。各入口的模型和 region 路径可按 README 覆盖：

```bash
NUM_EPOCHS=20 RESUME_FROM_CHECKPOINT=none \
OUTPUT_DIR=outputs/latent-halt-llama1b \
  bash scripts/train_llama1b_joint_simcot.sh

# 3B
NUM_EPOCHS=20 RESUME_FROM_CHECKPOINT=none \
LLAMA3B_TRAIN_FILE="$TRAIN_FILE" LLAMA3B_VALIDATION_FILE="$VALIDATION_FILE" \
OUTPUT_DIR=outputs/latent-halt-llama3b \
  bash scripts/train_llama3b_joint_simcot.sh

# 8B
NUM_EPOCHS=20 RESUME_FROM_CHECKPOINT=none \
OUTPUT_DIR=outputs/latent-halt-llama8b \
  bash scripts/train_llama8b_joint_simcot.sh
```

用训练完成后的 `base_model` 评测，并使用同尺寸的 region。以下以 8B 为例：

```bash
MODEL_PATH=outputs/latent-halt-llama8b/base_model \
TOKENIZER_PATH=outputs/latent-halt-llama8b/base_model \
THINK_REGION_FILE=results/analysis/think_hidden_geometry_llama8b/think_region.safetensors \
RESULTS_DIR=results/latent-halt-llama8b \
  bash scripts/eval_llama1b_latent.sh --datasets gsm8k gsm-hard multi-arith svamp
```

全量评测时不要设置 `--max_samples`。评测入口保存逐题输出及汇总指标，
便于在自己的运行环境中核对。

## 记录与核对

保留实际运行的命令、环境变量覆盖、训练配置、模型导出目录和 region 的
`manifest.json`。更改 batch size、梯度累计或 GPU 数量时，记录有效 batch
为 `每卡 batch × 梯度累计 × GPU 数`。消融实验入口见 `Ablations/`；比较同一
训练预算时，显式使用 `NUM_EPOCHS=20`，并核对各消融入口的其他默认参数。
