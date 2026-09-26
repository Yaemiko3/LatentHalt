# 训练结果对比

更新日期：2026-09-03

## 阅读口径

- 主指标是 GSM8K 测试集 exact-match accuracy，共 1319 条。Latent-Halt 保存时评估默认也用这 1319 条做 checkpoint 选择。
- `best checkpoint` 是已有评测记录中的最高点；`训练末 step/epoch` 是该训练记录实际达到的末端。两者可能不同。
- `overall` 只在有完整四数据集结果时列出，数据集为 GSM8K、GSM-Hard、Multi-Arith、SVAMP，共 3818 条。
- 失败、dry-run、smoke 和纯评测运行不放入主比较表；有有效 checkpoint 评测的恢复/重启运行会保留并标注状态。
- 有效 batch 的计算为 `每卡 batch × 梯度累计 × GPU 数`。没有记录 GPU 数时不臆测全局 batch。

## SFT

| 模型/变体 | 每卡 batch | 累计 | 有效 batch | learning rate | 训练末 step / epoch | 最佳 checkpoint | GSM8K 最佳 acc | 完整四集 overall |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Llama-1B SFT（最终运行） | 32 | 1 | 128（4 GPU） | `1e-4` | 9039 / 3.00 | `checkpoint-9039` | **52.24%**（689/1319） | 47.43%（1811/3818） |
| Llama-3B SFT（默认 lr） | 32 | 2 | 64（1 GPU） | `1e-4` | 18075 / 3.00 | `checkpoint-18000` | 70.74%（933/1319） | - |
| Llama-3B SFT（lr 对照） | 32 | 2 | 64（1 GPU） | `2e-5` | 18075 / 3.00 | `checkpoint-17500` | **74.75%**（986/1319） | - |
| Llama-3B SFT（lr 对照） | 32 | 2 | 64（1 GPU） | `5e-5` | 18075 / 3.00 | `checkpoint-14500` | 73.77%（973/1319） | - |
| Llama-8B SFT | 16 | 8 | 未记录 | `2e-5` | 2259 / 3.00 | `checkpoint-2259` | 67.70%（893/1319） | 51.49%（1966/3818） |

3B 学习率对照中，`2e-5` 在已评测 checkpoint 上最高；`5e-5` 次之，`1e-4` 最低。1B 和 8B 的完整四集结果来自最终模型目录，3B 当前保留的逐 checkpoint 结果主要是 GSM8K。

## Latent-Halt / SIM-CoT

表中的 `base lr / decoder lr` 分别是主模型和辅助 decoder 的学习率。训练从对应 SFT 初始化，默认 curriculum 为 33 epoch、每 500 step 保存并在 GSM8K 上评估。

| 模型/变体 | 每卡 batch | 累计 | GPU / 有效 batch | base lr / decoder lr | 训练末 step / epoch | 最佳 checkpoint | GSM8K 最佳 acc | 状态/备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Llama-1B Latent-Halt | 32 | 2 | 未记录 | `1e-4 / 1e-5` | 约 197,695 / 32.81 | `checkpoint-123000` | **37.00%**（488/1319） | 四集 overall 32.45%（1239/3818） |
| Llama-3B Latent-Halt（restart，bucket-aligned） | 16 | 1 | 4 / 64 | `5e-5 / 1e-5` | 约 92,000 / 15.27 | `checkpoint-92000` | **39.50%**（521/1319） | 索引状态 failed；评测日志有效 |
| Llama-3B Latent-Halt（b8-ga2，recovery） | 8 | 2 | 4 / 64 | `5e-5 / 1e-5` | 198825 / 33.00 | `checkpoint-172500` | 38.89%（513/1319） | 索引状态 completed |
| Llama-8B Latent-Halt（b8-ga2，FSDP/layout-fix） | 8 | 2 | 4 / 64 | `2e-5 / 1e-5` | 约 72,000 / 11.99 | `checkpoint-66500` | **39.58%**（522/1319） | 评测日志可见的末端为 step 72000 |

3B 的两个恢复链路全局 batch 都是 64，但局部 batch/累计方式不同；已有记录显示 `b16 × 1` 的峰值略高于 `b8 × 2`。8B 的最佳记录与 3B restart 峰值接近，当前记录最高为 8B `checkpoint-66500` 的 39.58%。

## COCONUT（旧基线）

项目还保留一条 1B COCONUT 链路。两次正式记录均为 4 GPU，源码参数为每卡 batch 8、梯度累计 16、learning rate `1e-4`、30 epoch，有效 batch 为 512；训练末为 22590 step / 30.00 epoch。结果目录只有最终 `base_model` 的评测，没有逐 checkpoint accuracy，因此不能可靠地给出逐 checkpoint 的最佳点。

最终 `base_model` 在 GSM8K 上为 10.69%（141/1319），四集 overall 为 6.39%（244/3818）；固定使用 10 个 latent block，并发生全量 max-budget fallback。该结果只作为历史基线，不与动态 halt 的 Latent-Halt 主表排名。

## 同尺寸的直观比较

以下差值仅用于定位，不代表严格的同协议因果比较：SFT 使用显式 CoT、答案上限 256；Latent-Halt 使用连续 latent block、greedy 解码和答案上限 64。

| 尺寸 | SFT 最佳 GSM8K | Latent-Halt 最佳 GSM8K | acc 差值（latent - SFT） |
| --- | ---: | ---: | ---: |
| 1B | 52.24% | 37.00% | -15.24 pp |
| 3B | 74.75%（lr `2e-5`） | 39.50%（restart 峰值） | -35.25 pp |
| 8B | 67.70% | 39.58% | -28.12 pp |

当前结果显示，主要瓶颈在 latent 推理路径的训练/解码一致性，而不是 SFT 的显式答案生成能力。若要作严格结论，应固定答案长度、解码策略和数据集后重新评测。

## 数据来源

- 训练运行索引：`log/experiments.jsonl`
- SFT 结果：`/data-juice-nfs/latent-halt/results/eval-sft-cot-llama3b-*-checkpoint-*/summary.json`、`/data-juice-nfs/latent-halt/results/eval-sft-cot-debug/summary.json`、`/data-juice-nfs/latent-halt/results/eval-sft-cot-full/summary.json`、`/data-juice-nfs/latent-halt/results/eval-sft-cot-llama8b/summary.json`
- Latent-Halt 最高点：`log/latent-halt-3b-base5e-5-dec1e-5-restart/checkpoint_eval.log`、`log/latent-halt-3b-base5e-5-dec1e-5-b8-ga2-recovery109000/checkpoint_eval.log`、`log/latent-halt-8b-base2e-5-dec1e-5-b8-ga2-fsdp-layoutfix2/checkpoint_eval.log`，以及 `log/experiments.jsonl` 中的 1B monitor 记录
- 训练超参：各输出目录的 `training_args.bin` 与 `simcot_config.json`
- 默认值和 batch 语义：`docs/training_workflow.md`、`scripts/train_latent_halt.py`、`src/workflow_backend.py`
