# 下一步实验计划

> 基于第一轮训练+评测（[`ae_training_and_eval.md`](ae_training_and_eval.md)）与 Gen 评测差距的排查（[`gen_eval_gap_analysis.md`](gen_eval_gap_analysis.md)），本文档只列下一轮要跑的消融实验：目标、改动点、预期信号。**不重复前两份文档的事实与数据**。
>
> 范围限定：仅针对生成任务（Table 1）。AE 部分在 [`dataset_investigation.md`](dataset_investigation.md) 中已经完成归因（数据集差异 → 数量级不可比），没有后续实验计划。

---

## 0. 目标

定位并缩小本轮 FlowVAE / GaussianVAE 与论文 Table 1 在 COV / 1-NNA / JSD 上的差距。每次消融只改一个维度，保证归因可分离。

不追求完全复现论文 Table 1 的绝对数字 —— 数据集差异（PC15k 本地不可得）是一个无法消除的混淆变量，详见 [`gen_eval_gap_analysis.md`](gen_eval_gap_analysis.md) §6.4。

---

## 1. 优先级与实验序列

| 优先级 | 消融维度                        | 预期信号                                                       | 训练时间影响 |
|--------|--------------------------------|----------------------------------------------------------------|--------------|
| P0     | 训练类别 `all 55 → airplane` only | COV↑ / 1-NNA→50% / JSD 下降若干个量级                          | ↓（数据量降）|
| P1     | LR schedule 加 plateau + 衰减段 | 同样 epoch 下 loss 更低、评测指标进一步收敛                    | 不变          |
| P2     | Flow 扩容至 14 层 / hidden 256  | 仅 FlowVAE 提升（GaussianVAE 不适用）                           | ↑（显存+算力）|
| P3     | Encoder 加 BN、FC head 改三层   | 可能改善 latent 表达，收益未知                                  | 不变          |

说明：P3 是原仓库结构，但当前仓库刻意选了简洁版作为学习读物（[`dataset_investigation.md`](dataset_investigation.md) §8.1）。仅在 P0-P2 都做完后仍有显著残差时才考虑。

---

## 2. P0：airplane-only 训练

### 改动点

仅改训练脚本，加一个 `--cates` 参数（与 `scripts/eval_gen.py` 的 `--cates` 对齐），透传给 `ShapeNetDataset(..., cates=...)`。不动模型结构、不动 LR schedule、不动 Flow 超参。

### 训练命令草案

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --cates airplane \
    --batch_size 384 \
    --model flow
```

airplane train 约 3438 shapes，batch 384 → 每 epoch 9 batch。若保持 2000 epoch，总 iter 数 ≈ 18K，远小于本轮全 55 类的 226K iter，需考虑提高 epoch 上限（例如 20000）以对齐 iter 数量级。

### 成功判据

- COV-CD 从 13% 上升到至少 30% 以上（减半的差距再拉回来一半）
- 1-NNA-CD 从 90% 下降向 50%（理想平衡点）
- JSD 从 0.44 下降至少一个数量级（从 10⁻¹ 进入 10⁻²）

如果达到以上信号，可以认为 §6.1 的归因成立，主因被锁定；否则还要继续剥离 P1-P3。

---

## 3. P1：LR schedule 对齐论文

在 P0 基础上，把 `LinearLR(1.0→0.0)` 换成"前 N iter plateau + 后 M iter 线性衰减"两段式，对齐论文 `sched_start=200K, sched_end=400K, end_lr=1e-4`。airplane-only 18K iter 的量级下，按比例缩放两个 iter 阈值（例如 plateau 9K + 衰减 9K）。

成功判据：同 epoch 下 val loss 更低、评测指标相对 P0 进一步改善。

---

## 4. P2：Flow 扩容（仅 FlowVAE）

在 P0 / P1 基础上，把 `--flow_layers 4 --flow_hidden_dim 128` 改为 `14` / `256`，对齐 `GEN_airplane.pt` 的 `latent_flow_depth=14, latent_flow_hidden_dim=256`。

GaussianVAE 无 flow，不做此实验。

成功判据：FlowVAE 在 P1 基础上 JSD / COV 进一步改善，且 FlowVAE 相对 GaussianVAE 拉开差距（验证可学习先验的价值）。

---

## 5. 不做的事

- **改用 PC15k 数据集**：不可执行，发布包未包含，本地无该数据集
- **Encoder 改回原仓库 BN + 三层 FC head**：本仓库定位是学习用途的简化实现，非必要不复杂化；仅在 P0-P2 都不足以解释残差时再回来动
- **kl_weight 调参**：`kl_weight=0.001` 与论文 ckpt 一致，不是差异来源
- **跑 55 类生成模型作为 Table 1 外的 ablation**：论文没有这一行，没有 baseline 可比，跑了也只能孤立地看数字

---

## 6. 代码改动清单（仅 P0 一步所需）

- `scripts/train_gen.py`：新增 `--cates` 参数，默认 `None`（保持当前行为），传给 `ShapeNetDataset`
- `scripts/train_ae.py`：建议同步加（对称性、日后可能用到），非本轮必要

其它维度的代码改动清单留到对应 P 阶段再补。
