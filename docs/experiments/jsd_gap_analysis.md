# Table 1 JSD 与论文差距的归因

> 范围：只讨论 Table 1 的 JSD 列。
> CD 系列和 EMD 系列的归因另篇：[gen_eval_gap_analysis.md](gen_eval_gap_analysis.md)、[emd_sinkhorn_decision.md](emd_sinkhorn_decision.md)。
>
> 与 [ae_eval_gap_analysis.md](ae_eval_gap_analysis.md) 的调查套路同构：用**纯数据实验**把 gap 拆成"数据集结构性差异"与"模型残差"两块。

---

## 1. 现象

airplane-only FlowVAE @ 20000 epoch，[scripts/eval_gen.py](../../scripts/eval_gen.py) 输出的 JSD = **0.011**。

论文 Table 1 Airplane 同列（还原到原始量纲，原表标注 JSD × 10³）：

| 行 | JSD |
|:-:|:-:|
| PC-GAN | 0.006188 |
| GCN-GAN | 0.006669 |
| TreeGAN | 0.015646 |
| PointFlow | 0.001536 |
| ShapeGF | 0.001059 |
| **Ours（论文）** | **0.001067** |
| **Train（论文 lower bound）** | **0.000809** |

我们 0.011 落在 GAN baseline 附近，比论文 "Ours" 大 10×、比论文的 lower bound "Train" 大 14×。此前 CD / EMD 分析已经把这套类问题明确为"数据集差异主导"（[ae_eval_gap_analysis.md](ae_eval_gap_analysis.md) §6），本文档用同一方法对 JSD 做一次。

---

## 2. 结论（先给出）

**0.011 的 JSD 由两部分叠加构成，我们实际能压缩的只有一半**：
$$
\text{JSD}_\text{我们实测} \approx \underbrace{\text{JSD}_\text{数据下限}}_{\text{~0.007，不可压缩}} + \underbrace{\text{JSD}_\text{模型残差}}_{\text{~0.004，训练可改善}}
$$

- **数据下限 ~0.007**：在 `shapenet.hdf5` 上，airplane 类 train vs test 本身的 JSD 就是 ~0.007，和有没有模型无关。论文在 `PC15k.Resplit` 上的同量是 0.000809，相差 9×，**这 9× 是两个数据集的结构性差异，我们手里无 PC15k，无法跨越**

- **模型残差 ~0.004**：airplane-only FlowVAE 生成集相对同尺寸真实子集多出的那部分 JSD，由训练配置与模型容量决定，可以通过 [next_experiments.md](next_experiments.md) 里的 P1（LR plateau）+ P2（Flow 扩容）继续收

---

## 3. 证据：数据下限 JSD

调查脚本 [scripts/investigate_jsd_gap.py](../../scripts/investigate_jsd_gap.py)。**完全不涉及任何模型采样**，仅读取 `data/shapenet/shapenet.hdf5` 里 airplane 类的 train / test 两个 split，用 eval 里同一套 `normalize_to_bbox` + `jsd_between_point_cloud_sets` 流程计算下述几个量：

### 3.1 Oracle 1 — 同尺寸 train 子集 vs test

从 3438 个 airplane train 样本里随机抽 607 个（与 test 等大），与全体 test 算 JSD。这是**与论文 "Train" 行定义最接近的量**。

| 指标 | 值 |
|:-:|:-:|
| JSD(train₆₀₇, test₆₀₇) on `shapenet.hdf5` | **0.00697** |
| JSD("Train", test) on PC15k（论文 Table 1） | 0.000809 |
| 比值 | **8.6×** |

**解读**：即便没有任何模型参与，仅把 airplane 的 train 与 test 对比，我们数据给出的 JSD 就已经是论文 lower bound 的 8.6 倍。论文 0.001 级别的 JSD 在 `shapenet.hdf5` 上从起点就不可达。

### 3.2 Oracle 4 — 样本量扫描

排除"是不是样本数不够导致直方图太稀"这个假设。固定 test 为 607 样本，train 侧从 200 扫到 3438（我们能拿到的全部 airplane train）：

| train 样本数 | JSD |
|:-:|:-:|
| 200 | 0.01566 |
| 500 | 0.00857 |
| 1000 | 0.00665 |
| 2000 | 0.00586 |
| 3438（全部） | 0.00572 |

**解读**：样本量给的改善在 2000 之后趋于饱和。即便用满 airplane train 全集，JSD 仍在 0.0057，距离 0.000809 还有 7×。**结论：样本量不是 gap 的成因**。

### 3.3 Oracle 5 — 体素分辨率扫描

排除"论文可能用了不同 resolution"这个假设。固定 607 train vs 607 test、point-level 变体：

| resolution | JSD |
|:-:|:-:|
| 16 | 0.00432 |
| 20 | 0.00542 |
| 28 | 0.00812 |
| 32 | 0.01021 |
| 48 | 0.01764 |
| 64 | 0.02845 |

**解读**：降分辨率能让 JSD 下降，但即使 resolution = 16（远低于参考仓库默认 28）仍是 0.0043，**比论文 0.000809 高 5×**。resolution 选择也无法弥合 gap。

### 3.4 Oracle 6 — 归一化方式

排除"论文可能用了全局 bbox 归一化而非 per-shape"这个假设：

| 归一化方式 | JSD |
|:-:|:-:|
| per-shape bbox（eval_gen 当前实现） | 0.00812 |
| global bbox（全体点云共用一个 bbox） | 0.14796 |

**解读**：per-shape bbox 是正确选择，global bbox 会导致 JSD 剧增 18×。我们当前的归一化与论文 Sec 5.2 "normalize both generated and reference into [−1, 1]³" 的协议一致，**不是 gap 来源**。

### 3.5 小结：数据下限锁死在 0.006–0.008

在 607 vs 607 的评测配置下（与 eval_gen.py airplane-only 评测等大），无论怎么调实现细节，**`shapenet.hdf5` 上 airplane 类的 train/test JSD 下限大约是 0.007**。这是"数据的噪声本底"，任何模型再好也跨不过去。

---

## 4. 证据：模型残差

我们 eval 实测 JSD = 0.011，Oracle 1 数据下限 ~0.007，差值 ~0.004 即为当前 airplane-only FlowVAE 相对完美先验的剩余误差。

分解：

$$
0.011 \approx 0.007_{\text{数据下限}} + 0.004_{\text{模型残差}}
$$


这 0.004 的剩余压缩空间对应 [next_experiments.md](next_experiments.md) 里尚未消耗完的维度：

- **Flow 扩容到 14 层 / hidden 256（P2）**：论文 `GEN_airplane.pt` 的 `latent_flow_depth=14, latent_flow_hidden_dim=256`，当前 4 / 128。扩容后先验容量提升，生成分布能更好覆盖真实分布的尾部
- **其它训练配置细节**：本轮 airplane-only 已采用线性 LR 衰减（commit `d26c6a2`），与论文 "200K iter plateau + 200K 衰减" 两段式仍有差异。若残差需要进一步压缩，可再做两段式对齐实验

---

## 5. 为什么 PC15k 能给出 0.001 级别的 JSD

这个问题我们**无法直接回答**——PC15k 未随 Google Drive 发布包提供，本地不可得。只能列出它可能偏更"干净"的几个结构性原因（均属**推测，未经本地实测**，不入结论链）：

- PC15k airplane 子集可能体量更大（更多样本 → 直方图更平滑）
- PC15k 的 train/test split 可能按某种形状相似度对齐（覆盖面更一致）
- PC15k 的 2048 点来自对 15k 稠密点云的二次下采样，与 `shapenet.hdf5` 里直接储存的 2048 点统计性质可能不同

此条与 [ae_eval_gap_analysis.md](ae_eval_gap_analysis.md) §6 完全对称：坐标量纲差异造成 AE CD ×10，JSD 差异造成 ×9—— **两条 gap 的根都指向"我们用 shapenet.hdf5，论文用 PC15k"这一事实，无法消除**。

---

## 6. 复现实验的命令

纯数据调查脚本（不需要 GPU，也不读 checkpoint）：

```bash
python scripts/investigate_jsd_gap.py
```

输出覆盖本文 §3 全部 Oracle 数字。

---

## 7. 报告用法（读 Table 1 JSD 列时）

- **CD 系列指标与论文在量纲上可比**，bbox 归一化消除了坐标尺度差异
- **EMD 系列绝对值不可比**（Sinkhorn vs approxmatch，见 [emd_sinkhorn_decision.md](emd_sinkhorn_decision.md)）
- **JSD 绝对值不可比**（数据下限差 9×，见本文档）
  - 我们能对齐的是相对排序：FlowVAE 应优于 GaussianVAE、airplane-only 应优于全 55 类合训
  - 我们能压的是模型残差（~0.004），通过 P1 / P2 训练配置调整
  - 数据下限差（~0.007→0.001）不是我们能通过训练或评测代码改善的

这三条都属于同一个元问题"数据集差异"（`shapenet.hdf5` vs `PC15k.Resplit`），在 [ae_eval_gap_analysis.md](ae_eval_gap_analysis.md) 已形成证据链、本文档为 JSD 列补齐对应的数据证据。
