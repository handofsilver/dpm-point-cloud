# 复现路线图

> Paper: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021
> 目标: 独立复现，理解每个组件的设计动机，而不是跑通代码。

---

## 工作模式约定

每个组件按如下节奏推进，**不跳步**：

1. **概念先行**：用对话解释"这个组件做什么、为什么这样设计"，对齐直觉
2. **写代码**：在 `model.py` 中新增对应的 `nn.Module`，逐行注释 tensor shape 和语义
3. **写测试**：在 `tests/` 下新增独立脚本，验证数学性质和形状正确性
4. **写文档**：在 `docs/code_guide/` 下新增对应的导读 `.md`，LaTeX + 代码块有机结合

**步子大小参考**：单次 session 实现且验证一个 `nn.Module`（约 30–80 行代码）。

---

## 整体依赖结构

```
VarianceSchedule                  ← 纯数学常数表，无参数
       │
       ├──► ConcatSquashLinear    ← 条件注入的基本层（单层）
       │           │
       │           └──► PointwiseNet     ← 6 层堆叠 + 时间嵌入 + 残差
       │                       │
       │                       └──► DiffusionPoint   ← 前向加噪 + 逆向采样
       │                                   │
       └──────────────────────────────────►│
                                           │
                              PointNetEncoder     ← 点云 → shape latent z
                                           │
                                    AutoEncoder   ← 第一个端到端可训练模型
                                           │
                                      FlowVAE     ← 最终生成模型
```

---

## 详细进度

### Phase 1 — 基础组件

| # | 组件 | 文件 | 测试 | 文档 | 状态 |
|---|---|---|---|---|---|
| 1-A | `VarianceSchedule` | `model.py` | `tests/test_variance_schedule.py` | `docs/code_guide/01_variance_schedule.md` | ✅ 完成 |
| 1-B | `ConcatSquashLinear` | `model.py` | `tests/test_concat_squash_linear.py` | `docs/code_guide/02_concat_squash_linear.md` | ✅ 完成 |
| 1-C | `PointwiseNet` | `model.py` | `tests/test_pointwise_net.py` | `docs/code_guide/03_pointwise_net.md` | ✅ 完成 |

**已记录概念**：
- 时间嵌入 `[β, sin(β), cos(β)]` 设计原理 → `paper_deep_dive.md §2`
- 残差连接 → `pytorch_notes.md §2`
- `LeakyReLU` vs `ReLU` → `pytorch_notes.md §3`

---

### Phase 2 — 扩散核心

| # | 组件 | 文件 | 测试 | 文档 | 状态 |
|---|---|---|---|---|---|
| 2-A | `DiffusionPoint.get_loss` | `model.py` | `tests/test_diffusion_point.py` | `docs/code_guide/04_diffusion_point.md` | ✅ 完成 |
| 2-B | `DiffusionPoint.sample` | `model.py` | （同上） | （同上） | ✅ 完成 |

**关键概念**：
- `get_loss`：对应 Algorithm 1，一步前向加噪（Eq.13）+ MSE 损失
- `sample`：逆向 Markov chain，T → T-1 → … → 0 逐步去噪
- 为什么训练时随机采 t 而不是遍历所有 t？（重要性采样直觉）

---

### Phase 3 — 编码器

| # | 组件 | 文件 | 测试 | 文档 | 状态 |
|---|---|---|---|---|---|
| 3-A | `PointNetEncoder` | `model.py` | `tests/test_encoder.py` | `docs/code_guide/05_pointnet_encoder.md` | ✅ 完成 |

**关键概念**：
- 为什么用 `Conv1d` 而不是 `Conv2d` 处理点云？（首次出现，需解释）
- MaxPool over N：如何把变长点集压缩成固定维度向量
- 双头输出 `(mu, log_var)` 与重参数化技巧
- AutoEncoder 模式只用 `mu`，为什么？

---

### Phase 4 — AutoEncoder（第一个可训练的完整模型）

| # | 组件 | 文件 | 测试/训练 | 文档 | 状态 |
|---|---|---|---|---|---|
| 4-A | `AutoEncoder` | `model.py` | `scripts/train_ae.py` | `docs/code_guide/06_autoencoder.md` | ✅ 完成 |
| 4-B | ShapeNet 数据加载 | `dataset.py` | — | — | ✅ 完成 |
| 4-C | 训练循环 + 可视化 | `scripts/train_ae.py` | — | — | ✅ 完成 |

**关键概念**：
- Chamfer Distance（CD）作为可视化评估指标（不作为训练损失）
- 点云归一化到零均值单位方差
- Adam + 梯度裁剪（`max_norm=10`）
- 超参: `T=200, β_T=0.05, lr=1e-3`

---

### Phase 5 — 生成模型（FlowVAE / GaussianVAE）

| # | 组件 | 文件 | 测试/训练 | 文档 | 状态 |
|---|---|---|---|---|---|
| 5-A | `GaussianVAE`（简单版，p(z)=N(0,I)） | `model.py` | `scripts/train_gen.py` | `docs/code_guide/07_gaussian_vae.md` | ✅ 完成 |
| 5-B-i | `AffineCouplingLayer` | `model.py` | — | `docs/code_guide/08_affine_coupling_layer.md` | ✅ 完成 |
| 5-B-ii | `NormalizingFlow`（K 层堆叠） | `model.py` | — | `docs/code_guide/09_normalizing_flow.md` | ✅ 完成 |
| 5-C | `FlowVAE`（完整生成模型） | `model.py` | `scripts/train_gen.py` | `docs/code_guide/10_flow_vae.md` | ✅ 完成 |

**关键概念**：
- 重参数化技巧：为什么 `z = mu + std * eps` 而不是直接 `sample(mu, std)`？
- KL 散度 closed-form 推导，以及为什么能写成 `-0.5 * (1 + log_var - mu² - exp(log_var))`
- `kl_weight=0.001`：量纲差异——扩散损失是 per-point MSE，KL 是 per-latent-dim 求和，权重平衡两者
- 超参: `T=100, β_T=0.02, lr=2e-3, kl_weight=0.001`（与 AutoEncoder 的对比）
- `mu ^ 2` vs `mu ** 2`：Python 按位异或陷阱（已记录）
- `flip` vs `reverse`：两个独立维度，混淆会导致 log_det 符号错误（已记录）

---

### Phase 6 — 评估与可视化（定性，已完成）

| # | 任务 | 文件 | 状态 |
|---|---|---|---|
| 6-A | 点云 3D 可视化 | `visualize.py` | ✅ 完成 |
| 6-B | Chamfer Distance 实现 | `metrics.py` | ✅ 完成 |
| 6-C | 重建效果定性对比（输入 vs 重建） | `scripts/reconstruct.py` | ✅ 完成 |
| 6-D | 生成多样性定性展示 | `scripts/generate.py` | ✅ 完成 |

**补齐测试**：

| 组件 | 测试文件 | 状态 |
|---|---|---|
| `DiffusionPoint` | `tests/test_diffusion_point.py` | ✅ 完成 |

---

### Phase 7 — 定量复现（Table 1 & Table 2）

> **目标**：复现论文 Table 1（生成模型，FlowVAE）和 Table 2（AutoEncoder 重建）的数字。
> 其他方法（AtlasNet、PointFlow 等）的数字直接从论文抄录，只跑 Ours 行。

#### 7-A：CD 和 EMD 底层实现（前提）

原始仓库的 `emd-cd` 分支依赖一个需手动编译的 CUDA 扩展（`approxmatch.cu`），其 Makefile 硬编码了 CUDA 10.0 / Python 3.7 / `caffe2` 库，**在 PyTorch 2.x + CUDA 13.0 环境下无法编译**。

**选定方案：用现代 pip 库替代，精度等价，无需编译 CUDA 扩展。**

| 指标 | 原始实现 | 替代方案 | 精度说明 |
|---|---|---|---|
| CD | `nndistance.cu`（双向最近邻） | `chamfer-distance`（pip 直装） | 完全等价，同一算法 |
| EMD | `approxmatch.cu`（近似匈牙利） | `geomloss` 的 `SamplesLoss("sinkhorn")` | 同为近似最优传输，误差量级相当，数值可比 |

> **注**：Table 1 中 DPM-3D 的核心优势体现在 EMD 列，不能省略。报告中需加脚注说明 EMD 使用 Sinkhorn 近似替代原 approxmatch kernel，这是复现工作的标准做法。

| 任务 | 文件 | 状态 |
|---|---|---|
| 安装 `chamfer-distance` 和 `geomloss` | `requirements.txt` 或 pip | ⬜ 待完成 |
| 用新库重写 `metrics.py` 中的 CD，新增 EMD 函数 | `metrics.py` | ⬜ 待实现 |

#### 7-B：Table 2 定量评估（AutoEncoder，重建质量）

Table 2 报告每个类别在测试集上的 mean CD 和 mean EMD（×10³ 等比例放大）。

| 任务 | 文件 | 状态 |
|---|---|---|
| 实现全测试集评估脚本 | `scripts/eval_ae.py` | ⬜ 待实现 |
| 在 chair/airplane/car 上跑出数字 | — | ⬜ 待训练 + 推理 |

**`eval_ae.py` 职责**：遍历测试集 → encode → diffusion sample → 逐样本 CD/EMD → 打印 per-category 均值（对照论文放大系数）

#### 7-C：Table 1 所需集合级指标（生成质量）

Table 1 使用 6 个数字：MMD-CD、MMD-EMD、COV-CD、COV-EMD、1-NNA-CD、1-NNA-EMD。

这三个指标均为**集合对集合**的比较，不是 per-sample 的距离。

| 指标 | 含义 | 状态 |
|---|---|---|
| **MMD**（Min. Matching Distance） | 每个生成样本找测试集中最近邻，取均值；衡量生成质量/保真度 | ⬜ 待实现（`metrics.py`）|
| **COV**（Coverage） | 测试集中被至少一个生成样本"命中"的比例；衡量多样性 | ⬜ 待实现（`metrics.py`）|
| **1-NNA**（1-Nearest Neighbor Accuracy） | 生成集 + 测试集混合，用 1-NN 分类器判断能否区分；越接近 50% 越好 | ⬜ 待实现（`metrics.py`）|

#### 7-D：Table 1 生成评估脚本

| 任务 | 文件 | 状态 |
|---|---|---|
| 实现 `eval_gen.py`：大批量采样 → 计算 MMD/COV/1-NNA | `scripts/eval_gen.py` | ⬜ 待实现 |
| 在 chair/airplane/car 上跑出数字 | — | ⬜ 待训练 + 推理 |

**`eval_gen.py` 职责**：
1. 从训练好的 FlowVAE 先验采样生成集 S_g（通常 2048 个形状）
2. 加载对应类别测试集 S_r
3. 调用 MMD/COV/1-NNA 函数，分别用 CD 和 EMD 各算一遍
4. 打印结果表格

#### 工作量汇总

| 阶段 | 核心工作 | 预估难度 |
|---|---|---|
| 7-A | EMD 实现（近似版本） | ★★★（算法复杂，需借助外部实现或自写近似） |
| 7-B | `eval_ae.py` + 跑 Table 2 | ★★（脚本简单，主要是 compute time） |
| 7-C | MMD / COV / 1-NNA 实现 | ★★（逻辑不复杂，但要处理大矩阵，注意内存） |
| 7-D | `eval_gen.py` + 跑 Table 1 | ★★（脚本简单，主要是 compute time） |

> **关键路径**：7-A（EMD）是阻塞项，7-B / 7-C / 7-D 都依赖它。建议先搞定 EMD，再并行推进其余。

---

## 文件结构（目标状态）

```
DPM_3D/
├── model.py              所有 nn.Module 定义
├── dataset.py            ShapeNet 数据加载
├── metrics.py            CD、EMD、MMD、COV、1-NNA
├── docs/
│   ├── code_guide/       每个组件的导读文档（LaTeX + 代码）
│   ├── notes/            学习笔记（pytorch_notes, paper_deep_dive, 本文档）
│   └── paper/            原论文 PDF + AI 摘要
├── tests/                每个组件的独立验证脚本
└── scripts/
    ├── train_ae.py       AutoEncoder 训练入口
    ├── train_gen.py      生成模型训练入口
    ├── eval_ae.py        Table 2 定量评估（重建 CD/EMD）
    └── eval_gen.py       Table 1 定量评估（MMD/COV/1-NNA）
```

---

## 如何续接上下文

如果切换了 session，告诉新的 agent：

> "我们正在复现 DPM-3D（Luo & Hu 2021）。当前进度见 `docs/notes/roadmap.md`。
> 工作模式：概念→代码→测试→文档，每次一个组件。
> 代码在 `model.py`，测试在 `tests/`，导读在 `docs/code_guide/`。
> 请先读 `docs/notes/roadmap.md` 和 `model.py` 了解当前状态，然后继续下一个待实现组件。"
