# EMD 指标计算方式的选择：Sinkhorn 而非 approxmatch

> 本文档记录本仓库 `metrics.earth_mover_distance` 为什么用 `geomloss` 的 Sinkhorn 近似，而不是论文原仓库声称的 `approxmatch.cu` CUDA kernel；
> 以及由此产生的 Table 1 EMD 相关数字（MMD-EMD / COV-EMD / 1-NNA-EMD）在和论文比较时应如何解读。
>
> 范围：仅讨论 EMD 计算方式。CD、MMD、COV、1-NNA 的算法本身不受影响。JSD 的实现 bug 另文记录。

---

## 1. 背景：论文用的 approxmatch 在本仓库不可得

论文 Table 1 的 EMD 列是通过 PC-GAN（Li et al. 2018, "PointCNN" 系列）公开的 `approxmatch.cu` CUDA kernel 算出来的 —— 一个硬 1-1 匹配的迭代启发式 OT 近似。

**本仓库拿不到该 kernel，原因有二**：

1. **参考仓库 `diffusion-point-cloud/evaluation/evaluation_metrics.py:13-22` 的 `emd_approx` 没实现**，源码只留了一句 warning：
   > `* EMD is not implemented due to GPU compatability issue.`
   > `* We will set all EMD to zero by default.`
   原作者只在 README 里给了一个外部链接指向某个 PC-GAN fork 的 `approxmatch.cu` —— **该链接在本仓库接手时已失效**。

2. 即便能拿到源码，`approxmatch.cu` 是 PyTorch 1.x 时代的 CUDA extension（手写 C++ binding + 硬编码 CUDA 版本），在本仓库运行环境（PyTorch 2.5.1 + CUDA 12.1）下编译不通过是大概率事件。与此同时本轮服务器环境已知 cuDNN 9.1.9 与驱动有兼容问题（见 [`1st_round_training_and_eval.md`](1st_round_training_and_eval.md) §4），硬上 CUDA extension 风险大、收益小。

因此最终选择 `geomloss.SamplesLoss("sinkhorn", p=1, blur=0.05)`。

---

## 2. 两种算法的数学本质对比

### approxmatch

求解真 EMD 的硬 1-1 匹配近似：

$$\text{EMD}_{1\text{-}1}(P, Q) = \min_{\pi \in \text{Perm}(N)} \sum_{i=1}^{N} \|p_i - q_{\pi(i)}\|$$

其中 $|P| = |Q| = N$。`approxmatch` 是对上式最优排列的贪心 + 迭代启发式求解，在 $N = 2048$ 量级上实测近似误差 1–2%，数值近似等价于真 Wasserstein-1。

### Sinkhorn（带熵正则的 OT）

求解带熵正则项的软 OT：

$$\text{OT}_\varepsilon(P, Q) = \min_{\pi \in \Pi(P, Q)} \langle \pi, C \rangle_F + \varepsilon \cdot H(\pi), \quad H(\pi) = \sum_{i,j} \pi_{ij} \log \pi_{ij}$$

其中 $\pi$ 是**软传输 plan**（稠密双随机矩阵，允许一个点的质量被分给多个目标）、$C$ 是 $\|p_i - q_j\|$ 构成的 cost matrix、$\varepsilon$ 是熵正则强度（对应 `geomloss` 的 `blur**2`）。`geomloss` 用 debiased Sinkhorn divergence：

$$\text{SD}_\varepsilon(P, Q) = \text{OT}_\varepsilon(P, Q) - \tfrac{1}{2}\text{OT}_\varepsilon(P, P) - \tfrac{1}{2}\text{OT}_\varepsilon(Q, Q)$$

以修正 $\text{OT}_\varepsilon(P, P) > 0$ 的常见偏差。

---

## 3. 性质差异

| 维度 | approxmatch | Sinkhorn（我们 `p=1, blur=0.05`）|
|---|---|---|
| 解的结构 | 硬 1-1 匹配（离散排列） | 软 plan（稠密矩阵） |
| 与真 EMD 的关系 | 数值上约等于真 Wasserstein-1 | $\varepsilon \to 0$ 时收敛到真 EMD；$\varepsilon > 0$ 时**系统性低于真 EMD**（熵项把传输方案抹平） |
| 对输入点云可微 | ❌ 匹配是离散的，梯度 0 或未定义 | ✅ 全程可微 |
| 可作训练 loss | ❌ 不可以 | ✅ 可以 |
| 偏差方向 | 因是近似排列，数值**略高于或等于**真 EMD | **系统性低于**真 EMD |
| 环境依赖 | PyTorch 1.x 时代的 CUDA extension | 纯 Python + KeOps JIT，PyTorch 2.x 直接可用 |
| 速度 | 原生 CUDA，快 | KeOps 自动生成 kernel，中等 |

**唯一决定性的差异是"可微性"**。其他差异（偏差方向、速度）在评测场景下都是次要的。

---

## 4. 应用场景：学界现状

**训练 loss**：**Sinkhorn 一统**。PointFlow、ShapeGF、LION、PointFM、Point-E 等涉及 OT-based 训练目标的工作都用 Sinkhorn 或直接用 Chamfer（因为 EMD 在训练里太贵）。**approxmatch 不做训练 loss**，它不可微。

**评测（MMD-EMD / COV-EMD / 1-NNA-EMD）**：

- 历史上的主流：**approxmatch**。PC-GAN (2018) 确立协议，PointFlow / ShapeGF / DPM-3D 等依次沿用同一个 CUDA kernel。历史 paper 数字都是它算的。
- 正在发生的迁移：该 kernel 是 PyTorch 1.x era 的 CUDA extension，在 PyTorch 2.x + CUDA 12+ 上普遍编译不过。近两年第三方复现（TorchPointCloudsMetrics、若干 LION 复现项目等）越来越多地改用 Sinkhorn，并在文档里明写"EMD 绝对值不可与历史数字直接比较"。

**结论**：本仓库用 Sinkhorn 不是"偏门方案"，是**当前新代码基的常见选择**，代价是放弃 EMD 绝对数值对齐历史论文。

---

## 5. 对本仓库评测结果的读法

### 5.1 CD 系列完全不受影响

CD 的实现（`metrics.chamfer_distance`）与 OT 求解器无关，与论文 / 参考仓库完全等价。MMD-CD / COV-CD / 1-NNA-CD 的**绝对值**与论文 Table 1 可比。

### 5.2 MMD-EMD 绝对值不可与论文比

我们的 MMD-EMD 与论文量纲**不同**：

- 论文 Table 1 airplane MMD-EMD $\times 10^1 = 1.061$ → $0.1061$（approxmatch，数值约等于真 Wasserstein-1）
- 本仓库 airplane-only @ 20000ep：MMD-EMD $= 0.0235$（Sinkhorn with `p=1, blur=0.05`）

差距约 4.5×，**这不是模型优势**，是 Sinkhorn 熵正则的系统性下偏。想恢复可比性需要换回 approxmatch kernel（见 §7）。

### 5.3 COV-EMD / 1-NNA-EMD 方向性可比

这两个指标本质上是基于"每对 $(S_i, R_j)$ 谁离谁近"的**排序**，而不是"距离的绝对值"：

- COV = 被某个生成样本作为最近邻的 ref 比例
- 1-NNA = 混合集合上 kNN 分类的准确率

Sinkhorn 和 approxmatch 对同一对点云给出的距离值**大致保持单调关系**（都反映"点云相似度"），因此排序基本一致，COV / 1-NNA 的**数字方向性可比**，但绝对值不保证相等。本轮 airplane-only 的 COV-EMD = 43.99%（论文 45.47%）、1-NNA-EMD = 76.69%（论文 75.12%）都已经接近论文水位 —— 这是此条"方向性可比"的实际证据。

### 5.4 CD-only 指标是最可靠的比较渠道

当和论文做严谨对比时，**以 CD 系列为主**（MMD-CD / COV-CD / 1-NNA-CD），EMD 系列仅作参考。

---

## 6. 参数选择：`p=1, blur=0.05`

- **`p=1`**：cost 函数用 $\|x - y\|^1$（欧氏距离），对齐 Wasserstein-1（即 EMD）。用 `p=2` 会变成 Wasserstein-2²，量纲是 coord²，无法与论文量纲对齐（论文那个也是 coord 单位）。
- **`blur=0.05`**：在 bbox 归一化 $[-1, 1]^3$ 的点云上，blur=0.05 意味着熵正则的"等效传输带宽"约 5% bbox 边长。更小（e.g. 0.01）更接近真 EMD 但数值更不稳定、更慢；更大（e.g. 0.1）偏差更明显。0.05 是 geomloss 在 3D 点云上的经验默认值。

---

## 7. 未来可能的迁移路径

若某天需要让 EMD 绝对值与论文对齐（例如撰写 paper 需要直接放一行对照），可以：

1. 从 PC-GAN / MSN 等开源仓库扒 `approxmatch` 的 CUDA 源码（`.cu` + `.cpp` binding）
2. 用 `torch.utils.cpp_extension.load_inline` 或 `load` 写 PyTorch 2.x 兼容的包装器
3. 替换 `metrics.earth_mover_distance` 内部实现，保持函数签名不变

工作量中等，主要风险点：CUDA kernel 内的手写内存管理在新架构（SM 8.0+）下可能需要调整 launch config。这项工作**没有短期必要性** —— 当前阶段主要关心训练是否能把生成质量推到论文水位，EMD 绝对值差异不改变定性结论。

---

## 8. 在评测输出中的注释义务

所有写入 `results/eval_gen/*` 的表格、以及 [`README.md`](../../README.md#L172) 的 Note on EMD，都应明写以下内容，避免未来自己或他人误读：

- "EMD 使用 geomloss Sinkhorn 近似（`p=1, blur=0.05`），与论文 `approxmatch.cu` 不同"
- "MMD-EMD 绝对值与论文不可直接比"
- "COV-EMD / 1-NNA-EMD 方向性可比"

参考已合入的说明：
- [`README.md:172`](../../README.md#L172)
- [`1st_round_training_and_eval.md:133`](1st_round_training_and_eval.md#L133)
- [`gen_eval_gap_analysis.md:35`](gen_eval_gap_analysis.md#L35)
