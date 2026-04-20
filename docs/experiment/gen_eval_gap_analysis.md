# Gen 评测与论文 Table 1 差距的排查

> 记录"为什么 FlowVAE / GaussianVAE 在 Table 1 评测协议下的 COV / 1-NNA / JSD 与论文相差悬殊"的排查过程与归因。
> 范围限定：仅针对 **生成任务（Table 1）**，不涉及 AE（Table 2 的 ~10× CD 差距归因见 [`dataset_investigation.md`](dataset_investigation.md)）。
>
> 第一轮完整训练 + 评测流程见 [`training_and_eval.md`](training_and_eval.md)；下一轮实验计划见 [`next_experiments.md`](next_experiments.md)。

---

## 1. 问题

第一轮评测（FlowVAE / GaussianVAE @ `epoch_2000.pt`，全 55 类合训，bbox 归一化后指标）相对论文 Table 1 系统性偏差。

论文 Table 1 "Ours" 换算到原始 scale（原表：CD×10³, EMD×10¹, JSD×10³）：

| Category | MMD-CD   | MMD-EMD | COV-CD | COV-EMD | 1-NNA-CD | 1-NNA-EMD | JSD      |
|----------|----------|---------|--------|---------|----------|-----------|----------|
| Airplane | 0.003276 | 0.1061  | 48.71% | 45.47%  | 64.83%   | 75.12%    | 0.001067 |
| Chair    | 0.012276 | 0.1784  | 48.94% | 47.52%  | 60.11%   | 69.06%    | 0.007797 |

我们 FlowVAE 结果：

| Category | MMD-CD   | MMD-EMD  | COV-CD | COV-EMD | 1-NNA-CD | 1-NNA-EMD | JSD      |
|----------|----------|----------|--------|---------|----------|-----------|----------|
| Airplane | 0.004911 | 0.035263 | 13.34% | 16.15%  | 90.44%   | 92.09%    | 0.441765 |
| Chair    | 0.016547 | 0.070243 | 20.83% | 21.33%  | 84.73%   | 89.23%    | 0.104457 |

比值（我们 / 论文）：

| Category | MMD-CD | MMD-EMD | COV-CD   | COV-EMD  | 1-NNA-CD | 1-NNA-EMD | JSD      |
|----------|--------|---------|----------|----------|----------|-----------|----------|
| Airplane | 1.50×  | 0.33×*  | 0.27×    | 0.35×    | 1.39×    | 1.23×     | **414×** |
| Chair    | 1.35×  | 0.39×*  | 0.43×    | 0.45×    | 1.41×    | 1.29×     | **13×**  |

\* EMD 用 `geomloss` Sinkhorn 近似替代论文 `approxmatch.cu`，绝对值不直接可比，列在这里只作记录，不参与归因。

可读出的三个方向：
- **MMD-CD 只偏大 35–50%**，不是灾难级
- **COV 减半以上、1-NNA 向 100% 靠拢**：分布覆盖差、生成样本与真实样本可区分性高，mode collapse 特征
- **JSD 数量级错位**（airplane 约 400× 差），分布匹配程度严重不足

---

## 2. 排查路径

1. 先排除实现错误：指标公式、bbox 归一化是否按论文协议
2. 从原作者发布的 pretrained ckpt 的 `args` 反推论文实际训练配置
3. 对齐配置差异，判定哪些可归因、哪些待消融实验验证

---

## 3. 排除实现错误

### 3.1 JSD 已补

`metrics.py` 第一版未实现 JSD。commit 3b06280 补上，实现按论文 Sec 5.2 通用做法：把两侧点云体素化到固定 grid（28³），归一化成概率分布后算 $\text{JSD}(P_g \| P_r) = \tfrac{1}{2}\text{KL}(P_g \| M) + \tfrac{1}{2}\text{KL}(P_r \| M)$，其中 $M = \tfrac{1}{2}(P_g + P_r)$。

### 3.2 bbox 归一化已加

论文 Sec 5.2 明确要求评测前把 `S_g` 和 `S_r` 都归一化到 $[-1, 1]^3$ bbox：

> "Following ShapeGF, when evaluating each of the model, we normalize both generated point clouds and reference point clouds into a bounding box of $[-1, 1]^3$, and then compute the metric on the normalized point clouds."

`scripts/eval_gen.py` 的 `normalize_to_bbox` 在计算指标前对 `S_g` 和 `S_r` 各自做 per-shape bbox 归一化（取最长轴做均匀缩放），commit 3b06280 已合入。归一化后两侧坐标都在 $[-1, 1]^3$，因此 **MMD-CD 数量级与论文直接可比**，不像 Table 2 AE 评测那样受 `shape_unit` 反归一化坐标系差异影响。

### 3.3 指标公式与原仓库一致

MMD / COV / 1-NNA 的实现与原仓库 `evaluation/evaluation_metrics.py` 对齐（pairwise CD / EMD + min + mean / one-hot / kNN 分类），AE 调查（§3.1）已经验证过公式无差异。

至此可以说：**评测流程本身合规，问题在 `epoch_2000.pt` 本身生成质量与论文差距大**。

---

## 4. 溯源论文训练配置

Google Drive 发布包的 `pretrained/` 目录里有 6 个 ckpt。逐一读取 `args`：

```python
import torch
for name in ['AE_airplane.pt','AE_chair.pt','AE_car.pt','AE_all.pt',
             'GEN_airplane.pt','GEN_chair.pt']:
    ckpt = torch.load(f'pretrained/{name}', weights_only=False, map_location='cpu')
    print(name, vars(ckpt['args']))
```

关键字段汇总（所有值均来自 ckpt，非推测）：

| ckpt              | model   | categories     | dataset_dir                | latent_flow_depth | latent_flow_hidden_dim | sched_start_epoch | sched_end_epoch | max_iters |
|-------------------|---------|----------------|----------------------------|-------------------|------------------------|-------------------|-----------------|-----------|
| AE_airplane.pt    | —       | `['airplane']` | `PC15k.Resplit`            | —                 | —                      | —                 | —               | 1,000,000 |
| AE_chair.pt       | —       | `['chair']`    | `PC15k.Resplit`            | —                 | —                      | —                 | —               | 1,000,000 |
| AE_car.pt         | —       | `['car']`      | `PC15k.Resplit`            | —                 | —                      | —                 | —               | 1,000,000 |
| AE_all.pt         | —       | `['all']`      | `PC15k.Resplit`            | —                 | —                      | 200,000           | 400,000         | 1,000,000 |
| **GEN_airplane.pt** | `flow`  | `['airplane']` | `PC15k.Resplit`            | **14**            | **256**                | 200,000           | 400,000         | 5,000,000 |
| **GEN_chair.pt**    | `flow`  | `['chair']`    | `PC15k.Resplit`            | **14**            | **256**                | 200,000           | 400,000         | 5,000,000 |

三个直接事实：
1. **Table 1 用的是 FlowVAE，per-category 训练**：发布的两个 GEN ckpt `model='flow'`、`categories` 各一个类别，没有 `GEN_all.pt` 和 `GEN_car.pt`。论文 Table 1 正文也只报 Airplane 和 Chair
2. **Flow 容量：14 层、hidden 256**；仓库默认（本轮实验）4 层、hidden 128
3. **LR schedule：前 200K iter 保持 `lr=2e-3` 不衰减**，然后在 200K→400K 线性衰到 `end_lr=1e-4`；本轮实验使用 `LinearLR(start=1.0, end=0.0, total_iters=epochs)`，从 epoch 0 就开始衰减

此外 §4 之外的事实：
4. 数据集差异：论文 pretrained 用 `ShapeNetCore.v2.PC15k.Resplit`（15k pts / shape），本仓库用 `shapenet.hdf5`（2048 pts / shape）。PC15k 未随发布包放出，本地不可得

---

## 5. 差异定位

本轮实验与论文 Table 1 FlowVAE 训练配置的对齐情况：

| 维度                 | 论文 pretrained                  | 本轮实验                         | 已验证 |
|----------------------|-----------------------------------|----------------------------------|--------|
| 训练类别             | per-category（airplane / chair） | 全 55 类合训                     | ✓ |
| Flow depth / hidden  | 14 / 256                          | 4 / 128                          | ✓ |
| LR schedule          | 200K iter plateau + 200K 衰减     | 从 epoch 0 线性衰减到 0          | ✓ |
| 数据集               | PC15k.Resplit（15k pts）          | shapenet.hdf5（2048 pts）        | ✓（PC15k 本地不可得） |
| model / categories 代码接口 | 原仓库 `--categories ['airplane']` | 仓库无 `--cates` 参数           | ✓ |
| 其它超参（`num_steps=100`, `beta_T=0.02`, `latent_dim=256`, `kl_weight=0.001`, `residual=True`, `lr=2e-3`, `end_lr=1e-4`, `batch_size=128` → 384） | 对齐（batch 384 为本轮覆盖） | 对齐 | ✓ |

---

## 6. 结论与归因

**本轮 Gen 评测与论文 Table 1 的差距不能完全归因于单一原因**。按影响程度排列的主要差异：

### 6.1 主因（高嫌疑）：全 55 类合训 vs per-category 训练

- 论文用 airplane-only / chair-only ckpt；先验 $p(z)$ 只需覆盖单类形状分布
- 本轮 ckpt 用全 55 类训练；先验必须覆盖跨类形状空间。评测时从先验无条件采样 2048 个形状，其中约 1/55 ≈ 1.9% 才是 airplane；绝大部分样本落在 car / chair / lamp / ... 的形状簇里
- 评测取 airplane 测试集做 MMD / COV / 1-NNA / JSD：
  - MMD（最小匹配距离）受影响小：总能从 2048 个样本里找到几个偏 airplane 的来匹配
  - COV（测试集被覆盖比例）受影响大：airplane 测试样本不易被采样到的 airplane 样本"覆盖"
  - 1-NNA 受影响大：airplane 真实样本与生成样本混合，kNN 分类器很容易把"非 airplane 的生成样本"判对，准确率推高
  - JSD（体素分布距离）受影响最大：生成分布横跨 55 类，真实分布只含 airplane，voxel 直方图大面积不重合
- 这一条可以**单枪匹马解释观察到的"MMD 只偏小、COV/1-NNA/JSD 严重崩"方向性特征**

### 6.2 次因 1：Flow 容量不足（仅 FlowVAE 适用）

- 论文 FlowVAE 用 14 层 Affine Coupling、hidden 256；本轮用 4 层 / hidden 128
- 可学习先验容量不足，$p(z)$ 拟合形状隐空间真实分布的能力受限
- 但 GaussianVAE（固定 $\mathcal{N}(0, I)$ 先验，无 flow）在同一轮实验里结果同样崩，说明 Flow 容量**不是 GaussianVAE 失败的主因**；仅对 FlowVAE 有定量贡献

### 6.3 次因 2：LR schedule 未做 plateau

- 论文：前 200K iter `lr=2e-3` 全速，随后 200K iter 衰到 1e-4
- 本轮：LinearLR 从 epoch 0 就开始线性衰减到 0
- 本轮总训练 iter 约 2000 epoch × 113 batch/epoch（43,433 / 384）≈ 226K iter，平均 LR 约 1e-3
- 训练总量与论文前 200K 阶段同量级，但**缺少"先充分训练再精调"的两阶段结构**，可能导致收敛点偏离论文 ckpt

### 6.4 结构性未知项：PC15k vs shapenet.hdf5

- 论文 pretrained 训练在 PC15k（15k pts/shape），本轮在 shapenet.hdf5（2048 pts/shape）
- PC15k 本地不可得，数据集差异对生成质量的贡献**无法独立量化**
- AE 调查（[`dataset_investigation.md`](dataset_investigation.md) §6）证明 AE 评测的 ~10× CD gap 主因是此项；生成任务里 bbox 归一化消除了坐标尺度差异，但点云密度（2048 vs 15000）对训练过程本身的影响仍是混淆变量

---

## 7. 可验证性与下一步

本节的归因是**结构性推理**（config diff + 评测协议机制），不是消融实验证据。要把"55 类合训"定罪为主因，需要一个只改此维度的 airplane-only 实验。

下一轮实验按优先级依次消融上述差异，详见 [`next_experiments.md`](next_experiments.md)。
