# AE 评估 CD 差距的溯源调查

> 记录"为什么我们 AE 的 CD 与论文 Table 2 相差约 10 倍"的排查过程与最终归因。
> 范围限定：本次调查只针对 **AE（Table 2）**。生成模型评估（Table 1, FlowVAE / GaussianVAE）尚在进行，预测见 §7。

---

## 1. 问题

`eval_ae.py`（反归一化后）跑出的结果与论文 Table 2 相差约一个数量级：

| Category | 我们 `epoch_2000.pt` on `shapenet.hdf5` | 论文 Table 2 | 比值 |
|----------|---------------------------------------|-------------|------|
| Airplane | 0.178 (CD×10³) | 2.118 | 11.9× |
| Chair    | 0.511 | 5.677 | 11.1× |
| Car      | 0.577 | 5.493 |  9.5× |

三类都偏小约 10×，系统性差异，不是随机波动。

---

## 2. 排查路径

1. 先排除实现 bug —— 检查 CD 公式、反归一化操作
2. 测量本地 `shapenet.hdf5` 的坐标尺度
3. 溯源论文 pretrained ckpt 用的数据集
4. **决定性实验**：用论文官方权重在本地数据上跑，隔离"训练质量"和"数据集"这两个变量

---

## 3. 排除实现错误

### 3.1 CD 公式等价

**我们的实现**（`metrics.py`）：
```python
dist = ((p.unsqueeze(2) - q.unsqueeze(1)) ** 2).sum(dim=-1)  # ||x-y||²
d_pq = dist.min(dim=2).values.mean(dim=1)
d_qp = dist.min(dim=1).values.mean(dim=1)
return d_pq + d_qp
```

**原始 repo**（`evaluation/evaluation_metrics.py::distChamfer`）用矩阵展开形式 `‖x‖² + ‖y‖² − 2x·y` 得到 pairwise 距离，随后 `min(1)+min(2)` 再 `mean`，与上面等价。公式无差异。

### 3.2 反归一化一致

原始 `test_ae.py` 在计算 CD 前先做：
```python
ref    = ref    * scale + shift
recons = recons * scale + shift
metrics = EMD_CD(all_recons, all_ref, ...)
```

我们的 `eval_ae.py` 最初漏了这一步（直接在归一化坐标系下算 CD），补上后 airplane CD×10³ 从 13.48 降到 0.178。自此公式 / 反归一化均与原始 repo 对齐。

### 3.3 shapenet.hdf5 坐标尺度实测

```python
import h5py
with h5py.File('data/shapenet/shapenet.hdf5', 'r') as f:
    pcs = f['02691156']['test'][:5]  # airplane
# min/max: -0.407, 0.400
# per-shape std: [0.119, 0.120, 0.107, 0.116, 0.113]
```

`shapenet.hdf5` 中 airplane 的 per-shape std ≈ 0.115。`shape_unit` 归一化时 scale 取此值，反归一化时 CD 被乘以 `scale² ≈ 0.013`：13.48 × 0.013 ≈ 0.178，数字自洽。

至此说明：实现无 bug，0.178 是"在 shapenet.hdf5 原始坐标下的真实 CD"。剩下要问的是"为什么论文报告的 2.118 比我们大 10×"。

---

## 4. 溯源论文的训练数据

读取原作者发布的 pretrained ckpt（`diffusion-point-cloud/pretrained/AE_airplane.pt`）：

```python
ckpt = torch.load('AE_airplane.pt', weights_only=False)
print(vars(ckpt['args']))
# dataset_dir:  './data/ShapeNetCore.v2.PC15k.Resplit'
# scale_mode:   'shape_unit'
# max_iters:    1000000
```

三个直接结论（均来自 ckpt args，非推测）：

1. 论文 pretrained 模型训练用 `ShapeNetCore.v2.PC15k.Resplit`（15k points/shape），**不是** `shapenet.hdf5`
2. 论文完整训练 1,000,000 steps
3. PC15k 未随 Google Drive 发布包一同发布（发布包只含 `shapenet.hdf5` + `pretrained/*.pt`）—— 本地无该数据集，其坐标尺度无法直接实测

原始 `train_ae.py` 的默认 `--dataset_path` 又指向 `shapenet.hdf5`，即原始 repo 内部就存在 "训练脚本默认用的数据集" 与 "pretrained ckpt 实际用的数据集" 不一致。使用 `shapenet.hdf5` 做我们的训练是合理的（与原始 repo 的默认 training 脚本一致）。

---

## 5. 决定性实验：官方权重 × 本地数据

为了分清"10× 差距"到底来自训练不足还是数据集差异，把**论文官方的 `AE_all.pt`** 放到 `shapenet.hdf5` 上跑我们自己的 `eval_ae.py`。加载官方权重时需要做两处兼容处理：把编码器换成 BN + 三层 FC 双分支的版本（否则权重对不上键名），以及把 ckpt 里的 `var_sched` 从长度 201 截到 200（首项是占位的 ᾱ=1，不截会在采样最后一步除 0 得到 NaN）。这些只改"怎么把权重装进网络"，不动 CD / EMD 的定义、反归一化、聚合方式。

结果：

| Category | 官方 `AE_all.pt` on `shapenet.hdf5` | 我们 `epoch_2000.pt` on `shapenet.hdf5` | 论文 Table 2 |
|----------|-------------------------------------|----------------------------------------|-------------|
| Airplane | **0.1949** (CD×10³) | 0.178 | 2.118 |
| Chair    | **0.5521** | 0.511 | 5.677 |
| Car      | **0.5923** | 0.577 | 5.493 |

三个类别上，官方权重与我们自训权重在本地数据上的 CD 差异 <10%，而两者都约为论文的 1/10。

---

## 6. 结论

**~10× 的 CD 差距第一原因是评测所用数据集不同（`shapenet.hdf5` vs PC15k），不是我们模型训练不足。**

证据链：
- 同一套 eval 流程 + 同一官方权重 + 换成 `shapenet.hdf5` → CD 变成约 0.19（airplane），与我们自训结果几乎相同
- 官方权重本身经过 1M steps 训练，若"训练不足"是主因，它不应该也落到 0.19

**机制：** `scale_mode='shape_unit'` 做的是"按每个样本的原始坐标 std 归一化 → 训练 → 评测前用同一个 std 反归一化回去"。反归一化后的点云处在**原始坐标系**；CD 的量纲是坐标²。所以两个数据集的原始坐标 scale 不同，CD 就会差 scale² 倍。

- `shapenet.hdf5` airplane 原始坐标 per-shape std ≈ 0.115（实测）
- 论文结果 / 本地结果 ≈ 10× → 反推 PC15k 的 std ≈ 0.115 × √10 ≈ 0.36（仅为推断量级，PC15k 本地不可得，未实测）

次级因素（无法量化、但不是主因）：
- 训练步数差异（我们 ~62k vs 论文 1M）—— 被决定性实验证伪：官方 1M-step 权重在本地数据上也给出 0.19
- 两个数据集每 shape 点数不同（2048 vs 15000），以及可能的预处理差异

---

## 7. 范围限定与对 Table 1 评测的预测

**本次调查仅针对 AE（Table 2）。**

生成模型（FlowVAE / GaussianVAE，Table 1）的训练和评测仍在进行。基于同一机制（`shape_unit` + 反归一化 → 坐标系尺度差异），可以预期：

- **MMD-CD / MMD-EMD**（绝对距离类指标）：与 Table 2 类似，将比论文 Table 1 数值系统性偏小，因为 MMD 依赖 CD/EMD 本身的量纲。偏小幅度量级预计与 AE 一致（约 10×）。
- **COV-CD / COV-EMD / 1-NNA-CD / 1-NNA-EMD**（ratio / 分类准确率类指标）：**与 scale 无关**，因为它们是"是否某个阈值 / nearest-neighbor 归属"的相对关系，coord 整体伸缩不会改变排序。这些数值应与论文处于同一量级，可直接比较。

此外，论文 Table 1 的评测协议特别规定（Section 5.2）：
> "Following ShapeGF, when evaluating each of the model, we normalize both generated point clouds and reference point clouds into a bounding box of [−1, 1]³, and then compute the metric on the normalized point clouds."

这一步 **[−1, 1]³ bbox 归一化** 发生在 eval 时，独立于训练时的 `shape_unit`。当前 `scripts/eval_gen.py` 没有实现这步，所以即便在相同数据集上，我们的 MMD 也不会直接对齐论文 Table 1 的数字。若要完全复刻论文协议，需在 `eval_gen.py` 的指标计算前插入 bbox 归一化。暂不改动，视后续需要再说。

---

## 8. 附录：实现差异备查

### 8.1 当前仓库 vs 原始 GitHub repo

两者都是可运行代码，只在下列细节上不同：

| 组件 | 当前仓库 | 原始 repo |
|------|---------|-----------|
| PointNet Conv 层 | 无 BN | Conv + BN |
| PointNet FC head | 单层 `512→zdim` × 2（mu / logvar） | 三层 `512→256→128→zdim` × 2，中间带 BN |
| VarianceSchedule 长度 | 200 | 201（首项 ᾱ=1 为占位） |
| 时间嵌入 | `β_t`（噪声强度） | `β_t` —— 一致 |
| ConcatSquashLinear | `W1·h ⊙ σ(W2·c) + W3·c` | 一致 |
| PointwiseNet 激活 | LeakyReLU | LeakyReLU |

前两行是加载官方 pretrained 权重时必须解决的结构差异，已在 §5 讨论。后四行一致，不影响互操作。

### 8.2 关于 Google Drive 发布包里的 `suppl-v2.pdf`

作者把论文的补充材料 `suppl-v2.pdf` 和 pretrained ckpts 一起放在了 Google Drive 发布包里。**这份 PDF 对若干实现细节的描述与他们自己的 GitHub 代码也不完全一致**，例如：

- 时间嵌入写作 `c = [t, sin(t), cos(t), z]`，看上去是整数时间步 `t`；实际代码用的是 `β_t`
- PointNet 编码器结构在文中被略写为 `512-256-128-256` 单条 FC，实际代码是 mu / logvar 双分支、每支三层、带 BN
- 附录列出的 Flow 默认超参（14 层 coupling、hidden 256、F/G 为 128-256-256-128 MLP）与当前仓库默认（4 层、hidden 128）不同 —— 想复刻论文 Table 1 需按文档调大

因此遇到"文档说 A，代码是 B"的情况，**一切以代码 / ckpt args / 实测为准**，`suppl-v2.pdf` 只作为了解大致设计意图的参考材料使用。
