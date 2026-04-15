# 05 · PointNetEncoder — 点云编码器

> 对应代码: `model.py` → `class PointNetEncoder`
> 对应论文: Section 4.3, Eq. (6)(7)

---

## 这个模块做什么？

把一朵点云 $X = \{x_i\}_{i=1}^N \in \mathbb{R}^{N \times 3}$ 压缩成 shape latent $z$ 的**分布参数**：

$$
\mu,\ \log\sigma^2 = \text{PointNetEncoder}(X) \quad \in \mathbb{R}^{F}
$$

这是整个模型的"理解"端：Encoder 看懂一朵点云的形状，把它压缩成一个紧凑向量，再交给扩散模块去重建或生成。

---

## 为什么要输出分布参数，而不是直接输出 $z$？

| 模式 | 如何使用 Encoder 输出 |
|:-:|:-:|
| **AutoEncoder** | 直接取 $\mu$ 作为 $z$（不采样，无随机性）|
| **GaussianVAE** | 重参数化采样：$z = \mu + \varepsilon \cdot e^{\frac{1}{2}\log\sigma^2}$，$\varepsilon \sim \mathcal{N}(0, I)$（Eq. 7）|

AutoEncoder 只需要 $\mu$，但 Encoder 的结构始终输出双头，两种模式复用同一个 Encoder。使用 $\log\sigma^2$ 而非 $\sigma$ 是数值稳定性的惯常做法：$\log\sigma^2$ 可以取任意实数，而 $\sigma$ 必须为正。

---

## 网络结构

```
输入: X  (B, N, 3)

1 调整维度: permute(0,2,1)
  → (B, 3, N)

2 逐点特征提取（4 层 Conv1d，kernel=1）:
  (B,   3, N) →[ReLU]→ (B, 128, N)
  (B, 128, N) →[ReLU]→ (B, 128, N)
  (B, 128, N) →[ReLU]→ (B, 256, N)
  (B, 256, N) →[ReLU]→ (B, 512, N)

3 MaxPool over N:
  (B, 512, N) → (B, 512)   # 置换不变聚合

4 双头 FC:
  (B, 512) → mu      : (B, zdim)
  (B, 512) → log_var : (B, zdim)
```

---

## 为什么用 Conv1d 而不是 Linear？

直觉上，对 N 个点各做一次线性变换，用 `nn.Linear` 也能做到。两者在数学上等价，但有一个关键区别：

- `nn.Linear` 期望输入 `(B, N, C)`，可以直接用，但语义上"N 个点"和"序列"容易混淆
- `nn.Conv1d(kernel=1)` 期望 `(B, C, N)`，**kernel_size=1 意味着每个位置只看自己**，明确表达了"逐点独立处理"的语义

PointNet 原论文使用 Conv1d 的惯例已被广泛沿用，选择 Conv1d 也让代码更贴近文献实现。

---

## MaxPool：置换不变性的来源

点云没有固定顺序，同一朵椅子的 2048 个点可以有 $2048!$ 种排列。网络必须对所有排列给出相同的 $z$，即**置换不变**（Permutation Invariance）。

MaxPool 天然满足这个性质：

$$
\text{MaxPool}(\{f(x_1), f(x_2), \ldots, f(x_N)\}) = \text{MaxPool}(\{f(x_{\pi(1)}), \ldots, f(x_{\pi(N)})\})
$$

对任意排列 $\pi$ 成立。只要 MaxPool 之前的操作是**逐点独立**的（Conv1d kernel=1 保证了这一点），整个 Encoder 就是置换不变的。

---

## 与 DiffusionPoint 的接口

```
AutoEncoder 模式:
    mu, log_var = encoder(x0)
    z = mu                         # 直接取均值，无随机性
    loss = diffusion.get_loss(x0, z)

GaussianVAE 模式:
    mu, log_var = encoder(x0)
    eps = torch.randn_like(mu)
    z = mu + eps * (0.5 * log_var).exp()   # 重参数化
    loss_diffusion = diffusion.get_loss(x0, z)
    loss_kl = -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=-1).mean()
    loss = loss_diffusion + kl_weight * loss_kl
```

---

## 验证结果

```
输入 x   : [4, 2048, 3]
mu       : [4, 256]       ✓ 形状正确
log_var  : [4, 256]       ✓ 形状正确

打乱点顺序后 mu 不变     : True    ✓ 置换不变性
打乱点顺序后 log_var 不变: True    ✓ 置换不变性
不同输入 → 不同 mu       : True    ✓
mu ≠ log_var（双头独立） : True    ✓
```

---

## 与后续模块的关系

```
PointNetEncoder
    └── AutoEncoder      ← 取 mu 作为 z，接 DiffusionPoint.get_loss
    └── GaussianVAE      ← 重参数化采样 z，加 KL 损失项
```
