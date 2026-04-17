# 07 — GaussianVAE

> 对应 `model.py: GaussianVAE`，论文 Section 4.3

---

## 1. 它解决了什么问题？

AutoEncoder 能重建点云，但无法**无条件生成**新形状。原因是 AutoEncoder 里 $z = \mu$（确定性），推理时必须先给一个真实点云才能得到 $z$——没有"凭空造"的能力。

GaussianVAE 通过两处修改解决这个问题：

1. 训练时用**重参数化**采 $z$，引入随机性
2. 加**KL 正则**把 $q(z \mid x)$ 拉向标准正态 $p(z) = \mathcal{N}(0, I)$

训练收敛后，$z$ 的分布接近 $\mathcal{N}(0, I)$，推理时直接从标准正态采 $z$ 就能生成新点云。

---

## 2. 重参数化技巧

### 问题

采样操作 $z \sim \mathcal{N}(\mu, \sigma^2 I)$ 不可微——梯度无法穿过随机节点流向 $\mu$ 和 $\sigma$。

### 解法

把随机性分离到一个**与参数无关的噪声变量** $\varepsilon$ 上：

$$
z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

现在 $z$ 对 $\mu$ 和 $\sigma$ 是确定性的可微函数，梯度可以正常流过：

$$
\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \varepsilon
$$

### 为什么用 `log_var` 而不是 `sigma`？

编码器直接输出 $\log \sigma^2$（`log_var`），而不是 $\sigma$：

$$
\sigma = \exp\!\left(\tfrac{1}{2} \log \sigma^2\right)
$$

因为 $\sigma > 0$ 是约束，而 $\log \sigma^2 \in \mathbb{R}$ 没有约束——网络可以输出任意实数，训练更稳定，不用担心输出负数再开根号。

```python
std = torch.exp(0.5 * log_var)   # (B, zdim)，保证 std > 0
eps = torch.randn_like(std)       # (B, zdim)，与参数无关
z   = mu + std * eps              # (B, zdim)，可微
```

---

## 3. KL 散度

### 直觉

KL 散度衡量两个分布的"距离"。我们要最小化：

$$
\mathcal{L}_\text{KL} = \mathrm{KL}\!\left(q(z \mid x) \;\|\; p(z)\right)
$$

其中 $q(z \mid x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$，$p(z) = \mathcal{N}(0, I)$。

这一项把编码器的输出分布推向标准正态——训练好之后，从标准正态采出的 $z$ 落在解码器"见过"的区域里，生成质量有保障。

### Closed-form 推导

对角高斯与标准正态的 KL 有解析解（Eq. 4 in VAE 原论文）：

$$
\mathrm{KL}\!\left(\mathcal{N}(\mu, \sigma^2) \;\|\; \mathcal{N}(0,1)\right) = -\frac{1}{2}\left(1 + \log \sigma^2 - \mu^2 - \sigma^2\right)
$$

对所有 $d$ 个 latent 维度求和：

$$
\mathcal{L}_\text{KL} = -\frac{1}{2} \sum_{d=1}^{D}\!\left(1 + \log \sigma^2_d - \mu_d^2 - \sigma^2_d\right)
$$

在代码中，先对 `zdim` 维 `.sum(dim=1)`，再对 batch `.mean()`：

```python
loss_kl = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1).mean()
```

> **常见陷阱**：Python 中 `^` 是**按位异或**，不是幂运算。对 float Tensor 执行 `mu ^ 2` 会报 `RuntimeError`。应使用 `mu ** 2` 或 `mu.pow(2)`。

---

## 4. 总损失

$$
\mathcal{L} = \mathcal{L}_\text{diffusion} + \lambda_\text{KL} \cdot \mathcal{L}_\text{KL}
$$

`kl_weight`（即 $\lambda_\text{KL}$）设为 `0.001` 的原因：

- $\mathcal{L}_\text{diffusion}$ 是 per-point MSE，规模较小（点云归一化后坐标在 $[-1, 1]$）
- $\mathcal{L}_\text{KL}$ 对 `zdim=256` 个维度求和，规模约大 $256\times$
- `kl_weight=0.001` 让两项在数量级上大致对齐，防止 KL 项过强压制重建能力

---

## 5. 推理：无条件生成

训练阶段的 KL 正则使 $q(z|x) \approx \mathcal{N}(0,I)$，推理时直接：

```python
z = torch.randn(batch_size, zdim, device=device)   # 从先验采 z
x = diffusion.sample(z, num_points, flexibility)   # 逆向扩散
```

无需任何输入点云，可以生成全新的三维形状。

---

## 6. 与 AutoEncoder 对比

| | AutoEncoder | GaussianVAE |
|---|---|---|
| $z$ 的来源（训练） | $z = \mu$（确定） | $z = \mu + \sigma\varepsilon$（随机） |
| $z$ 的来源（推理） | 编码真实点云 | 从 $\mathcal{N}(0,I)$ 采样 |
| 损失 | $\mathcal{L}_\text{diffusion}$ | $\mathcal{L}_\text{diffusion} + \lambda_\text{KL}\mathcal{L}_\text{KL}$ |
| 能力 | 重建 | 重建 + 无条件生成 |
| `T` | 200 | 100 |
| `beta_T` | 0.05 | 0.02 |
| `lr` | 1e-3 | 2e-3 |

---

## 7. 数据流总览

```
训练:
  x0: (B, N, 3)
    → PointNetEncoder
    → mu: (B, zdim),  log_var: (B, zdim)
    → 重参数化: z = mu + exp(0.5*log_var) * eps
    → z: (B, zdim)
    → DiffusionPoint.get_loss(x0, z)  →  L_diffusion
    → KL closed-form                  →  L_KL
    → L = L_diffusion + 0.001 * L_KL

推理:
  z ~ N(0, I): (B, zdim)
    → DiffusionPoint.sample(z, N, flex)
    → x: (B, N, 3)
```
