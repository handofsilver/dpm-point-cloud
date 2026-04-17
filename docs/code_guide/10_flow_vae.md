# 10 — FlowVAE

> 对应 `model.py: FlowVAE`，论文 Section 4.4

---

## 1. 与 GaussianVAE 的关系

`FlowVAE` 是 `GaussianVAE` 的直接升级，只改了两处：

| | GaussianVAE | FlowVAE |
|---|---|---|
| 先验 $p(z)$ | 固定 $\mathcal{N}(0,I)$ | `NormalizingFlow` 参数化 |
| KL 计算 | closed-form | MC 估计：$\log q - \log p_\theta$ |
| 生成时采 $z$ | `torch.randn` 直接得 $z$ | 先采 $u$，再过 `flow(u)` 得 $z$ |
| 可训练参数 | Encoder + Diffusion | Encoder + Diffusion + **Flow** |

编码、重参数化、扩散损失三步完全不变。

---

## 2. MC KL 估计与 log_det 符号

### 为什么 KL 不再有 closed-form？

GaussianVAE 的 KL 是两个高斯之间的散度，有解析解。FlowVAE 的先验 $p_\theta(z)$ 由 Flow 参数化，形式复杂，没有解析解，改用单样本 MC 估计：

$$\mathcal{L}_\text{KL} = \mathbb{E}_{z \sim q(z|x)}\bigl[\log q(z|x) - \log p_\theta(z)\bigr] \approx \log q(z|x) - \log p_\theta(z)$$

$z$ 已经通过重参数化采出来，直接代入即可（梯度照样流过）。

### 换元公式的符号推导

`flow.inverse(z)` 返回的 `log_det` 是：

$$\text{log\_det} = \sum_k \log|\det J_k^{-1}| = \log|\det J_\text{total}^{-1}|$$

换元公式为：

$$\log p_\theta(z) = \log p_u(u) - \log|\det J^\text{forward}|$$

因为 $\log|\det J^\text{forward}| = -\log|\det J^{-1}|$，代入得：

$$\log p_\theta(z) = \log p_u(u) + \log|\det J^{-1}| = \texttt{log\_p\_u} + \texttt{log\_det}$$

**常见 bug**：写成 `log_p_u - log_det`——符号错误，相当于把先验密度算反了。

---

## 3. 四处实现陷阱

| 位置 | 正确写法 | 错误写法 | 原因 |
|---|---|---|---|
| Step 4c | `log_p_u + log_det` | `log_p_u - log_det` | `log_det` 已是 $\log\|\det J^{-1}\|$，需加不需减 |
| Step 5 | `(log_q - log_p_z).mean()` | `log_q - log_p_z` | 缺 `.mean()`，结果是 `(B,)` 向量，`backward()` 不接受 |
| `sample` Step 1 | `torch.randn(..., device=device)` | `torch.randn(...)` | 缺 `device`，GPU 上设备不匹配报错 |
| `sample` Step 2 | `z = self.flow(u)` | `z, _ = self.flow(u)` | `flow.forward` 返回单个 Tensor，不是 tuple |

---

## 4. 梯度流向

`loss.backward()` 会同时更新三组参数：

- **Encoder**：通过 $\mathcal{L}_\text{diffusion}$ 和 $\log q(z|x)$
- **DiffusionPoint**：通过 $\mathcal{L}_\text{diffusion}$
- **NormalizingFlow**：通过 $\log p_\theta(z)$（即 $\log p_u(u) + \log|\det J^{-1}|$）

Flow 只靠 KL 项来学习——它被训练去让 $\log p_\theta(z)$ 尽可能大，即把先验分布塑造成贴合 encoder 输出 $z$ 的经验分布。

---

## 5. 数据流总览

```
训练:
  x0: (B, N, 3)
    → Encoder → mu, log_var: 各 (B, zdim)
    → 重参数化 → z = mu + exp(0.5*log_var) * eps: (B, zdim)
    → DiffusionPoint.get_loss(x0, z) → L_diffusion  [标量]
    → log q(z|x) = -0.5 * Σ[log2π + log_var + eps²].sum(dim=-1): (B,)
    → flow.inverse(z) → u: (B, zdim), log_det: (B,)
    → log p_u(u) = -0.5 * Σ[log2π + u²].sum(dim=-1): (B,)
    → log p_θ(z) = log_p_u + log_det: (B,)
    → L_KL = (log_q - log_p_z).mean()  [标量]
    → L = L_diffusion + 0.001 * L_KL   [标量]

生成:
  u ~ N(0,I): (B, zdim)
    → flow(u) → z: (B, zdim)
    → DiffusionPoint.sample(z, N, flex)
    → x: (B, N, 3)
```
