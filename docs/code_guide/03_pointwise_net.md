# 03 · PointwiseNet — 逐点噪声预测网络

> 对应代码: `model.py` → `class PointwiseNet`
> 对应论文: Section 4.2, Algorithm 1

---

## 这个模块做什么？

给定加噪后的点云 $x^{(t)}$、当前时间步的噪声强度 $\beta_t$、以及 shape latent $z$，预测出每个点上的噪声 $\hat\varepsilon$：

$$\hat\varepsilon = \varepsilon_\theta(x^{(t)},\ \beta_t,\ z) \quad \in \mathbb{R}^{B \times N \times 3}$$

训练目标：让 $\hat\varepsilon$ 尽量接近真实噪声 $\varepsilon$（MSE 损失）。

**关键特点**：每个点**独立处理**，点之间的信息仅通过共享的 $z$ 间接交互。这也是"Pointwise"命名的原因。

---

## 网络结构

```
输入: x^(t)  (B, N, 3)
      β_t    (B,)
      z      (B, zdim)

① 时间嵌入: [β_t, sin(β_t), cos(β_t)] → (B, 1, 3)

② 条件向量: ctx = cat([time_emb, z]) → (B, 1, 3+zdim)

③ 6 层 ConcatSquashLinear:
   3 →[LeakyReLU]→ 128 →[LeakyReLU]→ 256 →[LeakyReLU]→ 512
     →[LeakyReLU]→ 256 →[LeakyReLU]→ 128 →            → 3

④ 残差: output = h + x^(t)

输出: ε_θ  (B, N, 3)
```

沙漏形（3→128→256→512→256→128→3）：先升维提取特征，再降维输出，是 MLP 的经典设计。

---

## 时间嵌入：为什么用 [β, sin(β), cos(β)]？

网络需要感知"当前噪声强度"，有两种选择：

| 方案 | 维度 | 设计思路 |
|---|---|---|
| DDPM：正弦位置编码 | 128 维 | $t$ 的多频率 sin/cos 指纹，适合 UNet 多尺度结构 |
| 本论文：直接嵌入 $\beta_t$ | **3 维** | $\beta_t$ 本身就是语义信息（噪声强度），sin/cos 补充非线性特征 |

使用 $\beta_t$ 而非 $t$ 的核心原因：**网络预测噪声，给它看噪声强度本身比给序号更直接**。

```python
beta_col = beta.unsqueeze(1)                          # (B, 1)
time_emb = torch.cat([
    beta_col,
    torch.sin(beta_col),
    torch.cos(beta_col),
], dim=-1).unsqueeze(1)                               # (B, 1, 3)
```

---

## 条件向量的广播机制

```python
ctx = cat([time_emb, z], dim=-1)   # (B, 1, 3+zdim)
#                 ↑ N=1
```

`ctx` 的 N 维是 1，在 `ConcatSquashLinear` 内部与点特征 `(B, N, dim)` 相乘时**自动广播**到 N 个点。这正是"所有点共享同一个条件"的实现。

---

## 残差连接

```python
# 网络 h 预测"偏差量"，加上原始 x^(t) 是最终输出
output = x + h    # if residual=True
```

- 网络只需学"x 基础上改多少"，比从零预测完整坐标更容易
- 梯度可沿 `+x` 这条捷径流回，缓解深层梯度消失
- 详见 `docs/notes/pytorch_notes.md §2`

---

## 最后一层不加激活

前 5 层后接 `LeakyReLU`，最后一层（输出 3 维坐标偏移）**不加任何激活**。

原因：预测的是噪声偏移量，可正可负，ReLU 或 LeakyReLU 都会截断负值，破坏输出分布。

---

## 验证结果

```
输入 x    : [4, 2048, 3]
时间 beta : [4]
latent z  : [4, 256]
输出 out  : [4, 2048, 3]   ✓ 形状正确

residual=True vs False：值不同，差值范数 ≈ x 的范数  ✓
不同 beta → 不同输出  ✓
```

---

## 与后续模块的关系

```
PointwiseNet
    └── DiffusionPoint
            ├── get_loss():  调用 PointwiseNet 预测噪声，计算 MSE
            └── sample():    循环调用 PointwiseNet，逐步去噪
```
