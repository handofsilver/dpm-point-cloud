# 04 · DiffusionPoint — 扩散过程调度器

> 对应代码: `model.py` → `class DiffusionPoint`
> 对应论文: Section 4.1, Algorithm 1 (训练), Algorithm 2 (采样)

---

## 这个模块做什么？

`DiffusionPoint` 是**扩散模型的核心调度器**。它本身不包含可学习参数——它持有 `VarianceSchedule`（噪声系数表）和 `PointwiseNet`（噪声预测网络），将它们串联起来完成两件事：

1. **训练时** (`get_loss`)：给干净点云加噪，让网络预测噪声，返回 MSE 损失
2. **推理时** (`sample`)：从纯噪声出发，逐步逆向去噪，生成点云

类比：`VarianceSchedule` 是乐谱，`PointwiseNet` 是乐手，`DiffusionPoint` 是指挥。

---

## get_loss — 训练时的单次迭代

对应 **Algorithm 1**。数据流：

**输入**: $x^{(0)}$ `(B, N, 3)` 干净点云，$z$ `(B, F)` shape latent

| 步骤 | 操作 | shape |
|:---:|------|:-----:|
| ① | $t \sim \text{Uniform}\{1, \ldots, T\}$ | `(B,)` |
| ② | $\varepsilon \sim \mathcal{N}(0, I)$ | `(B, N, 3)` |
| ③ | $x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\, \varepsilon$ — Eq.(13) | `(B, N, 3)` |
| ④ | $\hat\varepsilon = \varepsilon_\theta(x^{(t)},\, \beta_t,\, z)$（调用 PointwiseNet） | `(B, N, 3)` |
| ⑤ | $\mathcal{L} = \text{MSE}(\hat\varepsilon,\, \varepsilon)$ — Eq.(9) 简化形式 | 标量 |

**输出**: $\mathcal{L}$（标量）

### 为什么能"一步跳"到 x^(t)？

逐步加噪的 Markov chain 可以被解析地折叠（closed-form）为一步直达：

$$
x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$
这是因为高斯噪声的叠加仍然是高斯——所有中间步的噪声合并成一个等效的 $\varepsilon$，系数由 $\bar\alpha_t$ 完全决定。

这意味着训练时**不需要模拟前向过程**（跑 t 步），直接用 $\bar\alpha_t$ 一步生成 $x^{(t)}$。这是扩散模型训练高效的关键。

### 为什么随机采 t？

一个 batch 里每个样本各自独立采一个 t。相比遍历所有 t（每次迭代计算 T 次前向），随机采样等价于对 t 的均匀重要性采样，期望上覆盖所有时间步，计算成本从 O(T) 降到 O(1)。

### 广播陷阱

`alpha_bar` 从 `VarianceSchedule` 取出后 shape 是 `(B,)`，但 `x0` 是 `(B, N, 3)`。直接相乘会广播错误。需要 `.view(B, 1, 1)` 扩展到 `(B, 1, 1)` 才能正确逐样本广播：

```python
# alpha_bar: (B,) → (B, 1, 1)，与 x0: (B, N, 3) 广播
torch.sqrt(alpha_bar).view(B, 1, 1) * x0
```

---

## sample — 推理时的逆向去噪

对应 **Algorithm 2**。数据流：

**输入**: $z$ `(B, F)` shape latent，$N$ 点数，$f$ flexibility 系数

**Step 1** — 初始化纯噪声：

$$x^{(T)} \sim \mathcal{N}(0, I), \quad \text{shape: } (B, N, 3)$$

**Step 2** — 逆向循环 $t = T, T-1, \ldots, 1$：

$$\hat\varepsilon = \varepsilon_\theta(x^{(t)},\, \beta_t,\, z)$$

$$x^{(t-1)} = \frac{1}{\sqrt{\alpha_t}} \left( x^{(t)} - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\, \hat\varepsilon \right) + \sigma_t \cdot z_{\text{noise}} \quad \text{(if } t > 1\text{)}$$

> 上式对应 DDPM 论文 Eq.(11)，其中 $\sigma_t$ 由 `flexibility` 参数在 $\sigma_t^{\text{flex}}$ 和 $\sigma_t^{\text{inflex}}$ 之间插值（见 [01_variance_schedule.md](01_variance_schedule.md)）。

**输出**: $x^{(0)}$ `(B, N, 3)` 生成的点云

### 逆向公式的直觉

每一步做两件事：

1. **去噪**：从 $x^{(t)}$ 中减掉网络预测的噪声分量，除以 $\sqrt{\alpha_t}$ 缩放回来
2. **加噪**：加回一点随机性（$\sigma_t \cdot z_{\text{noise}}$），防止生成结果过于确定

最后一步（t=1 → t=0）不加噪声——此时我们要的就是最终确定的输出。

### sigma 的广播

`sigma` 来自 `get_sigmas(t, flexibility)`，shape 是 `(B,)`，而 `z_noise` 是 `(B, N, 3)`。与 `get_loss` 中 `alpha_bar` 的道理相同，需要 `.view(B, 1, 1)` 广播：

```python
sigma.view(B, 1, 1) * z_noise  # (B, 1, 1) * (B, N, 3) → (B, N, 3)
```

### beta 的标量→批量转换

循环中从 `var_sched.betas[t_int - 1]` 取出的是 0 维标量 Tensor，但 `PointwiseNet` 期望 `(B,)` 的 beta。用 `.expand(B)` 扩展（不复制内存）：

```python
beta = self.var_sched.betas[t_int - 1].expand(B)  # 标量 → (B,)
```

---

## 常见 Python/PyTorch 陷阱

### `^` vs `**`

Python 中 `^` 是**按位异或**（bitwise XOR），不是幂运算。对 float Tensor 会直接报 `RuntimeError`。

```python
# 错误（对 float Tensor 报错）
torch.mean((a - b) ^ 2)

# 正确写法
torch.mean((a - b) ** 2)   # 手写
F.mse_loss(a, b)            # 推荐：语义清晰，行为等价
```

### `expand` vs `repeat`

两者都能把 `(1,)` 变成 `(B,)`，但机制不同：

| | `expand(B)` | `repeat(B)` |
|---|---|---|
| 内存 | 共享，不复制 | 复制 B 份 |
| 可写 | 否（写入会影响所有位置） | 是 |
| 用途 | 只读广播（本场景） | 需要独立修改每个元素 |

推理阶段 beta 只做乘法，用 `expand` 即可。

---

## 验证要点

1. `get_loss` 返回正值标量
2. `get_loss` 梯度能正常回传（`loss.backward()` 不报错）
3. `sample` 输出 shape 为 `(B, N, 3)`
4. `sample` 输出值分布合理（非 NaN，非极端值）

---

## 与其他模块的关系

```
DiffusionPoint
    ├── 持有 VarianceSchedule
    │       ├── alpha_bars    → get_loss: Eq.13 前向加噪
    │       ├── betas         → get_loss/sample: 传入 PointwiseNet
    │       ├── alphas        → sample: 逆向均值计算
    │       └── get_sigmas()  → sample: 逆向加噪
    │
    ├── 持有 PointwiseNet
    │       ├── get_loss: 预测噪声 ε_θ
    │       └── sample:  每步循环中预测噪声
    │
    └── 被上层模块调用
            ├── AutoEncoder.forward()  → get_loss
            ├── AutoEncoder.sample()   → sample
            └── FlowVAE               → 同上 + KL 损失
```
