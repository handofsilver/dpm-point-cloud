# 论文细节深度理解笔记

> 本文档记录在复现 DPM-3D 过程中，对论文中"看起来简单但其实有讲究"的细节的深入理解。
> 是 `ai_summarized_note.md` 的补充，专注于"为什么"而非"是什么"。

---

## 1. 逆扩散采样中的两种方差：σ_flex 与 σ_inflex

### 背景

逆扩散的每一步是：

$$x^{(t-1)} = \frac{1}{\sqrt{\alpha_t}} \left( x^{(t)} - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}} \varepsilon_\theta \right) + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, I)$$

这里 σ_t 决定了每步加回多少噪声。问题是：σ_t **应该取多大**？

数学上有两个合法的选择，对应两种不同的理论推导：

---

### σ_inflex —— "窄"方差（后验方差，Eq.11）

$$\sigma_t^{\text{inflex}} = \sqrt{\gamma_t}, \quad \gamma_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$$

**推导来源**：这是真实后验 $q(x^{(t-1)} | x^{(t)}, x^{(0)})$ 的方差（Bayes 定理推导，见 DDPM Eq.7 或本论文 Eq.10-11）。

**直觉**：如果我们**完全知道 x^{(0)}**，那么每步逆向去噪的不确定性就精确是 γ_t。这是理论上最"紧"的方差——不多加一丁点不必要的噪声。

**缺点**：当 t 很小时（接近干净样本），γ_t → 0，几乎不加噪声，生成样本趋于确定性。

---

### σ_flex —— "宽"方差（前向过程方差）

$$\sigma_t^{\text{flex}} = \sqrt{\beta_t}$$

**推导来源**：这直接用前向过程的每步方差，是后验方差的一个**上界**（Ho et al. DDPM 论文附录 C 有证明）。

**直觉**：假设数据本身的先验方差很大（接近单位方差），在这个假设下逆向每步最多需要加 √β_t 的噪声。

**结果**：生成样本更加多样化（stochastic），但可能稍微不那么精确。

---

### 两者的实际差异

| | σ_inflex | σ_flex |
|---|---|---|
| 理论依据 | 精确后验（知道 x^0） | 前向方差上界 |
| 生成多样性 | 较低（确定性更强） | 较高（更随机） |
| t 较小时 | 几乎不加噪 | 仍加一定噪声 |
| 适用场景 | 追求精确重建 | 追求多样生成 |

Ho et al. (DDPM) 实验发现，两者在图像质量上差别不大，但对某些数据类型（如点云这种非结构化数据）可能存在差异。

---

### 这篇论文的做法：用 `flexibility` 插值

```python
# 采样时，用 flexibility ∈ [0, 1] 在两者之间插值
sigma = sigma_flex * flexibility + sigma_inflex * (1 - flexibility)
```

- `flexibility=0`：纯窄方差，更确定性
- `flexibility=1`：纯宽方差，更随机
- 默认值通常在 0~1 之间调节，给用户控制生成多样性的旋钮

这是一个很实用的工程设计：**不硬选一个，而是把选择权留给用户**。

---

## 2. 时间嵌入：为什么用 [β, sin(β), cos(β)]，而不是 t 本身？

### 问题背景

噪声预测网络 $\varepsilon_\theta(x^{(t)}, t, z)$ 用同一套权重处理所有时间步，因此必须从输入中感知"当前噪声强度"。怎么把时间步信息给到网络？

### DDPM 的做法：正弦位置编码（128 维）

$$\text{emb}[2i] = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{emb}[2i+1] = \cos\!\left(\frac{t}{10000^{2i/d}}\right), \quad d=128$$

多频率 sin/cos 让每个整数 $t$ 对应唯一且平滑的高维向量，随后通过 FC 层注入 UNet 每一层。

### 本论文的做法：直接嵌入 $\beta_t$，仅 3 维

$$\text{time\_emb} = [\,\beta_t,\ \sin(\beta_t),\ \cos(\beta_t)\,] \in \mathbb{R}^3$$

**为什么用 $\beta_t$ 而不是 $t$？**
$t$ 是抽象序号，$\beta_t$ 才是语义信息——它直接就是"这一步加了多少噪声"，网络预测的目标也是噪声，二者对齐。

**为什么加 sin/cos？**
$\beta_t \in [10^{-4}, 0.05]$ 范围很小，单独一个标量特征太单薄。$\sin(\beta_t)$ 和 $\cos(\beta_t)$ 提供平滑有界的非线性特征，且 $\sin^2 + \cos^2 = 1$ 给幅度约束，训练更稳定。

**为什么 3 维就够？**
`PointwiseNet` 是纯 MLP，没有 UNet 的多尺度结构。时间信息通过 `ConcatSquashLinear` 的 ctx 机制注入每一层，无需高维嵌入。

### 数据流

```
β_t (标量) → [β_t, sin(β_t), cos(β_t)] → (B, 1, 3)
z   (shape latent)                      → (B, 1, 256)
                   cat → ctx            → (B, 1, 259)
                   ↓
         ConcatSquashLinear 的每一层
```

---

## 3. 为什么训练时随机采 t 而不是遍历所有 t？

### 问题

训练损失是对所有时间步的期望：

$$L = \mathbb{E}_{t \sim \text{Uniform}\{1,\ldots,T\}} \left[ \|\varepsilon_\theta(x^{(t)}, t, z) - \varepsilon\|^2 \right]$$

看起来每次迭代应该遍历 t=1 到 T 全部算一遍？

### 为什么随机采样就够了

每次只采一个（或一小批）$t$，等价于对这个期望做**蒙特卡洛估计**。只要训练足够多轮，每个 $t$ 被采到的次数趋于均匀，梯度估计的期望和遍历所有 $t$ 是一样的。

好处很明显：每次迭代只做 1 次前向+反向传播，计算成本与 T 无关（O(1) vs O(T)）。

这和 SGD 随机采 mini-batch 是同一个思想——用噪声换效率，期望上正确。

---

## 4. （占位）更多话题待补充

- ConcatSquashLinear 相比简单拼接（concat）的优势是什么？
- Normalizing Flow 里 change-of-variable 公式的直觉
