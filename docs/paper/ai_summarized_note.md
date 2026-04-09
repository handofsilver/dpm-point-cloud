# DPM-3D 论文精读笔记

> Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021

## 1. 核心思想

将点云中的每个点视为热力学系统中的粒子。前向扩散过程逐步加噪，把有意义的点分布转化为标准正态噪声；**逆扩散过程**（生成过程）则从噪声出发，通过学到的 Markov chain 逐步恢复目标形状。

关键区别于 DDPM (图像): 这里每个点 $x_i$ 是独立地从 **点分布** $q(x_i^{(0)} | z)$ 中采样的，整个点云共享同一个 **shape latent** $z$。

## 2. 前向扩散过程 (Forward)

每个点独立加噪，Markov chain:

$$q(x^{(t)} | x^{(t-1)}) = \mathcal{N}(x^{(t)} \mid \sqrt{1-\beta_t}\, x^{(t-1)},\; \beta_t I) \quad \text{— Eq.(2)}$$

利用累积乘积可以一步采样到任意时刻（跳步采样）:

$$q(x^{(t)} | x^{(0)}) = \mathcal{N}(x^{(t)} \mid \sqrt{\bar\alpha_t}\, x^{(0)},\; (1-\bar\alpha_t) I) \quad \text{— Eq.(13)}$$

其中 $\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$。

## 3. 逆扩散过程 (Reverse)

以 shape latent $z$ 为条件:

$$p_\theta(x^{(t-1)} | x^{(t)}, z) = \mathcal{N}(x^{(t-1)} \mid \mu_\theta(x^{(t)}, t, z),\; \beta_t I) \quad \text{— Eq.(4)}$$

生成时: $x^{(T)} \sim \mathcal{N}(0, I)$，逐步通过逆 Markov chain 得到 $x^{(0)}$。

## 4. 训练目标 (ELBO)

变分下界展开为（Eq.9）:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_q \Big[ \sum_{t=2}^{T} \sum_{i=1}^{N} D_{KL}(q(x_i^{(t-1)} | x_i^{(t)}, x_i^{(0)}) \| p_\theta(x_i^{(t-1)} | x_i^{(t)}, z)) - \sum_{i=1}^{N} \log p_\theta(x_i^{(0)} | x_i^{(1)}, z) + D_{KL}(q_\phi(z|X^{(0)}) \| p(z)) \Big]$$

其中后验 $q(x^{(t-1)} | x^{(t)}, x^{(0)})$ 有闭式高斯解 — Eq.(10-11):
- 均值: $\mu_t(x^{(t)}, x^{(0)}) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x^{(0)} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1 - \bar\alpha_t} x^{(t)}$
- 方差: $\gamma_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$

### 简化训练算法 (Algorithm 1)

每步只随机采一个时刻 $t$，而非遍历所有 $t$:
1. 采样 $X^{(0)} \sim q_{data}$
2. 编码 $z \sim q_\phi(z | X^{(0)})$
3. 采样 $t \sim \text{Uniform}\{1, \ldots, T\}$
4. 用 Eq.(13) 一步采样 $x_i^{(t)}$
5. 计算 $L_t$（KL 散度项）和 $L_z$（先验 KL 项）
6. 梯度下降

### 实际实现中的简化

源码实际预测**噪声** $\epsilon$（而非均值 $\mu$），与 DDPM 相同:
- 前向: $x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\, \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,I)$
- 网络预测 $\epsilon_\theta(x^{(t)}, t, z)$
- 损失: $\text{MSE}(\epsilon_\theta, \epsilon)$（等价于 KL 散度项的简化）

采样过程 (DDPM 式):
$$x^{(t-1)} = \frac{1}{\sqrt{\alpha_t}} \big( x^{(t)} - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x^{(t)}, t, z) \big) + \sigma_t \cdot z_{\text{noise}}$$

## 5. 模型架构

### 5.1 PointNet Encoder (编码器)

```
输入: (B, N, 3)  — 点云
 → transpose → (B, 3, N)
 → Conv1d(3→128) + BN + ReLU
 → Conv1d(128→128) + BN + ReLU
 → Conv1d(128→256) + BN + ReLU
 → Conv1d(256→512) + BN
 → MaxPool over N → (B, 512)
 → 分两条路:
   mu 路:  FC(512→256) + BN + ReLU → FC(256→128) + BN + ReLU → FC(128→zdim)
   logvar 路: 同上结构
输出: mu (B, zdim), log_var (B, zdim)
```

- 在 AutoEncoder 模式下只使用 mu，忽略 log_var
- 在 VAE/Generation 模式下用重参数化采样 z

### 5.2 ConcatSquashLinear (条件注入层)

```python
gate = sigmoid(Linear_gate(ctx))     # ctx: (B, 1, F+3)
bias = Linear_bias(ctx)
output = Linear(x) * gate + bias     # 逐元素门控 + 偏置注入
```

这是将 (时间, shape latent) 注入到点特征中的核心机制，比简单拼接更有效。

### 5.3 PointwiseNet (噪声预测网络)

```
输入: x (B, N, 3), beta (B,), context (B, F)
 → 时间嵌入: [beta, sin(beta), cos(beta)] → (B, 1, 3)
 → 上下文嵌入: cat(time_emb, context) → (B, 1, F+3)
 → 6 层 ConcatSquashLinear:
   3 → 128 → 256 → 512 → 256 → 128 → 3
   (前 5 层后接 LeakyReLU)
 → 残差连接: output = x + net_output（如果 residual=True）
输出: (B, N, 3)  — 预测的噪声
```

**关键特点**: 每个点独立处理（pointwise），点之间的信息仅通过共享的 shape latent z 交互。

### 5.4 VarianceSchedule (方差调度)

- 线性调度: $\beta_1 = 10^{-4}$ 到 $\beta_T$
- 预计算: $\alpha_t, \bar\alpha_t, \sigma_t^{\text{flex}}, \sigma_t^{\text{inflex}}$
- `flexibility` 参数在采样时插值两种方差: $\sigma = \sigma^{\text{flex}} \cdot f + \sigma^{\text{inflex}} \cdot (1-f)$
  - $\sigma^{\text{flex}} = \sqrt{\beta_t}$（对应 Eq.4 中的方差）
  - $\sigma^{\text{inflex}} = \sqrt{\gamma_t}$（后验方差，Eq.11）

### 5.5 Normalizing Flow (仅用于生成模式)

- 一叠 Affine Coupling Layer（14层，hidden_dim=256）
- 用于参数化先验 $p(z)$: $w \sim \mathcal{N}(0,I) \xrightarrow{F_\alpha} z$
- 通过 change-of-variable 公式计算 $\log p(z)$

## 6. 两种工作模式

### 模式 A: AutoEncoder (自编码器)

```
训练: X → PointNet → z (确定性) → DiffusionPoint.get_loss(X, z)
重建: X → PointNet → z → 采样 x^(T) ~ N(0,I) → 逆扩散 → X̂
```

- 编码器确定性（只取 mu），无 KL 正则
- 损失仅有扩散重建损失
- 超参: T=200, β_1=1e-4, β_T=0.05, lr=1e-3

### 模式 B: Generation (FlowVAE / GaussianVAE)

```
训练: X → PointNet → (mu, logvar) → 重参数化 → z
      → 扩散损失 + KL(q(z|X) || p(z))
生成: w ~ N(0,I) → Flow → z → 采样 x^(T) → 逆扩散 → X
```

- **FlowVAE**: p(z) 由 Normalizing Flow 参数化，更灵活
  - KL = -entropy(q) - log p(z)，其中 log p(z) 通过 flow 计算
- **GaussianVAE**: p(z) = N(0, I)，更简单
- 超参: T=100, β_1=1e-4, β_T=0.02, lr=2e-3, kl_weight=0.001

## 7. 数据与评估

- **数据集**: ShapeNet (51,127 shapes, 55 categories)，每个形状采 2048 个点
- **归一化**: 零均值、单位方差 (shape_unit)
- **评估指标**:
  - 重建: Chamfer Distance (CD), Earth Mover's Distance (EMD)
  - 生成: MMD, Coverage (COV), 1-NNA, JSD

## 8. 实现路线图（复现计划）

建议分阶段实现:

1. **Phase 1 — 基础组件**: VarianceSchedule, ConcatSquashLinear, PointwiseNet
2. **Phase 2 — 扩散核心**: DiffusionPoint (get_loss + sample)
3. **Phase 3 — 编码器**: PointNetEncoder
4. **Phase 4 — AutoEncoder**: 组装编码器 + 扩散解码器，用 ShapeNet 数据训练
5. **Phase 5 — 生成模型**: Normalizing Flow + FlowVAE / GaussianVAE
6. **Phase 6 — 评估 & 可视化**: CD/EMD metrics, 3D 点云可视化
