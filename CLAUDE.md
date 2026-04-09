# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a personal **paper-reading companion repo**: simplified PyTorch implementations of classic model architectures, built for understanding data flow and architecture correspondence to the original papers — not for industrial-grade performance or production use.

Each subdirectory corresponds to one paper/topic (e.g., `VAE/`, `DDPM/`, `DPM_3D/`). Each directory contains:
- The original paper PDF
- The user's study notes (Chinese, Markdown)
- Implementation code

## Implementation Philosophy

- **Readability over performance**: code should read like annotated pseudocode corresponding to the paper's method. Naming should be clear, readable, and follow standard programming conventions.
- **Thorough comments**: every layer, forward pass step, and loss computation must document input/output tensor shapes and semantic meaning. This is the primary value of the repo — understanding data flow at a glance. Example:
  ```python
  # Encode input to latent distribution parameters
  # x: (batch_size, 784) -> mu: (batch_size, latent_dim), log_var: (batch_size, latent_dim)
  ```
- **Minimal and self-contained**: each implementation should be understandable on its own without jumping across many files. Avoid deep abstraction hierarchies.
- **Reference the paper where helpful**: cite equation numbers or sections at key steps (e.g., `# Eq. (7): reparameterization trick`), but don't overdo it — shape and semantics comments take priority.
- **Standard datasets**: use well-known datasets available via `torchvision.datasets` (e.g., MNIST) to keep setup trivial.

## Tech Stack

- Python 3, PyTorch (torch, torchvision)
- Minimal dependencies: numpy, matplotlib for visualization
- No custom training frameworks — plain training loops with `torch.optim`

## Paper-Specific Context

Each subdirectory contains the original paper PDF. Implementation should treat the PDF as the primary reference — do not rely on prior knowledge of the paper. If the current paper is not listed below, read its PDF directly.

### VAE (Kingma & Welling 2014/2022)

- Encoder outputs `mu` and `log_var`（不用 `sigma`），保证数值稳定性
- Loss 中 KL 项使用两个高斯分布的闭式解，不要用采样近似

### DDPM (Ho et al. 2020)

- 待补充（placeholder）

### DPM-3D (Luo & Hu 2021)

- 源仓库参考: `/home/eliosilver/Github_Projects/diffusion-point-cloud`（可读不可抄，复现要自己搭建思路）
- 详细论文笔记见 `notes.md`

#### 核心概念
- 点云中每个点独立采样自点分布 $q(x_i^{(0)} | z)$，通过共享的 shape latent $z$ 关联
- 前向扩散: $q(x^{(t)}|x^{(0)}) = \mathcal{N}(\sqrt{\bar\alpha_t} x^{(0)}, (1-\bar\alpha_t)I)$ — 可一步跳采
- 逆扩散（生成）: 以 $z$ 为条件的 Markov chain，网络预测噪声 $\epsilon_\theta$
- 损失: MSE($\epsilon_\theta$, $\epsilon$) — 等价于 ELBO 中 KL 项的简化

#### 架构要点
- **PointNetEncoder**: Conv1d(3→128→128→256→512) + MaxPool + 双头FC → (mu, log_var)
  - AutoEncoder 模式只用 mu（确定性编码），Generation 模式用重参数化
- **ConcatSquashLinear**: `output = Linear(x) * sigmoid(Linear_gate(ctx)) + Linear_bias(ctx)` — 条件注入的核心机制
- **PointwiseNet**: 6层 ConcatSquashLinear MLP (3→128→256→512→256→128→3)，每个点独立处理
  - 时间嵌入: [β, sin(β), cos(β)]（极简，仅3维）
  - 上下文: cat(time_emb, z) → (B, 1, F+3)
  - 带残差连接
- **VarianceSchedule**: 线性 β schedule，预计算 α_bar 等；采样时有 flexibility 参数插值两种方差
- **Normalizing Flow** (仅生成模式): Affine Coupling Layers 参数化 p(z)

#### 两种模式
- **AutoEncoder**: PointNet → z → Diffusion decode。无 KL 项。T=200, β_T=0.05, lr=1e-3
- **FlowVAE/GaussianVAE**: PointNet → (μ,σ) → z → Diffusion decode + KL(q||p)。T=100, β_T=0.02, lr=2e-3, kl_weight=0.001

#### 实现约束
- 数据: ShapeNet 点云，2048 points/shape，归一化到零均值单位方差
- Encoder 输出 mu 和 log_var（不是 sigma），与 VAE 目录一致
- 优化器: Adam，梯度裁剪 max_norm=10，线性 LR 衰减

## Conventions

- Training scripts: `train.py` as entry point
- Model definitions: `model.py` or similar, with each component (Encoder, Decoder, full VAE, etc.) as a separate `nn.Module`
- Keep the Encoder output as `mu` and `log_var` (not `sigma`), matching the standard parameterization for numerical stability
- Visualization / sampling scripts separate from training
