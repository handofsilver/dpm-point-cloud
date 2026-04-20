# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A **paper-reading companion repo**: a clean, heavily-annotated PyTorch reimplementation of one paper:

> Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021

Built for understanding data flow and architecture correspondence to the paper — not for production use or benchmarking.

## Implementation Philosophy

- **Readability over performance**: code should read like annotated pseudocode corresponding to the paper's method. Naming should be clear and follow standard conventions.
- **Thorough comments**: every layer, forward pass step, and loss computation must document input/output tensor shapes and semantic meaning. This is the primary value of the repo. Example:
  ```python
  # Encode input to latent distribution parameters
  # x: (B, N, 3) -> mu: (B, F), log_var: (B, F)
  ```
- **Minimal and self-contained**: each implementation should be understandable on its own without jumping across many files. Avoid deep abstraction hierarchies.
- **Reference the paper**: cite equation numbers at key steps (e.g., `# Eq. (13): one-step forward sampling`), but shape/semantics comments take priority.

## Tech Stack

- Python 3, PyTorch (torch, torchvision)
- Minimal dependencies: numpy, matplotlib for visualization
- No custom training frameworks — plain training loops with `torch.optim`

## Paper Context — DPM-3D (Luo & Hu 2021)

- Reference repo: `/home/eliosilver/Github_Projects/diffusion-point-cloud` (read-only reference; reimplementation must be independent)
- Detailed study notes: `docs/notes/`

### Core Concepts
- Each point in a point cloud is independently sampled from $q(x_i^{(0)} | z)$, linked through a shared shape latent $z$
- Forward diffusion: $q(x^{(t)}|x^{(0)}) = \mathcal{N}(\sqrt{\bar\alpha_t} x^{(0)}, (1-\bar\alpha_t)I)$ — one-step jump sampling
- Reverse diffusion (generation): $z$-conditioned Markov chain, network predicts noise $\epsilon_\theta$
- Loss: MSE($\epsilon_\theta$, $\epsilon$) — simplified from the KL term in the ELBO

### Architecture Summary
- **PointNetEncoder**: Conv1d(3→128→128→256→512) + MaxPool + dual-head FC → (mu, log_var)
- **ConcatSquashLinear**: `output = Linear(x) * sigmoid(Linear_gate(ctx)) + Linear_bias(ctx)`
- **PointwiseNet**: 6-layer ConcatSquashLinear MLP (3→128→256→512→256→128→3), processes each point independently
- **VarianceSchedule**: linear β schedule, precomputes α_bar etc.; flexibility parameter interpolates between two variance choices
- **Normalizing Flow** (generation mode only): Affine Coupling Layers parameterize p(z)

### Two Operating Modes
- **AutoEncoder**: PointNet → z → Diffusion decode. No KL term. T=200, β_T=0.05, lr=1e-3
- **FlowVAE/GaussianVAE**: PointNet → (μ,σ) → z → Diffusion decode + KL(q||p). T=100, β_T=0.02, lr=2e-3, kl_weight=0.001

### Paper Tables and Training Protocol

Each row in the paper's result tables corresponds to **one independently trained checkpoint**. Reconstructed from released pretrained ckpts' `args` (see `pretrained/*.pt`, fields `model` and `categories`):

- **Table 1 (Generation)** — FlowVAE only. Reports Airplane and Chair, each row is a per-category FlowVAE ckpt (`GEN_airplane.pt` / `GEN_chair.pt`, both with `model='flow'`). Evaluation normalizes both `S_g` and `S_r` into a `[-1, 1]^3` bbox before computing MMD / COV / 1-NNA / JSD (Sec 5.2).
- **Table 2 (Auto-encoding)** — AutoEncoder. Reports Airplane, Car, Chair, ShapeNet: first three rows are per-category AE ckpts (`AE_airplane.pt` / `AE_car.pt` / `AE_chair.pt`); the ShapeNet row is the all-55 AE ckpt (`AE_all.pt`).
- **GaussianVAE does not appear in any paper table.** It is a code-only ablation (fixed N(0,I) prior in place of the learned flow prior) for internal Flow-vs-Gaussian comparison.
- The `train_*.py` scripts default to **all-55-category joint training**, which does not match any Table 1 row's protocol. Reproducing a Table 1 row requires per-category training via a `--cates` argument.

Current reproduction status and next-step ablation plan: `docs/experiments/gen_eval_gap_analysis.md`, `docs/experiments/next_experiments.md`.

### Implementation Constraints
- Data: ShapeNet point clouds, 2048 points/shape, normalized to zero mean and unit variance
- Encoder outputs mu and log_var (not sigma) for numerical stability
- Optimizer: Adam, gradient clipping max_norm=10, linear LR decay

## Collaboration Mode: Scaffold → Fill-in → Review

Current collaboration workflow (adjustable as learning progresses):

1. **Concept explanation**: AI explains the next component's purpose and design motivation
2. **Scaffold**: AI writes class/method signatures, comments, and TODO placeholders in `model.py` — no implementation
   - Each TODO should be fine-grained (single responsibility) with input/output shape hints
   - Start with the simpler mode (e.g., `get_loss`) before the complex one (e.g., `sample`)
3. **User fills in**: User attempts implementation guided by the comments
4. **Diagnose & finalize**: AI reviews user code, flags bugs and non-ideal practices, finalizes the code
5. **Documentation**: AI writes `docs/code_guide/` walkthrough + updates `roadmap.md`

## Conventions

- Model definitions: `model.py`, each component as a separate `nn.Module`
- Encoder output: `mu` and `log_var` (not `sigma`) for numerical stability
- Training scripts: `scripts/train_ae.py`, `scripts/train_gen.py`
- Visualization / sampling scripts separate from training
