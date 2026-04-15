# DPM-3D — Annotated PyTorch Reimplementation

> **Paper**: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021
> [[Paper]](https://arxiv.org/abs/2103.01458) · [[Original Repo]](https://github.com/luost26/diffusion-point-cloud)

A clean, **heavily-annotated** PyTorch reimplementation built for understanding — not benchmarking. Every module is documented with tensor shapes, design rationale, and references to paper equations.

---

## What This Is (and Isn't)

**This is**: a learning-oriented reimplementation where the code reads like annotated pseudocode. Each component is built independently, tested, and documented before the next one begins.

**This is not**: a reproduction of the original codebase. Code is written independently from the paper, with the original repo used only as a sanity check.

---

## Architecture

The model generates 3D point clouds by learning to reverse a diffusion process conditioned on a shape latent $z$:

```
[Training]
x^(0)  ──► PointNetEncoder ──► z (shape latent)
                                │
              t ~ Uniform{1..T} │  ε ~ N(0,I)
                                ▼
        x^(t) = √ᾱ_t · x^(0) + √(1−ᾱ_t) · ε       ← Eq.(13), one-step forward
                                │
              PointwiseNet(x^(t), β_t, z) ──► ε_θ
                                │
                    Loss = MSE(ε_θ, ε)

[Sampling]
z ~ prior  ──►  x^(T) ~ N(0,I)
                for t = T..1:
                    ε_θ = PointwiseNet(x^(t), β_t, z)
                    x^(t−1) = denoise(x^(t), ε_θ, α_t, ᾱ_t) + σ_t · noise
                return x^(0)
```

**Core modules** (implemented in order):

| Module | Role | Docs |
|--------|------|------|
| `VarianceSchedule` | Precomputes β, α, ᾱ, σ for all timesteps | [01](docs/code_guide/01_variance_schedule.md) |
| `ConcatSquashLinear` | Conditional gated linear layer — injects time + shape latent | [02](docs/code_guide/02_concat_squash_linear.md) |
| `PointwiseNet` | 6-layer MLP noise predictor, processes each point independently | [03](docs/code_guide/03_pointwise_net.md) |
| `DiffusionPoint` | Orchestrates training (forward + loss) and sampling (reverse loop) | [04](docs/code_guide/04_diffusion_point.md) |
| `PointNetEncoder` | Point cloud → shape latent z via Conv1d + MaxPool | *(in progress)* |
| `AutoEncoder` | End-to-end: encode → diffusion decode, no KL | *(planned)* |
| `FlowVAE` | Full generative model with Normalizing Flow prior on z | *(planned)* |

See [`docs/architecture.md`](docs/architecture.md) for the full data flow with Mermaid diagrams.

---

## Two Operating Modes

| | AutoEncoder | FlowVAE / GaussianVAE |
|--|--|--|
| Encoder output | deterministic `mu` only | stochastic `(mu, log_var)` + reparameterize |
| Loss | diffusion MSE | diffusion MSE + KL |
| T / β_T | 200 / 0.05 | 100 / 0.02 |
| Learning rate | 1e-3 | 2e-3 |

---

## Documentation

Each module has a companion guide under `docs/code_guide/`:

- Mathematical derivation with paper equation references (LaTeX)
- Annotated data flow
- PyTorch engineering notes (e.g., why `register_buffer`, why `log_var` not `sigma`)

Additional notes:
- [`docs/notes/paper_deep_dive.md`](docs/notes/paper_deep_dive.md) — design rationale: two sigma variants, time embedding choice, why random-sample t, etc.
- [`docs/notes/pytorch_notes.md`](docs/notes/pytorch_notes.md) — PyTorch patterns encountered during implementation
- [`docs/notes/roadmap.md`](docs/notes/roadmap.md) — implementation progress tracker

---

## Repository Structure

```
DPM_3D/
├── model.py              All nn.Module definitions
├── tests/                Per-module verification scripts
├── docs/
│   ├── architecture.md   Global data flow + module I/O reference
│   ├── code_guide/       Per-module walkthroughs (math + code)
│   ├── notes/            Study notes: paper deep-dives, PyTorch patterns, roadmap
│   └── paper/            Original paper PDF + AI-summarized notes
└── scripts/              (planned) train_ae.py, train_gen.py
```

---

## Acknowledgements

- Paper: [Luo & Hu, CVPR 2021](https://arxiv.org/abs/2103.01458)
- Original implementation: [luost26/diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud)
