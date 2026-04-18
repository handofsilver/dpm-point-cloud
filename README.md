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

**Core modules** (all implemented in `model.py`):

| Module | Role | Docs |
|--------|------|------|
| `VarianceSchedule` | Precomputes β, α, ᾱ, σ for all timesteps | [01](docs/code_guide/01_variance_schedule.md) |
| `ConcatSquashLinear` | Conditional gated linear layer — injects time + shape latent | [02](docs/code_guide/02_concat_squash_linear.md) |
| `PointwiseNet` | 6-layer MLP noise predictor, processes each point independently | [03](docs/code_guide/03_pointwise_net.md) |
| `DiffusionPoint` | Orchestrates training (forward + loss) and sampling (reverse loop) | [04](docs/code_guide/04_diffusion_point.md) |
| `PointNetEncoder` | Point cloud → shape latent z via Conv1d + MaxPool | [05](docs/code_guide/05_pointnet_encoder.md) |
| `AutoEncoder` | End-to-end: encode → diffusion decode, no KL | [06](docs/code_guide/06_autoencoder.md) |
| `GaussianVAE` | VAE with fixed N(0,I) prior on z | [07](docs/code_guide/07_gaussian_vae.md) |
| `AffineCouplingLayer` | Single invertible affine coupling step | [08](docs/code_guide/08_affine_coupling_layer.md) |
| `NormalizingFlow` | K stacked coupling layers — parameterizes p(z) | [09](docs/code_guide/09_normalizing_flow.md) |
| `FlowVAE` | Full generative model with Normalizing Flow prior | [10](docs/code_guide/10_flow_vae.md) |

---

## Two Operating Modes

| | AutoEncoder | FlowVAE / GaussianVAE |
|--|--|--|
| Encoder output | deterministic `mu` only | stochastic `(mu, log_var)` + reparameterize |
| Loss | diffusion MSE | diffusion MSE + KL |
| T / β_T | 200 / 0.05 | 100 / 0.02 |
| Learning rate | 1e-3 | 2e-3 |

---

## Environment Setup

```bash
# Create the environment
conda env create -f environment.yml
# Activate the environment
conda activate dpm3d
```

---

## Data Preparation

Download `shapenet.hdf5` from the original authors' Google Drive:

**[https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ](https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ)**

Place the file at:

```
data/shapenet/shapenet.hdf5
```

The HDF5 file has the structure `{synsetid}/{split} → (N, 2048, 3)`, covering all ShapeNet categories across `train` / `val` / `test` splits.

---

## Usage

**Run component tests** (no data required):

```bash
python tests/test_variance_schedule.py
python tests/test_concat_squash_linear.py
python tests/test_pointwise_net.py
python tests/test_encoder.py
python tests/test_diffusion_point.py
```

**Train AutoEncoder** (Table 2):

```bash
python scripts/train_ae.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --save_dir checkpoints/ae
```

**Train generative model** (Table 1):

```bash
# GaussianVAE (simpler prior)
python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --model gaussian \
    --save_dir checkpoints/gen

# FlowVAE (learned prior, paper's main result)
python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --model flow \
    --save_dir checkpoints/gen
```

**Qualitative: reconstruct input point clouds**:

```bash
python scripts/reconstruct.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --ckpt checkpoints/ae/epoch_2000.pt \
    --out_dir results/reconstruct
```

**Qualitative: generate new shapes**:

```bash
python scripts/generate.py \
    --ckpt checkpoints/gen/flow_epoch_2000.pt \
    --model flow \
    --out_dir results/generate
```

**Quantitative: Table 2 — AutoEncoder reconstruction (CD / EMD)**:

```bash
python scripts/eval_ae.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --ckpt checkpoints/ae/epoch_2000.pt \
    --cates airplane chair car \
    --out_dir results/eval_ae
```

**Quantitative: Table 1 — Generation quality (MMD / COV / 1-NNA)**:

```bash
python scripts/eval_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --ckpt checkpoints/gen/flow_epoch_2000.pt \
    --model flow \
    --cates airplane chair car \
    --out_dir results/eval_gen
```

> **Note on EMD**: both eval scripts use `geomloss` Sinkhorn approximation in place of the original `approxmatch.cu` CUDA kernel (incompatible with PyTorch 2.x / CUDA 13+). Both are approximate optimal transport solvers of comparable accuracy. Pass `--no_emd` to skip EMD computation during debugging.

---

## Repository Structure

```
DPM_3D/
├── model.py              All nn.Module definitions
├── dataset.py            ShapeNet data loading (single HDF5 file)
├── metrics.py            CD, EMD, MMD, COV, 1-NNA
├── visualize.py          3D point cloud visualization
├── tests/                Per-module verification scripts
├── scripts/
│   ├── train_ae.py       AutoEncoder training
│   ├── train_gen.py      GaussianVAE / FlowVAE training
│   ├── reconstruct.py    Qualitative reconstruction (input vs. reconstructed)
│   ├── generate.py       Qualitative generation diversity showcase
│   ├── eval_ae.py        Table 2: per-category CD / EMD on test set
│   └── eval_gen.py       Table 1: MMD / COV / 1-NNA vs. test set
└── docs/
    ├── code_guide/       Per-module walkthroughs (math + code)
    └── notes/            Study notes: paper deep-dives, PyTorch patterns, roadmap
```

---

## Documentation

Each module has a companion guide under `docs/code_guide/` combining:
- Mathematical derivation with paper equation references (LaTeX)
- Annotated tensor data flow
- PyTorch engineering notes

Additional notes:
- [`docs/notes/paper_deep_dive.md`](docs/notes/paper_deep_dive.md) — design rationale behind key choices
- [`docs/notes/pytorch_notes.md`](docs/notes/pytorch_notes.md) — PyTorch patterns encountered during implementation
- [`docs/notes/roadmap.md`](docs/notes/roadmap.md) — implementation progress tracker

---

## Results vs. Paper

> **Note on comparability**: The paper's released pretrained weights (e.g. `AE_all.pt`) were trained on `ShapeNetCore.v2.PC15k.Resplit` — this is directly confirmed from the checkpoint's `args.dataset_dir`. Paper Table 2's reported numbers most plausibly come from evaluating those same checkpoints on the same dataset's test split (the paper text doesn't name the eval file explicitly, but this is the natural reading since that's where the released weights came from). `ShapeNetCore.v2.PC15k.Resplit` is **not** included in the Google Drive release — only `shapenet.hdf5` (2048 points/shape, airplane per-shape std ≈ 0.115, measured) is. This repo trains and evaluates on `shapenet.hdf5`.
>
> We empirically verified — by running the paper's official `AE_all.pt` through our eval pipeline on `shapenet.hdf5` — that the ~10× AE CD gap is attributable to **the evaluation dataset, not training quality**. See [`docs/notes/dataset_investigation.md`](docs/notes/dataset_investigation.md) §5 for the decisive experiment and §6 for the mechanism. **Relative trends across categories are comparable; absolute values are not.**

### Table 2 — AutoEncoder Reconstruction

Paper units: CD ×10³, EMD ×10². Our units: CD ×10³, EMD ×10³ (Sinkhorn approximation; see note below).

**Paper Table 2** (trained on PC15k, evaluated on PC15k):

| Category | CD (paper) | EMD (paper) | Oracle CD | Oracle EMD |
|----------|-----------|------------|----------|-----------|
| Airplane | 2.118 | 2.876 | 1.016 | 2.141 |
| Car      | 5.493 | 3.937 | 3.917 | 3.246 |
| Chair    | 5.677 | 4.153 | 3.221 | 3.281 |
| ShapeNet | 5.252 | 3.783 | 3.074 | 3.112 |

**On `shapenet.hdf5`** (after `shape_unit` denormalization — this repo's eval pipeline):

| Category | Our `epoch_2000.pt` CD×10³ | Our EMD×10³ | Official `AE_all.pt` CD×10³ |
|----------|:--------------------------:|:-----------:|:---------------------------:|
| Airplane | 0.178 | 1.871 | **0.1949** |
| Car      | 0.577 | 2.093 | **0.5923** |
| Chair    | 0.511 | 3.486 | **0.5521** |

The official weights (rightmost column) give the same order of magnitude as our self-trained model on `shapenet.hdf5`, while both are ~10× smaller than paper Table 2 — ruling out undertraining and pinning the gap to the dataset.

### Table 1 — Generation Quality (in progress)

Paper units: CD ×10³, EMD ×10¹, JSD ×10³.

| Category | MMD-CD | MMD-EMD | COV-CD (%) | COV-EMD (%) | 1-NNA-CD (%) | 1-NNA-EMD (%) | JSD   |
|----------|--------|---------|------------|-------------|--------------|---------------|-------|
| Airplane | 3.276  | 1.061   | 48.71      | 45.47       | 64.83        | 75.12         | 1.067 |
| Chair    | 12.276 | 1.784   | 48.94      | 47.52       | 60.11        | 69.06         | 7.797 |

> **Our Gen models are still training; results will be filled in when available.** Expected behavior (same mechanism as AE — see [investigation §7](docs/notes/dataset_investigation.md)):
> - **MMD-CD / MMD-EMD** — absolute distance metrics, will be systematically smaller than paper (same ~10× scale effect).
> - **COV / 1-NNA** — ratio / classification metrics, scale-invariant, should land close to paper numbers.
> - The paper's Table 1 protocol additionally normalizes both generated and reference point clouds into `[−1,1]³` bbox before computing metrics (Section 5.2, following ShapeGF). `scripts/eval_gen.py` does not currently do this — so MMD absolute values won't match paper even without the dataset-scale effect.

> **EMD note**: the paper uses the CUDA `approxmatch.cu` kernel (from PC-GAN); this repo uses `geomloss` Sinkhorn approximation for PyTorch 2.x compatibility. Both are approximate optimal transport solvers but produce systematically different values — EMD numbers are not directly comparable.

## Citation

```bibtex
@inproceedings{luo2021diffusion,
  author = {Luo, Shitong and Hu, Wei},
  title = {Diffusion Probabilistic Models for 3D Point Cloud Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021}
}
```
