# 训练与评测实验记录

> 这份文档记录本仓库在服务器上完成的一次完整训练 + 评测全过程：硬件 / 软件环境、与 `environment.yml` 的偏差、训练命令、评测命令与结果。
>
> 相关专题另有文档，不在这里重复：
> - 评测 CD 与论文 Table 2 相差 ~10× 的归因 → [`../notes/dataset_investigation.md`](../notes/dataset_investigation.md)
> - 加载原仓库发布的 `AE_all.pt` 所需的兼容改动（不入主线，仅备查） → [`checkpoint_loading_restore.md`](checkpoint_loading_restore.md)

---

## 1. 硬件

单卡 NVIDIA A800 80GB PCIe（服务器上有 4 张，每次训练 / 评测只用其中 1 张，通过 `CUDA_VISIBLE_DEVICES` 指定）。Driver 570.133.07，`nvidia-smi` 报 CUDA 12.8。

## 2. 系统与 Python

- OS: Ubuntu 20.04.6 LTS（kernel 5.15.0-67-generic）
- Python: 3.10.20（miniconda env `dpm3d`）

## 3. 深度学习栈

| 项 | 值 |
|---|---|
| PyTorch | 2.5.1+cu121 |
| `torch.version.cuda` | 12.1 |
| `nvcc` (CUDA Toolkit) | 12.1 V12.1.66 |
| cuDNN | 9.1.9（`torch.backends.cudnn.version()` = 91900） |
| GPU compute capability | (8, 0) |

关键 Python 包：`numpy 2.2.6`、`h5py 3.16.0`、`matplotlib 3.10.8`、`geomloss 0.2.6`、`scipy 1.15.3`、`tqdm 4.67.3`。

## 4. 与 `environment.yml` 的偏差与 workaround

服务器上实际环境**不是**从当前仓库的 `environment.yml` 干净复刻出来的，存在两点偏差：

1. **CUDA 版本**：驱动级 12.8，工具链 12.1，conda 里 pip 包混装了 `cu12` / `cu13` 两套 NVIDIA 运行时（例如 `nvidia-cudnn-cu12 9.1.0.70` 与 `nvidia-cudnn-cu13 9.19.0.56` 并存）。PyTorch 实际用的是 `cu121` 构建，对齐 `nvcc 12.1`。
2. **cuDNN 兼容性问题**：cuDNN 9.1.9 与当前驱动在本项目里触发错误，无法直接使用。**workaround**：在代码入口显式关闭 cuDNN，走 cuBLAS fallback：
   ```python
   torch.backends.cudnn.enabled = False  # cuDNN 9.1.9 与当前驱动存在兼容问题
   ```

这不影响最终数值结果，只牺牲一点 Conv 速度。后续若换环境，这一行可去。

## 5. 数据

- 文件：`data/shapenet/shapenet.hdf5`（1.4 GB，55 个类别，每类 `train/val/test` 三个 split，`(N, 2048, 3)`）
- 样例规模：airplane `02691156` 有 `train=3438`、`val=607`、`test=607`
- 训练用全部 55 类，不做类别过滤（`ShapeNetDataset(path, split='train')` 默认行为），训练集总量 43,433 shapes
- 归一化：`scale_mode='shape_unit'`（每 shape 按自身原始坐标的均值/std 归一化，评测时再反归一化回原坐标系）

## 6. 训练命令

三套模型分别在 A800 上单卡训练：

### 6.1 AutoEncoder

```bash
conda activate dpm3d
CUDA_VISIBLE_DEVICES=1 python scripts/train_ae.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --batch_size 384 \
    --num_workers 12 \
    --print_freq 200
```

### 6.2 GaussianVAE

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --batch_size 384 \
    --print_freq 200 \
    --model gaussian
```

### 6.3 FlowVAE

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --batch_size 384 \
    --print_freq 200 \
    --model flow
```

### 6.4 关键超参汇总

以下值来自各 `scripts/train_*.py` 的默认参数 + 上面命令行覆盖项。

| 参数 | AE | GaussianVAE / FlowVAE |
|---|---|---|
| `batch_size` | 384（命令行覆盖；默认 128） | 384（命令行覆盖） |
| `num_workers` | 12（命令行覆盖） | 默认 |
| `epochs` | 2000 | 2000 |
| `lr` | 1e-3 | 2e-3 |
| `zdim` | 256 | 256 |
| `T`（扩散步数） | 200 | 100 |
| `beta_T` | 0.05 | 0.02 |
| `grad_clip` | 10 | 10 |
| `scale_mode`（数据） | `shape_unit` | `shape_unit` |
| LR schedule | LinearLR 1.0→0.0，2000 epoch 线性衰减 | 同 |
| `kl_weight` | — | 0.001（Gen 模式） |
| `flow_layers` / `flow_hidden_dim` | — | 4 / 128（FlowVAE，仓库默认） |

产出的 checkpoint（每 100 epoch 存一次）：
- `checkpoints/ae/epoch_0100.pt … epoch_2000.pt`
- `checkpoints/gen/gaussian_epoch_0100.pt … gaussian_epoch_1800.pt`
- `checkpoints/gen/flow_epoch_0100.pt … flow_epoch_1800.pt`

## 7. 评测

### 7.1 AE（Table 2）— 已完成

命令：
```bash
python scripts/eval_ae.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --ckpt checkpoints/ae/epoch_2000.pt \
    --cates airplane chair car \
    --out_dir results/eval_ae
```

结果（反归一化后，`shape_unit` denorm 回原始坐标系）：

| Category | CD ×10³ | EMD ×10³ |
|---|---|---|
| Airplane | 0.178 | 1.871 |
| Car      | 0.577 | 2.093 |
| Chair    | 0.511 | 3.486 |

EMD 使用 `geomloss` Sinkhorn 近似（原仓库用 `approxmatch.cu` CUDA kernel，我们的环境不兼容），EMD 绝对值与论文不可直接比。

### 7.2 Gen（Table 1）— 进行中

训练已完成、checkpoint 已存（`checkpoints/gen/flow_epoch_1800.pt`、`gaussian_epoch_1800.pt` 等），评测**尚未跑**。评测前还有两处改动待加：

1. **补 JSD 指标** —— 之前 `metrics.py` 漏了 JSD（Jensen-Shannon Divergence between voxelized point cloud distributions），论文 Table 1 里有这一列。
2. **加 `[-1, 1]³` bbox 归一化** —— 论文 Sec 5.2 要求 `S_g` 和 `S_r` 都先归一化到 bbox 再算 MMD / COV / 1-NNA / JSD，当前 `scripts/eval_gen.py` 没做。

两项改完后再跑评测。

## 8. 备注

### 8.1 用官方 `AE_all.pt` 做对照实验

为验证 AE 的 ~10× CD gap 到底是"训练问题"还是"数据集问题"，把原作者发布的 `AE_all.pt` 放到我们的 `shapenet.hdf5` 上跑：

```bash
python scripts/eval_ae.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --ckpt checkpoints/pretrained/AE_all.pt \
    --cates airplane chair car \
    --out_dir results/original_eval_ae
```

`batch_size` 用脚本默认 32。结果：

| Category | CD ×10³ | EMD ×10³ |
|---|---|---|
| Airplane | 0.1949 | 1.3888 |
| Chair    | 0.5521 | 2.7122 |
| Car      | 0.5923 | 2.0631 |

官方权重在 `shapenet.hdf5` 上的结果与我们自训（7.1）同一量级，都远低于论文 Table 2 —— 证明 gap 来自数据集而非训练质量。完整分析在 [`../notes/dataset_investigation.md`](../notes/dataset_investigation.md) §5–6。

加载 `AE_all.pt` 需要在主线之外补一套"BN 编码器 + `var_sched` 长度 201→200"的兼容改动，这些改动**不会入主线仓库**，仅以补丁形式记录在 [`checkpoint_loading_restore.md`](checkpoint_loading_restore.md)，供需要时按文档还原。
