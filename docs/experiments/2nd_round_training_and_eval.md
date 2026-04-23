# 第二轮训练与评测记录

> 范围：**仅针对 Table 1（生成任务）**，按 [`next_experiments.md`](next_experiments.md) 的 P0 + P1 合并执行一轮 per-category 训练。AE 与 Table 2 相关内容本轮未重训，沿用 [`1st_round_training_and_eval.md`](1st_round_training_and_eval.md) §7.1 的结果。
>
> 相关专题另有文档，本文不再重复：
> - Gen 评测与论文 Table 1 的差距归因 → [`gen_eval_gap_analysis.md`](gen_eval_gap_analysis.md)
> - JSD 列数据下限与模型残差拆分 → [`jsd_gap_analysis.md`](jsd_gap_analysis.md)
> - EMD 用 Sinkhorn 而非 approxmatch 的决定 → [`emd_sinkhorn_decision.md`](emd_sinkhorn_decision.md)

---

## 1. 环境

基本与 [`1st_round_training_and_eval.md`](1st_round_training_and_eval.md) §1–§4 相同。唯一变化：服务器侧 cuDNN 兼容问题已修复，不再需要 `torch.backends.cudnn.enabled = False` 的 workaround。该 workaround 本身不影响数值结果（只影响 Conv 速度），修复前后评测数值可比。

## 2. 改动动机

Round 1（55 类合训 + LinearLR 1.0→0.0）生成评测远离论文水位，具体差距与归因见 [`gen_eval_gap_analysis.md`](gen_eval_gap_analysis.md)。本轮按 [`next_experiments.md`](next_experiments.md) 合并 P0 + P1 一次性落地：

- **P0 —— airplane / chair 各训一个 FlowVAE（per-category）**，对齐论文 Table 1 "Ours" 行的 `GEN_airplane.pt` / `GEN_chair.pt` 协议
- **P1 —— LR schedule 换成两段式 plateau + decay**，近似论文 `sched_start=200K, sched_end=400K, end_lr=1e-4` 的策略，按 airplane-only 的 iter 数量级缩放到 `plateau [0, 10000) lr=2e-3, decay [10000, 20000) → 1e-4, then hold`

此外，为验证可学习先验（Flow）相对固定先验（Gaussian）的价值，在同一协议下追加了 **GaussianVAE airplane / chair** 对照训练。

P2（Flow 扩容到 14 层 / hidden 256）与 P3（Encoder BN + 三层 FC head）本轮**未做**。

## 3. 训练命令

两个训练任务分别占用 A800 单卡，并行执行：

### 3.1 FlowVAE — airplane

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --cates airplane \
    --model flow \
    --epochs 20000 \
    --batch_size 384 \
    --print_freq 200 \
    --save_freq 500
```

启动日志：
```
使用设备：cuda | 模型：flow
训练数据：3438 shapes | cates=['airplane']
LR schedule: plateau [0, 10000) lr=2.00e-03, decay [10000, 20000) → 1.00e-04, then hold
```

### 3.2 FlowVAE — chair

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --cates chair \
    --model flow \
    --epochs 20000 \
    --batch_size 384 \
    --print_freq 200 \
    --save_freq 500
```

启动日志：
```
使用设备：cuda | 模型：flow
训练数据：5602 shapes | cates=['chair']
```

### 3.3 iter 数量级核算

- airplane：3438 / 384 ≈ 9 batch/epoch，20000 epoch ≈ **180K iter**
- chair：5602 / 384 ≈ 15 batch/epoch，20000 epoch ≈ **300K iter**

与论文 400K iter 量级同阶。LR schedule 的切换点 10000 epoch 对应 iter 90K（airplane）/ 150K（chair），与论文 200K 的 plateau→decay 切换点同量级。

产出 checkpoint：`checkpoints/gen/flow_airplane_epoch_*.pt` / `checkpoints/gen/flow_chair_epoch_*.pt`（每 500 epoch 一次）。

### 3.4 GaussianVAE — airplane / chair（FlowVAE 对照）

训练命令与上面 FlowVAE 完全一致，仅 `--model` 改为 `gaussian`（其他参数 batch_size / epochs / cates 等全部保持默认不变）：

```bash
# airplane
python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --cates airplane \
    --model gaussian \
    --epochs 20000 \
    --batch_size 384 \
    --print_freq 200 \
    --save_freq 500

# chair
python scripts/train_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --cates chair \
    --model gaussian \
    --epochs 20000 \
    --batch_size 384 \
    --print_freq 200 \
    --save_freq 500
```

产出 checkpoint：`checkpoints/gen/gaussian_airplane_epoch_*.pt` / `checkpoints/gen/gaussian_chair_epoch_*.pt`。

## 4. 评测命令

```bash
# airplane
CUDA_VISIBLE_DEVICES=1 python scripts/eval_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --ckpt checkpoints/gen/flow_airplane_epoch_20000.pt \
    --model flow \
    --cates airplane \
    --out_dir results/eval_gen_airplane

# chair
CUDA_VISIBLE_DEVICES=1 python scripts/eval_gen.py \
    --data_path data/shapenet/shapenet.hdf5 \
    --ckpt checkpoints/gen/flow_chair_epoch_20000.pt \
    --model flow \
    --cates chair \
    --out_dir results/eval_gen_chair
```

GaussianVAE 评测命令同上，替换 `--ckpt` 和 `--model gaussian` 即可。

评测协议沿用 Round 1：bbox 归一化后算 MMD/COV/1-NNA（CD 与 EMD 各一份）+ JSD，详见 [`1st_round_training_and_eval.md`](1st_round_training_and_eval.md) §7.2。

## 5. 结果

### 5.1 FlowVAE — airplane（已完成）

评测输出：
```
[airplane] 测试集大小: 607
[airplane] 从先验采样 607 个形状...
  MMD-CD: 0.003389
  COV-CD: 0.487644
  1-NNA-CD: 0.701812
  MMD-EMD: 0.023530
  COV-EMD: 0.439868
  1-NNA-EMD: 0.766886
  JSD: 0.011060
```

与 Round 1 / 论文 Table 1 对照：

| 指标       | Round 1（all-55 合训） | **Round 2（airplane-only + P1 LR）** | 论文 Table 1 Airplane |
|------------|------------------------|--------------------------------------|-----------------------|
| MMD-CD     | 0.004911               | **0.003389**                         | 0.003276              |
| COV-CD     | 13.34%                 | **48.76%**                           | 48.71%                |
| 1-NNA-CD   | 90.44%                 | **70.18%**                           | 64.83%                |
| MMD-EMD    | 0.035263               | **0.023530**                         | 0.1061（approxmatch，量纲不可比） |
| COV-EMD    | 16.14%                 | **43.99%**                           | 45.47%                |
| 1-NNA-EMD  | 92.09%                 | **76.69%**                           | 75.12%                |
| JSD        | 0.441765               | **0.011060**                         | 0.001067              |

### 5.2 FlowVAE — chair（已完成）

评测输出：
```
chair             0.012761      0.472194      0.614257      0.055418      0.441860      0.695652      0.010844
```

与 Round 1 / 论文 Table 1 对照：

| 指标       | Round 1（all-55 合训） | **Round 2（chair-only + P1 LR）** | 论文 Table 1 Chair |
|------------|------------------------|-----------------------------------|--------------------|
| MMD-CD     | 0.016547               | **0.012761**                      | 0.012276           |
| COV-CD     | 20.83%                 | **47.22%**                        | 48.94%             |
| 1-NNA-CD   | 84.73%                 | **61.43%**                        | 60.11%             |
| MMD-EMD    | 0.070243               | **0.055418**                      | 0.1784（approxmatch，量纲不可比） |
| COV-EMD    | 21.33%                 | **44.19%**                        | 47.52%             |
| 1-NNA-EMD  | 89.23%                 | **69.57%**                        | 69.06%             |
| JSD        | 0.104457               | **0.010844**                      | 0.007797           |

现象与 airplane 同构：CD 系列基本贴上论文（MMD-CD 相对差 3.9%、COV-CD 差 1.7 pt、1-NNA-CD 超出论文 1.3 pt 且距 50% 理想点尚有 ~11 pt），COV-EMD / 1-NNA-EMD 与论文相差 ≤3 pt。JSD 0.010844 量级与 airplane 0.011 一致，模型残差方向性相同。

### 5.3 GaussianVAE — airplane（FlowVAE 对照）

| 指标       | GaussianVAE | FlowVAE | 差值 |
|------------|-------------|---------|------|
| MMD-CD     | 0.003484    | 0.003389 | +2.8% |
| COV-CD     | 48.93%      | 48.76%  | +0.2 pt |
| 1-NNA-CD   | 70.76%      | 70.18%  | +0.6 pt |
| MMD-EMD    | 0.024364    | 0.023530 | +3.5% |
| COV-EMD    | 42.83%      | 43.99%  | -1.2 pt |
| 1-NNA-EMD  | 78.75%      | 76.69%  | +2.1 pt |
| JSD        | 0.013287    | 0.011060 | +20% |

### 5.4 GaussianVAE — chair（FlowVAE 对照）

| 指标       | GaussianVAE | FlowVAE | 差值 |
|------------|-------------|---------|------|
| MMD-CD     | 0.013336    | 0.012761 | +4.5% |
| COV-CD     | 43.07%      | 47.22%  | -4.1 pt |
| 1-NNA-CD   | 66.84%      | 61.43%  | +5.4 pt |
| MMD-EMD    | 0.057231    | 0.055418 | +3.3% |
| COV-EMD    | 43.48%      | 44.19%  | -0.7 pt |
| 1-NNA-EMD  | 72.65%      | 69.57%  | +3.1 pt |
| JSD        | 0.014029    | 0.010844 | +29% |

## 6. 分析

### 6.1 FlowVAE vs 论文 Table 1

**CD 系列基本对齐论文**。airplane / chair 两类 MMD-CD 相对论文差 ≤4%，COV-CD 相对论文差 ≤2 pt，1-NNA-CD 均比论文偏高 1–5 pt 且距 50% 理想点尚有距离。P0 + P1 的合并改动把 Round 1 的主要差距一次性压到论文水位，归因链 [`gen_eval_gap_analysis.md`](gen_eval_gap_analysis.md) §6.1（训练数据子集匹配）+ §6.3（LR schedule）都得到了实测验证。

**EMD 系列方向性对齐**。两类 COV-EMD / 1-NNA-EMD 与论文差 ≤3 pt。MMD-EMD 绝对值与论文差 ~3.2×–4.5×，是 Sinkhorn vs approxmatch 的系统性差异，非模型差距；详见 [`emd_sinkhorn_decision.md`](emd_sinkhorn_decision.md) §5.2。

**JSD 仍余 0.010 量级**。airplane 0.011 / chair 0.011 都与论文 0.001–0.008 相差一个量级，但这不是纯模型残差 —— [`jsd_gap_analysis.md`](jsd_gap_analysis.md) 已用纯数据 oracle 实验证明 `shapenet.hdf5` 上 airplane train/test 本身的 JSD 下限就是 ~0.007（相对论文 PC15k 下限 0.000809 已经有 8.6× 的数据结构性差距），剩余 ~0.004 才是模型残差。想进一步压这 0.004 需要跑 P2（Flow 扩容），本轮未动。

### 6.2 FlowVAE vs GaussianVAE（内部 ablation）

**FlowVAE 一致性地优于 GaussianVAE，但优势幅度有限**：

- **CD 系列**：airplane 上差异在噪声范围内（MMD-CD 差 2.8%，COV/1-NNA 基本持平）。chair 上 Flow 优势更明显 —— COV-CD 多 4.1 pt、1-NNA-CD 好 5.4 pt，这两个是分布质量最直接的反映。
- **JSD**：两类一致地 Flow 优 20–29%，是最稳定的 separation 信号。JSD 反映体素分辨率下的分布覆盖性，Flow prior 的可学习性在这里体现。
- **EMD 系列**方向性与 CD 一致。

**优势有限的原因**：本仓库 Flow 仅 4 层 / hidden 128，论文 `GEN_airplane.pt` 的 Flow 是 14 层 / hidden 256（容量差 ~10 倍）。小 Flow 拟合能力有限，先验 $p_\theta(z)$ 与 $\mathcal{N}(0, I)$ 差距不大，ablation 信号自然偏弱。论文在大 Flow 下 FlowVAE 与 GaussianVAE 的差距会更显著，这是 P2（Flow 扩容）的预期信号 —— 但在本仓库"学习读物"的定位下不追。

**结论**：可学习先验（Flow）相对固定先验（Gaussian）的价值方向性成立，两类一致。

## 7. 实验状态

本轮完成：
- P0 + P1 合并落地，FlowVAE airplane / chair 均出结果，CD 系列对齐论文
- GaussianVAE 同协议对照完成，FlowVAE > GaussianVAE 方向性验证

未做（定性为非必要）：
- P2（Flow 14 层 / hidden 256）：纯论文对齐维度，不影响定性结论
- P3（Encoder BN + 三层 FC head）：同上
