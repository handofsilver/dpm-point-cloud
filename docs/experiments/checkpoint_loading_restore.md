# Checkpoint 加载修复说明（可据此从当前代码复原出适配原仓库 AE_all.pt 的代码）

> **这份文档的用途**：当前仓库的主线实现只面向自训 ckpt（无 BN 编码器、单层 FC 头、`var_sched` 长 200、ckpt 键名 `model`），不能直接加载论文官方发布的 `AE_all.pt`（BN + 三层双分支编码器、`var_sched` 长 201、ckpt 键名 `state_dict`）。本文档把"在当前仓库状态基础上，补一套最小兼容改动让官方 `AE_all.pt` 跑通 `eval_ae.py` / `reconstruct.py` 等脚本"所需的每一处修改都列清楚 —— 一个 coding agent 读完本文档即可独立完成这套改动，不需要再读其它上下文。
>
> 本文档**不会**提交进仓库，只作为一次性的"补丁清单"留存。仓库主线保持"面向自训 ckpt 的简洁实现"这一定位。

## 1. 评测方式是否改变？

**没有改变 `eval_ae.py` 中的评测定义**，仅修改「如何把权重装进网络」：

| 环节 | 是否修改 |
|------|----------|
| `ShapeNetDataset`、按类别遍历测试集、`DataLoader` | 否 |
| 编码：`mu, _ = model.encoder(x0)`；重建：`model.sample(z=mu, ...)` | 否 |
| 反归一化：`x0_orig = x0 * s + sh`，`recon_orig = recon * s + sh` | 否 |
| 指标：`chamfer_distance(x0_orig, recon_orig)`、`earth_mover_distance(...)` | 否 |
| 表格：`mean_cd * 1e3`、各类别与 `mean`、写出 `table2_results.txt` | 否 |

**有改动的部分**：`torch.load` 之后如何取 `state_dict`、如何根据权重选择 Encoder 结构、官方 `AE_all.pt` 的 `var_sched` 在加载前是否裁掉首项（避免采样 NaN）。这些只影响「模型算出的重建点云」，不改变「CD/EMD 怎么定义、怎么聚合」。

`eval_gen.py` / `generate.py` 仅把 `ckpt["model"]` 改为兼容 `model` 与 `state_dict` 两种键名；生成与指标逻辑未改。

---

## 2. 新增文件：仓库根目录 `checkpoint_util.py`

在仓库根目录（与 `model.py` 同级）新建该文件，内容如下：

```python
"""从 checkpoint 取出 state_dict：本仓库 train_* 用键 'model'，官方预训练 AE_all.pt 用 'state_dict'。"""


def state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    raise KeyError(
        "checkpoint 需包含 'model'（本仓库训练脚本保存）或 'state_dict'（如官方 AE_all.pt）"
    )
```

---

## 3. 修改 `model.py`

### 3.1 在文件头部增加 `Optional` 导入

**修改前**（示意）：

```python
import math
import torch
```

**修改后**：

```python
import math
from typing import Optional

import torch
```

### 3.2 在 `PointNetEncoder` 类结束之后、`# Phase 4-A: AutoEncoder` 注释块之前插入以下整段

插入位置：紧接在 `PointNetEncoder.forward` 末尾 `return mu, log_var` 之后，在 `# =============================================================================` 与 `# Phase 4-A: AutoEncoder` 之前。

```python
class PointNetEncoderPretrained(nn.Module):
    """
    与官方预训练 AE_all.pt 一致的编码器：Conv+BN+ReLU，双分支 FC（含 BN）。
    本仓库默认的 PointNetEncoder 无 BN，无法加载该 checkpoint。
    """

    def __init__(self, zdim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        feat = torch.max(x, dim=2).values
        m = F.relu(self.fc_bn1_m(self.fc1_m(feat)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        mu = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(feat)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        log_var = self.fc3_v(v)
        return mu, log_var


def prepare_ae_state_dict_for_load(state_dict: dict) -> dict:
    """
    官方 AE_all.pt 的 var_sched 常为长度 201：第 0 项 ᾱ=1 为占位，若直接按本仓库
    sample() 下标使用会在最后一步除以 sqrt(1-ᾱ)=0 得到 NaN。裁掉首项后长度为 200，与训练脚本一致。
    """
    if "encoder.fc1_m.weight" not in state_dict:
        return state_dict
    betas = state_dict.get("diffusion.var_sched.betas")
    if betas is None or betas.shape[0] != 201:
        return state_dict
    out = dict(state_dict)
    for suffix in ("betas", "alphas", "alpha_bars", "sigmas_flex", "sigmas_inflex"):
        k = f"diffusion.var_sched.{suffix}"
        out[k] = out[k][1:].contiguous()
    return out


def build_autoencoder(
    args, device: torch.device, state_dict: Optional[dict] = None
) -> "AutoEncoder":
    """
    根据 state_dict 判断是官方 AE 权重（encoder.fc1_m）还是本仓库 train_ae 权重。
    官方权重扩散步数 T 与 var_sched 缓冲区长度由 checkpoint 决定（裁掉占位后通常为 200）。
    """
    if state_dict is not None and "encoder.fc1_m.weight" in state_dict:
        T = int(state_dict["diffusion.var_sched.betas"].shape[0])
        var_sched = VarianceSchedule(T=T, beta_T=args.beta_T)
        net = PointwiseNet(zdim=args.zdim, residual=True)
        diffusion = DiffusionPoint(net=net, var_sched=var_sched)
        encoder = PointNetEncoderPretrained(zdim=args.zdim)
        return AutoEncoder(encoder=encoder, diffusion=diffusion).to(device)

    var_sched = VarianceSchedule(T=args.T, beta_T=args.beta_T)
    net = PointwiseNet(zdim=args.zdim, residual=True)
    diffusion = DiffusionPoint(net=net, var_sched=var_sched)
    encoder = PointNetEncoder(zdim=args.zdim)
    return AutoEncoder(encoder=encoder, diffusion=diffusion).to(device)
```

**注意**：`build_autoencoder` 在源码中出现在 `class AutoEncoder` **之前**，但函数体内调用了 `AutoEncoder(...)`。这在 Python 中合法：只有**运行** `build_autoencoder` 时才会解析 `AutoEncoder`，此时类已定义完毕。

---

## 4. 修改 `scripts/eval_ae.py`

### 4.1 替换 import 段

**修改前**（示意）：

```python
from dataset import ShapeNetDataset
from model import VarianceSchedule, PointwiseNet, DiffusionPoint, PointNetEncoder, AutoEncoder
from metrics import chamfer_distance, earth_mover_distance
```

**修改后**：

```python
from checkpoint_util import state_dict_from_checkpoint
from dataset import ShapeNetDataset
from model import build_autoencoder, prepare_ae_state_dict_for_load
from metrics import chamfer_distance, earth_mover_distance
```

### 4.2 删除脚本内的 `build_model` 函数（整个函数定义）

### 4.3 替换 `main()` 里「加载模型」片段

**修改前**（示意）：

```python
    model = build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
```

**修改后**：

```python
    # --- 加载模型（train_ae 用 ckpt['model']；官方 AE_all.pt 用 ckpt['state_dict'] + 不同 Encoder）---
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = prepare_ae_state_dict_for_load(state_dict_from_checkpoint(ckpt))
    model = build_autoencoder(args, device, state_dict=sd)
    model.load_state_dict(sd)
```

`main()` 中其余代码（从 `model.eval()` 到文件末尾）与修改前一致。

---

## 5. 修改 `scripts/reconstruct.py`

### 5.1 在 `sys.path.insert` 之后增加 import

```python
from checkpoint_util import state_dict_from_checkpoint
from dataset import ShapeNetDataset
from model import build_autoencoder, prepare_ae_state_dict_for_load
```

并删除原先对 `VarianceSchedule, PointwiseNet, DiffusionPoint, PointNetEncoder, AutoEncoder` 的 import（若不再使用）。

### 5.2 删除本地的 `build_model` 函数

### 5.3 将加载模型部分改为

```python
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = prepare_ae_state_dict_for_load(state_dict_from_checkpoint(ckpt))
    model = build_autoencoder(args, device, state_dict=sd)
    model.load_state_dict(sd)
```

其余推理与可视化逻辑不变。

---

## 6. 修改 `scripts/eval_gen.py`

在 `sys.path.insert` 之后、`import torch` 之前增加：

```python
from checkpoint_util import state_dict_from_checkpoint
```

将：

```python
    model.load_state_dict(ckpt["model"])
```

改为：

```python
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
```

---

## 7. 修改 `scripts/generate.py`

同样在 `sys.path.insert` 之后增加：

```python
from checkpoint_util import state_dict_from_checkpoint
```

将：

```python
    model.load_state_dict(ckpt["model"])
```

改为：

```python
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
```

---

## 8. 校验清单

完成上述步骤后应满足：

1. 本仓库 `train_ae` 保存的 checkpoint（含 `ckpt["model"]`）仍可被 `eval_ae.py` / `reconstruct.py` 加载（走 `PointNetEncoder` + `build_autoencoder` 默认分支）。
2. 官方 `AE_all.pt`（含 `ckpt["state_dict"]` 与 `encoder.fc1_m.*`）可被加载；若 `betas` 长度为 201，会先裁成 200 再加载，避免采样 NaN。
3. `eval_ae.py` 输出的 Table 2 格式与修改前一致（同一套 per-sample CD、均值、×1e3）。

---

## 9. 文件变更汇总

| 路径 | 操作 |
|------|------|
| `checkpoint_util.py` | 新建 |
| `model.py` | 增加 `Optional` 导入；增加 `PointNetEncoderPretrained`、`prepare_ae_state_dict_for_load`、`build_autoencoder` |
| `scripts/eval_ae.py` | 加载逻辑与 import 调整；删除内联 `build_model` |
| `scripts/reconstruct.py` | 同上（适配官方 AE） |
| `scripts/eval_gen.py` | 通用 `state_dict` 键名 |
| `scripts/generate.py` | 通用 `state_dict` 键名 |
