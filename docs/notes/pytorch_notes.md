# PyTorch 通用知识笔记

> 本文档记录在复现 DPM-3D 过程中遇到的 PyTorch 工程模式与 API 用法。
> 按话题分节，动态更新。

---

## 1. `register_buffer` vs `nn.Parameter`

### 背景

在 `nn.Module` 中存储张量，有三种方式：

| 方式 | 梯度 | 被优化器更新 | 随模型 `.to(device)` 移动 | 保存进 `state_dict` |
|---|---|---|---|---|
| `self.x = tensor` | 否 | 否 | **否** | **否** |
| `self.register_buffer('x', tensor)` | 否 | **否** | **是** | **是** |
| `self.x = nn.Parameter(tensor)` | **是** | **是** | **是** | **是** |

### 详解

**`nn.Parameter`**：
- 告诉 PyTorch "这个张量是模型要学习的参数"
- 自动加入 `model.parameters()`，因此优化器会更新它
- 典型用途：线性层的权重 `weight`、偏置 `bias`

**`register_buffer`**：
- 告诉 PyTorch "这个张量是模型的**固定常量**，不用学，但要跟着模型走"
- 不加入 `model.parameters()`，优化器看不到它、不更新它
- 但会跟着 `model.to('cuda')` 自动搬到 GPU
- 也会被 `model.state_dict()` 保存（方便 checkpoint 恢复）
- 典型用途：预计算的常数表（如扩散过程的 β、ᾱ）

### 为什么 VarianceSchedule 用 `register_buffer`？

`VarianceSchedule` 里的 β、α、ᾱ 都是根据超参数预计算好的**固定数表**，训练过程中不应该被改变。但训练时点云数据在 GPU 上，所以这些表也要在 GPU 上，否则运算会报设备不匹配的错误。

`register_buffer` 正好满足：**不被学习，但跟着模型走**。

```python
class VarianceSchedule(nn.Module):
    def __init__(self, T, beta_T):
        super().__init__()
        betas = torch.linspace(1e-4, beta_T, T)   # 普通张量
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # 用 register_buffer 注册，而不是 self.betas = betas
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        # ...
```

---

## 2. 残差连接（Residual Connection）

### 用法

```python
# 不用残差：输出完全依赖网络
out = network(x)

# 用残差：网络只学"在 x 基础上改变多少"
out = network(x) + x
```

### 为什么有效？

- **梯度流动**：反向传播时，梯度可以沿 `+ x` 这条捷径直接流回输入层，缓解深层网络梯度消失
- **学习更容易**：网络只需预测"残差"（偏差量），不需要从零构建完整输出
- **退化保险**：若某层什么都学不到，令其输出为 0，整体仍等于恒等变换（不比没有这层更差）

### 在 PointwiseNet 中的含义

最终输出 = 网络预测增量 + 原始 $x^{(t)}$，网络专注于预测"需要减去多少噪声"，而不是直接预测去噪后的坐标。

---

## 3. ReLU vs LeakyReLU

```
ReLU:       f(x) = max(0, x)
LeakyReLU:  f(x) = x        if x > 0
                 = 0.01 * x  if x ≤ 0
```

### 为什么 PointwiseNet 用 LeakyReLU？

ReLU 在 $x < 0$ 时梯度为 0。如果某个神经元的输入长期为负，它的权重永远得不到梯度更新——"死亡神经元"（Dead ReLU）。

LeakyReLU 在负半轴保留一个小斜率（默认 0.01），让死亡神经元仍有微弱梯度，训练更稳健。

### 使用规则（PointwiseNet）

- 前 5 层（中间层）：`LeakyReLU` — 保持梯度流动
- 最后 1 层（输出层）：**不接激活函数** — 输出是坐标偏移量，可正可负，不应被截断

---

## 4. `^` vs `**` — Python 运算符陷阱

```python
# 按位异或（bitwise XOR）— 仅对整数有意义
3 ^ 2    # = 1（二进制 11 XOR 10 = 01）

# 幂运算（power）
3 ** 2   # = 9
```

在 PyTorch 中，对 float Tensor 使用 `^` 会直接 **报错**（`RuntimeError`）。

正确写法：

```python
# 手写 MSE
torch.mean((a - b) ** 2)

# 推荐：使用 F.mse_loss，语义清晰
F.mse_loss(a, b)
```

---

## 5. `expand` vs `repeat` — 广播工具

两者都能把 shape 扩展，但机制不同：

```python
x = torch.tensor([3.14])   # shape: (1,)

x.expand(4)   # shape: (4,)，共享内存，不复制
x.repeat(4)   # shape: (4,)，复制 4 份，独立内存
```

| | `expand` | `repeat` |
|---|---|---|
| 内存 | 共享底层数据（视图） | 真实复制 |
| 可安全写入？ | 否（写入影响所有元素） | 是 |
| 适用场景 | 只读广播（乘法、加法） | 需要独立修改每个元素 |

DiffusionPoint 采样循环中的 `beta.expand(B)` 就是典型用法：只做乘法，不需要复制。

---

## 6. Tensor 广播规则速查

当两个不同 shape 的 Tensor 做逐元素运算时，PyTorch 从**最右维**开始对齐：

- 维度相同 → OK
- 其中一个是 1 → 自动扩展到另一个的大小
- 维度数不同 → 在左边补 1

```python
alpha_bar = ...   # (B,)
x0 = ...          # (B, N, 3)

# 直接乘：(B,) 与 (B, N, 3)
# PyTorch 先把 (B,) 看成 (B, 1, 1)？不会！
# 它会看成 (1, 1, B)... 不对
# 实际：从右对齐，(B,) 对 (B, N, 3) 的最后一维 3：B ≠ 3，报错！

# 正确：手动扩维
alpha_bar.view(B, 1, 1) * x0   # (B, 1, 1) * (B, N, 3) → (B, N, 3) ✓
```

经验法则：**当标量/向量要与高维 Tensor 相乘时，总是显式 `.view()` 到正确的 shape**。不要依赖自动广播的隐含行为。

---

## 7. （占位）更多话题待补充

- `torch.cumprod` 的用法
- 为什么用 `log_var` 而不是 `sigma`
- `detach()` 在什么情况下必须用
- `Conv1d` 为什么比 `Conv2d` 更适合点云

---

## 8. Dataset + DataLoader — PyTorch 数据加载模式

### 两个组件的分工

| 组件 | 职责 | 你需要实现什么 |
|---|---|---|
| `Dataset` | 知道数据在哪、怎么读、怎么预处理 | `__len__` 和 `__getitem__` |
| `DataLoader` | 打包成 batch、随机打乱、多线程加载 | 不需要实现，直接用 |

```python
dataset = ShapeNetDataset(root="data/shapenet", split="train")
loader  = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

for x0 in loader:          # x0: (128, 2048, 3)，DataLoader 自动拼 batch
    loss = model.get_loss(x0)
```

`DataLoader` 内部调用 `dataset[i]` 若干次，把返回的 Tensor 沿第 0 维 stack 成 batch。你只需要保证 `__getitem__` 返回形状一致的 Tensor。

### `__len__` 和 `__getitem__` 的语义

```python
len(dataset)       # 数据集总样本数，DataLoader 用来决定一个 epoch 迭代多少次
dataset[42]        # 第 42 个样本（单条，无 batch 维）
```

---

## 9. `glob.glob` — 文件路径匹配

`glob.glob(pattern)` 返回所有匹配 pattern 的文件路径列表。支持 `*`（任意字符）和 `**`（递归）通配符。

```python
import glob, os

# 找到 data/ 目录下所有包含 "train" 的 .h5 文件
pattern  = os.path.join("data", "*train*.h5")
h5_files = sorted(glob.glob(pattern))
# sorted() 保证每次运行顺序一致（文件系统返回顺序不稳定）
```

常见陷阱：忘记 `sorted()`。不加排序时，不同机器或文件系统返回顺序可能不同，导致实验不可复现。

---

## 10. `np.concatenate` — 合并 numpy 数组列表

```python
import numpy as np

arrays = [np.ones((100, 2048, 3)),
          np.ones((200, 2048, 3)),
          np.ones((150, 2048, 3))]

result = np.concatenate(arrays, axis=0)   # shape: (450, 2048, 3)
```

接受的是**列表**，沿指定轴拼接。注意：

- 不能用 Python `+`（那只拼接 list 本身，不合并数组内容）
- 拼接轴以外的维度必须完全相同（这里 axis=0，所以 `2048, 3` 必须一致）
- 返回新数组，原列表中的数组不被修改

---

## 11. `np.random.choice` — 不放回随机采样

```python
# 从 [0, N) 中不放回地随机采 num_points 个整数索引
choice = np.random.choice(N, num_points, replace=False)
pc = pc[choice]   # fancy indexing，取出对应的行
```

三个参数：
- `a`（第 1 个）：采样范围。传整数 n 表示从 `range(n)` 里采；传数组则直接从数组元素里采
- `size`（第 2 个）：采多少个
- `replace`：`True` = 放回采样（可重复），`False` = 不放回（每个只出现一次）

点云采样用 `replace=False`：保证 2048 个点互不重复。

---

## 12. 归一化：为什么 `mean(axis=0)` 但 `std()` 全局？

```python
pc -= pc.mean(axis=0)   # 零均值：每个轴独立去均值
pc /= pc.std()          # 单位方差：全局一个标量缩放
```

**`mean(axis=0)`**：沿 N 个点求均值，结果是 `(3,)`，即 xyz 各自的重心坐标。减去它把整朵点云平移到原点。三轴分别处理是对的，因为重心的 x/y/z 各自独立。

**`std()` 不指定 axis**：如果改成 `std(axis=0)` 就会对 xyz 三轴分别缩放——椅背和椅腿会被独立拉伸到同样的"高度"，形状变形。用全局 std 是**均匀缩放**，只改尺寸，不改比例，形状保持不变。

```
错误：std(axis=0)         正确：std()（全局标量）
  z  ↑  ← 椅背被压短         z  ↑
     │  ← 椅腿被拉长             │  形状不变，只是整体缩小
─────┼─────                  ─────┼─────
```

---

## 13. `torch.from_numpy()` 和 `.float()`

```python
tensor = torch.from_numpy(pc).float()
```

**`torch.from_numpy(pc)`**：
- 零拷贝地把 numpy array 转成 Tensor，两者**共享底层内存**
- dtype 与 numpy 保持一致（h5py 读出来通常是 `float64`）

**`.float()`**：
- 等价于 `.to(torch.float32)`，转成 32 位浮点
- 必须转：模型权重默认是 `float32`，输入是 `float64` 会报类型不匹配错误
- 也节省显存（float64 是 float32 的两倍）

---

## 14. h5py — 读取 HDF5 文件

HDF5 是一种"文件里的文件系统"格式，用路径访问数据集（dataset）。

```python
import h5py

with h5py.File("data.h5", "r") as f:
    # 列出文件内的所有 key
    print(list(f.keys()))          # 例如 ['data', 'label']

    # 读取数据（加 [:] 转为 numpy array，否则是懒加载对象）
    points = f["data"][:]          # numpy array: (M, 2048, 3)
    labels = f["label"][:]         # numpy array: (M,)
```

关键细节：
- `f["data"]` 返回 HDF5 Dataset 对象（懒加载，未读入内存）
- `f["data"][:]` 或 `np.array(f["data"])` 才是真正读入内存的 numpy array
- 推荐用 `np.array(f["data"])`：Pylance 能推断返回类型，无需 `# type: ignore`
- `with` 语句保证文件正确关闭，避免文件句柄泄漏

---

## 15. PyTorch 训练循环 — 固定五步

每个 batch 的标准写法，顺序不能乱：

```python
optimizer.zero_grad()                          # 1. 清空上一步梯度（不清会累积）
loss = model.get_loss(x0)                      # 2. 前向传播，算 loss
loss.backward()                                # 3. 反向传播，算梯度
clip_grad_norm_(model.parameters(), max_norm)  # 4. 裁剪梯度（可选，防爆炸）
optimizer.step()                               # 5. 用梯度更新参数
```

外层套两层循环：

```python
for epoch in range(num_epochs):    # 外层：epoch 数
    for x0 in loader:              # 内层：每个 batch
        x0 = x0.to(device)        # 数据移到和模型相同的设备
        # 五步
    scheduler.step()               # 每 epoch 结束后更新 lr
```

---

## 16. 优化器与 LR Scheduler

```python
# Adam：最常用，自适应学习率，一般不需要调太多
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# LinearLR：从 lr 线性衰减到 0
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,   # 初始倍率（lr × 1.0 = lr）
    end_factor=0.0,     # 结束倍率（lr × 0.0 = 0）
    total_iters=2000,   # 衰减总步数（epoch 数）
)
scheduler.step()        # 每个 epoch 结束后调用一次
```

`model.parameters()` 返回模型所有可学习参数的迭代器，传给优化器后它会统一管理。

---

## 17. `model.state_dict()` 与 Checkpoint 保存

`state_dict` 是一个有序字典，key 是参数名，value 是对应的 Tensor。例如一个两层 MLP 的 state_dict 长这样：

```python
import torch.nn as nn

mlp = nn.Sequential(nn.Linear(3, 64), nn.Linear(64, 3))
print(mlp.state_dict().keys())
# odict_keys(['0.weight', '0.bias', '1.weight', '1.bias'])
#              ↑第0层权重  ↑第0层偏置  ↑第1层权重  ↑第1层偏置
```

AutoEncoder 的 state_dict key 长这样：

```
encoder.conv1.weight
encoder.conv1.bias
encoder.fc_mu.weight
diffusion.net.layers.0._layer.weight
diffusion.var_sched.betas          ← register_buffer 也在里面
...
```

**保存与恢复**：

```python
# 保存：用字典包装，同时存 epoch 方便续训
torch.save({"epoch": epoch, "model": model.state_dict()}, "epoch_0100.pt")

# 恢复：先建好模型结构，再把参数填进去
ckpt = torch.load("epoch_0100.pt")
model.load_state_dict(ckpt["model"])
start_epoch = ckpt["epoch"] + 1
```

只保存 `state_dict` 而非整个模型对象，原因：保存整个对象依赖 pickle + 代码路径，换个目录就可能加载失败；state_dict 只是纯数据，只要模型结构一致就能恢复。

---

## 18. `model.train()` 与 `model.eval()`

```python
model.train()   # 训练模式：Dropout 激活，BatchNorm 用 batch 统计量
model.eval()    # 推理模式：Dropout 关闭，BatchNorm 用运行均值
```

固定写法：
- 训练循环开始时调 `model.train()`
- 验证/推理时调 `model.eval()`，并用 `with torch.no_grad():` 关闭梯度计算节省显存

---

## 19. `clip_grad_norm_` — 梯度裁剪

```python
from torch.nn.utils import clip_grad_norm_

clip_grad_norm_(model.parameters(), max_norm=10.0)
```

把所有参数梯度的全局 L2 范数裁剪到 `max_norm` 以内。训练早期梯度可能很大，裁剪防止单步更新太猛把模型"踢飞"。调用位置在 `loss.backward()` 之后、`optimizer.step()` 之前。

---

## 20. `loss.item()` — 标量 Tensor 转 Python float

```python
loss = model.get_loss(x0)   # Tensor，带梯度
print(loss.item())          # Python float，不带梯度，可以打印/记录
```

`.item()` 只适用于单元素 Tensor。打印 loss 时必须用，否则每次打印都会保留计算图，造成显存泄漏。
