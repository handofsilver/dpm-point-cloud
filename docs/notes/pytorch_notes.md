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
