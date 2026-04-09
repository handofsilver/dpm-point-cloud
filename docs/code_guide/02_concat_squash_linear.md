# 02 · ConcatSquashLinear — 条件注入层

> 对应代码: `model.py` → `class ConcatSquashLinear`
> 对应论文: Section 4.2（PointwiseNet 描述中的条件注入机制）

---

## 这个模块解决什么问题？

`PointwiseNet` 是一个处理点坐标的 MLP，但它不能只靠点坐标预测噪声——它还需要知道：

- **当前时间步**：t=5 和 t=150 的噪声强度完全不同
- **shape latent z**：这个点云是椅子还是飞机？

这两个信息合并成一个**条件向量 ctx**，问题变成：**怎么把 ctx 注入到点特征里？**

---

## 方案对比

### 朴素做法：Concat 拼接

$$
\text{output} = W \cdot [x \,\|\, \text{ctx}] + b
$$

直接把 $x$ 和 ctx 拼起来，过一个线性层。简单，但条件和内容"平等地"混在一起，条件的影响方式固定。

### 这篇论文的做法：门控调制（Gated Modulation）

$$
\boxed{\text{output} = \underbrace{W_x \cdot x}_{\text{内容变换}} \odot \underbrace{\sigma(W_{\text{gate}} \cdot \text{ctx})}_{\text{门}} + \underbrace{W_{\text{bias}} \cdot \text{ctx}}_{\text{偏置注入}}}
$$

三个路径：

| 路径 | 公式 | 作用 |
|:---:|:---:|:---|
| 主变换 | $W_x \cdot x$ | 对点特征做线性变换 |
| 门控 | $\sigma(W_{\text{gate}} \cdot \text{ctx}) \in (0,1)$ | 条件决定"放大或压缩"哪些维度 |
| 偏置 | $W_{\text{bias}} \cdot \text{ctx}$ | 条件直接加性注入 |

**直觉**：门 $\sigma(\cdot)$ 让网络根据条件**动态地选择**哪些特征维度重要——不同时间步、不同形状，激活不同的特征通道。这比简单 concat 更有表达能力。

---

## 张量形状分析

```
x   : (B, N, in_dim)    — B 个点云，每个 N 个点，每点 in_dim 维特征
ctx : (B, 1, ctx_dim)   — 同一 batch 里，条件对所有 N 个点共享（N=1）

W_x(x)    : (B, N, out_dim)
gate(ctx) : (B, 1, out_dim)  ← N=1，广播到 N
bias(ctx) : (B, 1, out_dim)  ← N=1，广播到 N

output = W_x(x) * gate + bias : (B, N, out_dim)
```

**关键**：`ctx` 的 N 维是 1，与 `x` 的 N 维（=2048）相乘时自动**广播（broadcast）**。
这正是"2048 个点共享同一个条件"的实现方式——一个 ctx 管所有点。

---

## 代码实现

```python
class ConcatSquashLinear(nn.Module):
    def __init__(self, in_dim, out_dim, ctx_dim):
        super().__init__()
        self._layer      = nn.Linear(in_dim, out_dim)            # 主变换
        self._hyper_gate = nn.Linear(ctx_dim, out_dim)           # 门控路径
        self._hyper_bias = nn.Linear(ctx_dim, out_dim, bias=False)  # 偏置路径

    def forward(self, x, ctx):
        # x:   (B, N, in_dim)
        # ctx: (B, 1, ctx_dim)
        gate = torch.sigmoid(self._hyper_gate(ctx))   # (B, 1, out_dim)
        bias = self._hyper_bias(ctx)                  # (B, 1, out_dim)
        return self._layer(x) * gate + bias           # (B, N, out_dim)，广播
```

注意 `_hyper_bias` 的 `bias=False`：bias 本身就是在做"加性注入"，线性层自带的截距项是多余的。

---

## 验证结果

```
输入 x   shape: [4, 2048, 3]
条件 ctx shape: [4, 1, 259]     ← N=1，不是 2048
输出 out shape: [4, 2048, 128]  ← 正确广播
```

---

## 与后续模块的关系

```
ConcatSquashLinear
    └── PointwiseNet   — 6 层堆叠，搭建噪声预测 MLP
                         每层的 ctx_dim = len(time_emb) + zdim = 3 + 256 = 259
```
