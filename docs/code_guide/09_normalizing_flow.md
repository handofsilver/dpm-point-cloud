# 09 — NormalizingFlow

> 对应 `model.py: NormalizingFlow`，论文 Section 4.4

---

## 1. 职责

`NormalizingFlow` 是 `AffineCouplingLayer` 的调度者，职责只有三件事：

1. 持有 $K$ 个 `AffineCouplingLayer`，相邻层交替 `flip`
2. `inverse(z)` — 训练路径：逆序走完所有层，累加 `log_det`，返回 $(u, \log|\det J_\text{total}|)$
3. `forward(u)` — 生成路径：正序走完所有层，返回 $z$

loss 的计算、采样 $u \sim \mathcal{N}(0,I)$ 都不在这里，由上层 `FlowVAE` 负责。

---

## 2. 为什么逆序 ≠ 正序？

设正向变换为 $z = f_K \circ \cdots \circ f_1(u)$，则：

$$f_\text{total}^{-1} = f_1^{-1} \circ f_2^{-1} \circ \cdots \circ f_K^{-1}$$

即从第 $K$ 层向第 1 层依次拆，每层调用 `reverse=True`。

如果用正序 + `reverse=True`，得到的是 $f_K^{-1} \circ \cdots \circ f_1^{-1}$，这不是正向的逆，中间变量对不上，结果是错的。

| 方法 | 遍历顺序 | 每层 `reverse` | 物理含义 |
|---|---|---|---|
| `forward(u)` | $f_1 \to f_K$ | `False` | $u \to z$（生成） |
| `inverse(z)` | $f_K^{-1} \to f_1^{-1}$ | `True` | $z \to u$（密度估计） |

---

## 3. log_det 的累加

多层链式变换的 log 行列式等于各层之和：

$$\log|\det J_\text{total}| = \sum_{k=1}^{K} \log|\det J_k|$$

`inverse` 里每层返回的 `ld` 就是 $-\sum s_i$（因为 `reverse=True`），累加即得总的 $\log|\det J_\text{total}^{-1}|$，直接用于换元公式：

$$\log p_\theta(z) = \log p_u(u) - \log|\det J_\text{total}|$$

注意：`log_det` 初始化时需要携带 `device=z.device`，否则 GPU 训练时会因 device mismatch 报错。

---

## 4. 常见 bug：traversal 方向写反

`inverse` 和 `forward` 的遍历方向很容易写反（这是本次实现中实际出现的错误）：

```python
# 错误：inverse 用正序，forward 用逆序
def inverse(self, z):
    for layer in self.layers: ...          # ❌ 应该是 reversed(self.layers)

def forward(self, u):
    for layer in reversed(self.layers): ...  # ❌ 应该是 self.layers
```

记忆方法：**"正向正序，逆向逆序"**——`forward` 用 `self.layers`，`inverse` 用 `reversed(self.layers)`。

---

## 5. 数据流

```
训练（inverse）:
  z: (B, zdim)  ← 来自 encoder 重参数化
    → f_K^{-1}: (y, ld_K)
    → f_{K-1}^{-1}: (y, ld_{K-1})
    → ...
    → f_1^{-1}: (u, ld_1)
    → log_det = ld_K + ... + ld_1: (B,)
  return u: (B, zdim), log_det: (B,)

生成（forward）:
  u: (B, zdim)  ← 从 N(0,I) 采样
    → f_1 → f_2 → ... → f_K
  return z: (B, zdim)
```
