# 08 — AffineCouplingLayer

> 对应 `model.py: AffineCouplingLayer`，论文 Section 4.4 / Appendix

---

## 1. 职责

`AffineCouplingLayer` 是 `NormalizingFlow` 的基本单元，只负责**单步可逆仿射变换**——不管损失、不管堆叠，只做"进来一个向量，出去一个向量 + 一个标量"。

类比：`ConcatSquashLinear` 之于 `PointwiseNet`，`AffineCouplingLayer` 之于 `NormalizingFlow`。

**接口**：

```
输入:  x       (B, zdim)
输出:  y       (B, zdim)   变换后的向量
       log_det (B,)        log|det J|，每个样本一个标量
```

---

## 2. 为什么需要它？

Flow 用可逆变换把简单分布 $u \sim \mathcal{N}(0,I)$ 搬运成复杂分布 $p_\theta(z)$。要计算 $\log p_\theta(z)$，换元公式要求：

$$\log p_\theta(z) = \log p_u(f^{-1}(z)) - \log|\det J_f|$$

一般神经网络的 Jacobian 是稠密方阵，计算行列式是 $O(d^3)$（$d=256$ 时约 $1.7\times10^7$ 次运算，每步都要算）。`AffineCouplingLayer` 通过精心设计的结构，把 $\log|\det J|$ 的计算代价降到 $O(d)$。

---

## 3. 核心设计：让 Jacobian 天然三角化

把 $d$ 维输入劈成两半 $[x_1, x_2]$，变换只改变一半：

$$z_1 = x_1, \quad z_2 = x_2 \odot \exp\!\bigl(s(x_1)\bigr) + t(x_1)$$

Jacobian 分块写出：

$$J = \frac{\partial z}{\partial x} = \begin{pmatrix} I & 0 \\ \frac{\partial z_2}{\partial x_1} & \operatorname{diag}(\exp(s(x_1))) \end{pmatrix}$$

右上角是零（$z_1$ 不依赖 $x_2$），下三角分块矩阵的行列式 = 对角块乘积：

$$\det J = 1 \cdot \prod_i \exp(s_i) = \exp\!\Bigl(\sum_i s_i\Bigr)$$

$$\log|\det J| = \sum_i s_i(x_1)$$

就是 $s$ 网络输出各分量直接求和，$O(d)$，不需要显式构造 Jacobian 矩阵。

---

## 4. 可逆性：不需要"反转"神经网络

逆向公式直接解析得出：

$$x_1 = z_1, \quad x_2 = \bigl(z_2 - t(z_1)\bigr) \odot \exp\!\bigl(-s(z_1)\bigr)$$

关键：$s$ 和 $t$ 的输入在正逆向中都是"不动的那半"（正向为 $x_1$，逆向为 $z_1 = x_1$），**两者相同**。所以 $s_\text{net}$ 和 $t_\text{net}$ 始终做前向传播，可逆性来自仿射运算本身。

---

## 5. 两个参数的区分

| 参数 | 类型 | 含义 | 影响什么 |
|---|---|---|---|
| `flip` | 类属性 | 哪半不动（交替层用） | split 顺序 |
| `reverse` | 方法参数 | 正向还是逆向 | 变换公式 + `log_det` 符号 |

两者**相互独立**，常见 bug 是把 `log_det` 的符号绑到 `flip` 上——错误，符号只取决于 `reverse`。

---

## 6. `flip` 为何需要交替？

单层 coupling 的 $z_1 = x_1$ 意味着前半维完全未被变换，表达能力受限。通过在相邻层间交替 `flip`：

```
第 1 层 (flip=False): 前半不动，后半被前半调制
第 2 层 (flip=True) : 后半不动，前半被后半调制
第 3 层 (flip=False): 前半不动，后半被前半调制
第 4 层 (flip=True) : ...
```

经过 4 层后，每个维度都在某些层被变换、在另一些层充当调制信号，整体具有完整表达能力。

---

## 7. s_net 为何用 Tanh 限幅？

$s$ 网络的输出直接进入 $\exp(s)$。训练初期网络权重随机，$s$ 可能输出很大的值，$\exp(s)$ 爆炸会导致梯度消失/爆炸。`Tanh` 把 $s$ 限制在 $(-1, 1)$，对应 $\exp(s) \in (e^{-1}, e^1) \approx (0.37, 2.72)$，初始缩放幅度可控。

`t_net` 是平移偏置，不参与指数运算，无需限幅。

---

## 8. 数据流总览

```
正向（生成，reverse=False）：
  x: (B, zdim)
    → chunk → x_a, x_b: 各 (B, d_half)
    → flip 决定 x1（不动）, x2（被变换）
    → s = s_net(x1): (B, d_half)
    → t = t_net(x1): (B, d_half)
    → y2 = x2 ⊙ exp(s) + t: (B, d_half)
    → 还原顺序 cat → y: (B, zdim)
    → log_det = s.sum(dim=-1): (B,)

逆向（推断，reverse=True）：
  同上，Step 3 改为 y2 = (x2 - t) ⊙ exp(-s)
  log_det = -s.sum(dim=-1)
```
