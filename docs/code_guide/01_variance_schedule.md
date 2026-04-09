# 01 · VarianceSchedule — 噪声节奏表

> 对应代码: `model.py` → `class VarianceSchedule`
> 对应论文: Section 3, Eq.(2)(11)(13)

---

## 这个模块做什么？

在扩散模型里，"加噪"和"去噪"都需要知道**每个时间步 t 对应的噪声强度**。
`VarianceSchedule` 的唯一职责就是：**把这张噪声强度表提前算好，存起来备用**。

它本身不参与任何前向传播，也没有可学习的参数。它是整个模型的"配置表"。

---

## 核心数学

### β 线性调度

论文使用最简单的线性调度：

$$
\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1), \quad t = 1, \ldots, T
$$

其中 $\beta_1 = 10^{-4}$（极小），$\beta_T$ 是超参数（AutoEncoder: 0.05，Generation: 0.02）。

**直觉**：$\beta_t$ 越大，这一步加的噪声越多。线性增长意味着扩散早期（t 小）变化温和，后期（t 大）噪声越来越猛。

### α 与 ᾱ

$$
\alpha_t = 1 - \beta_t \qquad \bar\alpha_t = \prod_{s=1}^{t} \alpha_s
$$

$\alpha_t$ 是这一步"保留原始信息"的比例，$\bar\alpha_t$ 是从 $t=1$ 到 $t$ 的**累积保留比例**。

```python
alphas     = 1.0 - betas                        # α_t
alpha_bars = torch.cumprod(alphas, dim=0)        # ᾱ_t = α_1 · α_2 · … · α_t
```

`torch.cumprod` 做的就是：`[a, b, c, d, ...]` → `[a, ab, abc, abcd, ...]`

### 跳步采样公式（Eq.13）

有了 $\bar\alpha_t$，就可以从 $x^{(0)}$ **一步跳**到任意时刻 $x^{(t)}$，不需要逐步迭代：

$$
\boxed{x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)}
$$

验证两端：

| 时刻 | $\bar\alpha_t$ 约为 | $x^{(t)}$ 约等于 |
|:---:|:---:|:---:|
| $t = 1$ | $\approx 1$ | $\approx x^{(0)}$（原样） |
| $t = T$ | $\approx 0$ | $\approx \varepsilon$（纯噪声） |

这正是训练时加噪的核心操作（在 `DiffusionPoint` 里会用到）。

---

## 两种方差

逆向采样时，每步需要加回一点随机性，其强度由 $\sigma_t$ 控制。两种选择：

### σ_inflex（窄，后验方差，Eq.11）

$$
\sigma_t^{\text{inflex}} = \sqrt{\gamma_t}, \qquad \gamma_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \cdot \beta_t
$$

来自真实后验 $q(x^{(t-1)} \mid x^{(t)}, x^{(0)})$ 的方差。理论上最紧，$t \to 0$ 时趋近于 0。

计算时需要 $\bar\alpha_{t-1}$（前一步的累积乘积），代码里用拼接技巧处理 $t=1$ 的边界：

```python
alpha_bars_prev = torch.cat([
    torch.tensor([1.0]),   # ᾱ_0 定义为 1（无噪声的原始状态）
    alpha_bars[:-1]        # ᾱ_1, ᾱ_2, ..., ᾱ_{T-1}
], dim=0)

gammas = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas
sigmas_inflex = torch.sqrt(gammas.clamp(min=0))   # clamp 防止浮点负数
```

### σ_flex（宽，前向过程方差）

$$
\sigma_t^{\text{flex}} = \sqrt{\beta_t}
$$

直接用前向过程的每步方差，是 $\gamma_t$ 的上界。生成更多样化。

```python
sigmas_flex = torch.sqrt(betas)
```

### flexibility 插值旋钮

$$
\sigma_t = f \cdot \sigma_t^{\text{flex}} + (1 - f) \cdot \sigma_t^{\text{inflex}}, \quad f \in [0, 1]
$$

```python
def get_sigmas(self, t, flexibility):
    idx = t - 1    # 1-indexed → 0-indexed
    return flexibility * self.sigmas_flex[idx] + (1 - flexibility) * self.sigmas_inflex[idx]
```

---

## register_buffer 的必要性

VarianceSchedule 里所有张量都是**固定常数**，不应被优化器更新。但训练时数据在 GPU 上，这些常数也必须在 GPU 上，否则运算会因设备不匹配而报错。

```python
# 错误做法：self.betas = betas
#   → 不会随 model.to('cuda') 移动，训练时报错

# 正确做法：
self.register_buffer('betas', betas)
#   → 不被学习，但随模型走，也会被 state_dict 保存
```

详见 `docs/pytorch_notes.md §1`。

---

## 验证结果

```
alpha_bar[t=1]  : 0.999900   ✓ 接近 1
alpha_bar[t=T]  : 0.006122   ✓ 接近 0（不是精确 0，T 有限步）
sqrt(alpha_bar_T)    : 0.078  ✓ 接近 0
sqrt(1-alpha_bar_T)  : 0.997  ✓ 接近 1
flex >= inflex everywhere: True  ✓ 宽≥窄恒成立
t_samples ∈ [1, 200]         ✓ 均匀采样范围正确
```

---

## 与后续模块的关系

```
VarianceSchedule
    ├── alpha_bars   → DiffusionPoint.get_loss()   用于 Eq.(13) 前向加噪
    ├── alphas       → DiffusionPoint.sample()     用于逆向均值计算
    ├── betas        → PointwiseNet                用于时间嵌入 [β, sin(β), cos(β)]
    └── get_sigmas() → DiffusionPoint.sample()     用于逆向加噪
```
