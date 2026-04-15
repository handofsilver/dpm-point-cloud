# 06 · AutoEncoder — 第一个端到端可训练模型

> 对应代码: `model.py` → `class AutoEncoder`
> 对应论文: Section 4.3（AutoEncoder 模式）

---

## 这个模块做什么？

将 `PointNetEncoder` 和 `DiffusionPoint` 封装成一个统一接口，对外只暴露两个方法：

```
get_loss(x0)        → 训练损失（标量）
sample(z, ...)      → 生成点云 (B, N, 3)
```

训练脚本不需要知道内部是"先编码再扩散"，也不需要知道 AutoEncoder 模式下 `z = mu` 这个细节。

---

## AutoEncoder 模式的核心：z = mu

Encoder 始终输出双头 $(\mu, \log\sigma^2)$，但 AutoEncoder 模式**直接取 $\mu$ 作为 $z$**，丢弃 $\log\sigma^2$：

$$
z = \mu = \text{Encoder}(X)
$$

没有重参数化，没有 KL 项，损失完全来自扩散重建：

$$
\mathcal{L} = \mathbb{E}_{t,\varepsilon}\left[\|\varepsilon_\theta(x^{(t)}, \beta_t, z) - \varepsilon\|^2\right]
$$

这使得 AutoEncoder 比 VAE 更简单——适合先验验证整条数据流是否跑通。

---

## 训练超参（AutoEncoder 模式）

| 参数 | 值 |
|:-:|:-:|
| T | 200 |
| β_T | 0.05 |
| lr | 1e-3 |
| optimizer | Adam |
| grad clip | max_norm=10 |

---

## 与后续模块的关系

```
AutoEncoder
    ├── get_loss(x0)
    │       ├── encoder(x0)  → mu, _
    │       ├── z = mu
    │       └── diffusion.get_loss(x0, z)
    └── sample(z, num_points, flexibility)
            └── diffusion.sample(z, ...)

GaussianVAE（下一步）
    └── 在 AutoEncoder 基础上：重参数化采样 z + KL 损失项
```
