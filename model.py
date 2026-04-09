"""
DPM-3D 复现
Paper: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021
"""

import torch
import torch.nn as nn


# =============================================================================
# Phase 1-A: VarianceSchedule
# 作用: 管理扩散过程的"噪声节奏"
# 对应论文: Section 3, Eq.(2)(13)
# =============================================================================


class VarianceSchedule(nn.Module):
    """
    线性 beta schedule: 预计算扩散过程中所有时间步的噪声系数。

    扩散过程共 T 步，beta 从 beta_1=1e-4 线性增长到 beta_T。
    beta 越大 = 这一步加的噪声越多。

    预计算以下数组（长度均为 T），下标 i 对应时间步 t = i+1：
        betas[i]        = beta_{i+1}
        alphas[i]       = alpha_{i+1} = 1 - beta_{i+1}
        alpha_bars[i]   = alpha_bar_{i+1} = prod(alpha_1 ... alpha_{i+1})
        sigmas_flex[i]  = sqrt(beta_{i+1})           — "宽"方差
        sigmas_inflex[i]= sqrt(gamma_{i+1})           — "窄"方差（后验）
    """

    # 类型声明：告诉 Pylance 这些 register_buffer 属性确实是 Tensor
    # 不影响运行时行为，仅用于 IDE 类型检查
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sigmas_flex: torch.Tensor
    sigmas_inflex: torch.Tensor

    def __init__(self, T: int, beta_T: float):
        """
        Args:
            T       : 扩散总步数（AutoEncoder 模式用 200，Generation 模式用 100）
            beta_T  : 最大噪声强度（AutoEncoder: 0.05，Generation: 0.02）
        """
        super().__init__()

        self.T = T

        # --- Step 1: 线性 beta schedule ---
        # beta_1 = 1e-4（极小），beta_T 是超参数
        # linspace(start, end, steps) 生成等差数列，shape: (T,)
        betas = torch.linspace(1e-4, beta_T, T)  # (T,)

        # --- Step 2: alpha = 1 - beta ---
        # alpha_t 表示这一步"保留原始信息"的比例
        alphas = 1.0 - betas  # (T,)

        # --- Step 3: alpha_bar = 累积乘积 ---
        # alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
        # cumprod: cumulative product，第 i 个元素 = 前 i+1 个元素之积
        # alpha_bar_t 趋近 0 时，x_t 几乎全是噪声（Eq.13 验证）
        alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)

        # --- Step 4: 两种方差 ---
        # sigmas_flex[t] = sqrt(beta_t)  — 直接用前向过程方差
        sigmas_flex = torch.sqrt(betas)  # (T,)

        # sigmas_inflex[t] = sqrt(gamma_t)，gamma_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        # 注意: t=1 时 alpha_bar_0 定义为 1（无任何噪声的原始状态）
        # 所以我们在 alpha_bars 前面拼一个 1.0，方便取 t-1 的值
        alpha_bars_prev = torch.cat(
            [
                torch.tensor([1.0]),  # alpha_bar_0 = 1
                alpha_bars[:-1],  # alpha_bar_1 ... alpha_bar_{T-1}
            ],
            dim=0,
        )  # (T,)

        # gamma_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t  — Eq.(11)
        gammas = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas  # (T,)

        # t=1 时 gamma_1 = 0（alpha_bar_0=1，分子=0），sqrt(0)=0，没问题
        sigmas_inflex = torch.sqrt(gammas.clamp(min=0))  # (T,)

        # --- Step 5: 用 register_buffer 注册所有常数 ---
        # 这些是预计算的固定常数，不需要梯度，但需要跟着模型移动到 GPU
        # （详见 docs/pytorch_notes.md §1）
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size: int) -> torch.Tensor:
        """
        训练时随机采样时间步（Algorithm 1, Step 3）。

        从 {1, 2, ..., T} 中均匀随机采 batch_size 个整数。
        注意返回值是 1-indexed（t=1 到 T），与论文符号一致。

        Returns:
            t: (batch_size,)，dtype=torch.long
        """
        # randint(low, high) 生成 [low, high) 的随机整数
        # 我们想要 [1, T]，所以 high = T+1
        t = torch.randint(1, self.T + 1, size=(batch_size,))  # (B,)
        return t

    def get_sigmas(self, t: torch.Tensor, flexibility: float) -> torch.Tensor:
        """
        根据 flexibility 在两种方差之间插值，供逆向采样使用。

        Args:
            t          : (B,) 时间步，1-indexed
            flexibility: float ∈ [0, 1]，0=窄方差，1=宽方差

        Returns:
            sigmas: (B,) 每个样本对应的 sigma 值
        """
        assert 0 <= flexibility <= 1, "flexibility 必须在 [0, 1] 之间"

        # t 是 1-indexed，转换为 0-indexed 数组下标
        idx = t - 1  # (B,)

        sigma = (
            flexibility * self.sigmas_flex[idx]
            + (1 - flexibility) * self.sigmas_inflex[idx]
        )  # (B,)
        return sigma


# =============================================================================
# Phase 1-B: ConcatSquashLinear
# 作用: 带条件门控的线性层，将时间步和 shape latent 注入点特征
# 对应论文: Section 4.2（PointwiseNet 描述中的条件注入机制）
# =============================================================================


class ConcatSquashLinear(nn.Module):
    """
    条件门控线性层。

    普通线性层只做 y = Wx + b，无法感知外部条件。
    这一层将条件向量 ctx 以"门控 + 偏置"的方式注入：

        gate   = sigmoid( Linear_gate(ctx) )       # ∈ (0,1)，shape: (B, 1, out_dim)
        bias   = Linear_bias(ctx)                  # shape: (B, 1, out_dim)
        output = Linear(x) * gate + bias           # 逐元素乘法（广播）

    其中:
        x   : (B, N, in_dim)   — 点特征，N 个点各自独立处理
        ctx : (B, 1, ctx_dim)  — 条件向量，每个 batch 共享同一个条件
    """

    def __init__(self, in_dim: int, out_dim: int, ctx_dim: int):
        """
        Args:
            in_dim  : 输入特征维度
            out_dim : 输出特征维度
            ctx_dim : 条件向量维度（时间嵌入 + shape latent 拼接后的维度）
        """
        super().__init__()

        # 主变换：对点特征做线性变换（没有 bias，bias 由条件提供）
        self._layer = nn.Linear(in_dim, out_dim)

        # 门控路径：从条件向量生成 sigmoid 门，决定保留多少主变换的输出
        self._hyper_gate = nn.Linear(ctx_dim, out_dim)

        # 偏置路径：从条件向量直接生成加性偏置
        self._hyper_bias = nn.Linear(ctx_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, N, in_dim)   点特征
            ctx : (B, 1, ctx_dim)  条件向量

        Returns:
            out : (B, N, out_dim)
        """
        # 主变换: (B, N, in_dim) → (B, N, out_dim)
        gate = torch.sigmoid(self._hyper_gate(ctx))  # (B, 1, out_dim)
        bias = self._hyper_bias(ctx)  # (B, 1, out_dim)

        # gate 和 bias 的 N 维是 1，会自动广播到 x 的 N 维
        # 即：N 个点共享同一个 gate 和 bias（来自同一个 ctx）
        out = self._layer(x) * gate + bias  # (B, N, out_dim)
        return out


# =============================================================================
# Phase 1-C: PointwiseNet
# 作用: 以 (x^(t), β_t, z) 为输入，预测噪声 ε（每个点独立处理）
# 对应论文: Section 4.2, Algorithm 1
# =============================================================================


class PointwiseNet(nn.Module):
    """
    逐点噪声预测网络。

    对点云中每个点独立地预测噪声，点之间的信息仅通过共享的 shape latent z 间接交互。

    网络结构（6 层 ConcatSquashLinear）:
        3 → 128 → 256 → 512 → 256 → 128 → 3
        前 5 层后接 LeakyReLU，最后一层无激活

    时间嵌入（极简 3 维）:
        beta_t (标量) → [beta_t, sin(beta_t), cos(beta_t)] → (B, 1, 3)

    条件向量（ctx）= 时间嵌入 + shape latent z:
        ctx = cat([time_emb, z], dim=-1) → (B, 1, 3 + zdim)

    残差连接:
        最终输出 = net_output + x^(t)（当 residual=True）
        网络只需预测"偏差量"，而不是完整的去噪坐标
    """

    def __init__(self, zdim: int, residual: bool = True):
        """
        Args:
            zdim    : shape latent z 的维度（默认 256）
            residual: 是否使用残差连接（默认 True）
        """
        super().__init__()

        self.residual = residual

        # ctx 的维度 = 时间嵌入(3) + shape latent(zdim)
        ctx_dim = 3 + zdim

        # 6 层 ConcatSquashLinear，维度: 3→128→256→512→256→128→3
        self.layers = nn.ModuleList(
            [
                ConcatSquashLinear(3, 128, ctx_dim),
                ConcatSquashLinear(128, 256, ctx_dim),
                ConcatSquashLinear(256, 512, ctx_dim),
                ConcatSquashLinear(512, 256, ctx_dim),
                ConcatSquashLinear(256, 128, ctx_dim),
                ConcatSquashLinear(128, 3, ctx_dim),
            ]
        )

        # 前 5 层后接 LeakyReLU（负半轴保留小梯度，防止神经元死亡）
        # 最后一层（输出层）不加激活，输出可正可负的坐标偏移
        self.act = nn.LeakyReLU()

    def forward(
        self,
        x: torch.Tensor,  # (B, N, 3)    加噪后的点云 x^(t)
        beta: torch.Tensor,  # (B,)         当前时间步对应的 beta_t
        z: torch.Tensor,  # (B, zdim)    shape latent
    ) -> torch.Tensor:
        """
        Returns:
            eps_pred: (B, N, 3)  每个点的预测噪声 ε_θ
        """
        # --- Step 1: 构造时间嵌入 ---
        # beta: (B,) → unsqueeze 到 (B, 1)，再拼成 3 维嵌入 → (B, 1, 3)
        # 用 [β, sin(β), cos(β)] 编码噪声强度（见 paper_deep_dive.md §2）
        beta_col = beta.unsqueeze(1)  # (B, 1)
        time_emb = torch.cat(
            [
                beta_col,
                torch.sin(beta_col),
                torch.cos(beta_col),
            ],
            dim=-1,
        ).unsqueeze(
            1
        )  # (B, 1, 3)

        # --- Step 2: 构造条件向量 ctx ---
        # z: (B, zdim) → unsqueeze 到 (B, 1, zdim)
        # ctx = cat([time_emb, z], dim=-1) → (B, 1, 3+zdim)
        # 所有 N 个点共享同一个 ctx（通过 ConcatSquashLinear 广播）
        z_expanded = z.unsqueeze(1)  # (B, 1, zdim)
        ctx = torch.cat([time_emb, z_expanded], dim=-1)  # (B, 1, 3+zdim)

        # --- Step 3: 6 层前向传播 ---
        # 前 5 层: ConcatSquashLinear + LeakyReLU
        # 最后 1 层: ConcatSquashLinear（无激活）
        h = x  # (B, N, 3)
        for i, layer in enumerate(self.layers):
            h = layer(h, ctx)  # (B, N, dim_i)
            if i < len(self.layers) - 1:
                h = self.act(h)  # 前 5 层加激活

        # --- Step 4: 残差连接 ---
        # 网络输出是"偏差量"，加上原始 x^(t) 得到最终预测
        # （见 pytorch_notes.md §2：残差连接）
        if self.residual:
            return x + h  # (B, N, 3)
        return h  # (B, N, 3)
