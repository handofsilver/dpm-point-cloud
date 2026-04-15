"""
DPM-3D 复现
Paper: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            flexibility * self.sigmas_flex[idx] + (1 - flexibility) * self.sigmas_inflex[idx]
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
                ConcatSquashLinear(in_dim=3, out_dim=128, ctx_dim=ctx_dim),
                ConcatSquashLinear(in_dim=128, out_dim=256, ctx_dim=ctx_dim),
                ConcatSquashLinear(in_dim=256, out_dim=512, ctx_dim=ctx_dim),
                ConcatSquashLinear(in_dim=512, out_dim=256, ctx_dim=ctx_dim),
                ConcatSquashLinear(in_dim=256, out_dim=128, ctx_dim=ctx_dim),
                ConcatSquashLinear(in_dim=128, out_dim=3, ctx_dim=ctx_dim),
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
        ).unsqueeze(1)
        # (B, 1, 3)

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


# =============================================================================
# Phase 2: DiffusionPoint
# 作用: 扩散过程的调度器——训练时加噪+算损失，推理时逐步去噪
# 对应论文: Section 4.1, Algorithm 1 (训练), Algorithm 2 (采样)
# =============================================================================


class DiffusionPoint(nn.Module):
    """
    扩散过程的核心调度模块。

    持有 VarianceSchedule（噪声系数表）和 PointwiseNet（噪声预测网络），
    对外只暴露两个方法：
        - get_loss(x0, z) → 训练损失（标量）
        - sample(z)       → 生成点云 (B, N, 3)
    """

    def __init__(
        self,
        net: PointwiseNet,  # 噪声预测网络（已构建好的实例）
        var_sched: VarianceSchedule,  # 噪声系数表（已构建好的实例）
    ):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    # -------------------------------------------------------------------------
    # 训练接口
    # -------------------------------------------------------------------------

    def get_loss(
        self,
        x0: torch.Tensor,  # (B, N, 3)  干净点云
        z: torch.Tensor,  # (B, F)     shape latent（来自 PointNetEncoder）
    ) -> torch.Tensor:
        """
        前向加噪 + 预测噪声 + 返回 MSE 损失。
        对应 Algorithm 1（训练循环的单次迭代）。

        Returns:
            loss: 标量 Tensor，MSE(ε_θ, ε)
        """
        B, N, _ = x0.shape

        # --- Step 1: 随机采样时间步 t ---
        # 每个样本独立采一个 t ∈ {1, ..., T}
        # t: (B,)，1-indexed
        t = self.var_sched.uniform_sample_t(B)

        # --- Step 2: 取出当前时间步的系数 ---
        # t 是 1-indexed，数组是 0-indexed，所以下标 = t - 1
        alpha_bar = self.var_sched.alpha_bars[t - 1]  # (B,)
        beta = self.var_sched.betas[t - 1]  # (B,)

        # --- Step 3: 采样噪声 ε ~ N(0, I) ---
        eps = torch.randn_like(x0)  # (B, N, 3)

        # --- Step 4: 一步前向加噪 — Eq.(13) ---
        # x^(t) = sqrt(α_bar_t) · x^(0) + sqrt(1 - α_bar_t) · ε
        # alpha_bar: (B,) → view(B,1,1) 与 x0: (B,N,3) 广播
        x_noisy = (
            torch.sqrt(alpha_bar).view(B, 1, 1) * x0 + torch.sqrt(1 - alpha_bar).view(B, 1, 1) * eps
        )  # (B, N, 3)

        # --- Step 5: 预测噪声 ---
        # PointwiseNet 期望 beta: (B,)，这里 beta 已经是 (B,)，直接传入
        eps_pred = self.net(x_noisy, beta, z)  # (B, N, 3)

        # --- Step 6: MSE 损失 ---
        # Python 注意: ** 是幂运算，^ 是按位异或（对 float Tensor 会报错）
        loss = F.mse_loss(eps_pred, eps)
        return loss

    # -------------------------------------------------------------------------
    # 推理接口
    # -------------------------------------------------------------------------

    def sample(
        self,
        z: torch.Tensor,  # (B, F)  shape latent（来自先验或 encoder）
        num_points: int,  # N，生成点云的点数（通常 2048）
        flexibility: float,  # σ 插值系数，0=窄方差，1=宽方差
    ) -> torch.Tensor:
        """
        逆向 Markov chain，从纯噪声逐步去噪，生成点云。
        对应 Algorithm 2。

        Returns:
            x: (B, N, 3) 生成的点云
        """
        B = z.shape[0]

        # --- Step 1: 初始化 x^(T) ~ N(0, I) ---
        x = torch.randn(B, num_points, 3, device=z.device)  # (B, N, 3)

        # --- Step 2: 逆向循环 t = T, T-1, ..., 1 ---
        for t_int in range(self.var_sched.T, 0, -1):

            # 组装 batch tensor：所有样本在同一时间步
            t = torch.full((B,), t_int, dtype=torch.long, device=z.device)  # (B,)

            # 取出当前时间步的系数（标量 Tensor）
            alpha = self.var_sched.alphas[t_int - 1]
            alpha_bar = self.var_sched.alpha_bars[t_int - 1]
            sigma = self.var_sched.get_sigmas(t, flexibility)  # (B,)

            # 预测噪声
            # beta 是标量 Tensor，但 PointwiseNet 期望 (B,)，需要 expand
            beta = self.var_sched.betas[t_int - 1].expand(B)  # (B,)
            eps_pred = self.net(x, beta, z)  # (B, N, 3)

            # 逆向均值（去噪一步）
            # x_{t-1} = (1/√α_t) · (x_t − (1−α_t)/√(1−ᾱ_t) · ε_θ)
            x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * eps_pred)

            # 加回随机性（t=1 时不加，此时已经是最终输出）
            # sigma: (B,) → view(B,1,1) 与 z_noise: (B,N,3) 广播
            if t_int > 1:
                z_noise = torch.randn_like(x)
                x = x + sigma.view(B, 1, 1) * z_noise  # (B, N, 3)

        return x  # (B, N, 3)


# =============================================================================
# Phase 3-A: PointNetEncoder
# 作用: 把一朵点云压缩成 shape latent z 的分布参数 (mu, log_var)
# 对应论文: Section 4.3, Eq.(6)(7)
# =============================================================================


class PointNetEncoder(nn.Module):
    """
    点云编码器：PointNet 风格，输出 shape latent 的分布参数。

    核心思路：
        1. 用 Conv1d(kernel=1) 对每个点独立做特征提取（等价于共享权重的 MLP）
        2. MaxPool over N 个点 → 得到置换不变的全局特征向量
        3. 双头 FC 输出 mu 和 log_var（而不是 sigma，数值更稳定）

    为什么用 Conv1d 而不是 Linear？
        Conv1d(in, out, kernel_size=1) 作用在 (B, C, N) 上，
        等价于对 N 个位置各自独立做一次线性变换，且所有位置共享权重。
        这正是 PointNet "逐点 MLP" 的含义，天然满足置换等变性。

    数据流：
        (B, N, 3)
            → permute → (B, 3, N)          # Conv1d 要求 channel 在第二维
            → Conv1d ×4 + ReLU             # 逐点升维
            → (B, 512, N)
            → MaxPool over N               # 置换不变聚合
            → (B, 512)                     # 全局特征向量
            → FC → mu      : (B, zdim)
            → FC → log_var : (B, zdim)
    """

    def __init__(self, zdim: int):
        """
        Args:
            zdim: shape latent z 的维度（与 PointwiseNet 的 zdim 保持一致）
        """
        super().__init__()

        # --- 逐点特征提取：4 层 Conv1d，kernel_size=1 ---
        # kernel_size=1：每个点只看自己，不看邻居（置换等变）
        # 不需要 padding（padding 会改变序列长度 N，kernel=1 时无意义）
        # 维度变化：3 → 128 → 128 → 256 → 512
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1)  # (B,   3, N) → (B, 128, N)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1)  # (B, 128, N) → (B, 128, N)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)  # (B, 128, N) → (B, 256, N)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)  # (B, 256, N) → (B, 512, N)

        # conv 层统一使用 ReLU（与 PointwiseNet 的 LeakyReLU 区分：
        # Encoder 是分类/压缩任务，ReLU 足够；PointwiseNet 需要输出可负的坐标偏移）
        self.act = nn.ReLU()

        # 双头 FC：MaxPool 后全局特征 (B, 512) → 分别输出 mu 和 log_var
        # 两个头权重独立，不能共享（各自学习不同的投影方向）
        self.fc_mu = nn.Linear(512, zdim)  # (B, 512) → (B, zdim)
        self.fc_log_var = nn.Linear(512, zdim)  # (B, 512) → (B, zdim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, 3)  输入点云

        Returns:
            mu     : (B, zdim)  latent 均值
            log_var: (B, zdim)  latent 对数方差（log σ²）
        """
        # --- Step 1: 调整维度顺序以适配 Conv1d ---
        # Conv1d 期望 (B, C, L)，坐标维 3 需要换到第二位
        # permute(0, 2, 1)：batch 维不动，N 和 3 互换
        x = x.permute(0, 2, 1)  # (B, N, 3) → (B, 3, N)

        # --- Step 2: 4 层逐点特征提取 ---
        # 每层：Conv1d → ReLU，channel 逐步扩大以提取更抽象的特征
        x = self.act(self.conv1(x))  # (B,   3, N) → (B, 128, N)
        x = self.act(self.conv2(x))  # (B, 128, N) → (B, 128, N)
        x = self.act(self.conv3(x))  # (B, 128, N) → (B, 256, N)
        x = self.act(self.conv4(x))  # (B, 256, N) → (B, 512, N)

        # --- Step 3: MaxPool over N，得到全局特征 ---
        # torch.max(tensor, dim) 返回 namedtuple(values, indices)，取 .values
        # 置换不变性保证：无论点的排列顺序，MaxPool 结果不变
        feat = torch.max(x, dim=2).values  # (B, 512, N) → (B, 512)

        # --- Step 4: 双头 FC，输出分布参数 ---
        # log_var 不加激活：允许取任意实数，exp(log_var) 自然为正
        mu = self.fc_mu(feat)  # (B, 512) → (B, zdim)
        log_var = self.fc_log_var(feat)  # (B, 512) → (B, zdim)

        return mu, log_var


# =============================================================================
# Phase 4-A: AutoEncoder
# 作用: 第一个端到端可训练模型——Encoder + DiffusionPoint 的封装
# 对应论文: Section 4.3（AutoEncoder 模式）
# =============================================================================


class AutoEncoder(nn.Module):
    """
    点云 AutoEncoder。

    持有 PointNetEncoder 和 DiffusionPoint，对外只暴露：
        - get_loss(x0) → 训练损失（标量）
        - sample(z)    → 生成点云 (B, N, 3)

    AutoEncoder 模式的关键：z = mu（直接取 Encoder 输出的均值，不采样）。
    没有 KL 项，损失完全来自扩散重建。
    """

    def __init__(
        self,
        encoder: PointNetEncoder,
        diffusion: DiffusionPoint,
    ):
        """
        Args:
            encoder  : 已构建好的 PointNetEncoder 实例
            diffusion: 已构建好的 DiffusionPoint 实例
        """
        super().__init__()

        # encoder 和 diffusion 作为子模块注册，参数会被 optimizer 和 .to(device) 统一管理
        self.encoder = encoder
        self.diffusion = diffusion

    def get_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        编码点云 → 取 mu 作为 z → 计算扩散损失。

        Args:
            x0: (B, N, 3)  输入点云（干净，未加噪）

        Returns:
            loss: 标量 Tensor，MSE(ε_θ, ε)
        """
        # encoder 始终输出双头；AutoEncoder 模式只用 mu，log_var 被丢弃
        mu, _ = self.encoder(x0)

        # AutoEncoder 模式：z = mu，不采样，不引入随机性
        z = mu

        return self.diffusion.get_loss(x0, z)

    def sample(
        self,
        z: torch.Tensor,  # (B, F)  shape latent（来自外部，或训练集编码结果）
        num_points: int,  # 生成点数，通常 2048
        flexibility: float,  # 方差插值系数，0=窄，1=宽
    ) -> torch.Tensor:
        """
        给定 z，逆向扩散生成点云。

        sample 不调用 Encoder——z 由调用方提供。
        （重建时 z 来自 encoder(x0)；纯生成时 z 来自先验分布）

        Returns:
            x: (B, N, 3) 生成的点云
        """
        return self.diffusion.sample(z, num_points, flexibility)


# =============================================================================
# Phase 5-A: GaussianVAE
# 作用: 在 AutoEncoder 基础上引入 KL 正则，使 z 的先验逼近 N(0, I)
# 对应论文: Section 4.3（GaussianVAE 模式）
# =============================================================================


class GaussianVAE(nn.Module):
    """
    点云生成 VAE，先验 p(z) = N(0, I)。

    与 AutoEncoder 的两处关键区别：

    1. **重参数化采样**（训练时）
       AutoEncoder:   z = mu                           # 确定性，无法生成新形状
       GaussianVAE:   z = mu + std * eps, eps~N(0,I)   # 随机性，梯度仍可流过

    2. **KL 散度正则项**
       KL(q(z|x) || p(z)) 把 encoder 的输出分布拉向标准正态，
       使得推理时可以直接从 N(0, I) 采 z 来生成新形状（无需输入点云）。

    总损失 = L_diffusion + kl_weight * L_KL
    超参: T=100, beta_T=0.02, lr=2e-3, kl_weight=0.001
    """

    def __init__(
        self,
        encoder: PointNetEncoder,
        diffusion: DiffusionPoint,
    ):
        """
        Args:
            encoder  : 已构建好的 PointNetEncoder 实例
            diffusion: 已构建好的 DiffusionPoint 实例
        """
        super().__init__()
        self.encoder = encoder
        self.diffusion = diffusion

    def get_loss(self, x0: torch.Tensor, kl_weight: float) -> torch.Tensor:
        """
        编码点云 → 重参数化采 z → 扩散损失 + KL 损失。

        Args:
            x0        : (B, N, 3)  输入点云（干净，未加噪）
            kl_weight : float      KL 项的权重，典型值 0.001

        Returns:
            loss: 标量 Tensor，L_diffusion + kl_weight * L_KL
        """
        # --- Step 1: 编码点云，得到后验分布参数 ---
        # q(z|x) = N(mu, diag(exp(log_var)))
        # mu: (B, zdim), log_var: (B, zdim)
        mu, log_var = self.encoder(x0)

        # --- Step 2: 重参数化技巧，采样 z ~ q(z|x) ---
        # 直接采样不可微（梯度无法流过随机节点），重参数化将随机性移到 eps：
        #   std = exp(0.5 * log_var)    [用 log_var 而非 sigma，避免对负数开根号]
        #   eps ~ N(0, I)               [与参数无关，梯度不流过这里]
        #   z   = mu + std * eps        [梯度对 mu 和 std（即 log_var）均可流]
        # std: (B, zdim), eps: (B, zdim), z: (B, zdim)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # --- Step 3: 扩散损失 L_diffusion ---
        # 与 AutoEncoder.get_loss 完全一致，z 已采好，直接传入
        # 内部随机采 t、加噪、预测噪声、返回 MSE — 见 DiffusionPoint.get_loss
        # loss_diffusion: 标量
        loss_diffusion = self.diffusion.get_loss(x0, z)

        # --- Step 4: KL 散度 L_KL（closed-form，对角高斯 vs 标准正态）---
        # KL(N(mu, sigma^2) || N(0, I)) = -1/2 * sum_d(1 + log_var_d - mu_d^2 - exp(log_var_d))
        # 对 zdim 维先 sum，再对 batch 取 mean，得到标量
        # 注意：Python 中 ** 是幂运算，^ 是按位异或——对 float Tensor 用 ^ 会报错！
        loss_kl = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1).mean()

        # --- Step 5: 合并损失 ---
        # kl_weight 极小（0.001），防止 KL 项过强压制扩散重建能力
        loss = loss_diffusion + kl_weight * loss_kl
        return loss

    def sample(
        self,
        batch_size: int,  # 生成几个形状
        num_points: int,  # 每个形状多少点（通常 2048）
        flexibility: float,  # 方差插值系数，0=窄，1=宽
        device: torch.device,
    ) -> torch.Tensor:
        """
        从先验 p(z) = N(0, I) 采样 z，再逆向扩散生成点云。

        这里不调用 Encoder——z 直接从标准正态采样。
        KL 正则保证了 encoder 输出的分布接近 N(0, I)，
        从而使先验采样的 z 落在 decoder 见过的分布里。

        Returns:
            x: (B, N, 3) 生成的点云
        """
        # 从先验采样 z ~ N(0, I)
        # zdim 从 encoder 的输出头读取，无需手动传入
        zdim = self.encoder.fc_mu.out_features
        z = torch.randn(batch_size, zdim, device=device)  # (B, zdim)

        # 逆向扩散生成点云: z → x^(T) → x^(T-1) → ... → x^(0)
        return self.diffusion.sample(z, num_points, flexibility)  # (B, N, 3)
