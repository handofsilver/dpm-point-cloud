"""
评估指标：CD、EMD，以及集合级指标 MMD、COV、1-NNA。

注：EMD 使用 geomloss 的 Sinkhorn 近似（替代原论文的 approxmatch.cu CUDA kernel）。
两者均为近似最优传输，误差量级相当，数值可比。报告时需加脚注说明。
"""

import torch
from geomloss import SamplesLoss

# Sinkhorn EMD 全局实例：p=1 对应 Wasserstein-1（即 EMD），blur 控制近似精度
# blur=0.05 是点云评估的常用值，越小越精确但越慢、越容易数值不稳定
_sinkhorn = SamplesLoss("sinkhorn", p=1, blur=0.05)


# ---------------------------------------------------------------------------
# 逐样本距离
# ---------------------------------------------------------------------------


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    计算两组点云之间的 Chamfer Distance（对称，距离用平方）。

    CD(S1, S2) = mean_{x in S1} min_{y in S2} ||x-y||^2
               + mean_{y in S2} min_{x in S1} ||y-x||^2

    Args:
        p: (B, N, 3)  点云组 1（通常是输入/真值点云）
        q: (B, M, 3)  点云组 2（通常是重建/生成点云）

    Returns:
        cd: (B,)  每个样本的 Chamfer Distance
    """
    # 构造成对平方距离矩阵
    # p: (B, N, 1, 3), q: (B, 1, M, 3) → 广播差值 (B, N, M, 3) → (B, N, M)
    dist = ((p.unsqueeze(2) - q.unsqueeze(1)) ** 2).sum(dim=-1)  # (B, N, M)

    # p → q：每个 p_i 找最近的 q_j，再对 N 个点取均值
    # .min(dim=2) 返回 namedtuple(values, indices)，取 .values 得 Tensor
    d_pq = dist.min(dim=2).values.mean(dim=1)  # (B,)

    # q → p：每个 q_j 找最近的 p_i，再对 M 个点取均值
    d_qp = dist.min(dim=1).values.mean(dim=1)  # (B,)

    return d_pq + d_qp  # (B,)


def earth_mover_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    计算两组点云之间的近似 EMD（Wasserstein-1，Sinkhorn 近似）。

    EMD 要求点云之间一一对应的最优匹配，比 CD 对全局结构更敏感。
    p 和 q 的点数 N 必须相同（最优传输要求等大小）。

    Args:
        p: (B, N, 3)
        q: (B, N, 3)

    Returns:
        emd: (B,)
    """
    # _sinkhorn 对整个 batch 求均值，需逐样本调用才能拿到 (B,) 的结果
    emd = torch.stack(
        [_sinkhorn(p[i].unsqueeze(0), q[i].unsqueeze(0)).squeeze() for i in range(p.size(0))]
    )
    return emd  # (B,)


# ---------------------------------------------------------------------------
# 集合级成对距离矩阵（供 MMD / COV / 1-NNA 使用）
# ---------------------------------------------------------------------------


def _pairwise_cd(x: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """
    计算两个点云集合之间所有点对的 CD，返回距离矩阵。

    Args:
        x: (M, N, 3)
        y: (K, N, 3)
        batch_size: 每批处理 x 中多少个样本（控制显存）

    Returns:
        dist: (M, K)，dist[i, j] = CD(x[i], y[j])
    """
    M, K = x.size(0), y.size(0)
    dist = torch.zeros(M, K, device=x.device)

    for i in range(0, M, batch_size):
        x_batch = x[i : i + batch_size]  # (bs, N, 3)
        bs = x_batch.size(0)
        for j in range(K):
            # 把 y[j] 复制 bs 份，凑成 (bs, N, 3) 后批量算 CD
            yj = y[j].unsqueeze(0).expand(bs, -1, -1)  # (bs, N, 3)
            dist[i : i + bs, j] = chamfer_distance(x_batch, yj)

    return dist  # (M, K)


def _pairwise_emd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两个点云集合之间所有点对的 EMD，返回距离矩阵。
    比 _pairwise_cd 慢很多，建议只在最终评估时使用。

    Args:
        x: (M, N, 3)
        y: (K, N, 3)

    Returns:
        dist: (M, K)
    """
    M, K = x.size(0), y.size(0)
    dist = torch.zeros(M, K, device=x.device)

    for i in range(M):
        # 把 x[i] 复制 K 份，和整个 y 批量算 EMD
        xi = x[i].unsqueeze(0).expand(K, -1, -1)  # (K, N, 3)
        dist[i] = earth_mover_distance(xi, y)  # (K,)

    return dist  # (M, K)


# ---------------------------------------------------------------------------
# 集合级指标
# ---------------------------------------------------------------------------


def _mmd_cov_1nna(
    M_sr: torch.Tensor,
    M_rr: torch.Tensor,
    M_ss: torch.Tensor,
    suffix: str,
) -> dict:
    """
    给定成对距离矩阵，计算 MMD、COV、1-NNA。

    Args:
        M_sr: (S, R)  sample vs ref 成对距离
        M_rr: (R, R)  ref vs ref 成对距离（对角线需排除）
        M_ss: (S, S)  sample vs sample 成对距离（对角线需排除）
        suffix: "CD" 或 "EMD"

    Returns:
        {"MMD-CD": ..., "COV-CD": ..., "1-NNA-CD": ...}
    """
    S, R = M_sr.shape

    # --- MMD ---
    # M_sr.t() → (R, S)，每行是一个 ref 对所有 sample 的距离，取行最小值再求均值
    mmd = M_sr.t().min(dim=1).values.mean()

    # --- COV ---
    nearest_ref = M_sr.min(dim=1).indices  # (S,) 每个 sample 最近的 ref 索引
    cov = float(nearest_ref.unique().numel()) / R

    # --- 1-NNA ---
    # 把 ref 和 sample 混合，对每个点云找最近邻（排除自身），看标签一致性
    # 完整距离矩阵布局（R+S, R+S）：
    #   左上 M_rr (R,R)  右上 M_sr.t() (R,S)
    #   左下 M_sr (S,R)  右下 M_ss    (S,S)
    M_full = torch.cat(
        [
            torch.cat([M_rr, M_sr.t()], dim=1),  # (R, R+S)
            torch.cat([M_sr, M_ss], dim=1),  # (S, R+S)
        ],
        dim=0,
    )  # (R+S, R+S)

    M_full.fill_diagonal_(float("inf"))  # 排除自身

    # ref → label 1，sample → label 0
    labels = torch.cat(
        [
            torch.ones(R, device=M_sr.device),
            torch.zeros(S, device=M_sr.device),
        ]
    )  # (R+S,)

    nn_labels = labels[M_full.min(dim=1).indices]  # 每个点最近邻的标签
    acc = (nn_labels == labels).float().mean()

    return {
        f"MMD-{suffix}": mmd,
        f"COV-{suffix}": cov,
        f"1-NNA-{suffix}": acc,
    }


def compute_all_metrics(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 64,
    use_emd: bool = True,
) -> dict:
    """
    计算 Table 1 所需的全部集合级指标：MMD、COV、1-NNA × CD/EMD = 6 个数字。

    Args:
        sample_pcs: (S, N, 3)  生成集
        ref_pcs:    (R, N, 3)  参考集（测试集）
        batch_size: 成对 CD 计算的批大小
        use_emd:    是否同时计算 EMD 版指标（慢，调试时可设 False）

    Returns:
        dict，key 形如 "MMD-CD"、"COV-EMD" 等
    """
    results = {}

    print("Computing pairwise CD...")
    M_sr = _pairwise_cd(sample_pcs, ref_pcs, batch_size)
    M_rr = _pairwise_cd(ref_pcs, ref_pcs, batch_size)
    M_ss = _pairwise_cd(sample_pcs, sample_pcs, batch_size)
    results.update(_mmd_cov_1nna(M_sr, M_rr, M_ss, suffix="CD"))

    if use_emd:
        print("Computing pairwise EMD (slow)...")
        M_sr_e = _pairwise_emd(sample_pcs, ref_pcs)
        M_rr_e = _pairwise_emd(ref_pcs, ref_pcs)
        M_ss_e = _pairwise_emd(sample_pcs, sample_pcs)
        results.update(_mmd_cov_1nna(M_sr_e, M_rr_e, M_ss_e, suffix="EMD"))

    return results
