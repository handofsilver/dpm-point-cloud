"""
评估指标
目前实现：Chamfer Distance (CD)
"""

import torch


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
