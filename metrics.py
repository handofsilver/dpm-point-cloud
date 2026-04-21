"""
评估指标：CD、EMD，以及集合级指标 MMD、COV、1-NNA、JSD。

注：EMD 使用 geomloss 的 Sinkhorn 近似（替代原论文的 approxmatch.cu CUDA kernel）。
两者均为近似最优传输，误差量级相当，数值可比。报告时需加脚注说明。

JSD 来自 Achlioptas et al. 2018（原论文 [25]），把点云集合投影到
28^3 体素占据直方图，再算 Jensen-Shannon Divergence。
"""

import os
from typing import Optional

import numpy as np
import torch
from geomloss import SamplesLoss
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

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


# ---------------------------------------------------------------------------
# JSD：体素占据分布距离
# ---------------------------------------------------------------------------


def _unit_cube_grid(resolution: int, clip_sphere: bool = True) -> np.ndarray:
    """
    生成单位立方体 [-0.5, 0.5]^3 内均匀网格的中心坐标。

    Args:
        resolution:  每轴格数，总共 resolution^3 个格子
        clip_sphere: True 时裁掉外接球以外的角落格（与原论文实现一致）

    Returns:
        coords: (G, 3)  网格中心坐标，G <= resolution^3
    """
    spacing = 1.0 / float(resolution - 1)
    # 用 meshgrid 生成所有格子中心，坐标范围 [-0.5, 0.5]
    axes = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1)  # (R, R, R, 3)
    coords = grid.reshape(-1, 3)  # (R^3, 3)
    if clip_sphere:
        # 保留到原点距离 <= 0.5 的格子（与 bbox 内切球一致）
        coords = coords[np.linalg.norm(coords, axis=1) <= 0.5 + spacing]
    return coords  # (G, 3)


def _occupancy_distribution(
    pcs: np.ndarray,
    grid_coords: np.ndarray,
) -> np.ndarray:
    """
    给定点云集合，统计每个体素格子被多少个点云"碰到过"（Bernoulli 计数）。

    每个点云对每个格子最多贡献 1（不是总点数，是命中该格的点云数），
    这样得到的向量可以归一化为整个集合的空间占据概率分布。

    Args:
        pcs:         (S, N, 3)  S 个点云，每个 N 点，坐标在 [-0.5, 0.5]^3
        grid_coords: (G, 3)    体素格中心（来自 _unit_cube_grid）

    Returns:
        dist: (G,)  每格被命中的点云数（未归一化整数计数）
    """
    # 在格子中心上建 kd-tree，查询"每个点属于哪个格子"
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(grid_coords)

    dist = np.zeros(len(grid_coords), dtype=np.float32)  # (G,)

    for pc in pcs:  # pc: (N, 3)
        _, indices = nn.kneighbors(pc)  # indices: (N, 1) 每个点最近格子的编号
        # unique：同一格子只计一次（Bernoulli，不是频率计数）
        unique_indices = np.unique(indices)
        dist[unique_indices] += 1

    return dist  # (G,)


def _jensen_shannon_div(P: np.ndarray, Q: np.ndarray) -> float:
    """
    计算两个非负向量之间的 Jensen-Shannon Divergence（以 2 为底，单位 bit）。

    JSD(P||Q) = H( (P+Q)/2 ) - ( H(P) + H(Q) ) / 2

    其中 H 是香农熵。JSD 是 KL 散度的对称化版本，范围 [0, 1]。

    Args:
        P: (G,)  分布 1，未归一化非负向量
        Q: (G,)  分布 2，未归一化非负向量

    Returns:
        jsd: float ∈ [0, 1]
    """
    # 归一化为概率分布（各自除以自身总和）
    P_ = P / P.sum()  # (G,)
    Q_ = Q / Q.sum()  # (G,)

    # scipy.stats.entropy 自动跳过 p=0 的项（避免 log(0)），以 2 为底输出 bit
    H_P = entropy(P_, base=2)
    H_Q = entropy(Q_, base=2)
    H_mix = entropy((P_ + Q_) / 2.0, base=2)

    return float(H_mix - (H_P + H_Q) / 2.0)


def jsd_between_point_cloud_sets(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    resolution: int = 28,
) -> float:
    """
    计算两组点云集合之间的 JSD（论文 Table 1 第四列）。

    点云必须已归一化到 [-1, 1]^3（由 eval_gen.normalize_to_bbox 保证）。
    内部将坐标缩到 [-0.5, 0.5]^3 再与体素网格匹配（保持与原论文实现一致）。

    Args:
        sample_pcs: (S, N, 3)  生成集（torch.Tensor，CPU 或 GPU）
        ref_pcs:    (R, N, 3)  参考集（torch.Tensor，CPU 或 GPU）
        resolution: 体素网格每轴格数，默认 28（与原论文一致）

    Returns:
        jsd: float
    """
    # 转为 numpy，缩放 [-1,1]^3 → [-0.5,0.5]^3（与 _unit_cube_grid 的坐标系对齐）
    s_np = sample_pcs.cpu().numpy() / 2.0  # (S, N, 3)
    r_np = ref_pcs.cpu().numpy() / 2.0  # (R, N, 3)

    grid = _unit_cube_grid(resolution, clip_sphere=True)  # (G, 3)

    P = _occupancy_distribution(s_np, grid)  # (G,) sample 分布
    Q = _occupancy_distribution(r_np, grid)  # (G,) ref 分布

    return _jensen_shannon_div(P, Q)


def _load_pairwise_if_match(
    cache_path: str,
    expected_shape: tuple,
    device: torch.device,
) -> Optional[dict]:
    """
    若 cache_path 存在且三块矩阵形状满足期望，则加载；否则返回 None。

    expected_shape: (S, R)，用于校验 M_sr；M_rr 期望 (R, R)、M_ss 期望 (S, S)。
    形状不匹配通常意味着 num_samples 或 num_points 改了，缓存失效。
    """
    if not os.path.exists(cache_path):
        return None
    data = torch.load(cache_path, map_location=device)
    S, R = expected_shape
    if data["sr"].shape != (S, R) or data["rr"].shape != (R, R) or data["ss"].shape != (S, S):
        print(f"  缓存 {cache_path} 形状不符，忽略并重新计算")
        return None
    return data


def _save_pairwise(cache_path: str, sr: torch.Tensor, rr: torch.Tensor, ss: torch.Tensor):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"sr": sr.cpu(), "rr": rr.cpu(), "ss": ss.cpu()}, cache_path)


def compute_all_metrics(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 64,
    use_emd: bool = True,
    use_jsd: bool = True,
    cache_dir: Optional[str] = None,
) -> dict:
    """
    计算 Table 1 所需的全部集合级指标：
      MMD、COV、1-NNA × CD/EMD（6 个）+ JSD（1 个）= 7 个数字。

    Args:
        sample_pcs: (S, N, 3)  生成集，已归一化到 [-1,1]^3
        ref_pcs:    (R, N, 3)  参考集（测试集），已归一化到 [-1,1]^3
        batch_size: 成对 CD 计算的批大小
        use_emd:    是否同时计算 EMD 版指标（慢，调试时可设 False）
        use_jsd:    是否计算 JSD
        cache_dir:  若给定，将成对 CD / EMD 距离矩阵缓存到该目录；
                    下次调用时若 (S, R) 形状一致则直接加载、跳过重算。
                    主要用于：JSD 代码改了重跑、MMD/COV/1-NNA 逻辑改了重跑时
                    不必重新计算 pairwise EMD（这步最耗时）。
                    缓存**不校验**生成样本是否同源 —— eval_gen.py 层面通过
                    manifest.json 对生成样本做版本校验，调用方负责保持一致。

    Returns:
        dict，key 形如 "MMD-CD"、"COV-EMD"、"JSD" 等
    """
    results = {}
    S, R = sample_pcs.size(0), ref_pcs.size(0)
    device = sample_pcs.device

    # --- Pairwise CD ---
    cd_cache = os.path.join(cache_dir, "pairwise_cd.pt") if cache_dir else None
    cached = _load_pairwise_if_match(cd_cache, (S, R), device) if cd_cache else None
    if cached is not None:
        print("Loading cached pairwise CD...")
        M_sr, M_rr, M_ss = cached["sr"].to(device), cached["rr"].to(device), cached["ss"].to(device)
    else:
        print("Computing pairwise CD...")
        M_sr = _pairwise_cd(sample_pcs, ref_pcs, batch_size)
        M_rr = _pairwise_cd(ref_pcs, ref_pcs, batch_size)
        M_ss = _pairwise_cd(sample_pcs, sample_pcs, batch_size)
        if cd_cache:
            _save_pairwise(cd_cache, M_sr, M_rr, M_ss)
    results.update(_mmd_cov_1nna(M_sr, M_rr, M_ss, suffix="CD"))

    # --- Pairwise EMD ---
    if use_emd:
        emd_cache = os.path.join(cache_dir, "pairwise_emd.pt") if cache_dir else None
        cached = _load_pairwise_if_match(emd_cache, (S, R), device) if emd_cache else None
        if cached is not None:
            print("Loading cached pairwise EMD...")
            M_sr_e = cached["sr"].to(device)
            M_rr_e = cached["rr"].to(device)
            M_ss_e = cached["ss"].to(device)
        else:
            print("Computing pairwise EMD (slow)...")
            M_sr_e = _pairwise_emd(sample_pcs, ref_pcs)
            M_rr_e = _pairwise_emd(ref_pcs, ref_pcs)
            M_ss_e = _pairwise_emd(sample_pcs, sample_pcs)
            if emd_cache:
                _save_pairwise(emd_cache, M_sr_e, M_rr_e, M_ss_e)
        results.update(_mmd_cov_1nna(M_sr_e, M_rr_e, M_ss_e, suffix="EMD"))

    # --- JSD（快，不缓存；代码若改动 cache 反而会误导） ---
    if use_jsd:
        print("Computing JSD...")
        results["JSD"] = jsd_between_point_cloud_sets(sample_pcs, ref_pcs)

    return results
