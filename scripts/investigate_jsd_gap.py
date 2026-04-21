"""
调查脚本（read-only，不改动任何生产代码）：
定位我们 Table 1 JSD 与论文 10× gap 的真实来源。

不依赖 metrics.py / dataset.py —— JSD 实现就在本脚本内，避免被任何生产代码改动污染。

假设评估对象（按优先级）：
  H1. 数据集差异：shapenet.hdf5 与 PC15k 在 airplane 类上形状分布不同，oracle lower bound 就 ≠ 论文 0.000809
  H2. 实现细节：Bernoulli vs 点级直方图、是否 /2 缩放，这 4 种变体里有一种能对齐论文
  H3. 论文 `GEN_airplane.pt` 在我们数据 + 我们 pipeline 上也产出 10× 水位，说明 pipeline 系统性偏
  H4. 模型质量：我们的 airplane-only 模型真的比论文差（但 MMD/COV/1-NNA 已达论文水平，相对不太可能主导）

本脚本先跑 oracle 实验（纯数据，不需要任何模型）：
  JSD(airplane train subset, airplane test)，4 个实现变体 × 2 个归一化方式 × 2 个分辨率

论文 Table 1 Airplane "Train" 行 = 0.000809（×10^-3 缩放还原后）是**所有 Table 1 数字的 lower bound**。
  - 若我们 oracle ≈ 0.001：pipeline/数据/实现都 OK，10× gap 来自模型质量 → H4
  - 若我们 oracle ≈ 0.01：pipeline 或数据有结构性差异，模型再练也到不了 0.001 → H1 或 H2
  - 若 4 个变体里有一个能给 ≈ 0.001：H2 胜出，可以定位到具体实现细节

用法：
  conda run -n dpm3d python scripts/investigate_jsd_gap.py
  （或任何能用的 python；本脚本只依赖 numpy / h5py / scipy / sklearn，无 torch / 无 GPU）
"""

import os
import sys

import h5py
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

# ShapeNetCore synset id for airplane
SYNSET_AIRPLANE = "02691156"

HDF5_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "shapenet", "shapenet.hdf5"
)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def normalize_to_bbox(pcs: np.ndarray) -> np.ndarray:
    """Per-shape bbox → [-1, 1]^3，与 eval_gen.py 的 normalize_to_bbox 完全一致。"""
    pc_max = pcs.max(axis=1, keepdims=True)  # (B, 1, 3)
    pc_min = pcs.min(axis=1, keepdims=True)
    shift = (pc_min + pc_max) / 2.0  # (B, 1, 3)
    scale = (pc_max - pc_min).max(axis=-1, keepdims=True) / 2.0  # (B, 1, 1)
    return (pcs - shift) / scale


# ---------------------------------------------------------------------------
# 4 个 JSD 实现变体（全部自包含）
# ---------------------------------------------------------------------------
def _unit_cube_grid(resolution: int, clip_sphere: bool = True) -> np.ndarray:
    axes = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if clip_sphere:
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]
    return grid


def _occupancy(pcs: np.ndarray, grid: np.ndarray, mode: str) -> np.ndarray:
    """
    mode='bernoulli'：每点云对每格最多 +1（参考仓库的 grid_bernoulli_rvars）
    mode='point'：每点 +1 到最近格（参考仓库的 grid_counters，JSD 真正喂入的那一路）
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(grid)
    dist = np.zeros(len(grid), dtype=np.float64)
    for pc in pcs:
        _, idx = nn.kneighbors(pc)
        idx = idx.ravel()
        if mode == "bernoulli":
            dist[np.unique(idx)] += 1.0
        elif mode == "point":
            np.add.at(dist, idx, 1.0)
        else:
            raise ValueError(mode)
    return dist


def _jsd(P: np.ndarray, Q: np.ndarray) -> float:
    P_, Q_ = P / P.sum(), Q / Q.sum()
    return float(
        entropy((P_ + Q_) / 2.0, base=2) - (entropy(P_, base=2) + entropy(Q_, base=2)) / 2.0
    )


def jsd_variant(
    sample_pcs: np.ndarray,
    ref_pcs: np.ndarray,
    mode: str,
    scale_half: bool,
    resolution: int = 28,
) -> float:
    """
    mode:        'bernoulli' | 'point'
    scale_half:  True 表示 /2.0 把 [-1,1]^3 缩到 [-0.5,0.5]^3 再查表（我们原代码）
                 False 表示不缩放，直接把 [-1,1]^3 的点云喂进 [-0.5,0.5]^3 grid（参考仓库原样）
    """
    s = sample_pcs / 2.0 if scale_half else sample_pcs
    r = ref_pcs / 2.0 if scale_half else ref_pcs
    grid = _unit_cube_grid(resolution, clip_sphere=True)
    P = _occupancy(s, grid, mode=mode)
    Q = _occupancy(r, grid, mode=mode)
    return _jsd(P, Q)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_split(split: str, synset: str = SYNSET_AIRPLANE) -> np.ndarray:
    with h5py.File(HDF5_PATH, "r") as f:
        pcs = np.array(f[synset][split], dtype=np.float32)  # (N, 2048, 3)
    return pcs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(HDF5_PATH):
        print(f"错误：找不到 {HDF5_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"加载 {HDF5_PATH}")
    train = load_split("train")
    test = load_split("test")
    val = load_split("val")
    print(f"airplane splits (on shapenet.hdf5):  train={train.shape}  val={val.shape}  test={test.shape}")

    # ------------------------------------------------------------------
    # Oracle 1: train 子集（随机下采到 test 大小）vs test
    # 对应论文 Table 1 "Train" 行的 lower bound = 0.000809（PC15k 上）
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(train), size=len(test), replace=False)
    train_sub = train[idx]

    # normalize_to_bbox 是论文 Sec 5.2 协议（eval_gen.py 用的也是这个）
    train_sub_bbox = normalize_to_bbox(train_sub)
    test_bbox = normalize_to_bbox(test)

    print("\n" + "=" * 80)
    print("Oracle 1: airplane train (下采到 %d) vs test (%d)" % (len(test), len(test)))
    print("对照点：论文 Table 1 'Train' 行 = 0.000809（PC15k）")
    print("=" * 80)
    print(f"  {'mode':<11} {'scale_half':<11} {'res':<5} {'JSD':<12}")
    for mode in ["bernoulli", "point"]:
        for scale_half in [True, False]:
            for resolution in [28]:
                j = jsd_variant(train_sub_bbox, test_bbox, mode, scale_half, resolution)
                print(f"  {mode:<11} {str(scale_half):<11} {resolution:<5} {j:.6f}")

    # ------------------------------------------------------------------
    # Oracle 2: test 分成两半（互不重叠），做 self-JSD
    # 这个理论上是"同一分布的两个独立样本"的差异，应该 << train vs test
    # 若这一项也 >> 0.001，说明 2048-pt 量级下 JSD 的 noise floor 本身就 >> 0.001
    # ------------------------------------------------------------------
    perm = rng.permutation(len(test))
    half = len(test) // 2
    test_a = test[perm[:half]]
    test_b = test[perm[half : 2 * half]]
    test_a_bbox = normalize_to_bbox(test_a)
    test_b_bbox = normalize_to_bbox(test_b)

    print("\n" + "=" * 80)
    print("Oracle 2: test 一分为二（%d vs %d），self-JSD —— 同分布 noise floor" % (half, half))
    print("=" * 80)
    print(f"  {'mode':<11} {'scale_half':<11} {'res':<5} {'JSD':<12}")
    for mode in ["bernoulli", "point"]:
        for scale_half in [True, False]:
            for resolution in [28]:
                j = jsd_variant(test_a_bbox, test_b_bbox, mode, scale_half, resolution)
                print(f"  {mode:<11} {str(scale_half):<11} {resolution:<5} {j:.6f}")

    # ------------------------------------------------------------------
    # Oracle 4: 固定变体（point + scale_half=True, res=28 = 我们当前生产实现），
    # 扫 sample 数（先验方差 vs 样本量的 trade-off），看 JSD 能随样本量下降多少
    # 如果论文用了更大的 train 集做 "Train" 行，我们可能能推回 0.001 量级
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Oracle 4: 固定 (point, scale_half=True, res=28)，扫样本量（airplane train 子集 vs test 全量）")
    print("=" * 80)
    print(f"  {'train_n':<10} {'test_n':<10} {'JSD':<12}")
    for n in [200, 500, 1000, 2000, 3438]:
        n_eff = min(n, len(train))
        idx_n = rng.choice(len(train), size=n_eff, replace=False)
        tr = normalize_to_bbox(train[idx_n])
        j = jsd_variant(tr, test_bbox, mode="point", scale_half=True, resolution=28)
        print(f"  {n_eff:<10} {len(test):<10} {j:.6f}")

    # ------------------------------------------------------------------
    # Oracle 5: 固定样本量（607 train vs 607 test），扫 resolution
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Oracle 5: 固定样本量 607 vs 607，扫 voxel resolution")
    print("=" * 80)
    print(f"  {'res':<5} {'JSD':<12}")
    for resolution in [16, 20, 28, 32, 48, 64]:
        j = jsd_variant(
            train_sub_bbox, test_bbox, mode="point", scale_half=True, resolution=resolution
        )
        print(f"  {resolution:<5} {j:.6f}")

    # ------------------------------------------------------------------
    # Oracle 6: 不走 per-shape bbox 归一化，改用"整个集合一次 bbox"
    # 若论文用了这种全局归一化，单个 shape 在共享 bbox 里会贴合更紧（airplane 多数形状
    # 都在中心），分布可能更密集 → JSD 更小
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Oracle 6: 全局 bbox 归一化（全体点云共用一个 bbox），vs per-shape bbox")
    print("=" * 80)

    def normalize_global_bbox(pcs):
        """所有形状共用一个 bbox（center = all-points mean, scale = all-points half-range）。"""
        all_pts = pcs.reshape(-1, 3)
        pc_max = all_pts.max(axis=0)
        pc_min = all_pts.min(axis=0)
        shift = (pc_min + pc_max) / 2.0
        scale = float((pc_max - pc_min).max() / 2.0)
        return (pcs - shift) / scale

    tr_g = normalize_global_bbox(train_sub)
    te_g = normalize_global_bbox(test)
    print(f"  {'norm':<15} {'JSD':<12}")
    j_per = jsd_variant(train_sub_bbox, test_bbox, mode="point", scale_half=True, resolution=28)
    j_glob = jsd_variant(tr_g, te_g, mode="point", scale_half=True, resolution=28)
    print(f"  {'per_shape':<15} {j_per:.6f}")
    print(f"  {'global':<15} {j_glob:.6f}")


if __name__ == "__main__":
    main()
