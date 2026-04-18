"""
生成模型定量评估脚本（Table 1）

从训练好的 FlowVAE / GaussianVAE 先验采样生成集 S_g，
与对应类别测试集 S_r 比较，计算 MMD / COV / 1-NNA（各 CD 和 EMD 两种距离）。

关键细节：
- 生成集大小 num_samples 默认取与测试集相同（部分论文固定用 2048）
- 1-NNA 要求 S_g 和 S_r 大小相等，num_samples 需 >= len(test_set)
- 成对距离矩阵 O(S×R) 内存，S=R=2000 时约 32MB（float32），可接受

运行方式：
    conda run -n dpm3d python scripts/eval_gen.py \
        --data_path data/shapenet/shapenet.hdf5 \
        --ckpt checkpoints/gen/flow_epoch_2000.pt \
        --model flow \
        --cates airplane chair car \
        --out_dir results/eval_gen
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from dataset import ShapeNetDataset
from model import (
    VarianceSchedule,
    PointwiseNet,
    DiffusionPoint,
    PointNetEncoder,
    GaussianVAE,
    NormalizingFlow,
    FlowVAE,
)
from metrics import compute_all_metrics


def normalize_to_bbox(pcs: torch.Tensor) -> torch.Tensor:
    """
    Per-shape 归一化到 [-1, 1]^3 bbox（ShapeGF / 论文 Sec 5.2 协议）。

    取外接 bbox 最长轴做均匀缩放：最长轴正好落在 [-1, 1]，其他轴更短。
    ref 和 gen 必须各自做一次，保证指标只比较形状不比较绝对尺度。

    pcs: (B, N, 3)  ->  (B, N, 3)
    """
    pc_max = pcs.max(dim=1, keepdim=True).values  # (B, 1, 3)
    pc_min = pcs.min(dim=1, keepdim=True).values  # (B, 1, 3)
    shift = (pc_min + pc_max) / 2  # (B, 1, 3)  bbox 中心
    scale = (pc_max - pc_min).max(dim=-1, keepdim=True).values / 2  # (B, 1, 1) 最长轴半长
    return (pcs - shift) / scale


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="shapenet.hdf5 路径")
    parser.add_argument("--ckpt", type=str, required=True, help="生成模型 checkpoint 路径")
    parser.add_argument(
        "--model", type=str, default="flow", choices=["gaussian", "flow"], help="与训练时保持一致"
    )
    parser.add_argument(
        "--cates", nargs="+", default=["airplane", "chair", "car"], help="要评估的类别"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="生成样本数；None 表示自动取与测试集相同大小"
    )
    parser.add_argument("--zdim", type=int, default=256)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--flexibility", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64, help="采样时的批大小")
    parser.add_argument("--cd_batch_size", type=int, default=64, help="成对 CD 计算的批大小")
    parser.add_argument("--no_emd", action="store_true", help="跳过 EMD 版指标（调试用）")
    parser.add_argument("--out_dir", type=str, default="results/eval_gen")
    return parser.parse_args()


def build_model(args, device):
    var_sched = VarianceSchedule(T=args.T, beta_T=args.beta_T)
    net = PointwiseNet(zdim=args.zdim, residual=True)
    diffusion = DiffusionPoint(net=net, var_sched=var_sched)
    encoder = PointNetEncoder(zdim=args.zdim)

    if args.model == "flow":
        flow = NormalizingFlow(
            zdim=args.zdim,
            num_layers=args.flow_layers,
            hidden_dim=args.flow_hidden_dim,
        )
        return FlowVAE(encoder=encoder, diffusion=diffusion, flow=flow).to(device)
    else:
        return GaussianVAE(encoder=encoder, diffusion=diffusion).to(device)


def sample_generated(model, num_samples, args, device) -> torch.Tensor:
    """从先验分布批量采样，返回 (num_samples, N, 3)。"""
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        with torch.no_grad():
            batch = model.sample(
                batch_size=bs,
                num_points=args.num_points,
                flexibility=args.flexibility,
                device=device,
            )  # (bs, N, 3)
        all_samples.append(batch.cpu())
        remaining -= bs
    return torch.cat(all_samples, dim=0)  # (num_samples, N, 3)


def load_ref(data_path, cate, num_points) -> torch.Tensor:
    """加载指定类别的完整测试集，返回 (R, N, 3)。"""
    dataset = ShapeNetDataset(path=data_path, split="test", cates=[cate], num_points=num_points)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    pcs = [batch["pointcloud"] for batch in loader]
    return torch.cat(pcs, dim=0)  # (R, N, 3)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    model = build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if "args" in ckpt:
        saved = ckpt["args"]
        for key in ("zdim", "T", "beta_T"):
            if key in saved:
                setattr(args, key, saved[key])
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}

    for cate in args.cates:
        print(f"\n{'='*50}")
        print(f"[{cate}] 加载测试集...")
        ref_pcs = load_ref(args.data_path, cate, args.num_points)  # (R, N, 3)
        R = ref_pcs.shape[0]
        print(f"[{cate}] 测试集大小: {R}")

        # 生成集大小默认与测试集相同（1-NNA 要求两集合等大）
        num_samples = args.num_samples if args.num_samples is not None else R
        print(f"[{cate}] 从先验采样 {num_samples} 个形状...")
        sample_pcs = sample_generated(model, num_samples, args, device)  # (S, N, 3)

        print(f"[{cate}] 计算 MMD / COV / 1-NNA...")
        # 论文 Sec 5.2 协议：ref 和 gen 都归一化到 [-1,1]^3 bbox 再算指标
        # （JSD 的 voxel 直方图也依赖这组归一化，必须统一）
        sample_pcs = normalize_to_bbox(sample_pcs)
        ref_pcs = normalize_to_bbox(ref_pcs)
        results = compute_all_metrics(
            sample_pcs=sample_pcs.to(device),
            ref_pcs=ref_pcs.to(device),
            batch_size=args.cd_batch_size,
            use_emd=not args.no_emd,
        )
        all_results[cate] = results

        for k, v in results.items():
            print(f"  {k}: {v:.6f}")

    # --- 打印汇总表格 ---
    keys_cd = ["MMD-CD", "COV-CD", "1-NNA-CD"]
    keys_emd = ["MMD-EMD", "COV-EMD", "1-NNA-EMD"]
    use_emd = not args.no_emd

    col_keys = keys_cd + (keys_emd if use_emd else []) + ["JSD"]
    header = f"{'Category':<12}" + "".join(f"  {k:>12}" for k in col_keys)

    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    for cate in args.cates:
        row = f"{cate:<12}"
        for k in col_keys:
            v = all_results[cate].get(k, float("nan"))
            row += f"  {v:>12.6f}"
        print(row)
    print(f"{'='*len(header)}")

    # --- 保存到文件 ---
    out_path = os.path.join(args.out_dir, "table1_results.txt")
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for cate in args.cates:
            row = f"{cate:<12}"
            for k in col_keys:
                v = all_results[cate].get(k, float("nan"))
                row += f"  {v:>12.6f}"
            f.write(row + "\n")
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
