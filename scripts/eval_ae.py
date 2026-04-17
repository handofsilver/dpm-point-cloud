"""
AutoEncoder 定量评估脚本（Table 2）

对每个指定类别的完整测试集做：编码 → 扩散重建 → 计算 per-sample CD / EMD，
最终打印 per-category 均值，格式与论文 Table 2 一致（×10^3 放大）。

运行方式：
    conda run -n dpm3d python scripts/eval_ae.py \
        --data_path data/shapenet/shapenet.hdf5 \
        --ckpt checkpoints/ae/epoch_0100.pt \
        --cates airplane chair car \
        --out_dir results/eval_ae
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from dataset import ShapeNetDataset
from model import VarianceSchedule, PointwiseNet, DiffusionPoint, PointNetEncoder, AutoEncoder
from metrics import chamfer_distance, earth_mover_distance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="shapenet.hdf5 路径")
    parser.add_argument("--ckpt", type=str, required=True, help="AutoEncoder checkpoint 路径")
    parser.add_argument(
        "--cates", nargs="+", default=["airplane", "chair", "car"], help="要评估的类别"
    )
    parser.add_argument("--zdim", type=int, default=256)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--beta_T", type=float, default=0.05)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--flexibility", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_emd", action="store_true", help="跳过 EMD 计算（调试用）")
    parser.add_argument("--out_dir", type=str, default="results/eval_ae")
    return parser.parse_args()


def build_model(args, device):
    var_sched = VarianceSchedule(T=args.T, beta_T=args.beta_T)
    net = PointwiseNet(zdim=args.zdim, residual=True)
    diffusion = DiffusionPoint(net=net, var_sched=var_sched)
    encoder = PointNetEncoder(zdim=args.zdim)
    return AutoEncoder(encoder=encoder, diffusion=diffusion).to(device)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    model = build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # per-category 结果累积
    cd_by_cate = defaultdict(list)  # cate → list of per-sample CD
    emd_by_cate = defaultdict(list)  # cate → list of per-sample EMD

    # --- 按类别遍历测试集 ---
    for cate in args.cates:
        print(f"\n[{cate}] loading test set...")
        dataset = ShapeNetDataset(
            path=args.data_path, split="test", cates=[cate], num_points=args.num_points
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"[{cate}] {len(dataset)} samples")

        for batch in loader:
            x0 = batch["pointcloud"].to(device)  # (B, N, 3)

            with torch.no_grad():
                # AutoEncoder 模式：取 mu 作为确定性 latent
                mu, _ = model.encoder(x0)  # (B, zdim)
                recon = model.sample(
                    z=mu,
                    num_points=args.num_points,
                    flexibility=args.flexibility,
                )  # (B, N, 3)

            cd = chamfer_distance(x0, recon)  # (B,)
            cd_by_cate[cate].extend(cd.cpu().tolist())

            if not args.no_emd:
                emd = earth_mover_distance(x0, recon)  # (B,)
                emd_by_cate[cate].extend(emd.cpu().tolist())

    # --- 打印结果表格（×10^3，与论文一致）---
    scale = 1e3
    header = f"{'Category':<12} {'CD×1e3':>10}"
    if not args.no_emd:
        header += f"  {'EMD×1e3':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    all_cd, all_emd = [], []
    for cate in args.cates:
        mean_cd = sum(cd_by_cate[cate]) / len(cd_by_cate[cate])
        all_cd.extend(cd_by_cate[cate])
        row = f"{cate:<12} {mean_cd * scale:>10.4f}"

        if not args.no_emd:
            mean_emd = sum(emd_by_cate[cate]) / len(emd_by_cate[cate])
            all_emd.extend(emd_by_cate[cate])
            row += f"  {mean_emd * scale:>10.4f}"

        print(row)

    # 所有类别的总均值
    print("-" * len(header))
    mean_row = f"{'mean':<12} {(sum(all_cd) / len(all_cd)) * scale:>10.4f}"
    if not args.no_emd and all_emd:
        mean_row += f"  {(sum(all_emd) / len(all_emd)) * scale:>10.4f}"
    print(mean_row)
    print("=" * len(header))

    # --- 保存数值到文本文件 ---
    out_path = os.path.join(args.out_dir, "table2_results.txt")
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for cate in args.cates:
            mean_cd = sum(cd_by_cate[cate]) / len(cd_by_cate[cate])
            line = f"{cate:<12} {mean_cd * scale:>10.4f}"
            if not args.no_emd and emd_by_cate[cate]:
                mean_emd = sum(emd_by_cate[cate]) / len(emd_by_cate[cate])
                line += f"  {mean_emd * scale:>10.4f}"
            f.write(line + "\n")
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
