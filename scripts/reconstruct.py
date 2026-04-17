"""
AutoEncoder 重建效果评估脚本

对测试集中若干样本做：编码 → 扩散重建 → 计算 CD → 保存对比图

运行方式：
    conda run -n dpm3d python scripts/reconstruct.py \
        --data_path data/shapenet/shapenet.hdf5 \
        --ckpt checkpoints/ae/epoch_0100.pt \
        --num_samples 4 \
        --out_dir results/reconstruct
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from dataset import ShapeNetDataset
from model import VarianceSchedule, PointwiseNet, DiffusionPoint, PointNetEncoder, AutoEncoder
from metrics import chamfer_distance
from visualize import plot_reconstruction


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="shapenet.hdf5 文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="AutoEncoder checkpoint 路径")
    parser.add_argument("--num_samples", type=int, default=4, help="评估样本数")
    parser.add_argument("--zdim", type=int, default=256)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--beta_T", type=float, default=0.05)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--flexibility", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="results/reconstruct")
    return parser.parse_args()


def build_model(args, device):
    var_sched = VarianceSchedule(T=args.T, beta_T=args.beta_T)
    net = PointwiseNet(zdim=args.zdim, residual=True)
    diffusion = DiffusionPoint(net=net, var_sched=var_sched)
    encoder = PointNetEncoder(zdim=args.zdim)
    model = AutoEncoder(encoder=encoder, diffusion=diffusion).to(device)
    return model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    model = build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    # eval 模式关闭 Dropout/BatchNorm 的训练行为；no_grad 节省显存、加速推理
    model.eval()

    # --- 数据：取前 num_samples 个测试样本 ---
    dataset = ShapeNetDataset(path=args.data_path, split="test")
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=False)
    x0 = next(iter(loader))["pointcloud"].to(device)  # (B, N, 3)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 推理：编码 → 重建 ---
    with torch.no_grad():
        mu, _ = model.encoder(x0)  # AutoEncoder 模式取 mu 作为 z
        recon = model.sample(  # 逆向扩散生成点云
            z=mu,
            num_points=args.num_points,
            flexibility=args.flexibility,
        )  # (B, N, 3)

        # chamfer_distance 期望 (B, N, 3)，返回 (B,)
        cd = chamfer_distance(x0, recon)  # (B,)

    # --- 保存每个样本的对比图 ---
    for i in range(x0.shape[0]):
        cd_val = cd[i].item()
        fig = plot_reconstruction(
            input_pc=x0[i],
            recon_pc=recon[i],
            cd_value=cd_val,
        )
        save_path = os.path.join(args.out_dir, f"sample_{i:02d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[{i}] CD={cd_val:.6f}  →  {save_path}")

    print(f"\n平均 CD: {cd.mean().item():.6f}")


if __name__ == "__main__":
    main()
