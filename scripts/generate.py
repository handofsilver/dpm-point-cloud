"""
生成模型多样性展示脚本（GaussianVAE / FlowVAE）

从先验分布采样 z，逆向扩散生成一批全新形状，拼成网格图保存。

运行方式：
    conda run -n dpm3d python scripts/generate.py \
        --ckpt checkpoints/gen/flow_epoch_2000.pt \
        --model flow \
        --num_samples 8 \
        --out_dir results/generate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt

from model import (
    VarianceSchedule,
    PointwiseNet,
    DiffusionPoint,
    PointNetEncoder,
    GaussianVAE,
    NormalizingFlow,
    FlowVAE,
)
from visualize import plot_point_cloud


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="生成模型 checkpoint 路径")
    parser.add_argument(
        "--model", type=str, default="flow", choices=["gaussian", "flow"], help="与训练时保持一致"
    )
    parser.add_argument("--num_samples", type=int, default=8, help="生成形状数量")
    parser.add_argument("--zdim", type=int, default=256)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--flexibility", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="results/generate")
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
        model = FlowVAE(encoder=encoder, diffusion=diffusion, flow=flow)
    else:
        model = GaussianVAE(encoder=encoder, diffusion=diffusion)

    return model.to(device)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    model = build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    # checkpoint 里的 args 覆盖命令行，保证结构与训练时完全一致
    if "args" in ckpt:
        saved = ckpt["args"]
        args.zdim = saved.get("zdim", args.zdim)
        args.T = saved.get("T", args.T)
        args.beta_T = saved.get("beta_T", args.beta_T)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- 从先验采样，生成点云 ---
    with torch.no_grad():
        samples = model.sample(
            batch_size=args.num_samples,
            num_points=args.num_points,
            flexibility=args.flexibility,
            device=device,
        )  # (B, N, 3)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 拼成网格图：每行 4 个 ---
    ncols = 4
    nrows = (args.num_samples + ncols - 1) // ncols  # 向上取整
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for i in range(args.num_samples):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_point_cloud(samples[i], title=f"sample {i}", ax=ax)  # type: ignore

    fig.suptitle(f"{args.model.upper()} — {args.num_samples} generated shapes")
    plt.tight_layout()

    save_path = os.path.join(args.out_dir, f"{args.model}_samples.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"网格图已保存至 {save_path}")


if __name__ == "__main__":
    main()
