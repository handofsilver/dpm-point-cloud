"""
生成模型训练脚本（GaussianVAE / FlowVAE）
Paper: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021

超参（GaussianVAE / FlowVAE 共用）：
    T=100, beta_T=0.02, zdim=256, lr=2e-3, kl_weight=0.001, grad_clip=10, epochs=2000
运行方式：
    conda run -n dpm3d python scripts/train_gen.py --data_path data/shapenet/shapenet.hdf5 --model gaussian
    conda run -n dpm3d python scripts/train_gen.py --data_path data/shapenet/shapenet.hdf5 --model flow
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

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

# =============================================================================
# 超参配置
# =============================================================================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="shapenet.hdf5 文件路径")
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints/gen", help="checkpoint 保存目录"
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--zdim", type=int, default=256)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--save_freq", type=int, default=100, help="每隔多少 epoch 保存一次")
    parser.add_argument("--print_freq", type=int, default=100, help="每隔多少 step 打印一次 loss")
    parser.add_argument(
        "--model",
        type=str,
        default="gaussian",
        choices=["gaussian", "flow"],
        help="生成模型类型：gaussian=GaussianVAE，flow=FlowVAE",
    )
    parser.add_argument("--flow_layers", type=int, default=4, help="FlowVAE 的 Flow 层数")
    parser.add_argument("--flow_hidden_dim", type=int, default=128, help="Flow s/t 网络隐层宽度")
    # --- per-category 训练（对齐 paper Table 1 GEN_*.pt 协议）---
    parser.add_argument(
        "--cates",
        type=str,
        nargs="+",
        default=None,
        help="类别列表（如 airplane chair），默认 None=全 55 类",
    )
    # --- LR schedule：plateau → 线性衰减 → 保持 end_lr（对齐 paper sched_start/sched_end/end_lr）---
    parser.add_argument("--end_lr", type=float, default=1e-4, help="LR 衰减终点")
    parser.add_argument(
        "--sched_start_epoch",
        type=int,
        default=None,
        help="plateau 结束的 epoch；默认 epochs // 2",
    )
    parser.add_argument(
        "--sched_end_epoch",
        type=int,
        default=None,
        help="线性衰减结束的 epoch；默认 epochs（之后保持 end_lr）",
    )
    return parser.parse_args()


# =============================================================================
# 模型构建
# =============================================================================


def build_model(args, device):
    """
    按依赖顺序组装所有子模块，返回 GaussianVAE 或 FlowVAE 实例。

    共享子模块：
        VarianceSchedule → DiffusionPoint（含 PointwiseNet）
        PointNetEncoder

    FlowVAE 额外需要：
        NormalizingFlow（K 层 AffineCouplingLayer）
    """
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


# =============================================================================
# 训练主循环
# =============================================================================


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} | 模型: {args.model}")

    # --- 数据 ---
    dataset = ShapeNetDataset(path=args.data_path, split="train", cates=args.cates)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"训练数据: {len(dataset)} shapes | cates={args.cates or 'all 55'}")

    # --- 模型 ---
    model = build_model(args, device)

    # --- 优化器 & LR Scheduler ---
    # 三段式 schedule（paper 协议）：
    #   [0, sched_start)       plateau at args.lr
    #   [sched_start, sched_end)  线性衰减 args.lr → args.end_lr
    #   [sched_end, epochs]    保持 args.end_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched_start = args.sched_start_epoch if args.sched_start_epoch is not None else args.epochs // 2
    sched_end = args.sched_end_epoch if args.sched_end_epoch is not None else args.epochs
    end_factor = args.end_lr / args.lr

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < sched_start:
            return 1.0
        if epoch_idx < sched_end:
            progress = (epoch_idx - sched_start) / max(1, sched_end - sched_start)
            return 1.0 - progress * (1.0 - end_factor)
        return end_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(
        f"LR schedule: plateau [0, {sched_start}) lr={args.lr:.2e}, "
        f"decay [{sched_start}, {sched_end}) → {args.end_lr:.2e}, then hold"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # --- 训练循环 ---
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch in loader:
            x0 = batch["pointcloud"].to(device)  # (B, N, 3)

            optimizer.zero_grad()

            loss = model.get_loss(x0, kl_weight=args.kl_weight)

            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1

            if global_step % args.print_freq == 0:
                print(f"Epoch {epoch:04d} | Step {global_step:06d} | Loss {loss.item():.4f}")

        scheduler.step()

        if epoch % args.save_freq == 0:
            cates_tag = "_".join(args.cates) if args.cates else None
            fname = (
                f"{args.model}_{cates_tag}_epoch_{epoch:04d}.pt"
                if cates_tag
                else f"{args.model}_epoch_{epoch:04d}.pt"
            )
            ckpt_path = os.path.join(args.save_dir, fname)
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"Checkpoint 保存至 {ckpt_path}")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    args = get_args()
    train(args)
