"""
AutoEncoder 训练脚本
Paper: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021

超参（AutoEncoder 模式）：
    T=200, beta_T=0.05, zdim=256, lr=1e-3, grad_clip=10, epochs=2000
运行方式：
    conda run -n dpm3d python scripts/train_ae.py --data_path data/shapenet/shapenet.hdf5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from dataset import ShapeNetDataset
from model import VarianceSchedule, PointwiseNet, DiffusionPoint, PointNetEncoder, AutoEncoder

# =============================================================================
# 超参配置
# =============================================================================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="shapenet.hdf5 文件路径")
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints/ae", help="checkpoint 保存目录"
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--zdim", type=int, default=256)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--beta_T", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--save_freq", type=int, default=100, help="每隔多少 epoch 保存一次")
    parser.add_argument("--print_freq", type=int, default=100, help="每隔多少 step 打印一次 loss")
    return parser.parse_args()


# =============================================================================
# 模型构建
# =============================================================================


def build_model(args, device):
    """
    按依赖顺序组装所有子模块，返回 AutoEncoder 实例。

    依赖顺序：
        VarianceSchedule
            └── DiffusionPoint
                    └── PointwiseNet（需要 zdim）
        PointNetEncoder（需要 zdim）
            └── AutoEncoder
    """
    var_sched = VarianceSchedule(T=args.T, beta_T=args.beta_T)
    net = PointwiseNet(zdim=args.zdim, residual=True)
    diffusion = DiffusionPoint(net=net, var_sched=var_sched)
    encoder = PointNetEncoder(zdim=args.zdim)
    model = AutoEncoder(encoder=encoder, diffusion=diffusion).to(device)
    return model


# =============================================================================
# 训练主循环
# =============================================================================


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 数据 ---
    dataset = ShapeNetDataset(path=args.data_path, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # --- 模型 ---
    model = build_model(args, device)

    # --- 优化器：Adam，自适应学习率 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- LR Scheduler：从 lr 线性衰减到 0 ---
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=args.epochs,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # --- 训练循环 ---
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch in loader:
            x0 = batch["pointcloud"].to(device)  # (B, N, 3) → 训练设备

            # 五步训练固定套路
            optimizer.zero_grad()  # 1. 清空梯度
            loss = model.get_loss(x0)  # 2. 前向传播
            loss.backward()  # 3. 反向传播
            clip_grad_norm_(model.parameters(), args.grad_clip)  # 4. 梯度裁剪
            optimizer.step()  # 5. 更新参数

            global_step += 1

            if global_step % args.print_freq == 0:
                print(f"Epoch {epoch:04d} | Step {global_step:06d} | Loss {loss.item():.4f}")

        scheduler.step()  # 每 epoch 衰减学习率

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:04d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)
            print(f"Checkpoint 保存至 {ckpt_path}")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    args = get_args()
    train(args)
