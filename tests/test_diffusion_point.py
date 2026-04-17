"""
DiffusionPoint 验证脚本
运行方式: conda run -n dpm3d python tests/test_diffusion_point.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from model import VarianceSchedule, PointwiseNet, DiffusionPoint


def _build(T=100, beta_T=0.02, zdim=64):
    var_sched = VarianceSchedule(T=T, beta_T=beta_T)
    net = PointwiseNet(zdim=zdim, residual=True)
    return DiffusionPoint(net=net, var_sched=var_sched)


def test_get_loss_shape():
    """get_loss 应返回标量（零维 Tensor）"""
    B, N, zdim = 4, 128, 64
    diffusion = _build(zdim=zdim)

    x0 = torch.randn(B, N, 3)
    z = torch.randn(B, zdim)

    loss = diffusion.get_loss(x0, z)

    print(f"loss shape : {list(loss.shape)}  (期望 []，即标量)")
    print(f"loss value : {loss.item():.4f}   (期望 > 0)")

    assert loss.shape == torch.Size([]), f"loss 应为标量，实际 shape: {loss.shape}"
    assert loss.item() > 0, "MSE loss 应为正数"
    print("  ✓ 通过\n")


def test_get_loss_has_gradient():
    """loss 应能反向传播（梯度可流回网络参数）"""
    B, N, zdim = 2, 64, 64
    diffusion = _build(zdim=zdim)

    x0 = torch.randn(B, N, 3)
    z = torch.randn(B, zdim)

    loss = diffusion.get_loss(x0, z)
    loss.backward()

    # 随机检查一个参数的梯度是否非空且非全零
    param = next(diffusion.net.parameters())
    has_grad = param.grad is not None and param.grad.abs().sum().item() > 0

    print(f"网络参数有梯度: {has_grad}  (期望 True)")
    assert has_grad, "反向传播后网络参数应有非零梯度"
    print("  ✓ 通过\n")


def test_sample_shape():
    """sample 应返回 (B, N, 3) 的点云"""
    B, N, zdim = 3, 128, 64
    diffusion = _build(T=10, zdim=zdim)  # T=10 加速测试

    z = torch.randn(B, zdim)

    with torch.no_grad():
        x = diffusion.sample(z, num_points=N, flexibility=0.0)

    print(f"sample 输出 shape: {list(x.shape)}  (期望 [{B}, {N}, 3])")
    assert x.shape == (B, N, 3), f"sample 形状错误: {x.shape}"
    print("  ✓ 通过\n")


def test_sample_flexibility_bounds():
    """flexibility=0 和 1 都应能正常运行，输出形状相同"""
    B, N, zdim = 2, 64, 64
    diffusion = _build(T=5, zdim=zdim)
    z = torch.randn(B, zdim)

    with torch.no_grad():
        x0 = diffusion.sample(z, num_points=N, flexibility=0.0)
        x1 = diffusion.sample(z, num_points=N, flexibility=1.0)

    print(f"flexibility=0 shape: {list(x0.shape)}")
    print(f"flexibility=1 shape: {list(x1.shape)}")
    assert x0.shape == x1.shape == (B, N, 3)

    # 两种方差设置下结果不同（flexibility 确实在起作用）
    are_different = not torch.allclose(x0, x1)
    print(f"两种 flexibility 结果不同: {are_different}  (期望 True)")
    assert are_different, "不同 flexibility 应产生不同样本"
    print("  ✓ 通过\n")


def test_noisy_x_not_equal_x0():
    """前向加噪后的 x_noisy 应与原始 x0 不同（加噪确实发生了）"""
    B, N, zdim = 2, 64, 64
    T = 50
    var_sched = VarianceSchedule(T=T, beta_T=0.02)

    x0 = torch.randn(B, N, 3)
    t = torch.randint(1, T + 1, (B,))
    alpha_bar = var_sched.alpha_bars[t - 1].view(B, 1, 1)
    eps = torch.randn_like(x0)
    x_noisy = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps

    are_different = not torch.allclose(x_noisy, x0)
    print(f"加噪后 x_noisy ≠ x0: {are_different}  (期望 True)")
    assert are_different, "加噪后点云应与原始点云不同"
    print("  ✓ 通过\n")


if __name__ == "__main__":
    print("=" * 55)
    print("DiffusionPoint 验证")
    print("=" * 55 + "\n")

    test_get_loss_shape()
    test_get_loss_has_gradient()
    test_sample_shape()
    test_sample_flexibility_bounds()
    test_noisy_x_not_equal_x0()

    print("=" * 55)
    print("全部通过 ✓")
    print("=" * 55)
