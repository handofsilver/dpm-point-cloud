"""
PointwiseNet 验证脚本
运行方式: conda run -n dpm3d python tests/test_pointwise_net.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from model import PointwiseNet


def test_output_shape():
    """输出形状应与输入点云相同：(B, N, 3)"""
    B, N, zdim = 4, 2048, 256

    net = PointwiseNet(zdim=zdim, residual=True)
    x = torch.randn(B, N, 3)
    beta = torch.rand(B)  # β_t ∈ (0, 1)，模拟随机时间步
    z = torch.randn(B, zdim)

    out = net(x, beta, z)

    print(f"输入 x    : {list(x.shape)}")
    print(f"时间 beta : {list(beta.shape)}")
    print(f"latent z  : {list(z.shape)}")
    print(f"输出 out  : {list(out.shape)}  (期望 [{B}, {N}, 3])")

    assert out.shape == (B, N, 3), f"形状错误: {out.shape}"
    print("  ✓ 通过\n")


def test_residual_vs_no_residual():
    """残差连接开关应影响输出值，但不影响形状"""
    B, N, zdim = 2, 100, 256
    x = torch.randn(B, N, 3)
    beta = torch.rand(B)
    z = torch.randn(B, zdim)

    net_res = PointwiseNet(zdim=zdim, residual=True)
    net_nores = PointwiseNet(zdim=zdim, residual=False)

    # 共享权重以便对比（把 no-res 的权重复制给 res 版本的网络）
    net_res.load_state_dict(net_nores.state_dict())

    out_res = net_res(x, beta, z)
    out_nores = net_nores(x, beta, z)

    shape_ok = out_res.shape == out_nores.shape == (B, N, 3)
    values_diff = not torch.allclose(out_res, out_nores)

    print(f"形状相同: {shape_ok}      (期望 True)")
    print(f"值不同  : {values_diff}   (期望 True，残差会改变输出值)")
    print(f"差值范数: {(out_res - out_nores).norm().item():.4f}  (≈ x 的范数，即 x 被加进去的效果)")

    assert shape_ok, "形状应相同"
    assert values_diff, "残差开关应改变输出值"
    print("  ✓ 通过\n")


def test_different_beta_gives_different_output():
    """不同的时间步 beta 应产生不同的噪声预测"""
    B, N, zdim = 2, 100, 256
    net = PointwiseNet(zdim=zdim)
    x = torch.randn(B, N, 3)
    z = torch.randn(B, zdim)

    beta_early = torch.full((B,), 1e-4)  # t 很小，接近干净
    beta_late = torch.full((B,), 0.05)  # t 很大，接近纯噪声

    out_early = net(x, beta_early, z)
    out_late = net(x, beta_late, z)

    are_different = not torch.allclose(out_early, out_late)
    print(f"beta_early vs beta_late 输出不同: {are_different}  (期望 True)")

    assert are_different, "不同 beta 应给出不同预测"
    print("  ✓ 通过\n")


if __name__ == "__main__":
    print("=" * 55)
    print("PointwiseNet 验证")
    print("=" * 55 + "\n")

    test_output_shape()
    test_residual_vs_no_residual()
    test_different_beta_gives_different_output()

    print("=" * 55)
    print("全部通过 ✓")
    print("=" * 55)
