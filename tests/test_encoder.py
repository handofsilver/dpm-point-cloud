"""
PointNetEncoder 验证脚本
运行方式: conda run -n dpm3d python tests/test_encoder.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from model import PointNetEncoder


def test_output_shape():
    """mu 和 log_var 的形状应均为 (B, zdim)"""
    B, N, zdim = 4, 2048, 256

    enc = PointNetEncoder(zdim=zdim)
    x = torch.randn(B, N, 3)

    mu, log_var = enc(x)

    print(f"输入 x   : {list(x.shape)}")
    print(f"mu       : {list(mu.shape)}       (期望 [{B}, {zdim}])")
    print(f"log_var  : {list(log_var.shape)}   (期望 [{B}, {zdim}])")

    assert mu.shape == (B, zdim), f"mu 形状错误: {mu.shape}"
    assert log_var.shape == (B, zdim), f"log_var 形状错误: {log_var.shape}"
    print("  ✓ 通过\n")


def test_permutation_invariance():
    """
    置换不变性：打乱点的顺序，编码结果不变。
    这是 MaxPool 保证的核心性质。
    """
    B, N, zdim = 2, 512, 256

    enc = PointNetEncoder(zdim=zdim)
    enc.eval()

    x = torch.randn(B, N, 3)

    # 对每个 batch 独立生成随机排列索引
    perm = torch.randperm(N)
    x_perm = x[:, perm, :]  # 打乱 N 个点的顺序

    with torch.no_grad():
        mu, log_var = enc(x)
        mu_perm, log_var_perm = enc(x_perm)

    mu_ok = torch.allclose(mu, mu_perm, atol=1e-5)
    lv_ok = torch.allclose(log_var, log_var_perm, atol=1e-5)

    print(f"打乱点顺序后 mu 不变     : {mu_ok}    (期望 True)")
    print(f"打乱点顺序后 log_var 不变: {lv_ok}   (期望 True)")

    assert mu_ok, "mu 应对点的排列不变"
    assert lv_ok, "log_var 应对点的排列不变"
    print("  ✓ 通过\n")


def test_different_inputs_give_different_outputs():
    """不同的点云应产生不同的 latent"""
    B, N, zdim = 2, 256, 256

    enc = PointNetEncoder(zdim=zdim)
    enc.eval()

    x1 = torch.randn(B, N, 3)
    x2 = torch.randn(B, N, 3)

    with torch.no_grad():
        mu1, _ = enc(x1)
        mu2, _ = enc(x2)

    are_different = not torch.allclose(mu1, mu2)
    print(f"不同输入 → 不同 mu: {are_different}  (期望 True)")

    assert are_different, "不同点云应编码到不同的 latent"
    print("  ✓ 通过\n")


def test_dual_heads_are_independent():
    """
    mu 和 log_var 应由独立的 FC 头产生，输出值不同。
    如果两个头权重相同（或被错误共享），输出会完全一致。
    """
    B, N, zdim = 2, 256, 256

    enc = PointNetEncoder(zdim=zdim)
    enc.eval()

    x = torch.randn(B, N, 3)

    with torch.no_grad():
        mu, log_var = enc(x)

    heads_differ = not torch.allclose(mu, log_var)
    print(f"mu ≠ log_var（双头独立）: {heads_differ}  (期望 True)")

    assert heads_differ, "mu 和 log_var 应由独立的 FC 头产生"
    print("  ✓ 通过\n")


if __name__ == "__main__":
    print("=" * 55)
    print("PointNetEncoder 验证")
    print("=" * 55 + "\n")

    test_output_shape()
    test_permutation_invariance()
    test_different_inputs_give_different_outputs()
    test_dual_heads_are_independent()

    print("=" * 55)
    print("全部通过 ✓")
    print("=" * 55)
