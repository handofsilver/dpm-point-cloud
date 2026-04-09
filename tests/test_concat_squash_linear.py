"""
ConcatSquashLinear 验证脚本
运行方式: conda run -n dpm3d python tests/test_concat_squash_linear.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model import ConcatSquashLinear


def test_output_shape():
    """输出形状应为 (B, N, out_dim)，ctx 的 N=1 正确广播"""
    B, N        = 4, 2048
    in_dim      = 3
    out_dim     = 128
    ctx_dim     = 259   # 256（shape latent）+ 3（时间嵌入）

    layer = ConcatSquashLinear(in_dim, out_dim, ctx_dim)
    x     = torch.randn(B, N, in_dim)
    ctx   = torch.randn(B, 1, ctx_dim)   # N=1，会广播到 2048

    out = layer(x, ctx)

    print(f"输入 x   : {list(x.shape)}")
    print(f"条件 ctx : {list(ctx.shape)}  (N=1，广播到 {N})")
    print(f"输出 out : {list(out.shape)}  (期望 [{B}, {N}, {out_dim}])")

    assert out.shape == (B, N, out_dim), f"形状错误: {out.shape}"
    print("  ✓ 通过\n")


def test_ctx_affects_output():
    """不同的 ctx 应产生不同的输出（条件确实起到了作用）"""
    layer = ConcatSquashLinear(3, 64, 32)
    x     = torch.randn(2, 100, 3)

    ctx_a = torch.randn(2, 1, 32)
    ctx_b = torch.randn(2, 1, 32)   # 不同条件

    out_a = layer(x, ctx_a)
    out_b = layer(x, ctx_b)

    # 相同的 x，不同的 ctx，输出应该不同
    are_different = not torch.allclose(out_a, out_b)
    print(f"不同 ctx → 输出不同: {are_different}  (期望 True)")

    assert are_different, "不同 ctx 应产生不同输出，但结果相同"
    print("  ✓ 通过\n")


def test_gate_is_between_0_and_1():
    """门控值应在 (0, 1) 之间（sigmoid 输出范围）"""
    layer = ConcatSquashLinear(3, 64, 32)
    ctx   = torch.randn(2, 1, 32)

    gate = torch.sigmoid(layer._hyper_gate(ctx))
    print(f"gate 最小值: {gate.min().item():.4f}  (期望 > 0)")
    print(f"gate 最大值: {gate.max().item():.4f}  (期望 < 1)")

    assert (gate > 0).all() and (gate < 1).all(), "gate 应严格在 (0, 1) 内"
    print("  ✓ 通过\n")


if __name__ == '__main__':
    print("=" * 55)
    print("ConcatSquashLinear 验证")
    print("=" * 55 + "\n")

    test_output_shape()
    test_ctx_affects_output()
    test_gate_is_between_0_and_1()

    print("=" * 55)
    print("全部通过 ✓")
    print("=" * 55)
