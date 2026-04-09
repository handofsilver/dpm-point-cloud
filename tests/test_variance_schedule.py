"""
VarianceSchedule 验证脚本
运行方式: conda run -n dpm3d python tests/test_variance_schedule.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model import VarianceSchedule


def test_alpha_bar_trend():
    """alpha_bar 应从接近 1 单调递减到接近 0"""
    vs = VarianceSchedule(T=200, beta_T=0.05)

    ab_first = vs.alpha_bars[0].item()
    ab_last  = vs.alpha_bars[-1].item()

    print(f"alpha_bar[t=1]  : {ab_first:.6f}  (期望接近 1)")
    print(f"alpha_bar[t=T]  : {ab_last:.6f}  (期望接近 0)")

    assert ab_first > 0.99,  f"alpha_bar[1] 应接近 1，实际={ab_first}"
    assert ab_last  < 0.05,  f"alpha_bar[T] 应接近 0，实际={ab_last}"
    assert (vs.alpha_bars.diff() <= 0).all(), "alpha_bar 应单调不增"

    print("  ✓ 通过\n")


def test_eq13_boundary():
    """验证 Eq.(13) 两端：t=T 时应几乎是纯噪声"""
    vs = VarianceSchedule(T=200, beta_T=0.05)
    ab_T = vs.alpha_bars[-1]

    signal_coef = ab_T.sqrt().item()
    noise_coef  = (1 - ab_T).sqrt().item()

    print(f"sqrt(alpha_bar_T)   : {signal_coef:.6f}  (期望接近 0，信号几乎消失)")
    print(f"sqrt(1-alpha_bar_T) : {noise_coef:.6f}  (期望接近 1，几乎纯噪声)")

    assert signal_coef < 0.15, f"t=T 时信号系数应接近 0，实际={signal_coef}"
    assert noise_coef  > 0.98, f"t=T 时噪声系数应接近 1，实际={noise_coef}"

    print("  ✓ 通过\n")


def test_sigmas_ordering():
    """宽方差 >= 窄方差（数学性质，必须恒成立）"""
    vs = VarianceSchedule(T=200, beta_T=0.05)

    all_ge = (vs.sigmas_flex >= vs.sigmas_inflex).all().item()
    print(f"flex >= inflex everywhere: {all_ge}  (期望 True)")

    assert all_ge, "sigmas_flex 应在每一步都 >= sigmas_inflex"
    print("  ✓ 通过\n")


def test_uniform_sample_t():
    """uniform_sample_t 应在 [1, T] 内均匀采样"""
    vs = VarianceSchedule(T=200, beta_T=0.05)
    t  = vs.uniform_sample_t(batch_size=10000)

    t_min, t_max = t.min().item(), t.max().item()
    print(f"t 采样范围: [{t_min}, {t_max}]  (期望 [1, 200])")

    assert t_min >= 1,   f"t 最小值应 >= 1，实际={t_min}"
    assert t_max <= 200, f"t 最大值应 <= 200，实际={t_max}"
    assert t_min == 1 and t_max == 200, "10000 次采样应覆盖全范围"

    print("  ✓ 通过\n")


def test_get_sigmas_interpolation():
    """get_sigmas(flex=0.5) 应等于两端的平均值"""
    vs     = VarianceSchedule(T=200, beta_T=0.05)
    t_test = torch.tensor([100])

    s_flex   = vs.get_sigmas(t_test, flexibility=1.0).item()
    s_inflex = vs.get_sigmas(t_test, flexibility=0.0).item()
    s_mid    = vs.get_sigmas(t_test, flexibility=0.5).item()
    expected = (s_flex + s_inflex) / 2

    print(f"t=100, flex=1.0 : {s_flex:.6f}")
    print(f"t=100, flex=0.0 : {s_inflex:.6f}")
    print(f"t=100, flex=0.5 : {s_mid:.6f}  (期望 {expected:.6f})")

    assert abs(s_mid - expected) < 1e-6, f"插值结果不符，期望={expected}，实际={s_mid}"
    print("  ✓ 通过\n")


if __name__ == '__main__':
    print("=" * 55)
    print("VarianceSchedule 验证")
    print("=" * 55 + "\n")

    test_alpha_bar_trend()
    test_eq13_boundary()
    test_sigmas_ordering()
    test_uniform_sample_t()
    test_get_sigmas_interpolation()

    print("=" * 55)
    print("全部通过 ✓")
    print("=" * 55)
