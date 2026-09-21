"""Exact MASH line-spectrum anchors: the first-order sawtooth closed
form, Parseval, the exact period, and -- as a third independent path
to the additive-noise formula -- band-by-band agreement of the exact
lines with `dsm_phase_psd` for long periods."""
from math import gcd

import numpy as np
import pytest

from fracpll import (closed_loop_lines, dsm_phase_psd, loop_filter_impedance,
                     mash_line_spectrum, mash_period, mash_sequence,
                     open_loop)


@pytest.mark.parametrize("a,P", [(3, 8), (5, 16), (1, 12)])
def test_first_order_lines_equal_sawtooth_closed_form(a, P):
    """MASH-1 phase (cycles) is -{k a/P}: a permutation of j/P, whose
    DFT is exact: |c_n| = 1/(2 P sin(pi n'/P)), n' = n a^-1 mod P."""
    r = mash_line_spectrum(a, P, 1, 1.0)
    assert r["period"] == P
    k = np.arange(1, P // 2 + 1)
    kk = (k * pow(a, -1, P)) % P
    amp = 2 * np.pi / (2 * P * np.sin(np.pi * kk / P))
    closed = 2 * amp ** 2
    if P % 2 == 0:
        closed[-1] = amp[-1] ** 2
    assert np.allclose(r["line_rad2"], closed, rtol=1e-12)


def test_parseval_and_periods():
    for num, den, m in [(13, 64, 2), (13, 64, 3), (6, 16, 1)]:
        r = mash_line_spectrum(num, den, m, 1.0)
        assert np.isclose(r["line_rad2"].sum(), r["variance_rad2"],
                          rtol=1e-12)
    assert mash_period(6, 16, 1) == 16 // gcd(6, 16)
    # the output is periodic with the state period after start-up
    P = mash_period(13, 64, 3)
    d = mash_sequence(13, 64, 3, 3 * P + 2)
    assert np.array_equal(d[2:2 + P], d[2 + P:2 + 2 * P])
    with pytest.raises(ValueError, match="max_period"):
        mash_period(104857, 1 << 20, 3, max_period=1000)


@pytest.mark.parametrize("m", [2, 3])
def test_exact_lines_confirm_additive_formula_for_long_periods(m):
    r = mash_line_spectrum(6553, 1 << 16, m, 1.0)
    P = r["period"]
    psd = r["line_rad2"] * P          # lines per unit frequency
    f = r["f_hz"]
    edges = np.geomspace(0.02, 0.45, 8)
    for a, b in zip(edges[:-1], edges[1:]):
        s = (f >= a) & (f < b)
        ratio = psd[s].mean() / dsm_phase_psd(f[s], 1.0, m).mean()
        assert abs(ratio - 1.0) < 0.1


def test_first_order_now_answered_not_refused():
    with pytest.raises(ValueError, match="mash_line_spectrum"):
        dsm_phase_psd(np.array([0.1]), 1.0, 1)
    r = mash_line_spectrum(1, 10, 1, 50e6)
    assert np.isclose(r["f_hz"][0], 5e6)        # f_ref / period


def test_closed_loop_lines():
    r = mash_line_spectrum(3, 64, 1, 50e6)
    zf = lambda w: loop_filter_impedance(w, 100e-12, [(4.7e3, 1.5e-9)])
    lg = open_loop(2 * np.pi * r["f_hz"], 100e-6, 20e6, 40.0, zf)
    out = closed_loop_lines(r["f_hz"], r["line_rad2"], lg)
    hl2 = np.abs(lg / (1 + lg)) ** 2
    assert np.allclose(out["line_rad2"], hl2 * r["line_rad2"], rtol=1e-12)
    small = hl2 * r["line_rad2"] <= 0.02
    assert small.any()
    assert np.allclose(out["sideband_dbc"][small],
                       10 * np.log10(hl2 * r["line_rad2"] / 2)[small],
                       rtol=1e-12)
    assert np.all(np.isnan(out["sideband_dbc"][~small]))
    # open-loop MASH-1 divider lines are large-angle: no dBc claimed
    big = r["line_rad2"] > 0.02
    assert big.any() and np.all(np.isnan(r["sideband_dbc"][big]))
