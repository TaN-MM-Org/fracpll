"""0.6.0 capabilities: the second-order filter design for a target
bandwidth and phase margin, and the third-order (R3-C3) filter in the
averaged model.

Anchors: the design values are derived again inside the test by a
different route (the half-angle form of the phase maximum, and |L| = 1
solved with the textbook second-order impedance), and the designed
loop is read back by `stability`, a separate numerical search.  The
third-order transimpedance is held against the polynomial form derived
by nodal analysis, against a direct nodal-matrix solve of the circuit,
and against its R3 -> 0 limit; its loop stability against the roots of
the closed-loop characteristic polynomial."""
import math

import numpy as np
import pytest

from fracpll import (design_loop_filter, loop_filter_impedance, open_loop,
                     sampled_margins, second_order_impedance, stability,
                     third_order_impedance)

CASES = [  # f_c, PM, Icp, |Kvco|, N, f_ref
    (40e3, 55.0, 100e-6, 20e6, 40.0, 50e6),
    (1e3, 10.0, 5e-3, 50e6, 400.0, 10e6),
    (200e3, 70.0, 1e-3, 400e6, 100.0, 50e6),
    (2.5e6, 45.0, 0.5e-3, 400e6, 10.0, 50e6),
    (30e3, 89.0, 50e-6, 5e6, 25.5, 20e6),
]


def _independent_design(fc, pm, icp, kv, nd):
    """The same filter from the half-angle form of the phase maximum
    and |L| = 1 on the textbook impedance (no sec/tan formula)."""
    wc = 2 * math.pi * fc
    # phase of L: -180 + atan(w T2) - atan(w T1), maximal at
    # w = 1/sqrt(T1 T2); with x = wc T1 the margin is 90 - 2 atan(x)
    x = math.tan(math.radians(45.0 - pm / 2.0))
    t1 = x / wc
    t2 = 1.0 / (wc * x)
    # |L| = icp kv |1 + j wc T2| / (wc^2 Ctot N |1 + j wc T1|) = 1
    ctot = icp * kv * abs(1 + 1j * wc * t2) / (
        wc * wc * nd * abs(1 + 1j * wc * t1))
    c1 = ctot * t1 / t2          # T1/T2 = C1/(C1 + C2)
    c2 = ctot - c1
    return c1, t2 / c2, c2


@pytest.mark.parametrize("case", CASES)
def test_design_equals_independent_derivation(case):
    fc, pm, icp, kv, nd, f_ref = case
    d = design_loop_filter(fc, pm, icp, kv, nd, f_ref)
    c1, r2, c2 = _independent_design(fc, pm, icp, kv, nd)
    assert d["c1"] == pytest.approx(c1, rel=1e-12)
    assert d["r2"] == pytest.approx(r2, rel=1e-12)
    assert d["c2"] == pytest.approx(c2, rel=1e-12)
    assert d["c_shunt"] == d["c1"] and d["branches"] == [(d["r2"], d["c2"])]
    assert d["t1_s"] == pytest.approx(r2 * c1 * c2 / (c1 + c2), rel=1e-12)
    assert d["t2_s"] == pytest.approx(r2 * c2, rel=1e-12)
    # the loop gain on the textbook impedance: |L| = 1 and the phase
    # equals -180 + PM at f_c, and the phase is at its maximum there
    zf = lambda w: second_order_impedance(w, c1, r2, c2)
    wc = 2 * math.pi * fc
    lg = open_loop(wc * np.array([1 - 1e-4, 1.0, 1 + 1e-4]), icp, kv,
                   nd, zf)
    assert abs(lg[1]) == pytest.approx(1.0, rel=1e-12)
    assert 180 + np.degrees(np.angle(lg[1])) == pytest.approx(pm,
                                                              abs=1e-10)
    ph = np.degrees(np.angle(lg))
    assert abs(ph[2] - ph[0]) < 1e-6          # flat: the maximum
    assert ph[1] > ph[0] and ph[1] > ph[2]


@pytest.mark.parametrize("case", CASES)
def test_designed_loop_reads_back_with_stability(case):
    fc, pm, icp, kv, nd, f_ref = case
    d = design_loop_filter(fc, pm, icp, kv, nd, f_ref)
    zf = lambda w: loop_filter_impedance(w, d["c_shunt"], d["branches"])
    r = stability(icp, kv, nd, zf, f_ref)
    assert r["f_crossover_hz"] == pytest.approx(fc, rel=1e-9)
    assert r["phase_margin_deg"] == pytest.approx(pm, abs=1e-9)
    assert d["averaged"] == r
    m = sampled_margins(icp, kv, nd, f_ref, d["c_shunt"], d["branches"])
    for k in ("f_crossover_hz", "phase_margin_deg", "gain_margin_db"):
        assert d["per_cycle"][k] == m[k]


def test_per_cycle_check_reduces_to_the_target_for_a_slow_loop():
    """Far below the reference the exact per-cycle loop is the averaged
    one; near f_ref/10 its margin is visibly smaller (50.1 degrees for
    a 55 degree design at 4.99 MHz, as the README states)."""
    slow = design_loop_filter(1e3, 55.0, 100e-6, 20e6, 40.0, 50e6)
    assert slow["per_cycle"]["phase_margin_deg"] == pytest.approx(
        55.0, abs=1e-6)
    assert slow["per_cycle"]["f_crossover_hz"] == pytest.approx(
        1e3, rel=1e-6)
    fast = design_loop_filter(4.99e6, 55.0, 100e-6, 20e6, 40.0, 50e6)
    assert round(fast["per_cycle"]["phase_margin_deg"], 1) == 50.1
    assert fast["per_cycle"]["stable"]


def test_design_refusals():
    good = dict(f_c_hz=40e3, phase_margin_deg=55.0, icp=100e-6,
                kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=50e6)
    for key, bad in (("phase_margin_deg", 0.0), ("phase_margin_deg", 90.0),
                     ("phase_margin_deg", float("nan")),
                     ("f_c_hz", -1.0), ("icp", 0.0),
                     ("kvco_hz_per_v", -20e6), ("n_div", 0.5),
                     ("f_ref_hz", float("inf"))):
        with pytest.raises(ValueError):
            design_loop_filter(**{**good, key: bad})
    with pytest.raises(ValueError, match="Gardner"):
        design_loop_filter(**{**good, "f_c_hz": 5e6})


# ---------------------------------------------------------------------
# third-order filter
C1, R2, C2, R3, C3 = 100e-12, 4.7e3, 1.5e-9, 10e3, 10e-12
W = 2 * np.pi * np.geomspace(1e1, 1e9, 400)


def _poly_form(w, c1, r2, c2, r3, c3):
    s = 1j * w
    a0 = c1 + c2 + c3
    a1 = r2 * c2 * (c1 + c3) + r3 * c3 * (c1 + c2)
    a2 = c1 * c2 * c3 * r2 * r3
    return (1 + s * r2 * c2) / (s * (a0 + a1 * s + a2 * s * s))


def _nodal(w, c1, r2, c2, r3, c3):
    """Nodes: 1 pump, 2 between R2 and C2, 3 at C3.  Solve Y v = e1."""
    out = np.empty(w.size, dtype=complex)
    for k, wk in enumerate(w):
        s = 1j * wk
        g2, g3 = 1 / r2, 1 / r3
        y = np.array([[s * c1 + g2 + g3, -g2, -g3],
                      [-g2, g2 + s * c2, 0.0],
                      [-g3, 0.0, g3 + s * c3]], dtype=complex)
        out[k] = np.linalg.solve(y, np.array([1.0, 0.0, 0.0]))[2]
    return out


def test_third_order_equals_polynomial_and_nodal_solve():
    z = third_order_impedance(W, C1, R2, C2, R3, C3)
    assert np.allclose(z, _poly_form(W, C1, R2, C2, R3, C3), rtol=1e-11,
                       atol=0)
    assert np.allclose(z, _nodal(W, C1, R2, C2, R3, C3), rtol=1e-9,
                       atol=0)


def test_third_order_r3_to_zero_is_second_order_with_c1_plus_c3():
    z = third_order_impedance(W, C1, R2, C2, 0.0, C3)
    assert np.allclose(z, second_order_impedance(W, C1 + C3, R2, C2),
                       rtol=1e-12, atol=0)


def test_third_order_loop_margin_and_poles():
    """`stability` on the third-order loop: |L| = 1 and the margin from
    the polynomial form at the crossover, and all closed-loop poles in
    the left half plane (roots of N s^2 A(s) + K (1 + s T2))."""
    icp, kv, nd = 100e-6, 20e6, 40.0
    zf = lambda w: third_order_impedance(w, C1, R2, C2, R3, C3)
    r = stability(icp, kv, nd, zf, 50e6)
    wc = 2 * np.pi * r["f_crossover_hz"]
    lg = icp * kv * _poly_form(np.array([wc]), C1, R2, C2, R3, C3)[0] / (
        1j * wc * nd)
    assert abs(lg) == pytest.approx(1.0, rel=1e-12)
    assert r["phase_margin_deg"] == pytest.approx(
        180 + np.degrees(np.angle(lg)), abs=1e-9)
    # the R3-C3 pole (1.6 MHz) costs phase: below the second-order 54.2
    zf2 = lambda w: loop_filter_impedance(w, C1 + C3, [(R2, C2)])
    assert r["phase_margin_deg"] < stability(icp, kv, nd, zf2,
                                             50e6)["phase_margin_deg"]
    roots = _char_roots(icp * kv, nd, C1, R2, C2, R3, C3)
    assert np.all(roots.real < 0)


def _char_roots(k, nd, c1, r2, c2, r3, c3):
    a0 = c1 + c2 + c3
    a1 = r2 * c2 * (c1 + c3) + r3 * c3 * (c1 + c2)
    a2 = c1 * c2 * c3 * r2 * r3
    # N s^2 (a2 s^2 + a1 s + a0) + k (1 + s r2 c2) = 0
    return np.roots([nd * a2, nd * a1, nd * a0, k * r2 * c2, k])


def test_third_order_refusals():
    with pytest.raises(ValueError, match="c3"):
        third_order_impedance(W, C1, R2, C2, R3, 0.0)
    with pytest.raises(ValueError, match="r3"):
        third_order_impedance(W, C1, R2, C2, -1.0, C3)
    with pytest.raises(ValueError, match="r2"):
        third_order_impedance(W, C1, float("nan"), C2, R3, C3)
    with pytest.raises(ValueError, match="positive"):
        third_order_impedance(np.array([0.0]), C1, R2, C2, R3, C3)
