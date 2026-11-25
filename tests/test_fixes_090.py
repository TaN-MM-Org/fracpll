"""Regression tests for the 0.9.0 fixes.  Each fails with 0.8.0.

* `stability` read the loop phase at the crossover alone, modulo 360
  degrees.  A loop whose phase had fallen past -360 degrees there came
  out with a large positive margin.  Case: the filter of README
  example 1 followed by four extra real poles at 8 kHz (for example
  four buffered R-C sections).  0.8.0 returned a crossover of
  10658.85 Hz and a phase margin of 171.14 degrees, "stable"; the
  phase there is -368.86 degrees, the margin -188.86, and the closed
  loop has poles in the right half plane.  The phase is now followed
  continuously along the scan.
* Whole numbers went through int() in places the 0.8.0 fix did not
  reach: `n_grid` of `stability`, `closed_loop_response` and
  `sampled_margins`, `n_eval` of `lock_transient`, `n_mc` of
  `margin_uncertainty` (2.9 ran 2 draws), and the `columns` of the
  file readers (columns=(0.7, 1.9) read columns 0 and 1).  A fraction
  is now refused; 4001.0, a NumPy integer or "4001" still work."""
import math

import numpy as np
import pytest

from fracpll import (closed_loop_response, lock_transient,
                     loop_filter_impedance, margin_uncertainty,
                     read_noise_csv, fit_tuning, sampled_margins,
                     stability)

FR = 50e6
ICP, KV, N = 100e-6, 20e6, 40.0
C1, R2, C2 = 100e-12, 4.7e3, 1.5e-9
WP = 2 * math.pi * 8e3


def _z2(w):
    return loop_filter_impedance(w, C1, [(R2, C2)])


def _z_extra_poles(w):
    return _z2(w) / (1.0 + 1j * np.asarray(w) / WP) ** 4


def test_phase_below_minus_360_is_not_read_as_a_margin():
    with pytest.raises(ValueError, match="unstable") as err:
        stability(ICP, KV, N, _z_extra_poles, FR)
    msg = str(err.value)
    pm = float(msg.split("phase margin ")[1].split(" deg")[0])
    fc = float(msg.split("crossover ")[1].split(" Hz")[0])
    # the margin in closed form, with no wrapping at all:
    # 180 + phase = atan(w T2) - atan(w T1) - 4 atan(w / wp)
    w = 2 * math.pi * fc
    t2 = R2 * C2
    t1 = R2 * C1 * C2 / (C1 + C2)
    exact = math.degrees(math.atan(w * t2) - math.atan(w * t1)
                         - 4 * math.atan(w / WP))
    assert abs(pm - exact) < 0.01          # pm printed to 2 decimals
    assert pm < -180.0
    # and the loop really is unstable: roots of the closed loop,
    # s^2 N (C1 + C2)(1 + s T1)(1 + s/wp)^4 + Icp Kv (1 + s T2) = 0
    p = np.poly1d([N * (C1 + C2), 0.0, 0.0]) * np.poly1d([t1, 1.0]) \
        * np.poly1d([1.0 / WP, 1.0]) ** 4
    p = p + ICP * KV * np.poly1d([t2, 1.0])
    assert np.max(np.roots(p.coeffs).real) > 0.0
    # 0.8.0's rule, 180 + the principal angle, gave about +171 degrees
    from fracpll import open_loop
    lg = open_loop(w, ICP, KV, N, _z_extra_poles)[0]
    assert 170.0 < 180.0 + math.degrees(np.angle(lg)) < 172.0


def test_unchanged_where_the_old_rule_was_right():
    """For loop_filter_impedance filters and the stable third-order
    filters the margin equals 0.8.0's rule, 180 + the principal angle
    of L at the crossover, bit for bit (57 of 60 random loops are
    stable and inside the model: 28 second-order and 29 third-order
    filters; at least 40 are asked for)."""
    from fracpll import open_loop, third_order_impedance
    rng = np.random.default_rng(90)
    n = 0
    for _ in range(60):
        c1 = 10 ** rng.uniform(-11, -9)
        c2 = c1 * 10 ** rng.uniform(0.5, 1.5)
        r2 = 10 ** rng.uniform(3, 4.5)
        icp = 10 ** rng.uniform(-5, -3)
        if rng.uniform() < 0.5:
            zf = lambda w, c1=c1, r2=r2, c2=c2: loop_filter_impedance(
                w, c1, [(r2, c2)])
        else:
            zf = lambda w, c1=c1, r2=r2, c2=c2: third_order_impedance(
                w, c1, r2, c2, 3 * r2, c1 / 20)
        try:
            rep = stability(icp, KV, N, zf, FR)
        except ValueError:
            continue
        lg = open_loop(2 * math.pi * rep["f_crossover_hz"], icp, KV, N,
                       zf)[0]
        old = 180.0 + float(np.degrees(np.angle(lg)))
        if old > 180.0:
            old -= 360.0
        assert rep["phase_margin_deg"] == old
        n += 1
    assert n >= 40


@pytest.mark.parametrize("call", [
    lambda: stability(ICP, KV, N, _z2, FR, n_grid=4000.5),
    lambda: closed_loop_response(ICP, KV, N, _z2, FR, n_grid=4000.5),
    lambda: sampled_margins(ICP, KV, N, FR, C1, [(R2, C2)], n_grid=4000.5),
    lambda: margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], u_icp=0.02,
                               n_mc=2.9, seed=1),
    lambda: lock_transient(
        fit_tuning([0.0, 1.0, 2.0, 3.0], [4.8e9, 5.2e9, 5.6e9, 6.0e9],
                   reference="synthetic tuning points, fracpll tests"),
        100.0, FR, 200e-6, 50e-12, [(4.7e3, 1.5e-9)], n_eval=100.5),
])
def test_fractions_are_refused_not_cut(call):
    with pytest.raises(ValueError, match="whole number"):
        call()


def test_reader_columns_are_whole_numbers(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("1e3,-80,-81\n1e4,-100,-101\n1e5,-120,-121\n",
                 encoding="utf-8")
    ref = "synthetic file, fracpll test suite"
    # 0.8.0 read columns (0.7, 1.9) as (0, 1)
    with pytest.raises(ValueError, match="whole number"):
        read_noise_csv(p, ref, columns=(0.7, 1.9))
    with pytest.raises(ValueError, match="pair"):
        read_noise_csv(p, ref, columns=2)
    a = read_noise_csv(p, ref, columns=(0, 2))
    b = read_noise_csv(p, ref, columns=(0.0, np.int64(2)))
    assert a.dbc_per_hz == b.dbc_per_hz == (-81.0, -101.0, -121.0)


def test_whole_numbers_in_other_forms_still_work():
    a = stability(ICP, KV, N, _z2, FR)
    assert stability(ICP, KV, N, _z2, FR, n_grid=4001.0) == a
    assert stability(ICP, KV, N, _z2, FR, n_grid=np.int64(4001)) == a
    m = sampled_margins(ICP, KV, N, FR, C1, [(R2, C2)])
    assert sampled_margins(ICP, KV, N, FR, C1, [(R2, C2)],
                           n_grid="4001") == m
    u0 = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], u_icp=0.02,
                            n_mc=20, seed=1)
    u1 = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], u_icp=0.02,
                            n_mc=20.0, seed=1)
    assert u0["mc"] == u1["mc"]
