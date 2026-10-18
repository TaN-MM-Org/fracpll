"""Closed-loop bandwidth and peaking (`closed_loop_response`, 0.7.0).

* The textbook second-order type-2 loop.  With the filter a resistor R
  in series with a capacitor C (no shunt capacitor), the averaged loop
  gain is L = (Icp Kvco/N)(1 + s R C)/(s^2 C), so

      H = L/(1+L) = (2 zeta wn s + wn^2)/(s^2 + 2 zeta wn s + wn^2),
      wn^2 = Icp Kvco/(N C),   2 zeta wn = Icp Kvco R/N.

  Derived here: |H|^2 = (1 + 4 zeta^2 u)/((1 - u)^2 + 4 zeta^2 u) with
  u = (w/wn)^2.  |H|^2 = 1/2 gives u^2 - 2(1 + 2 zeta^2) u - 1 = 0, so
  w_3dB = wn sqrt(a + sqrt(a^2 + 1)), a = 1 + 2 zeta^2.  d|H|^2/du = 0
  gives 2 zeta^2 u^2 + u - 1 = 0, so the peak is at
  u* = (sqrt(1 + 8 zeta^2) - 1)/(4 zeta^2), with |H|^2 there by
  substitution.
* The filters of the package (second order, and third order through
  its polynomial form), against a dense scan done here with a separate
  impedance code path.

Tolerances.  The -3 dB point is a simple root (|H|^2 falls with a slope
of order 1 in ln f), found to rounding: 1e-12 relative.  The peak
VALUE is found to rounding as well (1e-9 dB asked).  The peak POSITION
sits on a flat maximum, where a change of x in ln f changes |H|^2 only
by order x^2: Brent's bounded method stops when its bracket is within
2 (sqrt(eps) |ln f| + xatol/3) of its best point, at most 4e-7 in
ln f for f below 1 MHz, so 2e-6 relative is asked (2.1e-7 observed
over 33 loops)."""
import math

import numpy as np
import pytest

from fracpll import (closed_loop_response, design_loop_filter,
                     loop_filter_impedance, open_loop,
                     second_order_impedance, third_order_impedance)

FR = 50e6
LOOP = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0)


@pytest.mark.parametrize("zeta", [0.3, 0.5, 1 / math.sqrt(2), 1.0, 2.0,
                                  5.0])
@pytest.mark.parametrize("fn", [3e3, 30e3, 300e3])
def test_second_order_type2_closed_forms(zeta, fn):
    icp, kv, nd = LOOP["icp"], LOOP["kvco_hz_per_v"], LOOP["n_div"]
    wn = 2 * math.pi * fn
    c = icp * kv / (nd * wn ** 2)
    r = 2 * zeta * wn * nd / (icp * kv)
    zf = lambda w: r + 1.0 / (1j * np.asarray(w) * c)
    got = closed_loop_response(icp, kv, nd, zf, FR)
    a = 1 + 2 * zeta ** 2
    f3 = wn * math.sqrt(a + math.sqrt(a * a + 1)) / (2 * math.pi)
    u = (math.sqrt(1 + 8 * zeta ** 2) - 1) / (4 * zeta ** 2)
    h2 = (1 + 4 * zeta ** 2 * u) / ((1 - u) ** 2 + 4 * zeta ** 2 * u)
    assert got["f_3db_hz"] == pytest.approx(f3, rel=1e-12)
    assert got["peaking_db"] == pytest.approx(10 * math.log10(h2),
                                              abs=1e-9)
    assert got["f_peak_hz"] == pytest.approx(fn * math.sqrt(u), rel=2e-6)
    assert got["peaking_db"] > 0.0


def _scan_check(zf_pkg, zf_indep, f_lo, f_hi):
    got = closed_loop_response(zfilter=zf_pkg, f_ref_hz=FR, **LOOP)

    def h2(f):
        lg = open_loop(2 * np.pi * np.atleast_1d(f), zfilter=zf_indep,
                       **LOOP)
        return np.abs(lg / (1 + lg)) ** 2

    # the peak: no point of a dense scan is higher, and the highest
    # scan point is below it by no more than the scan resolution allows
    # (step 3.7e-6 in ln f, so |H|^2 there is below the maximum by
    # about 1e-11 relative, 5e-11 dB)
    f = np.geomspace(f_lo, f_hi, 1_000_001)
    peak = 10 * np.log10(h2(f).max())
    assert got["peaking_db"] >= peak - 1e-12
    assert got["peaking_db"] - peak < 1e-9
    # the -3 dB point brackets the root
    f3 = got["f_3db_hz"]
    assert h2(f3 * (1 - 1e-10))[0] > 0.5 > h2(f3 * (1 + 1e-10))[0]
    return got


def test_second_order_filter_against_a_dense_scan():
    c1, r2, c2 = 100e-12, 4.7e3, 1.5e-9
    _scan_check(lambda w: loop_filter_impedance(w, c1, [(r2, c2)]),
                lambda w: second_order_impedance(w, c1, r2, c2), 5e3, 2e5)


def test_third_order_filter_against_a_dense_scan():
    c1, r2, c2, r3, c3 = 100e-12, 4.7e3, 1.5e-9, 10e3, 10e-12
    a0 = c1 + c2 + c3
    a1 = r2 * c2 * (c1 + c3) + r3 * c3 * (c1 + c2)
    a2 = c1 * c2 * c3 * r2 * r3

    def poly(w):                       # the nodal-analysis form
        s = 1j * np.asarray(w)
        return (1 + s * r2 * c2) / (s * (a0 + a1 * s + a2 * s * s))

    _scan_check(lambda w: third_order_impedance(w, c1, r2, c2, r3, c3),
                poly, 5e3, 2e5)


def test_refusals():
    # a filter that is only a resistor: type 1, |H| falls from 1, no peak
    with pytest.raises(ValueError, match="no peak inside"):
        closed_loop_response(zfilter=lambda w: 1e4 + 0 * np.asarray(w),
                             f_ref_hz=FR, **LOOP)
    # a loop designed at 0.08 f_ref with 45 degrees: crossover below
    # f_ref/10, but its -3 dB point above it
    d = design_loop_filter(0.08 * FR, 45.0, f_ref_hz=FR, **LOOP)
    zf = lambda w: loop_filter_impedance(w, d["c_shunt"], d["branches"])
    with pytest.raises(ValueError, match="-3 dB bandwidth"):
        closed_loop_response(zfilter=zf, f_ref_hz=FR, **LOOP)
    # what `stability` refuses is refused (README example 5's fast loop)
    fast = lambda w: loop_filter_impedance(w, 2e-12, [(20e3, 200e-12)])
    with pytest.raises(ValueError, match="exceeds f_ref/10"):
        closed_loop_response(0.5e-3, 400e6, 10, fast, FR)
    with pytest.raises(ValueError, match="negative"):
        closed_loop_response(100e-6, -20e6, 40.0, fast, FR)
