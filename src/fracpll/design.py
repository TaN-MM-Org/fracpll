"""Design a second-order loop filter for a target bandwidth and margin.

The inverse of `fracpll.loop.stability`: given the loop bandwidth
(unity-gain crossover f_c) and the phase margin you want, and the pump
current, Kvco and N you have, `design_loop_filter` returns the
classic second-order passive filter (C1 to ground in parallel with a
series R2-C2 branch) that meets both exactly in the averaged model.

Method (W. O. Keese, "An Analysis and Performance Evaluation of a
Passive Filter Design Technique for Charge Pump Phase-Locked Loops",
National Semiconductor Application Note AN-1001 (1996), eqs. 8-12):
with T1 = R2 C1 C2 / (C1 + C2) and T2 = R2 C2 the filter impedance is
Z(s) = (1 + s T2) / (s (C1 + C2) (1 + s T1)), so the loop gain
L = Icp Kvco Z / (s N) has the phase -180 deg + atan(w T2) - atan(w T1).
Putting the maximum of that phase at the crossover, w_c = 1/sqrt(T1 T2),
gives a phase margin atan(w_c T2) - atan(w_c T1) = 90 deg - 2 atan(w_c T1),
so

    T1 = (sec(PM) - tan(PM)) / w_c,        T2 = 1 / (w_c^2 T1),

and |L(j w_c)| = 1 fixes the total capacitance,

    C1 = (T1/T2) (Icp Kvco / (N w_c^2))
         sqrt((1 + (w_c T2)^2) / (1 + (w_c T1)^2)),
    C2 = C1 (T2/T1 - 1),      R2 = T2 / C2.

Here Icp Kvco is in A Hz/V, the product of AN-1001's K_phi (tabulated in mA/(2 pi rad))
and K_vco (tabulated in MHz/V).  The tests derive the same values independently
(the phase maximum and |L| = 1 solved in the test) and read the
designed loop back with `stability`, a separate numerical route.

Since 0.9.0 the same design rule also gives the third-order filter of
`fracpll.filters.third_order_impedance` (an R3-C3 section after the
second-order filter), when you choose two ratios: t3_over_t1, where
the section's pole sits (T3 = R3-C3 pole time constant over T1), and
c3_over_c1 (C3 = c3_over_c1 C1).  Its transimpedance is

    Z(s) = (1 + s T2) / (s A0 (1 + s T1)(1 + s T3)),

with A0 = C1 + C2 + C3, A0 (T1 + T3) = A1 and A0 T1 T3 = A2 of the
polynomial form in `fracpll.filters`.  The loop phase is
-180 deg + atan(w T2) - atan(w T1) - atan(w T3).  With T3 = r T1 and
x = w_c T1, the margin fixes w_c T2 = tan(PM + atan(x) + atan(r x)),
and a phase peak at w_c (zero slope there) is the condition

    sin(2 (PM + atan(x) + atan(r x))) / 2 = x/(1 + x^2) + r x/(1 + (r x)^2),

which is solved for x by root finding (its left side falls from
sin(2 PM)/2 > 0 to 0 while the right side stays positive, so a root
exists; for r = 0 it is the closed form above).  |L(j w_c)| = 1 then
gives A0, and so A1 and A2.  The five parts follow from
A0 = C1 + C2 + C3, R2 C2 = T2, A2 = C1 C2 C3 R2 R3 (so
R3 C3 = A2 / (C1 T2)) and A1 = T2 (C1 + C3) + R3 C3 (C1 + C2): with
C3 = c C1 these give the quadratic

    (1 + c) T2^2 C1^2 - (A1 T2 + c A2) C1 + A0 A2 = 0.

Both of its roots give the same transimpedance, and so the same loop;
the root with the larger C1 (and so the smaller R3, which adds less
resistor noise) is returned, if it gives a positive C2 (otherwise the
other one).  These steps are derived here from the polynomial form, not
taken from a published third-order procedure.  The tests read the
designed filter back with `third_order_impedance` and `stability`
(bandwidth and margin), check that the phase slope is zero at f_c, and
check the pole ratio and C3/C1 of the parts.  Not every pair of ratios
can be built: when the quadratic has no root with positive parts, the
design refuses and asks for smaller ratios.

The averaged model holds for loops well below the reference, so the
design refuses f_c at or above f_ref/10 (Gardner, IEEE Trans.
Commun. 28, 1849 (1980)), as `stability` does.  The returned dict also gives the
margins of the exact per-cycle loop (`sampled_margins`), which move
away from the targets as f_c approaches the reference: check them
before building.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.optimize import brentq

from .filters import loop_filter_impedance, third_order_impedance
from .loop import stability
from .sampled import sampled_margins

__all__ = ["design_loop_filter"]

_EPS = 2.220446049250313e-16        # the spacing of floats near 1


def design_loop_filter(f_c_hz: float, phase_margin_deg: float,
                       icp: float, kvco_hz_per_v: float, n_div: float,
                       f_ref_hz: float, t3_over_t1: float | None = None,
                       c3_over_c1: float | None = None) -> dict[str, Any]:
    """Second-order passive filter for a target bandwidth and margin.

    f_c_hz : the loop bandwidth (unity-gain crossover) to design for
        (Hz), below f_ref_hz / 10.
    phase_margin_deg : the phase margin to design for (degrees),
        strictly between 0 and 90.
    icp : pump current (A), > 0.
    kvco_hz_per_v : oscillator gain magnitude |Kvco| (Hz/V), > 0 (as
        for `stability`; a negative-Kvco oscillator with a reversed
        pump is the same loop).
    n_div : (mean) division ratio, >= 1.
    f_ref_hz : reference frequency (Hz), for the validity check and
        the per-cycle check.
    t3_over_t1, c3_over_c1 : give both (each finite and > 0) for the
        third-order filter with an R3-C3 section (since 0.9.0; module
        docstring): the section's time constant T3 = t3_over_t1 T1 and
        C3 = c3_over_c1 C1.  Leave both out for the second-order filter,
        computed as before 0.9.0.

    Returns a dict with
        c_shunt, branches : the filter, ready for `loop_filter_impedance`,
            `sampled_margins`, `simulate_pll` and the other models
            (c_shunt = C1, branches = [(R2, C2)]);
        c1, r2, c2 : the same values by name (F, ohm, F);
        t1_s, t2_s : the time constants T1 = R2 C1 C2/(C1 + C2) and
            T2 = R2 C2 (s);
        averaged : `stability` of the designed loop (f_crossover_hz,
            phase_margin_deg), equal to the targets to rounding;
        per_cycle : `sampled_margins` of the designed loop (the exact
            per-cycle crossover, phase margin and gain margin).
    With the R3-C3 section it also has r3, c3 (ohm, F, ready as the
    `r3` and `c3` of the other models) and t3_s, the section's time
    constant T3; `averaged` and `per_cycle` are then read with the
    section, and t1_s is the other pole's time constant.

    Refuses non-finite or non-positive inputs, a phase margin outside
    (0, 90) degrees (a second-order filter cannot give 90 degrees or
    more; at 90 its C1 would be zero), and f_c at or above f_ref/10.  The
    values are exact; round them to available parts and check the
    rounded filter with `stability` and `sampled_margins` (and
    `margin_uncertainty` for the part tolerances).
    """
    fc, pm = float(f_c_hz), float(phase_margin_deg)
    icp, kv = float(icp), float(kvco_hz_per_v)
    nd, f_ref = float(n_div), float(f_ref_hz)
    for name, v in (("f_c_hz", fc), ("icp", icp),
                    ("kvco_hz_per_v", kv), ("f_ref_hz", f_ref)):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be finite and positive, got "
                             f"{v!r}")
    if not (math.isfinite(nd) and nd >= 1.0):
        raise ValueError(f"n_div must be finite and >= 1, got {nd!r}")
    if not (math.isfinite(pm) and 0.0 < pm < 90.0):
        raise ValueError(
            f"phase_margin_deg must be strictly between 0 and 90, got "
            f"{pm!r}: the phase of a second-order loop peaks below "
            "-90 degrees, and at 90 the shunt capacitor C1 would vanish")
    # strictly below f_ref/10, with room for rounding, so that the
    # read-back by `stability` (which refuses above f_ref/10) succeeds
    if fc >= 0.1 * f_ref * (1.0 - 1e-9):
        raise ValueError(
            f"f_c_hz = {fc:.4g} Hz is not below f_ref/10 = "
            f"{f_ref / 10.0:.4g} Hz: the averaged model this design rests on is not "
            "trustworthy that close to the reference (Gardner, IEEE "
            "Trans. Commun. 28, 1849 (1980)). Choose a lower bandwidth, "
            "or check a candidate filter at any bandwidth with "
            "sampled_margins")
    if (t3_over_t1 is None) != (c3_over_c1 is None):
        raise ValueError("give both t3_over_t1 and c3_over_c1 for the "
                         "third-order filter, or neither")
    if t3_over_t1 is not None:
        return _design_third_order(fc, pm, icp, kv, nd, f_ref,
                                   float(t3_over_t1), float(c3_over_c1))
    wc = 2.0 * math.pi * fc
    phi = math.radians(pm)
    t1 = (1.0 / math.cos(phi) - math.tan(phi)) / wc
    t2 = 1.0 / (wc * wc * t1)
    c1 = (t1 / t2) * (icp * kv / (nd * wc * wc)) * math.sqrt(
        (1.0 + (wc * t2) ** 2) / (1.0 + (wc * t1) ** 2))
    c2 = c1 * (t2 / t1 - 1.0)
    r2 = t2 / c2
    zf = lambda w: loop_filter_impedance(w, c1, [(r2, c2)])  # noqa: E731
    avg = stability(icp, kv, nd, zf, f_ref)
    pc = sampled_margins(icp, kv, nd, f_ref, c1, [(r2, c2)])
    return {"c_shunt": c1, "branches": [(r2, c2)],
            "c1": c1, "r2": r2, "c2": c2, "t1_s": t1, "t2_s": t2,
            "averaged": avg,
            "per_cycle": {k: pc[k] for k in
                          ("f_crossover_hz", "phase_margin_deg",
                           "gain_margin_db", "f_phase_crossover_hz",
                           "stable")}}


def _design_third_order(fc, pm, icp, kv, nd, f_ref, r, c):
    """The third-order filter of the module docstring (inputs checked
    by `design_loop_filter`)."""
    for name, v in (("t3_over_t1", r), ("c3_over_c1", c)):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be finite and positive, got "
                             f"{v!r}")
    wc = 2.0 * math.pi * fc
    phi = math.radians(pm)

    def lead(x):                     # PM + atan(x) + atan(r x)
        return phi + math.atan(x) + math.atan(r * x)

    def slope(x):                    # zero where the phase peaks at w_c
        return (0.5 * math.sin(2.0 * lead(x)) - x / (1.0 + x * x)
                - r * x / (1.0 + (r * x) ** 2))

    # the largest x with lead(x) < 90 degrees (w_c T2 then stays finite)
    x_max = brentq(lambda x: lead(x) - 0.5 * math.pi, 0.0,
                   1.0 / math.tan(0.5 * phi) + 1.0, xtol=1e-300,
                   rtol=4 * _EPS)
    x = brentq(slope, 0.0, x_max, xtol=1e-300, rtol=4 * _EPS,
               maxiter=500)
    y = math.tan(lead(x))            # w_c T2
    t1, t3, t2 = x / wc, r * x / wc, y / wc
    a0 = (icp * kv / (nd * wc * wc)) * math.sqrt(
        (1.0 + y * y) / ((1.0 + x * x) * (1.0 + (r * x) ** 2)))
    a1, a2 = a0 * (t1 + t3), a0 * t1 * t3
    qa = (1.0 + c) * t2 * t2
    qb = a1 * t2 + c * a2
    disc = qb * qb - 4.0 * qa * a0 * a2
    found = None
    if disc >= 0.0:
        sq = math.sqrt(disc)
        # the larger root first; the smaller one by Vieta's rule, which
        # avoids the cancellation of qb - sq
        big = (qb + sq) / (2.0 * qa)
        for c1 in (big, a0 * a2 / (qa * big)):
            c3 = c * c1
            c2 = a0 - c1 - c3
            if c1 > 0.0 and c2 > 0.0:
                found = (c1, c2, c3)
                break
    if found is None:
        raise ValueError(
            f"no third-order filter with C3 = {c:g} C1 has these poles "
            f"(T3 = {r:g} T1) at {fc:.4g} Hz and {pm:g} degrees: the "
            "parts would not all be positive. Choose a smaller "
            "t3_over_t1 or c3_over_c1")
    c1, c2, c3 = found
    r2 = t2 / c2
    r3 = a2 / (c1 * t2 * c3)
    zf = lambda w: third_order_impedance(w, c1, r2, c2, r3, c3)  # noqa: E731
    avg = stability(icp, kv, nd, zf, f_ref)
    pc = sampled_margins(icp, kv, nd, f_ref, c1, [(r2, c2)], r3=r3, c3=c3)
    return {"c_shunt": c1, "branches": [(r2, c2)], "r3": r3, "c3": c3,
            "c1": c1, "r2": r2, "c2": c2, "t1_s": t1, "t2_s": t2,
            "t3_s": t3, "averaged": avg,
            "per_cycle": {k: pc[k] for k in
                          ("f_crossover_hz", "phase_margin_deg",
                           "gain_margin_db", "f_phase_crossover_hz",
                           "stable")}}
