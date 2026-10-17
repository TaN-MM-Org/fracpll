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

from .filters import loop_filter_impedance
from .loop import stability
from .sampled import sampled_margins

__all__ = ["design_loop_filter"]


def design_loop_filter(f_c_hz: float, phase_margin_deg: float,
                       icp: float, kvco_hz_per_v: float, n_div: float,
                       f_ref_hz: float) -> dict[str, Any]:
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
