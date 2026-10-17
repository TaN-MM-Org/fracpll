"""The linearized charge-pump PLL loop: gain, crossover, margin.

The averaged small-signal model of a charge-pump PLL is one formula,

    L(jw) = (Icp / 2 pi) * Z(jw) * (2 pi Kvco) / (jw N)
          = Icp * Kvco * Z(jw) / (jw N),

with Icp the pump current (A), Z the loop-filter impedance (ohm), Kvco
the oscillator gain in Hz/V and N the (mean) division ratio.  This is
exactly the loop expression of the GaN-on-SOI co-design study this
package is distilled from.

Stability is reported, not assumed: `stability` finds the unity-gain
crossover and the phase margin and refuses -- with the reason -- when
the loop has no crossover in the analysis band, when the margin is not
positive, and when the crossover is too close to the reference
frequency for the averaged continuous-time model to be trusted.  The
classical guidance, going back to the discrete-time analysis of
F. M. Gardner, "Charge-pump phase-lock loops", IEEE Trans. Commun. 28,
1849 (1980), is to keep the loop bandwidth about a decade below the
reference; this module refuses beyond f_ref/10 rather than silently
reporting numbers its own model no longer supports, and points to
`fracpll.sampled`, whose exact sampled-data map answers the stability
question at ANY bandwidth, and to `fracpll.eventsim` for the loop edge
by edge.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

__all__ = ["open_loop", "error_transfer", "lowpass_transfer",
           "stability"]


def open_loop(w: ArrayLike, icp: float, kvco_hz_per_v: float, n_div: float,
              zfilter: Callable[[np.ndarray], ArrayLike]) -> np.ndarray:
    """Open-loop gain L(jw).

    w : angular frequencies (rad/s), positive.
    icp : charge-pump current (A), > 0.
    kvco_hz_per_v : oscillator tuning gain (Hz/V), > 0.  A loop built
        around a negative-Kvco oscillator (like the co-designed GaN
        ring) reverses the pump polarity, so the loop sees |Kvco|: pass
        the magnitude here.  A negative value is refused, because it
        would give L < 0 at DC -- a loop with positive feedback -- and
        every result derived from it would be meaningless.  (The
        per-cycle and edge-level models take the sign directly, with
        pump_polarity=-1.)
    n_div : division ratio, >= 1.
    zfilter : callable w -> complex impedance (e.g. a lambda over
        `fracpll.filters.loop_filter_impedance`).
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    icp = float(icp)
    kv = float(kvco_hz_per_v)
    n_div = float(n_div)
    if not (np.isfinite(icp) and icp > 0.0):
        raise ValueError("icp must be finite and positive")
    if not (np.isfinite(kv) and kv != 0.0):
        raise ValueError("kvco_hz_per_v must be finite and nonzero")
    if kv < 0.0:
        raise ValueError(
            "kvco_hz_per_v is negative: a working loop pairs a "
            "negative-Kvco oscillator with a reversed pump, so the loop "
            "sees |Kvco|. Pass the magnitude here (the per-cycle and "
            "edge-level models take the sign with pump_polarity=-1)")
    if not (np.isfinite(n_div) and n_div >= 1.0):
        raise ValueError("n_div must be finite and >= 1")
    z = np.asarray(zfilter(w), dtype=complex)
    return icp * kv * z / (1j * w * n_div)


def error_transfer(lg: ArrayLike) -> np.ndarray:
    """VCO-noise (high-pass) transfer 1/(1+L)."""
    return 1.0 / (1.0 + np.asarray(lg, dtype=complex))


def lowpass_transfer(lg: ArrayLike) -> np.ndarray:
    """Input-referred (low-pass) transfer L/(1+L).

    The two transfers sum to one exactly; the tests hold that
    identity."""
    lg = np.asarray(lg, dtype=complex)
    return lg / (1.0 + lg)


def _refine_crossover(icp, kvco_hz_per_v, n_div, zfilter, a, b):
    """The unity crossing of |L| between grid points a < b (Hz).

    |L(a)| > 1 >= |L(b)|.  Brent's method on ln|L| against ln f,
    converged to the floating-point resolution of f (xtol 4e-16 in
    ln f).  Before 0.6.0 this was 80 steps of geometric bisection
    (80 evaluations of the filter); Brent's method needs 5 or 6 in
    typical loops and lands on the same root to within a few units in
    the last place (the tests keep the bisection as the reference)."""
    def g(x):
        return float(np.log(np.abs(open_loop(
            2.0 * np.pi * np.exp(x), icp, kvco_hz_per_v, n_div,
            zfilter))[0]))
    xa, xb = float(np.log(a)), float(np.log(b))
    x = brentq(g, xa, xb, xtol=4e-16 * max(abs(xa), abs(xb), 1.0),
               rtol=4 * np.finfo(float).eps, maxiter=200)
    return float(np.exp(x))


def stability(icp: float, kvco_hz_per_v: float, n_div: float,
              zfilter: Callable[[np.ndarray], ArrayLike], f_ref_hz: float,
              f_lo: float | None = None, f_hi: float | None = None,
              n_grid: int = 4001) -> dict[str, float]:
    """Crossover frequency and phase margin, with refusals.

    Scans |L| on a log grid from f_lo to f_hi (defaults: f_ref/1e6 to
    f_ref/2), locates the last downward unity-gain crossing, refines it
    by root finding, and returns dict(f_crossover_hz,
    phase_margin_deg), the margin being 180 deg + arg L at the
    crossover, in (-180, 180].

    zfilter may be any transimpedance from the pump current to the
    oscillator control voltage, for example `third_order_impedance`.

    Refuses when |L| never crosses unity in the band, when the phase
    margin is not positive (the linearized loop is unstable), and when
    the crossover exceeds f_ref/10, where the averaged continuous-time
    model itself stops being trustworthy (Gardner 1980).
    """
    f_ref = float(f_ref_hz)
    if not (np.isfinite(f_ref) and f_ref > 0.0):
        raise ValueError("f_ref_hz must be finite and positive")
    f_lo = f_ref * 1e-6 if f_lo is None else float(f_lo)
    f_hi = f_ref * 0.5 if f_hi is None else float(f_hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi) and 0.0 < f_lo < f_hi):
        raise ValueError("need 0 < f_lo < f_hi for the scanned band")
    if int(n_grid) < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid!r}")
    f = np.geomspace(f_lo, f_hi, int(n_grid))
    w = 2.0 * np.pi * f
    mag = np.abs(open_loop(w, icp, kvco_hz_per_v, n_div, zfilter))
    above = mag > 1.0
    if above.all() or not above.any():
        raise ValueError(
            "|L| does not cross unity between "
            f"{f_lo:.3g} and {f_hi:.3g} Hz: no crossover to report. "
            "Check the gain (icp*kvco/N) and the filter values")
    # last index where |L| >= 1 followed by |L| < 1
    idx = np.flatnonzero(above[:-1] & ~above[1:])
    if idx.size == 0:
        raise ValueError(
            "|L| crosses unity only upward in the band; the loop gain "
            "is not monotone here and a single crossover cannot be "
            "reported. Inspect |L(f)| directly")
    i = int(idx[-1])
    fc = _refine_crossover(icp, kvco_hz_per_v, n_div, zfilter,
                           float(f[i]), float(f[i + 1]))
    lg = open_loop(2.0 * np.pi * fc, icp, kvco_hz_per_v, n_div,
                   zfilter)[0]
    # 180 deg + arg L, reported in (-180, 180] (the convention of
    # `sampled_margins`).  np.angle returns arg L in (-180, 180], so a
    # loop gain whose phase has passed -180 deg (for example a
    # third-order filter whose R3-C3 pole sits below the crossover)
    # came out before 0.6.0 as a margin above 180 deg (336.85 instead
    # of -23.15) and was reported as stable.
    pm = 180.0 + float(np.degrees(np.angle(lg)))
    if pm > 180.0:
        pm -= 360.0
    if pm <= 0.0:
        raise ValueError(
            f"phase margin {pm:.2f} deg at crossover {fc:.4g} Hz: the "
            "linearized loop is unstable. Add zero resistance (R2) or "
            "reduce the loop gain")
    if fc > f_ref / 10.0:
        raise ValueError(
            f"crossover {fc:.4g} Hz exceeds f_ref/10 = "
            f"{f_ref / 10.0:.4g} Hz: the averaged continuous-time "
            "model is no longer trustworthy this close to the "
            "reference (Gardner, IEEE Trans. Commun. 28, 1849 (1980)); "
            "this averaged model refuses here. Use "
            "fracpll.sampled_stability for the exact sampled-data "
            "answer at any bandwidth, or fracpll.simulate_pll for the "
            "edge-level loop")
    return {"f_crossover_hz": float(fc),
            "phase_margin_deg": float(pm)}
