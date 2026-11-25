"""The linearized charge-pump PLL loop: gain, crossover, margin.

The averaged small-signal model of a charge-pump PLL is one formula,

    L(jw) = (Icp / 2 pi) * Z(jw) * (2 pi Kvco) / (jw N)
          = Icp * Kvco * Z(jw) / (jw N),

with Icp the pump current (A), Z the loop-filter impedance (ohm), Kvco
the oscillator gain in Hz/V and N the (mean) division ratio.  This is
exactly the loop expression of the GaN-on-SOI PLL study this package
is distilled from (its iteration-1 loop model; the README gives the
citation).

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

Since 0.7.0, `closed_loop_response` reads the closed loop L/(1+L):
its -3 dB bandwidth and its peaking, with the same refusals.

Since 0.9.0, `stability` follows the loop phase continuously along its
scan, from the low end of the band up to the crossover, instead of
reading it modulo 360 degrees at the crossover alone.  A loop whose
phase had fallen past -360 degrees there (a filter with several poles
below the crossover, or a loop delay) was reported with a large
positive margin before; it now gets its true, negative margin and is
refused.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize_scalar

from ._checks import whole

__all__ = ["open_loop", "error_transfer", "lowpass_transfer",
           "stability", "closed_loop_response"]


def open_loop(w: ArrayLike, icp: float, kvco_hz_per_v: float, n_div: float,
              zfilter: Callable[[np.ndarray], ArrayLike]) -> np.ndarray:
    """Open-loop gain L(jw).

    w : angular frequencies (rad/s), positive.
    icp : charge-pump current (A), > 0.
    kvco_hz_per_v : oscillator tuning gain (Hz/V), > 0.  A loop built
        around a negative-Kvco oscillator reverses the pump polarity, so the loop sees |Kvco|: pass
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
    phase_margin_deg), the margin being 180 deg + the phase of L at the
    crossover.

    The phase is followed continuously along the scan (since 0.9.0):
    it is taken between -360 and 0 degrees at f_lo, where the
    oscillator's integration alone gives -90, and then followed from
    grid point to grid point up to the crossover, so that a phase that
    has fallen past -360 degrees is not mistaken for one near 0.  This
    needs the phase to turn by less than 180 degrees between two grid
    points; for a very sharp resonance in the filter, raise n_grid.
    For the filters of this package (phase between -180 and -90
    degrees) the result is the same, bit for bit, as reading the phase
    at the crossover alone, as before 0.9.0.

    zfilter may be any transimpedance from the pump current to the
    oscillator control voltage, for example `third_order_impedance`.
    n_grid is a whole number >= 2.

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
    n_grid = whole(n_grid, "n_grid")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid!r}")
    f = np.geomspace(f_lo, f_hi, n_grid)
    w = 2.0 * np.pi * f
    lg_grid = open_loop(w, icp, kvco_hz_per_v, n_div, zfilter)
    mag = np.abs(lg_grid)
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
    pm = 180.0 + float(np.degrees(_continuous_phase(lg_grid[:i + 1],
                                                     lg)))
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


def _continuous_phase(lg_scan, lg_end):
    """The phase (rad) of lg_end, followed continuously along lg_scan.

    lg_scan holds the loop gain on the scan grid from its low end up to
    the last grid point before the crossover, and lg_end is the gain at
    the crossover.  The phase at the low end is taken in (-2 pi, 0]
    (the oscillator integrates, so a loop gain starts near -90 or -180
    degrees), then unwrapped from point to point.

    History: before 0.6.0 the margin was 180 deg + np.angle(L) with
    np.angle in (-180, 180], so a loop whose phase had passed -180 deg
    (example 12's refused third-order filter) got 336.85 instead of
    -23.15 degrees.  0.6.0 moved the margin into (-180, 180], which
    still misread a phase below -360 deg: a filter with four extra
    poles at 8 kHz (phase -368.86 deg at its 10.66 kHz crossover) was
    reported stable with 171.14 degrees.  Where the old rule was right
    every correction here is exactly zero, so the result is unchanged,
    bit for bit."""
    a = np.angle(np.asarray(lg_scan, dtype=complex))
    two_pi = 2.0 * np.pi
    if a[0] > 0.0:
        a = a.copy()
        a[0] -= two_pi
    ph = np.unwrap(a)
    a_end = float(np.angle(lg_end))
    return a_end + two_pi * round((float(ph[-1]) - a_end) / two_pi)


def closed_loop_response(icp: float, kvco_hz_per_v: float, n_div: float,
                         zfilter: Callable[[np.ndarray], ArrayLike],
                         f_ref_hz: float, f_lo: float | None = None,
                         n_grid: int = 4001) -> dict[str, float]:
    """Closed-loop bandwidth and peaking of H = L/(1+L) (averaged model).

    H is the transfer from the reference (or divider) phase to the
    output phase, divided by N: what a jitter-transfer or closed-loop
    response measurement shows.  A charge-pump loop with a capacitor
    in its filter has two integrators (it is "type 2"), so |H| rises
    above 1 before it falls: the peaking.

    The loop is first checked with `stability` (same inputs and scan,
    same refusals).  Then |H| is scanned on the same log grid from f_lo
    (default f_ref/1e6) to f_ref/2; its largest value is refined by
    Brent's bounded method on ln f between the neighbors of the largest
    grid value, and the -3 dB point (the last offset where |H|^2 falls
    through 1/2) by root finding on ln f.

    Returns dict(f_3db_hz, peaking_db, f_peak_hz, f_crossover_hz,
    phase_margin_deg): the -3 dB bandwidth (Hz), the peak of |H| in dB
    (20 log10 max |H|, >= 0) and where it is (Hz), and the crossover
    and phase margin from `stability`.  The peak value is accurate to
    rounding; its position, at the top of a flat maximum, only to a few
    parts in 10^7.

    Refuses what `stability` refuses, a largest |H| at an end of the
    scanned band (no peak inside it: lower f_lo, or the loop is not
    type 2), |H| that does not fall through -3 dB in the band, and a
    -3 dB point above f_ref/10, where the averaged model is no longer
    trustworthy (Gardner, IEEE Trans. Commun. 28, 1849 (1980)).
    """
    rep = stability(icp, kvco_hz_per_v, n_div, zfilter, f_ref_hz,
                    f_lo=f_lo, n_grid=n_grid)
    f_ref = float(f_ref_hz)
    f_lo = f_ref * 1e-6 if f_lo is None else float(f_lo)
    f = np.geomspace(f_lo, f_ref * 0.5, whole(n_grid, "n_grid"))
    h2 = np.abs(lowpass_transfer(open_loop(2.0 * np.pi * f, icp,
                                           kvco_hz_per_v, n_div,
                                           zfilter))) ** 2

    def h2_at(x):                       # |H|^2 at f = exp(x)
        lg = open_loop(2.0 * np.pi * np.exp(x), icp, kvco_hz_per_v,
                       n_div, zfilter)[0]
        return float(np.abs(lg / (1.0 + lg)) ** 2)

    i = int(np.argmax(h2))
    if i == 0 or i == f.size - 1:
        end = "low" if i == 0 else "high"
        raise ValueError(
            f"the largest |H| in the band is at its {end} "
            f"end ({f[i]:.4g} Hz): there is no peak inside "
            f"{f_lo:.3g} to {f[-1]:.3g} Hz. Lower f_lo; a loop whose "
            "filter has no capacitor (type 1) has no peaking")
    lo, hi = float(np.log(f[i - 1])), float(np.log(f[i + 1]))
    opt = minimize_scalar(lambda x: -h2_at(x), bounds=(lo, hi),
                          method="bounded",
                          options={"xatol": 1e-12 * max(abs(lo), 1.0)})
    if -float(opt.fun) >= h2[i]:
        peak2, f_peak = -float(opt.fun), float(np.exp(opt.x))
    else:                                 # the grid point is higher
        peak2, f_peak = float(h2[i]), float(f[i])
    below = h2 < 0.5
    idx = np.flatnonzero(~below[:-1] & below[1:])
    idx = idx[idx >= i]
    if idx.size == 0:
        raise ValueError(
            "|H| does not fall through -3 dB above its peak between "
            f"{f[i]:.4g} and {f[-1]:.4g} Hz: no closed-loop bandwidth "
            "to report")
    j = int(idx[-1])
    xa, xb = float(np.log(f[j])), float(np.log(f[j + 1]))
    x3 = brentq(lambda x: h2_at(x) - 0.5, xa, xb,
                xtol=4e-16 * max(abs(xa), abs(xb), 1.0),
                rtol=4 * np.finfo(float).eps, maxiter=200)
    f3 = float(np.exp(x3))
    if f3 > f_ref / 10.0:
        raise ValueError(
            f"the -3 dB bandwidth {f3:.4g} Hz exceeds f_ref/10 = "
            f"{f_ref / 10.0:.4g} Hz: the averaged continuous-time model "
            "is no longer trustworthy there (Gardner, IEEE Trans. "
            "Commun. 28, 1849 (1980)); this averaged model refuses. "
            "Lower the loop bandwidth, or use the per-cycle model "
            "(sampled_open_loop) for the loop gain at any bandwidth")
    return {"f_3db_hz": f3, "peaking_db": float(10.0 * np.log10(peak2)),
            "f_peak_hz": f_peak,
            "f_crossover_hz": rep["f_crossover_hz"],
            "phase_margin_deg": rep["phase_margin_deg"]}
