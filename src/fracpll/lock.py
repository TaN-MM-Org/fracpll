"""Averaged nonlinear lock transients, and what leakage does at lock.

The averaged large-signal model of the study: the tri-state PFD plus
charge pump becomes a saturating phase detector,

    i(t) = Icp * ( tanh(phi_e / 2 pi) + mismatch ) - I_leak,

linear for small phase error and bounded at +-Icp, driving the passive
loop filter of `fracpll.filters`; the oscillator follows its measured
tuning curve; the phase error integrates the frequency difference,

    dphi_e/dt = 2 pi ( f_ref - f_vco(Vc) / N ).

`lock_transient` integrates this system (stiff-safe LSODA) and reports
the trajectory, whether the loop settled with bounded phase, the
settle time, and the static phase offset.

The static offset has a closed form: at lock the pump must cancel the
leakage-and-mismatch disturbance, so

    tanh(phi_ss / 2 pi) = I_leak/Icp - mismatch,
    phi_ss = 2 pi atanh(I_leak/Icp - mismatch),

reducing to phi_ss ~ 2 pi (I_leak/Icp - mismatch) in the linear range
-- the small-signal statement of the study.  `static_offset` computes
it and refuses when |I_leak/Icp - mismatch| >= 1: the pump then cannot
cancel the disturbance at any phase, and no lock point exists.  The
transient is held against this closed form in the tests (two code
paths: the integrated trajectory and the closed-form offset).

The tanh characteristic is the smooth averaged surrogate used in the
source study; a real tri-state PFD is LINEAR in the phase error over
(-2 pi, 2 pi) and is resolved edge by edge in `fracpll.eventsim`.  Its
exact static offset (with mismatch, reset delay and dead zone) is
`fracpll.pfd.pfd_static_offset`; the two agree to first order in
I_leak/Icp, which the tests check.

Honest limits: this is the AVERAGED model -- per-edge behavior (dead
zone, cycle-slip granularity, reset overlap) is below its resolution
(use `fracpll.eventsim` for it), and it says nothing about a loop whose crossover violates the
`fracpll.loop.stability` validity refusal.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

__all__ = ["static_offset", "lock_transient"]


def static_offset(icp, i_leak=0.0, mismatch=0.0):
    """Closed-form static phase offset (rad) at lock.

    icp : pump current (A); i_leak : leakage current (A);
    mismatch : fractional UP/DN mismatch (dimensionless).

    Refuses when the pump cannot cancel the disturbance at any phase
    (|I_leak/Icp - mismatch| >= 1)."""
    icp = float(icp)
    if not (np.isfinite(icp) and icp > 0.0):
        raise ValueError("icp must be finite and positive")
    rho = float(i_leak) / icp - float(mismatch)
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        raise ValueError(
            f"leakage/mismatch ratio {rho:.4g} reaches the pump "
            "saturation: the charge pump cannot cancel the "
            "disturbance at any phase, so no locked operating point "
            "exists. Reduce the leakage or raise Icp")
    return float(2.0 * np.pi * np.arctanh(rho))


def lock_transient(curve, n_div, f_ref_hz, icp, c_shunt, branches=(),
                   i_leak=0.0, mismatch=0.0, vc0=None, phi0=0.0,
                   t_end=None, n_eval=2000, settle_tol_rad=0.05):
    """Integrate the averaged loop from a cold start.

    curve : a `fracpll.tuning.TuningCurve` (the measured oscillator).
    n_div : division ratio (the lock target is f_vco = N f_ref).
    f_ref_hz, icp : reference frequency (Hz) and pump current (A).
    c_shunt, branches : the loop filter of `fracpll.filters` (every
        branch needs R > 0; an R = 0 branch belongs in c_shunt).
    i_leak, mismatch : pump non-idealities (A and fractional).
    vc0 : starting control voltage (V), inside the measured range
        (default: its low edge, the precharge-low start of the study).
    phi0 : starting phase error (reference radians).
    t_end : integration time in seconds (default 1e5 N / (2 pi f_ref),
        about 32 ms for N = 100 at 50 MHz -- long, so that the
        `locked` test below sees a settled tail).
    n_eval : number of equally spaced output samples over [0, t_end].
    settle_tol_rad : the settling band around the static offset
        (reference radians).

    Returns dict(t, phi_e, vc, locked, t_settle, phi_static).
    `locked` requires the phase error to stay within settle_tol_rad of
    the closed-form static offset over the trailing 20 % of the output
    samples.  `t_settle` is the last time the phase error is outside
    that band, located on the integrator's continuous solution by root
    finding (so it does not depend on n_eval; before 0.4.0 it was
    rounded UP to the next output sample); None when not locked, 0.0
    when the run starts inside the band and never leaves it.

    Refuses when the lock target frequency lies outside the measured
    tuning range: the oscillator cannot reach it, and integrating
    longer will not change that.  Also refuses a negative-Kvco curve,
    which this averaged model does not handle (use `simulate_pll` with
    pump_polarity=-1), a start voltage outside the measured range (the
    curve does not extrapolate), and non-physical filter values.
    """
    f_ref = float(f_ref_hz)
    n_div = float(n_div)
    if not (np.isfinite(f_ref) and f_ref > 0.0):
        raise ValueError("f_ref_hz must be finite and positive")
    if not (np.isfinite(n_div) and n_div >= 1.0):
        raise ValueError("n_div must be finite and >= 1")
    f_target = n_div * f_ref
    f_lo = float(min(curve.f_hz[0], curve.f_hz[-1]))
    f_hi = float(max(curve.f_hz[0], curve.f_hz[-1]))
    if not (f_lo <= f_target <= f_hi):
        raise ValueError(
            f"lock target N*f_ref = {f_target:.6g} Hz lies outside "
            f"the measured tuning range [{f_lo:.6g}, {f_hi:.6g}] Hz; "
            "the oscillator cannot reach it at any control voltage")
    if curve.f_hz[-1] < curve.f_hz[0]:
        raise ValueError(
            "this averaged lock model is written for a positive-Kvco "
            "oscillator; for a negative-Kvco curve use "
            "fracpll.simulate_pll(curve, ..., pump_polarity=-1)")
    phi_ss = static_offset(icp, i_leak, mismatch)

    icp = float(icp)
    c_sh = float(c_shunt)
    if not (np.isfinite(c_sh) and c_sh > 0.0):
        raise ValueError("c_shunt must be finite and positive")
    br = [(float(r), float(c)) for (r, c) in branches]
    for r, c in br:
        if not (np.isfinite(r) and r > 0.0):
            raise ValueError("zero-resistance branches: lump the "
                             "capacitor into c_shunt instead (an R=0 "
                             "branch is the same node); R must be "
                             "finite and > 0")
        if not (np.isfinite(c) and c > 0.0):
            raise ValueError("branch capacitors must be finite and > 0")
    v_lo, v_hi = float(curve.vc[0]), float(curve.vc[-1])
    vc0 = v_lo if vc0 is None else float(vc0)
    if not (v_lo - 1e-12 <= vc0 <= v_hi + 1e-12):
        raise ValueError(
            f"vc0 = {vc0:.4g} V lies outside the measured range "
            f"[{v_lo:.4g}, {v_hi:.4g}] V of the tuning curve, which "
            "does not extrapolate")
    tol = float(settle_tol_rad)
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError("settle_tol_rad must be finite and positive")
    n_eval = int(n_eval)
    if n_eval < 2:
        raise ValueError("n_eval must be >= 2")

    # states: [phi_e, v_shunt, v_c(branch capacitors)...]
    def rhs(t, y):
        phi, vsh = y[0], y[1]
        vcap = y[2:]
        vc = min(max(vsh, v_lo), v_hi)   # rail at the measured range
        i = icp * (np.tanh(phi / (2.0 * np.pi)) + mismatch) \
            - float(i_leak)
        # branch currents into the node
        dvcap = np.empty_like(vcap)
        i_br = 0.0
        for k, (r, c) in enumerate(br):
            ib = (vsh - vcap[k]) / r
            dvcap[k] = ib / c
            i_br += ib
        dvsh = (i - i_br) / c_sh
        f_v = float(np.asarray(curve.frequency(vc)).ravel()[0])
        dphi = 2.0 * np.pi * (f_ref - f_v / n_div)
        return np.concatenate(([dphi, dvsh], dvcap))

    if t_end is None:
        t_end = 2000.0 / (2.0 * np.pi * f_ref) * n_div * 50.0
    t_end = float(t_end)
    if not (np.isfinite(t_end) and t_end > 0.0):
        raise ValueError("t_end must be finite and positive")
    y0 = np.concatenate(([float(phi0), vc0],
                         np.full(len(br), vc0)))
    t_eval = np.linspace(0.0, t_end, n_eval)
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                    t_eval=t_eval, rtol=1e-8, atol=1e-10,
                    dense_output=True)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    phi = sol.y[0]
    vc = np.clip(sol.y[1], v_lo, v_hi)
    tail = slice(int(0.8 * phi.size), None)
    locked = bool(np.all(np.abs(phi[tail] - phi_ss) < tol))
    t_settle = None
    if locked:
        t_settle = _last_exit(sol.sol, phi_ss, tol, t_eval)
    return {"t": sol.t, "phi_e": phi, "vc": vc, "locked": locked,
            "t_settle": t_settle, "phi_static": phi_ss}


def _last_exit(dense, phi_ss, tol, t_eval):
    """Last time |phi(t) - phi_ss| >= tol on the continuous solution.

    Samples the solver's interpolant at its own step boundaries, at 8
    evenly spaced points inside every step, and at the output samples,
    takes the last sample outside the band, and refines the exit by
    Brent's method on the interpolant.  (An excursion out of the band
    shorter than about 1/9 of one solver step, falling between those
    samples, would not be seen; the solver keeps its steps short
    wherever the solution changes quickly.)"""
    ts = np.asarray(dense.ts, dtype=float)
    inner = (ts[:-1, None]
             + np.diff(ts)[:, None] * np.arange(1, 9)[None, :] / 9.0)
    grid = np.unique(np.concatenate((ts, inner.ravel(), t_eval)))
    g = np.abs(dense(grid)[0] - phi_ss) - tol
    out = np.flatnonzero(g >= 0.0)
    if out.size == 0:
        return 0.0
    i = int(out[-1])
    if i + 1 >= grid.size:
        return float(grid[-1])
    a, b = grid[i], grid[i + 1]
    fun = lambda t: abs(float(dense(t)[0]) - phi_ss) - tol
    return float(brentq(fun, a, b, xtol=1e-15 * max(b, 1e-300),
                        rtol=1e-12))
