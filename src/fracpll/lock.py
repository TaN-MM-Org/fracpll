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
paths), and against a frequency-domain synthesis of the linearized
step response (two independent numerical routes to one trajectory).

Honest limits: this is the AVERAGED model -- per-edge behavior (dead
zone, cycle-slip granularity, reset overlap) is below its resolution,
and it says nothing about a loop whose crossover violates the
`fracpll.loop.stability` validity refusal.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["static_offset", "lock_transient"]


def static_offset(icp, i_leak=0.0, mismatch=0.0):
    """Closed-form static phase offset (rad) at lock.

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
    c_shunt, branches : the loop filter of `fracpll.filters`.
    i_leak, mismatch : pump non-idealities (A and fractional).
    vc0 : starting control voltage (default: low edge of the measured
        range, the precharge-low start of the study).
    t_end : integration time (default: 400 reference-scaled loop
        time constants, coarse but sufficient for the tests).

    Returns dict(t, phi_e, vc, locked, t_settle, phi_static) where
    `locked` requires the phase error to stay within settle_tol_rad of
    the closed-form static offset over the trailing 20 % of the run.

    Refuses when the lock target frequency lies outside the measured
    tuning range: the oscillator cannot reach it, and integrating
    longer will not change that.
    """
    f_ref = float(f_ref_hz)
    n_div = float(n_div)
    f_target = n_div * f_ref
    f_lo = float(min(curve.f_hz[0], curve.f_hz[-1]))
    f_hi = float(max(curve.f_hz[0], curve.f_hz[-1]))
    if not (f_lo <= f_target <= f_hi):
        raise ValueError(
            f"lock target N*f_ref = {f_target:.6g} Hz lies outside "
            f"the measured tuning range [{f_lo:.6g}, {f_hi:.6g}] Hz; "
            "the oscillator cannot reach it at any control voltage")
    phi_ss = static_offset(icp, i_leak, mismatch)

    icp = float(icp)
    br = [(float(r), float(c)) for (r, c) in branches]
    c_sh = float(c_shunt)
    v_lo, v_hi = float(curve.vc[0]), float(curve.vc[-1])
    vc0 = v_lo if vc0 is None else float(vc0)

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
            if r > 0.0:
                ib = (vsh - vcap[k]) / r
            else:
                ib = 0.0  # handled by lumping below
            dvcap[k] = ib / c
            i_br += ib
        dvsh = (i - i_br) / c_sh
        f_v = float(np.asarray(curve.frequency(vc)).ravel()[0])
        dphi = 2.0 * np.pi * (f_ref - f_v / n_div)
        return np.concatenate(([dphi, dvsh], dvcap))

    for r, c in br:
        if r <= 0.0:
            raise ValueError("zero-resistance branches: lump the "
                             "capacitor into c_shunt instead (an R=0 "
                             "branch is the same node)")

    if t_end is None:
        t_end = 2000.0 / (2.0 * np.pi * f_ref) * n_div * 50.0
    y0 = np.concatenate(([float(phi0), vc0],
                         np.full(len(br), vc0)))
    t_eval = np.linspace(0.0, float(t_end), int(n_eval))
    sol = solve_ivp(rhs, (0.0, float(t_end)), y0, method="LSODA",
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    phi = sol.y[0]
    vc = np.clip(sol.y[1], v_lo, v_hi)
    tail = slice(int(0.8 * phi.size), None)
    locked = bool(np.all(np.abs(phi[tail] - phi_ss)
                         < float(settle_tol_rad)))
    t_settle = None
    if locked:
        off = np.abs(phi - phi_ss) >= float(settle_tol_rad)
        idx = np.flatnonzero(off)
        t_settle = float(sol.t[idx[-1] + 1]) if idx.size else 0.0
    return {"t": sol.t, "phi_e": phi, "vc": vc, "locked": locked,
            "t_settle": t_settle, "phi_static": phi_ss}
