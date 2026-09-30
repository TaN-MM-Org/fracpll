"""The exact sampled-data (discrete-time) model of a charge-pump PLL.

A charge-pump loop is not continuous-time: the detector acts once per
reference period, as a pulse of charge.  The averaged gain L(jw) of
`fracpll.loop` is its low-bandwidth approximation, and that is why
`fracpll.loop.stability` refuses beyond f_ref/10.  This module removes
that limit with the linearised map the edge dynamics actually obey.

Linearise about lock.  Between reference edges the filter state x
(capacitor voltages, see `fracpll._network`) and the oscillator excess
phase psi (in oscillator CYCLES) evolve with no pump current,

    dx/dt = A x,      dpsi/dt = Kvco * v_c = Kvco * c.x,

so over one period s = [x, psi] maps by Phi = expm(M T),
M = [[A, 0], [Kvco c, 0]].  At each reference edge the detector
injects the charge of one pulse.  An oscillator lead psi makes the
divider edge early by psi/(N f_ref) seconds, and a pulse of that
width carries

    q = -pol * Icp * psi / (N f_ref)

into the control node (pol = pump polarity; matched pumps -- a reset
overlap then cancels exactly).  The kick is instantaneous to first
order: its spread over a pulse of width |dt| alters the next state
only at second order in dt.  So, EXACTLY to first order,

    s_{k+1} = Phi (I - pol Icp/(N f_ref) B e_psi^T) s_k,
    B = [b, 0],  b = e_0 / C_shunt.

This is the impulse-invariant charge-pump model going back to
F. M. Gardner, IEEE Trans. Commun. 28, 1849 (1980): here it is built
numerically for ANY passive filter of `fracpll.filters` and ANY loop
bandwidth.  It is checked two ways in the tests: against the
edge-accurate simulator of `fracpll.eventsim` cycle by cycle (two
independent code paths, one trajectory), and, at low bandwidth, its
poles z are checked against exp(s T) of the continuous closed-loop
poles, which recovers the averaged model where it is valid.

Since 0.4.0 the same map also gives the exact per-cycle loop gain
G(z) (`sampled_open_loop`) and, from it, phase and gain margins at any
bandwidth (`sampled_margins`).  G is held in the tests against the
aliased sum of the averaged gain, sum_m L(j(w + m 2 pi f_ref)), the
margins against eigenvalue routes, and G/(1+G) against the per-cycle
simulation in the time domain.

Honest limits: this is the SMALL-SIGNAL map about lock, with matched
pumps and no dead zone (a dead zone has zero gain at the origin, so
there is nothing to linearise; `fracpll.eventsim` handles it).  It
makes no claim about pull-in or cycle slips -- those are large-signal,
and belong to the event-driven simulator.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from ._network import FilterModes

__all__ = ["sampled_loop_map", "sampled_stability",
           "sampled_open_loop", "sampled_margins",
           "continuous_closed_loop_poles", "simulate_sampled",
           "pulse_doublet_vector"]


def sampled_loop_map(icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                     branches=(), pump_polarity=1):
    """The one-period map s_{k+1} = F s_k, s = [x..., psi (cycles)].

    Returns dict(F, Phi, kick) with F = Phi @ kick.
    """
    icp, kv = float(icp), float(kvco_hz_per_v)
    nd, f_ref = float(n_div), float(f_ref_hz)
    pol = int(pump_polarity)
    if icp <= 0.0 or nd < 1.0 or f_ref <= 0.0 or kv == 0.0:
        raise ValueError("need icp > 0, n_div >= 1, f_ref > 0, kvco != 0")
    if pol not in (1, -1):
        raise ValueError("pump_polarity must be +1 or -1")
    if pol * kv < 0.0:
        raise ValueError("pump polarity and Kvco sign give positive "
                         "feedback; use pump_polarity=-1 for a "
                         "negative-Kvco oscillator")
    fm = FilterModes(c_shunt, branches)
    a, b, c = fm.state_matrices()
    n = fm.n
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = a
    M[n, :n] = kv * c
    Phi = expm(M / f_ref)
    kick = np.eye(n + 1)
    kick[:n, n] -= pol * icp / (nd * f_ref) * b
    return {"F": Phi @ kick, "Phi": Phi, "kick": kick}


def sampled_stability(icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                      branches=(), pump_polarity=1):
    """Exact small-signal stability at ANY bandwidth.

    Returns dict(stable, spectral_radius, poles).  Stable means every
    pole of the one-period map lies strictly inside the unit circle.
    (The filter's charge-conserving mode is an integrator, but with
    the loop closed it is fed back through psi, so no pole is left on
    the unit circle: the continuous characteristic polynomial below
    has constant term Icp*Kvco != 0, hence no root at s = 0.)
    """
    F = sampled_loop_map(icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                         branches, pump_polarity)["F"]
    z = np.linalg.eigvals(F)
    rho = float(np.max(np.abs(z)))
    return {"stable": bool(rho < 1.0), "spectral_radius": rho,
            "poles": z}


def sampled_open_loop(f_hz, icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                      branches=(), pump_polarity=1):
    """Exact per-cycle open-loop gain G(z) on the unit circle.

    Breaking the loop of `sampled_loop_map` at the phase detector, a
    divider-phase error e_k (oscillator cycles) at reference edge k
    makes the charge pol Icp e_k / (N f_ref), and the oscillator phase
    sampled at the following edges answers with

        G(z) = (pol Icp / (N f_ref)) e_psi^T (z I - Phi)^(-1) Phi B,

    so the closed loop from the divider phase u to the sampled
    oscillator phase psi is G/(1+G), and 1 + G(z) = 0 exactly at the
    poles of the one-period map F (matrix determinant lemma).  Returned
    at z = exp(j 2 pi f / f_ref) for 0 < f <= f_ref/2; at f = f_ref/2
    (z = -1) it is real.

    G is the exact discrete-time counterpart of the averaged loop gain
    L(jw) of `fracpll.loop.open_loop`: by the Poisson sum it equals
    sum over all integers m of L(j(w + m 2 pi f_ref)) -- the averaged
    gain plus all of its aliases -- which the tests check to 1e-10.
    At offsets far below f_ref the m = 0 term dominates and G -> L.

    Sign convention: as in `sampled_loop_map`, pass the signed Kvco
    with pump_polarity=-1 for a negative-Kvco oscillator; G then equals
    the positive-Kvco loop of the same |Kvco|.
    """
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    f_ref = float(f_ref_hz)
    if not (np.isfinite(f_ref) and f_ref > 0.0):
        raise ValueError("f_ref_hz must be finite and positive")
    if not np.all(np.isfinite(f)) or np.any(f <= 0.0) \
            or np.any(f > 0.5 * f_ref):
        raise ValueError("offsets must satisfy 0 < f <= f_ref/2: a "
                         "sampled loop has no separate response beyond "
                         "Nyquist (f and f_ref - f are the same point "
                         "of the unit circle)")
    m = sampled_loop_map(icp, kvco_hz_per_v, n_div, f_ref, c_shunt,
                         branches, pump_polarity)
    Phi = m["Phi"]
    n1 = Phi.shape[0]
    b = FilterModes(c_shunt, branches).state_matrices()[1]
    PB = Phi @ np.r_[b, 0.0]
    z = np.exp(2j * np.pi * f / f_ref)
    z[f == 0.5 * f_ref] = -1.0            # exactly real at Nyquist
    a = z[:, None, None] * np.eye(n1)[None, :, :] - Phi[None, :, :]
    rhs = np.broadcast_to(PB.astype(complex), (f.size, n1))[:, :, None]
    x = np.linalg.solve(a, rhs)[:, :, 0]
    k = int(pump_polarity) * float(icp) / (float(n_div) * f_ref)
    return k * x[:, -1]


def sampled_margins(icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                    branches=(), pump_polarity=1, f_lo=None,
                    n_grid=4001):
    """Phase margin and gain margin of the exact per-cycle loop.

    Works at ANY loop bandwidth, including the fast loops that
    `fracpll.loop.stability` refuses (crossover above f_ref/10).  It
    reads the exact loop gain G of `sampled_open_loop` on a log grid
    from f_lo (default f_ref/1e6) to f_ref/2, refines by root finding,
    and returns

    * f_crossover_hz, phase_margin_deg: where |G| last falls through 1,
      and 180 deg + arg G there (in (-180, 180]);
    * f_phase_crossover_hz, gain_margin_db: the first frequency at or
      above the crossover where G is real and negative (its phase is
      -180 deg), and -20 log10 |G| there.  A sampled loop always has one by
      f_ref/2 when G(z=-1) < 0, so unlike the averaged model it has a
      finite gain margin: the factor 10^(gain_margin_db/20) is how much
      Icp can grow before a pole leaves the unit circle there (the
      tests check it against the eigenvalue route to 1e-9).  If G is
      never real and negative above the crossover, gain_margin_db is
      +inf and f_phase_crossover_hz is NaN;
    * stable, spectral_radius: from the eigenvalues of the one-period
      map (`sampled_stability`) -- the authority on stability; the
      margins say how far the loop is from losing it.

    Refuses when |G| does not fall through 1 between f_lo and f_ref/2,
    in particular when |G| >= 1 still at f_ref/2.  Such a loop is
    unstable: for this filter family every alias term of G(-1) has a
    negative real part (the filter impedance is capacitive), so
    G(-1) <= -1, and a Schur-stable characteristic polynomial p of
    degree d needs (-1)^d p(-1) > 0, i.e. 1 + G(-1) > 0 here.

    Honest limits: the same small-signal model as `sampled_loop_map`
    (matched pumps, no dead zone, constant Kvco, linear in the pulse
    width).
    """
    f_ref = float(f_ref_hz)
    if not (np.isfinite(f_ref) and f_ref > 0.0):
        raise ValueError("f_ref_hz must be finite and positive")
    f_hi = 0.5 * f_ref
    f_lo = f_ref * 1e-6 if f_lo is None else float(f_lo)
    if not (np.isfinite(f_lo) and 0.0 < f_lo < f_hi):
        raise ValueError("need 0 < f_lo < f_ref/2")
    args = (icp, kvco_hz_per_v, n_div, f_ref, c_shunt, branches,
            pump_polarity)
    f = np.geomspace(f_lo, f_hi, int(n_grid))
    f[-1] = f_hi
    g = sampled_open_loop(f, *args)
    mag = np.abs(g)

    def gain(x):
        return sampled_open_loop(min(x, f_hi), *args)[0]

    if mag[-1] >= 1.0:
        raise ValueError(
            f"|G| = {mag[-1]:.4g} >= 1 still at f_ref/2: the loop gain "
            "is too high for a phase detector that acts once per "
            "reference cycle, and the loop is unstable (gain margin "
            f"{-20.0 * np.log10(mag[-1]):.3g} dB). Reduce Icp or Kvco, "
            "or raise N")
    above = mag > 1.0
    idx = np.flatnonzero(above[:-1] & ~above[1:])
    if idx.size == 0:
        raise ValueError(
            f"|G| does not fall through 1 between {f_lo:.3g} and "
            f"{f_hi:.3g} Hz: no crossover to report. Check the gain "
            "(icp*kvco/N) and the filter values, or lower f_lo")
    i = int(idx[-1])
    fc = brentq(lambda x: abs(gain(x)) - 1.0, f[i], f[i + 1],
                xtol=1e-13 * f[i], rtol=1e-15, maxiter=200)
    pm = 180.0 + float(np.degrees(np.angle(gain(fc))))
    if pm > 180.0:
        pm -= 360.0
    # phase crossover: G real and negative, the first one at or above
    # fc.  The scan starts AT fc (then the grid points above it), so
    # the stretch between fc and the next grid point is not skipped.
    # An imaginary part below 1e-9 |G| counts as zero, so rounding
    # cannot fake a sign change; a filter with no resistor has G real
    # and negative everywhere, so its phase crossover is fc itself and
    # its gain margin 0 dB (the loop is marginal).
    fs = np.concatenate(([fc], f[i + 1:]))
    gs = np.concatenate(([gain(fc)], g[i + 1:]))
    im = np.where(np.abs(gs.imag) <= 1e-9 * np.abs(gs), 0.0, gs.imag)
    f_pc = np.nan
    for j in range(fs.size - 1):
        if im[j] == 0.0 and gs.real[j] < 0.0:
            f_pc = float(fs[j])
            break
        if im[j] * im[j + 1] < 0.0:
            r = brentq(lambda x: gain(x).imag, fs[j], fs[j + 1],
                       xtol=1e-13 * fs[j], rtol=1e-15, maxiter=200)
            if gain(r).real < 0.0:
                f_pc = float(r)
                break
    if np.isnan(f_pc) and gs.real[-1] < 0.0:
        f_pc = f_hi                        # G(z = -1) is real
    gm_db = np.inf if np.isnan(f_pc) else \
        float(-20.0 * np.log10(abs(gain(f_pc))))
    st = sampled_stability(*args)
    return {"f_crossover_hz": float(fc), "phase_margin_deg": float(pm),
            "f_phase_crossover_hz": f_pc, "gain_margin_db": gm_db,
            "stable": st["stable"],
            "spectral_radius": st["spectral_radius"]}


def continuous_closed_loop_poles(icp, kvco_hz_per_v, n_div, c_shunt,
                                 branches=()):
    """Roots of 1 + L(s) = 0 for the averaged model (rad/s).

    With Z = 1/Y and Y(s) = s C0 + sum_k s C_k / (1 + s R_k C_k),
    1 + Icp Kvco Z/(s N) = 0 becomes the polynomial
    s N Y(s) prod_k (1 + s R_k C_k) + Icp Kvco prod_k (1 + s R_k C_k) = 0.
    Used to show the sampled map reduces to the averaged model at low
    bandwidth (z = exp(s T)).  Like `fracpll.loop.open_loop`, it takes
    the magnitude |Kvco| the loop sees and refuses a negative value.
    """
    if not (np.isfinite(float(kvco_hz_per_v)) and float(kvco_hz_per_v) > 0.0):
        raise ValueError("pass the magnitude |Kvco| (> 0) the loop sees; "
                         "the sign goes in pump_polarity for the "
                         "per-cycle and edge-level models")
    P = np.poly1d([1.0])
    for r, cc in branches:
        P = P * np.poly1d([r * cc, 1.0])
    Ypoly = np.poly1d([float(c_shunt), 0.0]) * P
    for k, (r, cc) in enumerate(branches):
        others = np.poly1d([1.0])
        for m, (r2, c2) in enumerate(branches):
            if m != k:
                others = others * np.poly1d([r2 * c2, 1.0])
        Ypoly = Ypoly + np.poly1d([cc, 0.0]) * others
    char = np.poly1d([float(n_div), 0.0]) * Ypoly \
        + float(icp) * float(kvco_hz_per_v) * P
    return np.roots(char.coeffs)


def pulse_doublet_vector(kvco_hz_per_v, c_shunt, branches=()):
    """The second-order pulse-width correction direction, M B.

    A pump pulse is a RECTANGLE of charge q = pol Icp tau lasting |tau|
    on one side of the reference edge, not an impulse at the edge.
    Expanding the exact response e^{M(t-s)} B i(s) over the pulse, the
    zeroth moment gives the impulse B q and the first moment gives

        delta s = -(q tau / 2) M B = -(pol Icp tau^2 / 2) M B,

    the same for both edge orders (the charge centroid sits at kT +
    tau/2 either way).  Returns M B = [A b ; Kvco / C_shunt]: its
    oscillator row is Kvco times the filter's control-voltage impulse
    response at t = 0+, i.e. Kvco / C_shunt (the in-pulse voltage
    ramp), and since the oscillator row of e^{Mt} M B is Kvco times that
    impulse response at time t, the correction decays to the LASTING
    phase error -Kvco pol Icp tau^2 / (2 C_total), independent of
    C_shunt (asserted in the tests).
    """
    fm = FilterModes(c_shunt, branches)
    a, b, c = fm.state_matrices()
    n = fm.n
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = a
    M[n, :n] = float(kvco_hz_per_v) * c
    B = np.r_[b, 0.0]
    return M @ B


def simulate_sampled(icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                     branches=(), u_cycles=None, n_int=None, num=0,
                     den=2, mash_order=0, n_cycles=None, order=2,
                     pump_polarity=1):
    """Fast per-cycle simulation of the charge-pump loop, to 1st or 2nd
    order in the PFD pulse width.

    order=1 is the exact linearised sampled-data map of
    `sampled_loop_map`.  order=2 adds every second-order effect of the
    finite pulse width, computed self-consistently from the map's own
    state (nothing is taken from the edge-level simulator):

    * the pulse-centroid ("doublet") correction -(pol Icp tau^2/2) M B
      of `pulse_doublet_vector`, applied on the correct side of the
      sample for reference-first and divider-first cycles;
    * the divider edge fired at the oscillator's ACTUAL phase: its
      frequency offset Kvco dv_c over the pulse, and, when the reference
      leads, the control-voltage ramp during the UP pulse.

    With a delta-sigma divider, order 2 reproduces the in-band noise
    that the edge-level simulator `fracpll.eventsim.simulate_pll` shows
    and order 1 misses: the tests hold the two to 1e-3 of the effect in
    the time domain and to 0.5 dB band by band.  The mechanism is the
    tau^2 of the pulse centroid: tau is driven by the shaped
    delta-sigma phase, so tau^2 carries difference-frequency content
    that lands in the loop band -- even with perfectly matched pumps.

    Divider sequence: pass u_cycles (u[k] = sum_{j<=k} (d_j - frac), the
    divider phase in oscillator cycles, aligned with the k-th reference
    edge as in `simulate_pll`), or n_int/num/den/mash_order/n_cycles to
    generate it from `fracpll.mash.mash_sequence`.  n_div is the MEAN
    ratio N + num/den.

    Returns dict(psi, tau): oscillator excess phase at each reference
    edge (cycles) and the pulse width divider-minus-reference (s).

    Honest limits: matched pumps, no dead zone, constant Kvco (the
    small-signal slope at lock); third-order terms are neglected -- the
    tests show them 60 dB or more below the second-order effect for the
    designs checked.  For mismatch, dead zones, measured tuning curves
    and large signals, use the edge-level simulator.
    """
    icp, kv = float(icp), float(kvco_hz_per_v)
    nd, f_ref = float(n_div), float(f_ref_hz)
    pol = int(pump_polarity)
    if int(order) not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if pol * kv <= 0.0:
        raise ValueError("pump polarity and Kvco sign must give negative "
                         "feedback (pump_polarity=-1 for negative Kvco)")
    if u_cycles is None:
        if n_cycles is None:
            raise ValueError("pass u_cycles, or num/den/mash_order and "
                             "n_cycles")
        from .mash import mash_sequence
        m = int(mash_order)
        if m == 0:
            u = np.zeros(int(n_cycles))
        else:
            d = mash_sequence(int(num), int(den), m, int(n_cycles) + 4)
            u = np.cumsum(d - int(num) / int(den))[:int(n_cycles)]
        if n_int is not None and not np.isclose(
                nd, int(n_int) + (int(num) / int(den) if m else 0.0)):
            raise ValueError("n_div must equal n_int + num/den (the mean "
                             "division ratio)")
    else:
        u = np.asarray(u_cycles, dtype=float)
    fm = FilterModes(c_shunt, branches)
    a, b, c = fm.state_matrices()
    n = fm.n
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = a
    M[n, :n] = kv * c
    Phi = expm(M / f_ref)
    B = np.r_[b, 0.0]
    MB = M @ B
    c0 = float(c_shunt)
    two = int(order) == 2
    s = np.zeros(n + 1)
    K = u.size
    psi = np.empty(K)
    taus = np.empty(K)
    nf = nd * f_ref
    for k in range(K):
        s = Phi @ s
        tau = (u[k] - s[-1]) / nf
        if two:
            dpsi = kv * s[0] * tau
            if tau > 0.0:
                dpsi += kv * pol * icp * tau * tau / (2.0 * c0)
            tau = (u[k] - s[-1] - dpsi) / nf
            v = -(pol * icp * tau * tau / 2.0) * MB
            if tau < 0.0:            # divider-first: pulse precedes edge
                s[-1] += v[-1]
        psi[k] = s[-1]
        taus[k] = tau
        s = s + B * (pol * icp * tau)
        if two:
            if tau < 0.0:
                s[:-1] += v[:-1]
            else:
                s += v
    return {"psi": psi, "tau": taus}
