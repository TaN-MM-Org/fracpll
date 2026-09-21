"""Edge-accurate, event-driven simulation of a charge-pump PLL.

The averaged model of `fracpll.loop` and `fracpll.lock` treats the
phase detector as a continuous-time gain.  That is its limit: it cannot
see individual edges, so it says nothing about the dead zone, reset
overlap, pump mismatch at the edge level, cycle slips, or a loop whose
bandwidth approaches the reference.  This module removes that limit by
simulating the loop EDGE BY EDGE:

* a tri-state phase-frequency detector: the reference edge sets UP,
  the divider edge sets DN, and when both are set they reset together
  after a reset delay t_reset -- so both pumps conduct for t_reset every
  cycle (the reset overlap), and an edge that arrives while its flop is
  already set is lost (a cycle slip);
* a charge pump with separate UP and DN currents (mismatch), a
  constant leakage, a pump polarity (negative-Kvco oscillators need
  the pump inverted), and an optional dead zone modelled as a turn-on
  delay t_deadzone: a pump delivers current only after it has been
  commanded on for t_deadzone, so pulses shorter than that deliver
  nothing.  This dead-zone model is a behavioural choice, stated as
  such -- not a device model;
* the passive loop filter of `fracpll.filters`, integrated EXACTLY
  between edges: the pump current is constant there, and the filter's
  modal closed form (`fracpll._network`) gives the control voltage and
  its time integral with no time step at all;
* the oscillator either as a linear tuning law f = f0 + Kvco v_c
  (`LinearVCO`), whose phase is then also exact in closed form, or as
  a MEASURED tuning curve (`fracpll.tuning.TuningCurve`), whose phase
  is integrated by Gauss-Legendre quadrature over the exact v_c(t);
* a divider that counts oscillator cycles and emits its edge when the
  oscillator phase reaches the programmed count -- located by
  root-finding on the exact phase, not by sampling -- with the count
  sequence N + d_k taken from `fracpll.mash.mash_sequence` for
  fractional-N operation.

What it is checked against (see tests/test_eventsim.py): the exact
charge balance of the periodic steady state (leakage, mismatch, reset
delay and dead zone together), the exact linearised sampled-data map of
`fracpll.sampled` cycle by cycle, the averaged model in its own regime,
the delta-sigma noise formula of `fracpll.mash` through the loop, and
the exact mean division ratio of a fractional-N lock.

What it shows that the linear models cannot: with a delta-sigma
divider, the edge-level loop carries an extra in-band noise floor at
offsets below a few times the loop bandwidth, even with perfectly
matched pumps.  It is circuit behaviour of this model, not numerics:
it is unchanged (to 0.001 dB) when the time resolution is improved
10^5-fold, independent of the Welch window, and it falls about
5.5 dB per halving of the PFD pulse width while the linear band
stays fixed (asserted in the tests).  Single-pulse runs show the
edge-level map has second-order terms of size ~Kvco Icp e^2/(2 C0)
whose coefficient depends on which edge leads, i.e. the PFD/pump/
oscillator sampling is weakly nonlinear even when matched; the floor
itself, however, does not scale with C0, so the precise mechanism is
NOT identified here and no claim is made about it.  With pump
MISMATCH the simulator reproduces the well-documented folding of
shaped quantization noise into the band and its reduction by an
offset current (T.-H. Lin, C.-L. Ti and Y.-H. Liu, IEEE Trans.
Circuits Syst. I 56, 877 (2009)).

Honest limits: the pump switches instantaneously (apart from the
dead-zone delay), the divider and PFD are ideal logic apart from the
stated delays, and the filter elements are linear.  Noise sources other
than the delta-sigma divider and an optional user-supplied reference
jitter are not injected here; `fracpll.noise` handles those in the
frequency domain.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from ._network import FilterModes
from .mash import mash_sequence

__all__ = ["LinearVCO", "SimResult", "simulate_pll"]


@dataclass(frozen=True)
class LinearVCO:
    """f = f0_hz + kvco_hz_per_v * v_c, with provenance.

    A design-value or fitted linear tuning law; `reference` says where
    the numbers come from (mandatory, >= 8 characters).
    """
    f0_hz: float
    kvco_hz_per_v: float
    reference: str

    def __post_init__(self):
        if not isinstance(self.reference, str) \
                or len(self.reference.strip()) < 8:
            raise ValueError("a LinearVCO requires a real reference "
                             "(datasheet, fit, or design note)")
        if not np.isfinite(self.f0_hz) or not np.isfinite(
                self.kvco_hz_per_v) or self.kvco_hz_per_v == 0.0:
            raise ValueError("f0_hz must be finite and kvco nonzero")

    def frequency(self, vc):
        return self.f0_hz + self.kvco_hz_per_v * np.asarray(vc)


@dataclass
class SimResult:
    """Everything the edge-level run records, one entry per edge."""
    t_ref: np.ndarray        # reference edge times (s)
    t_div: np.ndarray        # divider edge times (s)
    vc_ref: np.ndarray       # control voltage at each reference edge (V)
    psi_ref: np.ndarray      # oscillator excess phase at each reference
                             # edge, in oscillator CYCLES, relative to the
                             # exact mean ratio (N + num/den) f_ref t
    n_ratio: float           # the programmed mean ratio N + num/den
    f_ref: float
    slips: int               # edges lost by the PFD (cycle slips)
    counts: np.ndarray       # division count used for each divider edge
    _ref_frame: np.ndarray = None   # frame index + in-frame time: the
    _ref_rel: np.ndarray = None     # full-precision form of the edge
    _div_frame: np.ndarray = None   # times (absolute floats lose
    _div_rel: np.ndarray = None     # resolution on long runs)

    def time_error(self):
        """t_div[k] - t_ref[k] (s) over the common edges, at full
        precision (computed from frame index and in-frame time).

        Pairs edges by index, which is right while no edge has been
        lost; after cycle slips use `nearest_edge_error`."""
        n = min(self.t_ref.size, self.t_div.size)
        T = 1.0 / self.f_ref
        return ((self._div_frame[:n] - self._ref_frame[:n]) * T
                + (self._div_rel[:n] - self._ref_rel[:n]))

    def nearest_edge_error(self):
        """Each divider edge minus the NEAREST reference edge (s).

        Insensitive to cycle slips: once locked it is the static
        offset whatever happened during pull-in."""
        tr, td = self.t_ref, self.t_div
        if tr.size == 0 or td.size == 0:
            return np.zeros(0)
        idx = np.clip(np.searchsorted(tr, td), 1, tr.size - 1)
        left, right = tr[idx - 1], tr[idx]
        pick = np.where(np.abs(td - left) <= np.abs(td - right),
                        left, right)
        return td - pick

    def phase_rad(self):
        """Oscillator excess phase at the reference edges (rad)."""
        return 2.0 * np.pi * self.psi_ref


_GL_CACHE = {}


def _gl(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def simulate_pll(vco, n_int, f_ref_hz, icp, c_shunt, branches=(),
                 n_cycles=1000, num=0, den=2, mash_order=0,
                 i_dn=None, i_leak=0.0, t_reset=0.0, t_deadzone=0.0,
                 pump_polarity=1, vc0=None, phase0_cycles=0.0,
                 ref_jitter_s=None, quad_nodes=16):
    """Run the loop edge by edge for n_cycles reference periods.

    vco : `LinearVCO` (exact phase) or a `TuningCurve` (measured,
        Gauss-Legendre phase over the exact control voltage; the run
        refuses if v_c leaves the measured range).
    n_int : integer part N of the division ratio.
    num, den, mash_order : fractional part num/den generated by a
        MASH-`mash_order` cascade (0 = integer-N).
    icp : UP pump current (A); i_dn : DN current (default = icp).
    i_leak : constant leakage OUT of the control node (A).
    t_reset, t_deadzone : PFD reset delay and pump turn-on delay (s).
    pump_polarity : +1, or -1 for a negative-Kvco oscillator.
    vc0 : starting control voltage (default: the exact lock voltage
        of a LinearVCO; required for a measured curve).
    phase0_cycles : starting oscillator phase lead, in cycles.
    ref_jitter_s : optional array of per-edge reference time offsets.
    """
    f_ref = float(f_ref_hz)
    T = 1.0 / f_ref
    n_int = int(n_int)
    num, den, m = int(num), int(den), int(mash_order)
    if n_int < 1 or f_ref <= 0.0:
        raise ValueError("need n_int >= 1 and f_ref > 0")
    if m == 0 and num != 0:
        raise ValueError("a fractional part needs mash_order 1, 2 or 3")
    icp = float(icp)
    i_dn = icp if i_dn is None else float(i_dn)
    if icp <= 0.0 or i_dn <= 0.0:
        raise ValueError("pump currents must be positive")
    pol = int(pump_polarity)
    if pol not in (1, -1):
        raise ValueError("pump_polarity must be +1 or -1")
    t_reset, t_dz = float(t_reset), float(t_deadzone)
    if t_reset < 0.0 or t_dz < 0.0:
        raise ValueError("delays must be >= 0")
    n_cycles = int(n_cycles)

    fm = FilterModes(c_shunt, branches)
    linear = isinstance(vco, LinearVCO)
    n_avg = n_int + (num / den if m else 0.0)
    if linear:
        f0, kv = float(vco.f0_hz), float(vco.kvco_hz_per_v)
        if pol * kv < 0.0:
            raise ValueError(
                "pump polarity and Kvco sign give POSITIVE feedback: a "
                "negative-Kvco oscillator needs pump_polarity=-1")
        v_lock = (n_avg * f_ref - f0) / kv
        vc0 = v_lock if vc0 is None else float(vc0)
    else:
        if vc0 is None:
            raise ValueError("a measured tuning curve needs an explicit "
                             "vc0 inside its measured range")
        vc0 = float(vc0)
        k_sign = float(np.sign(np.ravel(vco.kvco(vc0))[0]))
        if pol * k_sign < 0.0:
            raise ValueError(
                "pump polarity and the measured Kvco sign give POSITIVE "
                "feedback: use pump_polarity=-1 for this oscillator")
        gx, gw = _gl(int(quad_nodes))

    if m:
        d = mash_sequence(num, den, m, n_cycles + 8)
    else:
        d = np.zeros(n_cycles + 8, dtype=np.int64)

    jit = None
    if ref_jitter_s is not None:
        jit = np.asarray(ref_jitter_s, dtype=float)
        if jit.size < n_cycles + 1:
            raise ValueError("ref_jitter_s needs >= n_cycles + 1 entries")

    # oscillator phase increment over [0, dt] from modal state w, current i
    def inc(w, i, dt):
        if dt <= 0.0:
            return 0.0
        if linear:
            return f0 * dt + kv * fm.vc_integral(w, i, dt)
        ts = 0.5 * dt * (gx + 1.0)
        v = np.array([fm.vc_at(w, i, s) for s in ts])
        return 0.5 * dt * float(np.dot(gw, vco.frequency(v)))

    # --- state -------------------------------------------------------
    w = fm.modal_from_voltages([vc0] * fm.n)   # branches at equilibrium
    t = 0.0
    theta = float(phase0_cycles)   # cycles since the last divider target
    count = 0                      # cycles completed at divider edges
    j = 0                          # divider edge index
    target = n_int + int(d[0])
    k_ref = 0                      # reference edges so far
    # Time is kept RELATIVE to the current frame origin k_ref * T
    # (nominal), so its float resolution stays ~1e-24 s however long the
    # run: every pending event time is shifted by -T at each reference
    # edge.  Absolute times are formed only for the output record.

    def jit_at(k):
        return float(jit[k]) if jit is not None else 0.0

    next_ref = T + jit_at(1)
    up = dn = False
    up_on = dn_on = False
    t_up_start = t_dn_start = t_rst_at = None
    slips = 0

    t_ref_l, t_div_l, vc_l, psi_l, cnt_l = [], [], [], [], []
    rf_l, rr_l, df_l, dr_l = [], [], [], []

    def current():
        return pol * (icp * up_on - i_dn * dn_on) - i_leak

    def set_up(tnow):
        nonlocal up, up_on, t_up_start, t_rst_at, slips
        if up:
            slips += 1
            return
        up = True
        if t_dz == 0.0:
            up_on = True
        else:
            t_up_start = tnow + t_dz
        if dn:
            t_rst_at = tnow + t_reset

    def set_dn(tnow):
        nonlocal dn, dn_on, t_dn_start, t_rst_at, slips
        if dn:
            slips += 1
            return
        dn = True
        if t_dz == 0.0:
            dn_on = True
        else:
            t_dn_start = tnow + t_dz
        if up:
            t_rst_at = tnow + t_reset

    def do_reset():
        nonlocal up, dn, up_on, dn_on, t_up_start, t_dn_start, t_rst_at
        up = dn = up_on = dn_on = False
        t_up_start = t_dn_start = t_rst_at = None

    while k_ref < n_cycles:
        i = current()
        cands = [next_ref]
        for tc in (t_rst_at, t_up_start, t_dn_start):
            if tc is not None:
                cands.append(tc)
        t_fix = min(cands)
        dt_fix = t_fix - t
        if dt_fix < 0.0:
            dt_fix = 0.0
        dth = inc(w, i, dt_fix)
        if theta + dth >= target:
            # divider edge inside this constant-current interval
            g0 = theta - target
            if g0 >= 0.0:
                dts = 0.0
            else:
                dts = brentq(lambda s: theta + inc(w, i, s) - target,
                             0.0, dt_fix, xtol=1e-22, rtol=1e-15,
                             maxiter=200)
            w = fm.advance(w, i, dts)
            t += dts
            theta = 0.0
            count += target
            t_div_l.append(k_ref * T + t)
            df_l.append(k_ref)
            dr_l.append(t)
            cnt_l.append(target)
            j += 1
            target = n_int + int(d[j]) if j < d.size else n_int
            set_dn(t)
            if t_rst_at is not None and t_rst_at <= t:
                do_reset()
            continue
        # advance to the fixed event
        w = fm.advance(w, i, dt_fix)
        theta += dth
        t = t_fix
        if not linear:
            v = fm.vc(w)
            lo, hi = vco.vc[0], vco.vc[-1]
            if v < lo - 1e-12 or v > hi + 1e-12:
                raise ValueError(
                    f"control voltage {v:.4g} V left the measured tuning "
                    f"range [{lo:.4g}, {hi:.4g}] V at t = "
                    f"{k_ref * T + t:.4g} s; the "
                    "oscillator is not characterised there")
        if t_rst_at is not None and t_rst_at <= t:
            do_reset()
        if t_up_start is not None and t_up_start <= t:
            t_up_start = None
            if up:
                up_on = True
        if t_dn_start is not None and t_dn_start <= t:
            t_dn_start = None
            if dn:
                dn_on = True
        if next_ref <= t:
            t_ref_l.append(k_ref * T + t)
            rf_l.append(k_ref)
            rr_l.append(t)
            k_ref += 1
            vc_l.append(fm.vc(w))
            # excess phase vs the exact mean ratio (integer arithmetic)
            if m:
                ex = (count * den - (n_int * den + num) * k_ref) / den
            else:
                ex = float(count - n_int * k_ref)
            ex += theta - (n_avg * f_ref * (t - T))
            psi_l.append(ex)
            set_up(t)
            if t_rst_at is not None and t_rst_at <= t:
                do_reset()
            # shift the frame origin by one period
            t -= T
            if t_rst_at is not None:
                t_rst_at -= T
            if t_up_start is not None:
                t_up_start -= T
            if t_dn_start is not None:
                t_dn_start -= T
            next_ref = T + jit_at(k_ref + 1)

    return SimResult(t_ref=np.array(t_ref_l), t_div=np.array(t_div_l),
                     vc_ref=np.array(vc_l), psi_ref=np.array(psi_l),
                     n_ratio=n_avg, f_ref=f_ref, slips=slips,
                     counts=np.array(cnt_l, dtype=np.int64),
                     _ref_frame=np.array(rf_l, dtype=np.int64),
                     _ref_rel=np.array(rr_l),
                     _div_frame=np.array(df_l, dtype=np.int64),
                     _div_rel=np.array(dr_l))
