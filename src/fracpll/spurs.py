"""Reference spurs of a locked integer-N loop, from leakage and pump
mismatch.

Once an integer-N loop is locked, every reference period looks the
same: the phase detector fires the same pump pulses at the same times,
and leakage drains the filter at the same rate.  The pump current is
therefore a periodic waveform with period T = 1/f_ref, and so is the
control voltage.  Its ripple frequency-modulates the oscillator and
puts tones (reference spurs) at k f_ref on both sides of the carrier.
`reference_spurs` computes them from that periodic steady state, with
no small-pulse or averaging approximation:

1. The locked timing offset dt (divider edge minus reference edge)
   comes from the exact charge balance of `pfd_static_offset`: the net
   pump charge per period cancels the leakage.  The pulses are then
   known exactly, with the tri-state detector of `fracpll.pfd`: the
   pump set by the earlier edge is on from that edge (plus the dead-zone
   delay t_deadzone) until t_reset after the later edge, and the other
   pump from the later edge (plus t_deadzone) until the same reset.
2. A rectangular pulse of current I from t = a to t = b has the Fourier
   coefficient (I/T) (b - a) sinc(k f_ref (b - a)) exp(-j k w_ref (a + b)/2)
   at harmonic k (w_ref = 2 pi f_ref, sinc(x) = sin(pi x)/(pi x)); the
   constant leakage has none for k >= 1.  Their sum is c_k (A).
3. The ripple of the control voltage at harmonic k is V_k = Z(j k w_ref)
   c_k, with Z the filter's transimpedance from pump current to the
   oscillator's control voltage (`loop_filter_impedance`, or the
   third-order filter of `third_order_impedance` when r3 and c3 are
   given).  This is exact for the periodic steady state: in it the
   phase detector sees the same edges every cycle, so the loop
   corrects nothing at k f_ref, and the ripple is the filter's response
   to the known current.
4. The oscillator phase is 2 pi Kvco times the integral of the control
   voltage, so its line at k f_ref has the complex amplitude
   Phi_k = |Kvco| V_k / (j k f_ref) (oscillator radians), the one-sided
   line power 2 |Phi_k|^2 (rad^2, as in `mash_line_spectrum`), and the
   single-sideband level 10 log10(|Phi_k|^2) dBc, reported where the
   small-angle rule holds (a phase amplitude of 0.2 rad or less, as
   for every dBc in this package).

What the tests check: the harmonics against a separate time-domain
calculation of the periodic steady state (matrix exponentials of the
filter's state equations over each constant-current interval, then
the Fourier integral by quadrature), for second- and third-order
filters, with leakage, mismatch, reset delay, dead zone and both pump
polarities; and that steady state against the edge-by-edge simulator
`simulate_pll` (control voltage at the reference edge and the locked
timing offset).

Honest limits: integer-N only (a fractional-N divider changes the pulse
width every cycle); the pump switches instantly apart from the stated
delays, as in `simulate_pll`; Kvco is the slope at the lock voltage,
so the result is exact for a straight-line tuning law and a
small-ripple approximation for a measured curve.  The loop must be
locked and stable: without r3 and c3 this is checked with the exact
per-cycle model (`sampled_stability`), at any bandwidth; with them,
with the averaged model (`stability`), which refuses beyond f_ref/10.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .filters import loop_filter_impedance
from .loop import stability
from .mash import _small_angle_dbc
from .pfd import pfd_static_offset
from .sampled import _polarity, sampled_stability

__all__ = ["reference_spurs"]


def _pulses(f_ref, icp, i_dn, i_leak, t_reset, t_dz, pol):
    """The locked timing offset dt and the pump pulses of one period.

    Returns (dt, [(a, b, I), ...]) with the reference edge at t = 0, the
    divider edge at t = dt, and I the current INTO the control node
    (leakage, which is constant, left out).  For pump_polarity = -1 the
    pump that adds charge is DN (set by the divider edge): the loop is
    the mirror image of a +1 loop with the two pump currents exchanged,
    so its offset is minus that loop's."""
    if pol == 1:
        dt = pfd_static_offset(f_ref, icp, i_dn=i_dn, i_leak=i_leak,
                               t_reset=t_reset, t_deadzone=t_dz)["dt_s"]
    else:
        dt = -pfd_static_offset(f_ref, i_dn, i_dn=icp, i_leak=i_leak,
                                t_reset=t_reset, t_deadzone=t_dz)["dt_s"]
    if dt >= 0.0:            # reference first: UP from 0, DN from dt
        up = (t_dz, dt + t_reset)
        dn = (dt + t_dz, dt + t_reset)
    else:                    # divider first: DN from dt, UP from 0
        dn = (dt + t_dz, t_reset)
        up = (t_dz, t_reset)
    pulses = []
    if up[1] > up[0]:
        pulses.append((up[0], up[1], pol * icp))
    if dn[1] > dn[0]:
        pulses.append((dn[0], dn[1], -pol * i_dn))
    return dt, pulses


def reference_spurs(f_ref_hz: float, icp: float, kvco_hz_per_v: float,
                    n_div: float, c_shunt: float,
                    branches: Sequence[tuple[float, float]] = (),
                    i_dn: float | None = None, i_leak: float = 0.0,
                    t_reset: float = 0.0, t_deadzone: float = 0.0,
                    pump_polarity: int = 1, r3: float | None = None,
                    c3: float | None = None,
                    n_harmonics: int = 5) -> dict[str, Any]:
    """Reference spurs of a locked integer-N loop (module docstring).

    f_ref_hz : reference frequency (Hz).
    icp, i_dn : UP and DN pump currents (A); i_dn defaults to icp.
    kvco_hz_per_v : oscillator gain at the lock voltage (Hz/V), signed
        as for `simulate_pll`: negative together with pump_polarity=-1.
    n_div : division ratio, used only to check that the loop is stable.
    c_shunt, branches : the loop filter, as for `loop_filter_impedance`.
    r3, c3 : optional R3-C3 section after it (ohm, F), as in
        `third_order_impedance`; the oscillator is then driven by C3.
    i_leak : leakage OUT of the control node (A); t_reset, t_deadzone :
        PFD reset delay and pump turn-on delay (s), as for
        `pfd_static_offset` and `simulate_pll`.
    n_harmonics : how many harmonics k = 1, 2, ... to return (>= 1).

    Returns a dict with
        k, f_hz : harmonic numbers and spur offsets k f_ref (Hz);
        sideband_dbc : single-sideband spur level (dBc) at each offset;
            -inf where the line is exactly zero (no net pulse current),
            NaN where the phase amplitude exceeds 0.2 rad (no dBc);
        line_rad2 : one-sided power of the oscillator phase at each
            offset (rad^2), as in `mash_line_spectrum`;
        ripple_v : amplitude (peak) of the control-voltage ripple at
            each harmonic (V);
        dt_s : the locked divider-minus-reference timing offset (s);
        check : how stability was checked ("per-cycle" with the
            spectral radius, or "averaged" with the crossover and
            phase margin).

    Refuses what `pfd_static_offset` refuses (no locked operating
    point), non-finite or non-physical inputs, a pump polarity and Kvco
    sign that give positive feedback, r3 without c3 (or the reverse),
    and a loop that is not stable (or, with r3 and c3, that the averaged
    model cannot judge): it has no periodic steady state.  The stability
    check uses the pump current whose pulse the phase error moves: UP
    when the reference edge leads, DN when the divider edge leads, both
    when the offset is exactly zero.
    """
    f_ref = float(f_ref_hz)
    icp = float(icp)
    idn = icp if i_dn is None else float(i_dn)
    kv, nd = float(kvco_hz_per_v), float(n_div)
    pol = _polarity(pump_polarity)
    if not (math.isfinite(kv) and kv != 0.0):
        raise ValueError(f"kvco_hz_per_v must be finite and nonzero, got "
                         f"{kvco_hz_per_v!r}")
    if pol * kv < 0.0:
        raise ValueError("pump polarity and Kvco sign give POSITIVE "
                         "feedback: a negative-Kvco oscillator needs "
                         "pump_polarity=-1")
    if not (math.isfinite(nd) and nd >= 1.0):
        raise ValueError(f"n_div must be finite and >= 1, got {n_div!r}")
    nh = n_harmonics
    if isinstance(nh, bool) or not float(nh).is_integer() \
            or not 1 <= int(nh) <= 10 ** 6:
        raise ValueError(f"n_harmonics must be a whole number from 1 to "
                         f"10^6, got {n_harmonics!r}")
    nh = int(nh)
    if (r3 is None) != (c3 is None):
        raise ValueError("give both r3 and c3 for the third-order "
                         "section, or neither")
    br = [(float(r), float(c)) for r, c in branches]
    # the filter's element checks (finite, R >= 0, C > 0)
    loop_filter_impedance(1.0, c_shunt, br)
    if r3 is not None:
        r3, c3 = float(r3), float(c3)
        if not (math.isfinite(r3) and r3 >= 0.0):
            raise ValueError(f"r3 must be finite and >= 0, got {r3!r}")
        if not (math.isfinite(c3) and c3 > 0.0):
            raise ValueError(f"c3 must be finite and positive, got {c3!r}")

    def zfilter(w):
        w = np.atleast_1d(np.asarray(w, dtype=float))
        if r3 is None:
            return loop_filter_impedance(w, c_shunt, br)
        return (loop_filter_impedance(w, c_shunt, br + [(r3, c3)])
                / (1.0 + 1j * w * r3 * c3))

    dt, pulses = _pulses(f_ref, icp, idn, float(i_leak), float(t_reset),
                         float(t_deadzone), pol)

    # the loop must be locked and stable for a periodic steady state
    gains = [icp, idn] if dt == 0.0 else [icp if dt > 0.0 else idn]
    if r3 is None:
        for g in gains:
            st = sampled_stability(g, kv, nd, f_ref, c_shunt, br, pol)
            if not st["stable"]:
                raise ValueError(
                    f"the per-cycle loop is unstable (spectral radius "
                    f"{st['spectral_radius']:.6g} with a pump current of "
                    f"{g:.4g} A): it has no locked steady state, so no "
                    "reference spur to report")
        check = {"model": "per-cycle",
                 "spectral_radius": st["spectral_radius"]}
    else:
        for g in gains:
            try:
                rep = stability(g, abs(kv), nd, zfilter, f_ref)
            except ValueError as err:
                raise ValueError(
                    f"the averaged model cannot confirm a stable lock "
                    f"with the R3-C3 section ({err}); without a stable "
                    "lock there is no reference spur to report") from None
        check = {"model": "averaged", **rep}

    k = np.arange(1, nh + 1)
    w = 2.0 * math.pi * f_ref * k
    ck = np.zeros(nh, dtype=complex)
    for a, b, cur in pulses:
        width = b - a
        ck += (cur * f_ref * width) * np.sinc(w * width / (2.0 * math.pi)) \
            * np.exp(-0.5j * w * (a + b))
    vk = zfilter(w) * ck
    phik = abs(kv) * vk / (1j * k * f_ref)
    line = 2.0 * np.abs(phik) ** 2
    dbc = _small_angle_dbc(line)
    dbc[line == 0.0] = -np.inf
    return {"k": k, "f_hz": k * f_ref, "sideband_dbc": dbc,
            "line_rad2": line, "ripple_v": 2.0 * np.abs(vk),
            "dt_s": float(dt), "check": check}

