"""Exact static phase offset of a tri-state PFD / charge pump.

In the periodic steady state of an integer-N loop the filter's
integrating capacitor returns to the same charge every reference
period, so the NET CHARGE the pump delivers per period must cancel the
leakage exactly.  With the divider edge a time dt after the reference
edge (dt > 0: reference leads), the tri-state PFD holds UP for
dt + t_reset and DN for t_reset (both reset together t_reset after the
later edge), and with a pump turn-on (dead-zone) delay t_dz each pump
delivers current only for max(0, on-time - t_dz).  Charge balance is

    I_up max(0, dt + t_r - t_dz) - I_dn max(0, t_r - t_dz) = I_leak T

for dt >= 0, and symmetrically, with DN leading, for dt < 0:

    I_up max(0, t_r - t_dz) - I_dn max(0, |dt| + t_r - t_dz) = I_leak T.

`pfd_static_offset` solves this EXACTLY (it is piecewise linear), with
no small-signal or averaging assumption.  It is the edge-level answer
that the averaged tanh detector of `fracpll.lock.static_offset`
approximates: the two agree to first order in I_leak/Icp, and the
tests hold the event-driven simulator of `fracpll.eventsim` against
this formula, including mismatch, reset delay and dead zone together.

It refuses where no single operating point exists: inside a dead zone
with no leakage to bias out of it (every dt with |dt| < t_dz - t_r
delivers zero charge, so the loop has no restoring force there and the
phase wanders), and when the demanded leakage charge exceeds what one
full reference period of pumping can supply.
"""
from __future__ import annotations

import math

__all__ = ["pfd_static_offset"]


def pfd_static_offset(f_ref_hz, icp, i_dn=None, i_leak=0.0,
                      t_reset=0.0, t_deadzone=0.0):
    """Exact steady-state divider-minus-reference time offset.

    Returns dict(dt_s, phase_ref_rad) with phase_ref_rad = 2 pi dt f_ref,
    the static phase error in reference radians (positive: the
    reference edge leads).  Leakage flows OUT of the control node.
    """
    f = float(f_ref_hz)
    T = 1.0 / f
    iu = float(icp)
    idn = iu if i_dn is None else float(i_dn)
    il, tr, tdz = float(i_leak), float(t_reset), float(t_deadzone)
    if f <= 0.0 or iu <= 0.0 or idn <= 0.0 or tr < 0.0 or tdz < 0.0:
        raise ValueError("need f_ref, currents > 0 and delays >= 0")
    base = iu * max(0.0, tr - tdz) - idn * max(0.0, tr - tdz)
    need = il * T  # charge the pump must deliver per period
    if need == base:
        if tdz > tr:
            raise ValueError(
                "inside the dead zone with nothing to bias the loop out "
                f"of it: every |dt| < t_deadzone - t_reset = {tdz - tr:.3g}"
                " s delivers zero net charge, so there is no single "
                "operating point -- the phase wanders. Make t_reset >= "
                "t_deadzone (the standard cure) or state the leakage")
        dt = 0.0
    elif need > base:
        # reference leads: UP on for dt + tr
        # iu * (dt + tr - tdz) - idn * max(0, tr - tdz) = need
        dt = (need + idn * max(0.0, tr - tdz)) / iu - tr + tdz
        if dt < 0.0:
            dt = 0.0
    else:
        # divider leads: DN on for |dt| + tr
        # iu * max(0, tr - tdz) - idn * (|dt| + tr - tdz) = need
        adt = (iu * max(0.0, tr - tdz) - need) / idn - tr + tdz
        dt = -max(adt, 0.0)
    if abs(dt) + tr >= T:
        raise ValueError(
            f"the required pump on-time ({abs(dt) + tr:.3g} s) reaches "
            f"the reference period ({T:.3g} s): the pump cannot cancel "
            "this leakage/mismatch within one cycle, so no locked "
            "operating point exists")
    return {"dt_s": dt, "phase_ref_rad": 2.0 * math.pi * dt * f}
