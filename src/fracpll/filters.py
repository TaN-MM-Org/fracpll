"""Passive charge-pump loop filters, as impedances.

The filter of a charge-pump PLL is a one-port: the pump injects its
current into it and the control voltage is the voltage across it, so
everything the loop needs to know is the impedance Z(jw).  This module
computes Z(jw) for the family that covers the standard second- and
third-order passive filters and the varactor-branch filter of the
GaN-on-SOI study this package is distilled from: a shunt capacitor in
parallel with any number of series R-C branches,

    Z(jw) = 1 / ( jw C_shunt + sum_k 1/(R_k + 1/(jw C_k)) ).

The classic second-order filter (integrating C1 shunted by R2+C2) is
the special case of one branch, and its textbook closed form

    Z(s) = (1 + s R2 C2) / ( s (C1+C2) (1 + s R2 C1C2/(C1+C2)) )

is held against the general expression in the tests -- two code paths,
one impedance.

Honest limits: these are ideal passive elements.  Element temperature
or voltage dependence (a varactor C2, say) enters by evaluating the
filter at the operating point; the tests and `fracpll.lock` treat the
elements as constant over one analysis.
"""
from __future__ import annotations

import numpy as np

__all__ = ["loop_filter_impedance", "second_order_impedance"]


def _check_w(w):
    w = np.atleast_1d(np.asarray(w, dtype=float))
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("angular frequencies w must be finite and "
                         "positive (evaluate DC limits analytically, "
                         "not numerically)")
    return w


def loop_filter_impedance(w, c_shunt, branches=()):
    """Impedance of a shunt capacitor with series R-C branches.

    w : angular frequency array (rad/s), positive.
    c_shunt : the shunt capacitance to ground (F), > 0.
    branches : sequence of (R, C) series branches in parallel with it,
        R >= 0 (ohm), C > 0 (F).

    Returns complex Z(jw) with the same shape as w.
    """
    w = _check_w(w)
    c_shunt = float(c_shunt)
    if not (np.isfinite(c_shunt) and c_shunt > 0.0):
        raise ValueError("c_shunt must be finite and positive; a loop "
                         "filter with no shunt path has no defined "
                         "high-frequency impedance")
    s = 1j * w
    y = s * c_shunt
    for k, (r, c) in enumerate(branches):
        r, c = float(r), float(c)
        if not (np.isfinite(r) and r >= 0.0):
            raise ValueError(f"branch {k}: R must be finite and >= 0")
        if not (np.isfinite(c) and c > 0.0):
            raise ValueError(f"branch {k}: C must be finite and > 0")
        y = y + 1.0 / (r + 1.0 / (s * c))
    return 1.0 / y


def second_order_impedance(w, c1, r2, c2):
    """The textbook closed form of the classic second-order filter.

    C1 to ground in parallel with a series R2-C2 branch:

        Z(s) = (1 + s R2 C2) / ( s (C1+C2) (1 + s R2 Cp) ),
        Cp   = C1 C2 / (C1 + C2).

    Kept as an independent code path for the tests; use
    `loop_filter_impedance` for computation.
    """
    w = _check_w(w)
    for name, v in (("c1", c1), ("c2", c2)):
        if not (np.isfinite(float(v)) and float(v) > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if not (np.isfinite(float(r2)) and float(r2) >= 0.0):
        raise ValueError("r2 must be finite and >= 0")
    s = 1j * w
    cp = c1 * c2 / (c1 + c2)
    return (1.0 + s * r2 * c2) / (s * (c1 + c2) * (1.0 + s * r2 * cp))
