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

Since 0.6.0, `third_order_impedance` adds the common third-order
filter: the second-order filter followed by an R3-C3 low-pass section
at the oscillator input (the topology of W. O. Keese, "An Analysis and
Performance Evaluation of a Passive Filter Design Technique for Charge
Pump Phase-Locked Loops", National Semiconductor Application Note
AN-1001 (1996)).  It is a two-port, so what the loop needs is its
TRANSIMPEDANCE, control voltage at C3 per pump current,

    Z(s) = (1 + s R2 C2) / ( s (A0 + A1 s + A2 s^2) ),
    A0 = C1 + C2 + C3,
    A1 = R2 C2 (C1 + C3) + R3 C3 (C1 + C2),
    A2 = C1 C2 C3 R2 R3,

found by nodal analysis; the tests hold the code against this
polynomial, against a nodal-matrix solve and against the R3 -> 0
limit.  It plugs into the averaged model (`open_loop`, `stability`,
`synthesizer_psd`); the per-cycle and edge-by-edge models take only
the shunt-C + R-C family.

Honest limits: these are ideal passive elements.  Element temperature
or voltage dependence (a varactor C2, say) enters by evaluating the
filter at the operating point; the tests and `fracpll.lock` treat the
elements as constant over one analysis.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["loop_filter_impedance", "second_order_impedance",
           "third_order_impedance"]


def _check_w(w):
    w = np.atleast_1d(np.asarray(w, dtype=float))
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("angular frequencies w must be finite and "
                         "positive (evaluate DC limits analytically, "
                         "not numerically)")
    return w


def loop_filter_impedance(w: ArrayLike, c_shunt: float,
                          branches: Sequence[tuple[float, float]] = ()
                          ) -> np.ndarray:
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


def second_order_impedance(w: ArrayLike, c1: float, r2: float,
                           c2: float) -> np.ndarray:
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


def third_order_impedance(w: ArrayLike, c1: float, r2: float, c2: float,
                          r3: float, c3: float) -> np.ndarray:
    """Transimpedance of the third-order passive loop filter.

    C1 to ground, in parallel with a series R2-C2 branch, at the pump
    node; then R3 in series to C3 to ground, whose voltage drives the
    oscillator.  Returns V(C3) / I(pump) (ohm) at the angular
    frequencies w (rad/s), as a complex array:

        Z = Z_node / (1 + j w R3 C3),

    with Z_node the impedance at the pump node, C1 || (R2 + C2) ||
    (R3 + C3) (`loop_filter_impedance` with both branches).  The closed
    polynomial form is in the module docstring.  R3 = 0 gives the
    second-order filter with C1 + C3 to ground.

    c1, c2, c3 : capacitances (F), finite and > 0.
    r2, r3 : resistances (ohm), finite and >= 0.

    Use it as `zfilter` of the averaged model, for example
    ``stability(icp, kvco, n, lambda w: third_order_impedance(w, ...),
    f_ref)``.  The per-cycle (`sampled_*`, `simulate_sampled`) and
    edge-by-edge (`simulate_pll`) models do not take this filter.
    """
    w = _check_w(w)
    for name, v in (("c1", c1), ("c2", c2), ("c3", c3)):
        if not (np.isfinite(float(v)) and float(v) > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    for name, v in (("r2", r2), ("r3", r3)):
        if not (np.isfinite(float(v)) and float(v) >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0")
    z_node = loop_filter_impedance(w, c1, [(r2, c2), (r3, c3)])
    return z_node / (1.0 + 1j * w * float(r3) * float(c3))
