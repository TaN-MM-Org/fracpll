"""Exact MASH delta-sigma division sequences and their phase noise.

A fractional-N synthesizer reaches a non-integer mean division ratio
N + num/den by dithering the integer divider with the carry pattern of
a MASH accumulator cascade.  `mash_sequence` generates that pattern
with EXACT integer arithmetic -- no floating point enters, so the mean
division offset over one period equals num/den exactly, and the tests
check it with `fractions.Fraction`.

`dsm_phase_psd` is the one-sided PSD of the resulting divider phase
error,

    S_phi(f) = (2 pi)^2 / (6 f_ref) * (2 sin(pi f/f_ref))^(2(m-1))
                                                        [rad^2/Hz],

which follows from quantization noise of variance 1/12 cycle^2 spread
over the two-sided reference band and shaped by the (m-1) net
differentiations that survive the frequency-to-phase integration.
The factor is a CONVENTION statement: this package is one-sided
throughout (integrate S once, never twice), which is why the
constant is (2 pi)^2/6 and not the (2 pi)^2/12 seen in two-sided
writing.  The formula is not taken on authority: the test suite
generates exact MASH-2 and MASH-3 sequences, measures their Welch
PSD, and holds it against this expression.

Honest limits, stated plainly: for m = 1 a plain accumulator produces
DISCRETE SPURS, not noise-shaped quantization noise -- the white-noise
model behind the formula simply does not apply, and our own simulation
shows it (the smooth formula misses the line spectrum entirely).
`dsm_phase_psd` therefore refuses m = 1 instead of returning a number
that the measurement would contradict.  Delta-sigma fractional-N
synthesis itself goes back to Riley, Copeland and Kwasniewski, IEEE J.
Solid-State Circuits 28, 553 (1993).
"""
from __future__ import annotations

import numpy as np

__all__ = ["mash_sequence", "dsm_phase_psd", "MASH_RANGE"]

#: inclusive (min, max) division offset of a MASH-m cascade
MASH_RANGE = {1: (0, 1), 2: (-1, 2), 3: (-3, 4)}


def mash_sequence(num, den, m, n_steps):
    """Integer division offsets d[k] of a MASH-m cascade.

    num, den : the fraction num/den in [0, 1), integers, den >= 2.
    m : modulator order, 1 <= m <= 3.
    n_steps : number of reference cycles to generate.

    Accumulators start at zero.  Returns an int64 array; add it to the
    integer part N to get the per-cycle division ratio.
    """
    num, den, m, n_steps = int(num), int(den), int(m), int(n_steps)
    if den < 2 or not (0 <= num < den):
        raise ValueError("need integers 0 <= num < den with den >= 2")
    if m not in (1, 2, 3):
        raise ValueError("MASH order m must be 1, 2 or 3")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    acc = [0] * m
    carries = np.zeros((m, n_steps), dtype=np.int64)
    for k in range(n_steps):
        inp = num
        for i in range(m):
            acc[i] += inp
            if acc[i] >= den:
                acc[i] -= den
                carries[i, k] = 1
            inp = acc[i]
    y = carries[0].copy()
    for i in range(1, m):
        d = carries[i].copy()
        for _ in range(i):
            d = np.diff(d, prepend=0)
        y += d
    return y


def dsm_phase_psd(f_hz, f_ref_hz, m):
    """One-sided divider phase-noise PSD (rad^2/Hz) of a MASH-m.

    Valid for m = 2 or 3.  m = 1 is refused: a first-order accumulator
    produces discrete spurs and the white-quantization-noise model
    does not describe it (see the module docstring; our simulation
    shows the failure directly).
    """
    m = int(m)
    if m == 1:
        raise ValueError(
            "a first-order accumulator produces discrete spurs, not "
            "noise-shaped quantization noise; the white-noise PSD "
            "model does not apply at m = 1 and this package refuses "
            "to report it. Use m >= 2, or analyze the spur lines of "
            "the exact sequence from mash_sequence")
    if m not in (2, 3):
        raise ValueError("MASH order m must be 2 or 3 here")
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    f_ref = float(f_ref_hz)
    if not (np.isfinite(f_ref) and f_ref > 0.0):
        raise ValueError("f_ref_hz must be finite and positive")
    if not np.all(np.isfinite(f)) or np.any(f <= 0.0) \
            or np.any(f > f_ref / 2.0):
        raise ValueError("offsets must satisfy 0 < f <= f_ref/2 (the "
                         "sampled spectrum has no content beyond "
                         "Nyquist)")
    return (2.0 * np.pi) ** 2 / (6.0 * f_ref) \
        * (2.0 * np.sin(np.pi * f / f_ref)) ** (2 * (m - 1))
