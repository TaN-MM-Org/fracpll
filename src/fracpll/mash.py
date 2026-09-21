"""Exact MASH delta-sigma division sequences and their phase noise.

A fractional-N synthesizer reaches a non-integer mean division ratio
N + num/den by dithering the integer divider with the carry pattern of
a MASH accumulator cascade.  `mash_sequence` generates that pattern
with EXACT integer arithmetic -- no floating point enters, so the mean
division offset over one period equals num/den exactly, and the tests
check it with `fractions.Fraction`.

Two descriptions of the resulting phase error are provided.

1. THE EXACT LINE SPECTRUM (`mash_line_spectrum`).  The MASH state map
   (acc_1 <- acc_1 + num, acc_i <- acc_i + acc_(i-1), all mod den) is a
   bijection on a finite set, so the sequence from the zero state is
   PERIODIC; its period is found exactly by iterating the integer
   state, and the discrete Fourier transform over one period gives the
   phase error as a set of spectral LINES at k f_ref / P with exact
   powers -- no statistical model at all.  This holds for every order,
   including m = 1, whose output is a spur spectrum rather than noise.
   It is checked against the closed form of the first-order sawtooth
   and by Parseval (line powers sum to the time-domain variance).

2. THE ADDITIVE-NOISE PSD (`dsm_phase_psd`), for m = 2, 3 with long
   periods, where the lines are dense enough to act as a continuum:

    S_phi(f) = (2 pi)^2 / (6 f_ref) * (2 sin(pi f/f_ref))^(2(m-1))
                                                        [rad^2/Hz].

   Derivation: the divider count error is the quantization error e of
   an m-th order noise-shaper, NTF (1 - z^-1)^m, with e modelled as
   white with variance 1/12 (count^2).  A count error of dN cycles in
   one period is a frequency error dN f_ref at the OSCILLATOR, and the
   phase is its running sum, 2 pi sum(dN) oscillator radians -- one
   factor (1 - z^-1) cancels, leaving (2 sin(pi f/f_ref))^(2(m-1)).
   The two-sided density of the white sequence is (1/12)/f_ref; one
   side doubles it.  This is equation (3.7) of W. Rhee, "Multi-Bit
   Delta-Sigma Modulation Technique for Fractional-N Frequency
   Synthesizers", Ph.D. thesis, University of Illinois at
   Urbana-Champaign (2001), written there as the single-sideband
   L(f) = (2 pi)^2/(12 f_PD) (2 sin(pi f/f_PD))^(2(m-1)); L(f) is half
   of the one-sided S_phi used throughout this package.

   REFERRAL (stated because it is easy to get wrong): this PSD is
   already OSCILLATOR-referred.  In the closed loop it reaches the
   output through |L/(1+L)|^2 alone, with NO factor N^2 -- see
   `fracpll.noise.synthesizer_psd`.  fracpll 0.1.0 applied N^2 here, a
   bug that overstated delta-sigma noise by N^2; the edge-level
   simulator of `fracpll.eventsim` measures the no-N^2 result, and the
   tests hold it.

The formula is not taken on authority: the test suite also measures
the Welch PSD of exact MASH-2 and MASH-3 sequences against it.  For
m = 1 the additive model fails (the output is a line spectrum), so
`dsm_phase_psd` refuses m = 1 and points to `mash_line_spectrum`.
Delta-sigma fractional-N synthesis itself goes back to Riley, Copeland
and Kwasniewski, IEEE J. Solid-State Circuits 28, 553 (1993).
"""
from __future__ import annotations

import numpy as np

__all__ = ["mash_sequence", "dsm_phase_psd", "MASH_RANGE",
           "mash_period", "mash_line_spectrum"]

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
            "to report it. Use mash_line_spectrum for the exact spur "
            "lines, or m >= 2")
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


#: largest one-sided line power (rad^2) reported in dBc: amplitude
#: sqrt(2 * 0.02) = 0.2 rad, where J1(x) ~ x/2 holds to 0.5 %
SMALL_ANGLE_MAX_RAD2 = 0.02


def _small_angle_dbc(line_rad2):
    line = np.asarray(line_rad2, dtype=float)
    out = np.full(line.shape, np.nan)
    ok = (line > 0.0) & (line <= SMALL_ANGLE_MAX_RAD2)
    out[ok] = 10.0 * np.log10(line[ok] / 2.0)
    return out


def _mash_state_step(acc, num, den):
    inp = num
    out = []
    for a in acc:
        a = (a + inp) % den
        out.append(a)
        inp = a
    return out


def mash_period(num, den, m, max_period=1 << 22):
    """Exact period of the MASH-m state from the zero state.

    The state map is a bijection on a finite set, so the orbit of zero
    returns to zero; the first return is the period.  Refuses beyond
    `max_period` rather than guessing.
    """
    num, den, m = int(num), int(den), int(m)
    if den < 2 or not (0 <= num < den) or m not in (1, 2, 3):
        raise ValueError("need 0 <= num < den, den >= 2, m in {1,2,3}")
    acc = [0] * m
    for k in range(1, int(max_period) + 1):
        acc = _mash_state_step(acc, num, den)
        if all(a == 0 for a in acc):
            return k
    raise ValueError(f"period exceeds max_period = {max_period}; use "
                     "dsm_phase_psd (continuum model) for long periods")


def mash_line_spectrum(num, den, m, f_ref_hz, max_period=1 << 22):
    """Exact line spectrum of the MASH-m divider phase error.

    The phase is 2 pi sum_(j<k) (d_j - num/den), in OSCILLATOR radians
    (the same referral as `dsm_phase_psd`), over one exact period
    starting after the (m - 1)-cycle start-up of the differencing.

    Returns dict(period, f_hz, line_rad2, sideband_dbc, variance_rad2):
    one-sided line powers (rad^2) at f = k f_ref / period for
    k = 1..period//2, their single-sideband level 10 log10(line_rad2/2)
    in dBc -- reported only where the small-angle approximation holds
    (line_rad2 <= 0.02, a phase amplitude of 0.2 rad or less) and NaN
    elsewhere, because a dBc number is meaningless for large phase
    modulation (open-loop divider lines of a MASH-1 are of order 1 rad;
    pass them through `fracpll.noise.closed_loop_lines` first) -- and
    the phase variance, which the line powers sum to exactly
    (Parseval; asserted in the tests).
    """
    P = mash_period(num, den, m, max_period)
    f_ref = float(f_ref_hz)
    if not (np.isfinite(f_ref) and f_ref > 0.0):
        raise ValueError("f_ref_hz must be finite and positive")
    s0 = int(m) - 1
    d = mash_sequence(num, den, m, s0 + 2 * P)
    if not np.array_equal(d[s0:s0 + P], d[s0 + P:s0 + 2 * P]):
        raise RuntimeError("sequence not periodic with the state period "
                           "(internal consistency check failed)")
    win = d[s0:s0 + P]
    # exact integer running sum of den*(d - num/den)
    cum = np.concatenate(([0], np.cumsum(win * int(den) - int(num))[:-1]))
    if int(np.sum(win * int(den) - int(num))) != 0:
        raise RuntimeError("period does not return the phase exactly")
    phase = 2.0 * np.pi * cum.astype(float) / float(den)
    c = np.fft.fft(phase) / P
    kmax = P // 2
    k = np.arange(1, kmax + 1)
    line = 2.0 * np.abs(c[1:kmax + 1]) ** 2
    if P % 2 == 0:
        line[-1] = np.abs(c[kmax]) ** 2
    var = float(np.mean((phase - phase.mean()) ** 2))
    dbc = _small_angle_dbc(line)
    return {"period": P, "f_hz": k * f_ref / P, "line_rad2": line,
            "sideband_dbc": dbc, "variance_rad2": var}
