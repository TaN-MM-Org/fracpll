"""Plan the jitter measurement before spending the analyzer time.

An averaged phase-noise measurement estimates each PSD bin with a
relative scatter of 1/sqrt(n_avg).  The jitter integral inherits that
scatter in closed form: writing the estimate J^2 = sum_i w_i S_i
(trapezoid weights w_i) with independent per-bin estimates
S_i_hat = S_i (1 + e_i/sqrt(n)), e_i unit variance,

    relative sigma of J = (1/2) * sqrt( sum (w_i S_i)^2 )
                          / ( sum w_i S_i ) / sqrt(n_avg),

so `averages_for_jitter` inverts this exactly for the number of
averages a target error bar costs -- the plan and the estimator are
two readings of the same weights.  A seeded Monte Carlo with
exponentially distributed periodogram bins (the actual statistics of
a one-shot periodogram) holds the closed form in the tests.

Honest limits: the bins are treated as independent, which holds for
non-overlapping averages; heavily overlapped Welch segments correlate
neighboring estimates and the true error is then somewhat smaller
than planned -- the plan errs on the safe (conservative) side.
"""
from __future__ import annotations

import numpy as np

__all__ = ["jitter_relative_sigma", "averages_for_jitter"]


def _weights(f):
    f = np.asarray(f, dtype=float)
    if f.size < 3 or np.any(np.diff(f) <= 0.0):
        raise ValueError("need an increasing offset grid with >= 3 "
                         "points")
    w = np.empty_like(f)
    w[0] = 0.5 * (f[1] - f[0])
    w[-1] = 0.5 * (f[-1] - f[-2])
    w[1:-1] = 0.5 * (f[2:] - f[:-2])
    return w


def jitter_relative_sigma(f_hz, s_out, n_avg):
    """Predicted relative 1-sigma of the RMS-jitter estimate."""
    n = int(n_avg)
    if n < 1:
        raise ValueError("n_avg must be >= 1")
    s = np.asarray(s_out, dtype=float)
    w = _weights(f_hz)
    if s.shape != w.shape or np.any(s < 0.0):
        raise ValueError("s_out must be >= 0 on the same grid as f")
    num = np.sqrt(np.sum((w * s) ** 2))
    den = np.sum(w * s)
    if den <= 0.0:
        raise ValueError("the PSD integrates to zero; there is no "
                         "jitter to estimate")
    return float(0.5 * num / den / np.sqrt(n))


def averages_for_jitter(target_rel_sigma, f_hz, s_out):
    """Averages needed for a target relative jitter error bar.

    Exact inversion of `jitter_relative_sigma` (1/sqrt(n) law):
    returns (n, achieved) with achieved <= target and n-1 failing it,
    which the tests verify on both sides.
    """
    tgt = float(target_rel_sigma)
    if not (np.isfinite(tgt) and tgt > 0.0):
        raise ValueError("target_rel_sigma must be positive")
    one = jitter_relative_sigma(f_hz, s_out, 1)
    n = int(np.ceil((one / tgt) ** 2))
    return n, jitter_relative_sigma(f_hz, s_out, n)
