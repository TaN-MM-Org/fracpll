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

Since 0.9.0 both functions also take method="loglog", for a jitter
integrated with `rms_jitter(..., method="loglog")` (straight lines on
log-log axes between the points, the right rule for a few points far
apart).  That integral is not a weighted sum of the S_i, so the
weights w_i S_i above become its first-order sensitivities
g_i = S_i dJ^2/dS_i = dJ^2/d(ln S_i), taken in closed form segment by
segment (for the trapezoid rule g_i = w_i S_i, the same formula).
The log-log integral scales with the PSD (doubling every S_i doubles
it), so sum_i g_i = J^2 exactly, which the tests check, together with
the g_i against finite differences and the error bar against a seeded
Monte Carlo.  Being first order, it holds for small scatter, that is
for a few averages or more.

Honest limits: the bins are treated as independent, which holds for
non-overlapping averages; heavily overlapped Welch segments correlate
neighboring estimates and the true error is then somewhat smaller
than planned -- the plan errs on the safe (conservative) side.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._checks import whole

__all__ = ["jitter_relative_sigma", "averages_for_jitter"]


def _weights(f):
    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or f.size < 3 or not np.all(np.isfinite(f)) \
            or np.any(np.diff(f) <= 0.0):
        raise ValueError("need a finite, strictly increasing offset grid "
                         "with >= 3 points")
    w = np.empty_like(f)
    w[0] = 0.5 * (f[1] - f[0])
    w[-1] = 0.5 * (f[-1] - f[-2])
    w[1:-1] = 0.5 * (f[2:] - f[:-2])
    return w


def _loglog_sensitivities(f, s):
    """g_i = d I / d(ln S_i) of the log-log integral I of
    `fracpll.noise.rms_jitter(method="loglog")`.

    Segment i is I_i = A E(x) with A = S_i f_i L, L = ln(f_(i+1)/f_i),
    x = ln(S_(i+1) f_(i+1) / (S_i f_i)) and E(x) = (e^x - 1)/x.  Since
    dA/d(ln S_i) = A, dx/d(ln S_i) = -1 and dx/d(ln S_(i+1)) = +1, it
    adds A (E - E') to g_i and A E' to g_(i+1), with
    E'(x) = (x e^x - e^x + 1)/x^2 (series 1/2 + x/3 + x^2/8 near 0)."""
    L = np.log(f[1:] / f[:-1])
    x = np.log((s[1:] * f[1:]) / (s[:-1] * f[:-1]))
    a = s[:-1] * f[:-1] * L
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    e = np.where(small, 1.0 + x / 2.0 + x * x / 6.0, np.expm1(xs) / xs)
    de = np.where(small, 0.5 + x / 3.0 + x * x / 8.0,
                  (xs * np.exp(xs) - np.expm1(xs)) / (xs * xs))
    g = np.zeros_like(s)
    g[:-1] += a * (e - de)
    g[1:] += a * de
    return g


def jitter_relative_sigma(f_hz: ArrayLike, s_out: ArrayLike,
                          n_avg: int, method: str = "trapezoid") -> float:
    """Predicted relative 1-sigma of the RMS-jitter estimate.

    f_hz : offset grid (Hz), increasing; each point is one analyzer
        reading, averaged n_avg times, independent of the others.
    s_out : one-sided phase-noise PSD on that grid (rad^2/Hz).
    n_avg : number of averages, a whole number >= 1 (before 0.8.0 a
        fraction such as 2.5 was cut to 2 without a word).
    method : the rule the jitter is integrated with, as for
        `rms_jitter`: "trapezoid" (the default, as before 0.9.0) or
        "loglog" (needs S > 0 at every point; module docstring).
    Returns sigma(J)/J, dimensionless.
    """
    n = whole(n_avg, "n_avg")
    if n < 1:
        raise ValueError(f"n_avg must be >= 1, got {n_avg!r}")
    if method not in ("trapezoid", "loglog"):
        raise ValueError(f"method must be 'trapezoid' or 'loglog', got "
                         f"{method!r}")
    s = np.asarray(s_out, dtype=float)
    w = _weights(f_hz)
    if s.shape != w.shape or np.any(s < 0.0) \
            or not np.all(np.isfinite(s)):
        raise ValueError("s_out must be finite and >= 0 on the same "
                         "grid as f")
    if method == "loglog":
        if np.any(s <= 0.0):
            raise ValueError("method='loglog' needs a positive PSD at "
                             "every point (a power law cannot pass "
                             "through zero)")
        g = _loglog_sensitivities(np.asarray(f_hz, dtype=float), s)
        return float(0.5 * np.sqrt(np.sum(g ** 2)) / np.sum(g)
                     / np.sqrt(n))
    num = np.sqrt(np.sum((w * s) ** 2))
    den = np.sum(w * s)
    if den <= 0.0:
        raise ValueError("the PSD integrates to zero; there is no "
                         "jitter to estimate")
    return float(0.5 * num / den / np.sqrt(n))


def averages_for_jitter(target_rel_sigma: float, f_hz: ArrayLike,
                        s_out: ArrayLike,
                        method: str = "trapezoid") -> tuple[int, float]:
    """Averages needed for a target relative jitter error bar.

    Exact inversion of `jitter_relative_sigma` (1/sqrt(n) law):
    returns (n, achieved) with achieved <= target and n-1 failing it,
    which the tests verify on both sides.  The estimate
    ceil((sigma_1/target)^2) is checked against `jitter_relative_sigma`
    itself and moved by one where rounding put it on the wrong side
    (before 0.5.0 a target equal to the error bar of exactly n
    averages could return n + 1).  method : as for
    `jitter_relative_sigma` ("trapezoid" or "loglog").
    """
    tgt = float(target_rel_sigma)
    if not (np.isfinite(tgt) and tgt > 0.0):
        raise ValueError("target_rel_sigma must be finite and positive, "
                         f"got {target_rel_sigma!r}")
    one = jitter_relative_sigma(f_hz, s_out, 1, method)
    est = (one / tgt) ** 2
    if not est < 1e15:
        raise ValueError(
            f"target_rel_sigma = {tgt:.3g} needs about {est:.3g} "
            "averages, more than any measurement can make")
    n = max(1, int(np.ceil(est)))
    while n > 1 and jitter_relative_sigma(f_hz, s_out, n - 1,
                                          method) <= tgt:
        n -= 1
    while jitter_relative_sigma(f_hz, s_out, n, method) > tgt:
        n += 1
    return n, jitter_relative_sigma(f_hz, s_out, n, method)
