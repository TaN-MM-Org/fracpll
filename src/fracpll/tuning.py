"""Oscillator tuning curves from YOUR measurements.

The oscillator model in this package is the measured tuning curve
f(Vc) itself -- not a shipped parameter set.  `fit_tuning` fits a
shape-preserving monotone cubic (PCHIP) through your measured points
and exposes frequency and gain Kvco = df/dVc; `TuningFamily` carries
one fitted curve per measured temperature and interpolates linearly in
T between them, refusing outside the measured hull.

Refusals do the honest work: non-monotonic measured data is refused
with the offending interval named (a varactor-tuned ring past its
usable bias range does exactly this, and averaging over the fold would
report a Kvco the hardware does not have); evaluation outside the
measured voltage or temperature range is refused rather than
extrapolated.

The gain is computed two ways in the tests -- the analytic derivative
of the fitted PCHIP and a central finite difference across it -- two
code paths, one number.

Since 0.5.0, `fit_linear_vco` fits a straight line f = f0 + Kvco Vc
through measured points, by (weighted) least squares, and returns Kvco
WITH its standard uncertainty, plus a ready `LinearVCO`.  PCHIP passes
through every point and so has no uncertainty of its own; the straight
line averages the scatter of the points and says how well Kvco is
known.  It refuses when a straight line does not describe the points:
by a chi-square test when you state the frequency uncertainty, and by
an F-test for a quadratic term when you do not (standard
least-squares results, for example P. R. Bevington and D. K.
Robinson, "Data Reduction and Error Analysis for the Physical
Sciences", 3rd ed., McGraw-Hill (2003), chapters 6 and 11).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.interpolate import PchipInterpolator

__all__ = ["fit_tuning", "fit_linear_vco", "TuningCurve", "TuningFamily"]


@dataclass(frozen=True)
class TuningCurve:
    """A fitted monotone tuning curve at one condition."""
    vc: np.ndarray
    f_hz: np.ndarray
    reference: str
    label: str = ""
    _interp: PchipInterpolator = field(init=False, repr=False,
                                       compare=False, default=None)

    def __post_init__(self):
        interp = PchipInterpolator(self.vc, self.f_hz)
        object.__setattr__(self, "_interp", interp)

    def _check(self, vc):
        v = np.atleast_1d(np.asarray(vc, dtype=float))
        lo, hi = self.vc[0], self.vc[-1]
        if np.any(v < lo - 1e-12) or np.any(v > hi + 1e-12):
            raise ValueError(
                f"control voltage outside the measured range "
                f"[{lo:.4g}, {hi:.4g}] V ({self.reference}); this "
                "curve does not extrapolate")
        return v

    def frequency(self, vc: ArrayLike) -> np.ndarray:
        """f(Vc) in Hz on the measured range."""
        return self._interp(self._check(vc))

    def kvco(self, vc: ArrayLike) -> np.ndarray:
        """Tuning gain df/dVc in Hz/V (signed)."""
        return self._interp.derivative()(self._check(vc))


def fit_tuning(vc: ArrayLike, f_hz: ArrayLike, reference: str,
               label: str = "") -> TuningCurve:
    """Fit a monotone tuning curve through measured (Vc, f) points.

    vc : control voltages (V), strictly increasing, >= 4 points.
    f_hz : measured frequencies (Hz), strictly monotone (either
        direction -- negative-Kvco oscillators are first-class here,
        the co-designed GaN ring is one).
    reference : where the measurement comes from (mandatory, >= 8
        characters).
    """
    v = np.asarray(vc, dtype=float)
    f = np.asarray(f_hz, dtype=float)
    if not isinstance(reference, str) or len(reference.strip()) < 8:
        raise ValueError("a tuning curve requires a real reference "
                         "(instrument+date or citation)")
    if v.size < 4 or v.size != f.size:
        raise ValueError("need >= 4 (Vc, f) points of equal length "
                         "(fewer cannot expose curvature in Kvco)")
    if not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0.0):
        raise ValueError("Vc must be finite and strictly increasing")
    if not np.all(np.isfinite(f)) or np.any(f <= 0.0):
        raise ValueError("frequencies must be finite and positive")
    df = np.diff(f)
    if np.all(df > 0.0):
        pass
    elif np.all(df < 0.0):
        pass
    else:
        k = int(np.flatnonzero(np.sign(df[1:]) != np.sign(df[0]))[0]) + 1
        raise ValueError(
            "the measured tuning is not monotone: the slope changes "
            f"sign near Vc = {v[k]:.4g} V. A folded tuning curve has "
            "no single-valued Kvco there; restrict the fit to the "
            "monotone bias range the loop will actually use")
    return TuningCurve(vc=v.copy(), f_hz=f.copy(),
                       reference=reference, label=label)


def fit_linear_vco(vc: ArrayLike, f_hz: ArrayLike, reference: str,
                   f_sigma_hz: float | ArrayLike | None = None,
                   vc_window: tuple[float, float] | None = None,
                   p_min: float = 1e-3) -> dict[str, Any]:
    """Straight-line fit f = f0 + Kvco Vc, with honest uncertainties.

    vc : measured control voltages (V), strictly increasing.
    f_hz : measured frequencies (Hz) at those voltages.
    reference : where the measurement comes from (mandatory, >= 8
        characters), passed on to the returned `LinearVCO`.
    f_sigma_hz : the standard uncertainty of each frequency reading
        (Hz), one number or one per point.  If given, the fit is
        weighted by 1/sigma^2, the parameter uncertainties follow from
        these sigmas, and a chi-square test checks that the points
        scatter about the line as much as the sigmas say.  If None,
        the uncertainties are estimated from the scatter of the points
        about the line (residuals, n - 2 degrees of freedom), and an
        F-test checks for curvature (a quadratic term).
    vc_window : (lo, hi) in V: fit only the points with lo <= Vc <= hi,
        for example a window around the lock voltage, where a straight
        line describes a curved tuning law well enough.
    p_min : the refusal threshold for both tests (default 1e-3).

    Returns a dict with
        kvco_hz_per_v, u_kvco_hz_per_v : slope and standard uncertainty;
        f0_hz, u_f0_hz : the frequency at Vc = 0 and its uncertainty;
        cov : 2x2 covariance of (f0_hz, kvco_hz_per_v);
        vc_mean, f_at_vc_mean_hz, u_f_at_vc_mean_hz : the (weighted)
            mean voltage and the fitted frequency there, which is
            uncorrelated with the slope;
        n_points, dof (n - 2), residuals_hz;
        chi2, p_value : the chi-square test (stated sigmas) or the
            curvature F-test (no sigmas), and `test` naming which;
        sigma_source : "stated" or "residuals";
        vco : a `LinearVCO` with the fitted f0 and Kvco.
    With sigma_source "residuals", the uncertainties have only n - 2
    degrees of freedom; for a coverage interval use Student's t with
    `dof` degrees of freedom, not a Gaussian factor.

    Refuses fewer than 3 points with stated sigmas (4 without, so that
    curvature can be tested), non-finite values, a non-increasing Vc,
    non-positive sigmas, points that lie exactly on a line with no
    stated sigma (zero scatter gives no uncertainty), and data that a
    straight line does not describe (test p-value below p_min); then it
    suggests a narrower vc_window or `fit_tuning`.
    """
    from .eventsim import LinearVCO
    x = np.asarray(vc, dtype=float).ravel()
    y = np.asarray(f_hz, dtype=float).ravel()
    if not isinstance(reference, str) or len(reference.strip()) < 8:
        raise ValueError("a tuning fit requires a real reference "
                         "(instrument+date or citation, >= 8 characters)")
    if x.size != y.size:
        raise ValueError(f"vc and f_hz differ in length ({x.size} and "
                         f"{y.size})")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("vc and f_hz must be finite")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("vc must be strictly increasing")
    sig = None
    if f_sigma_hz is not None:
        sig = np.broadcast_to(np.asarray(f_sigma_hz, dtype=float),
                              x.shape).copy()
        if not (np.all(np.isfinite(sig)) and np.all(sig > 0.0)):
            raise ValueError("f_sigma_hz must be finite and positive")
    pm = float(p_min)
    if not (0.0 <= pm < 1.0):
        raise ValueError(f"p_min must be in [0, 1), got {p_min!r}")
    if vc_window is not None:
        lo, hi = (float(v) for v in vc_window)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError("vc_window must be (lo, hi) with lo < hi")
        keep = (x >= lo) & (x <= hi)
        x, y = x[keep], y[keep]
        if sig is not None:
            sig = sig[keep]
    n_min = 3 if sig is not None else 4
    if x.size < n_min:
        raise ValueError(
            f"need >= {n_min} points (in the window) for a straight-line "
            "fit with an uncertainty and a test of the line "
            f"({'3 with stated sigmas' if sig is not None else '4 without stated sigmas, to test for curvature'}); "
            f"got {x.size}")
    w = np.ones_like(x) if sig is None else 1.0 / sig ** 2
    sw = float(np.sum(w))
    xm = float(np.sum(w * x) / sw)
    ym = float(np.sum(w * y) / sw)
    dx = x - xm
    sxx = float(np.sum(w * dx * dx))
    kv = float(np.sum(w * dx * (y - ym)) / sxx)
    resid = y - (ym + kv * dx)
    dof = x.size - 2
    if sig is not None:
        scale = 1.0
        chi2 = float(np.sum(w * resid ** 2))
        p = float(stats.chi2.sf(chi2, dof))
        test = "chi-square of the line with the stated sigmas"
        if p < pm:
            raise ValueError(
                f"a straight line does not describe these points within "
                f"their stated uncertainties (chi-square {chi2:.4g} for "
                f"{dof} degrees of freedom, p = {p:.3g} < p_min = "
                f"{pm:g}). The tuning law is curved here, or the sigmas "
                "are too small: fit a narrower vc_window around the "
                "operating point, or use fit_tuning for the whole curve")
    else:
        rss1 = float(np.sum(resid ** 2))
        if rss1 == 0.0:
            raise ValueError(
                "the points lie exactly on a straight line, so their "
                "scatter gives no uncertainty; state the frequency "
                "uncertainty with f_sigma_hz")
        scale = rss1 / dof
        # nested quadratic fit, in the centered variable for conditioning
        a = np.column_stack((np.ones_like(dx), dx, dx * dx))
        coef = np.linalg.lstsq(a, y - ym, rcond=None)[0]
        rss2 = float(np.sum((y - ym - a @ coef) ** 2))
        d2 = x.size - 3
        chi2 = np.inf if rss2 == 0.0 else (rss1 - rss2) / (rss2 / d2)
        p = float(stats.f.sf(chi2, 1, d2)) if np.isfinite(chi2) else 0.0
        chi2 = float(chi2)
        test = "F-test for a quadratic term (no stated sigmas)"
        if p < pm:
            raise ValueError(
                f"the points curve: a quadratic term is significant "
                f"(F = {chi2:.4g} with 1 and {d2} degrees of freedom, "
                f"p = {p:.3g} < p_min = {pm:g}), so one straight-line "
                "Kvco would misdescribe them. Fit a narrower vc_window "
                "around the operating point, or use fit_tuning for the "
                "whole curve")
    var_k = scale / sxx
    var_c = scale / sw
    f0 = ym - kv * xm
    cov = np.array([[var_c + xm * xm * var_k, -xm * var_k],
                    [-xm * var_k, var_k]])
    vco = LinearVCO(f0_hz=float(f0), kvco_hz_per_v=kv, reference=reference)
    return {"kvco_hz_per_v": kv, "u_kvco_hz_per_v": float(np.sqrt(var_k)),
            "f0_hz": float(f0), "u_f0_hz": float(np.sqrt(cov[0, 0])),
            "cov": cov, "vc_mean": xm, "f_at_vc_mean_hz": ym,
            "u_f_at_vc_mean_hz": float(np.sqrt(var_c)),
            "n_points": int(x.size), "dof": int(dof),
            "residuals_hz": resid, "chi2": chi2, "p_value": p,
            "test": test,
            "sigma_source": "stated" if sig is not None else "residuals",
            "vco": vco}


@dataclass(frozen=True)
class TuningFamily:
    """Measured tuning curves at several temperatures.

    curves : mapping T (K) -> TuningCurve.  Interpolation in T is
    linear between measured corners; outside the measured span it
    refuses.
    """
    curves: Mapping[float, TuningCurve]

    def __post_init__(self):
        if len(self.curves) < 2:
            raise ValueError("a family needs >= 2 measured "
                             "temperatures (one curve is a "
                             "TuningCurve, use it directly)")
        for t in self.curves:
            float(t)

    def _pair(self, t_k):
        ts = np.array(sorted(float(t) for t in self.curves))
        t = float(t_k)
        if t < ts[0] - 1e-9 or t > ts[-1] + 1e-9:
            raise ValueError(
                f"temperature {t:.4g} K outside the measured span "
                f"[{ts[0]:.4g}, {ts[-1]:.4g}] K; this family does not "
                "extrapolate in temperature")
        i = int(np.clip(np.searchsorted(ts, t) - 1, 0, ts.size - 2))
        t0, t1 = ts[i], ts[i + 1]
        x = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return self.curves[t0], self.curves[t1], x

    def frequency(self, vc: ArrayLike, t_k: float) -> np.ndarray:
        """f(Vc) in Hz at temperature t_k (K), linear in T between the
        two nearest measured curves."""
        c0, c1, x = self._pair(t_k)
        return (1.0 - x) * c0.frequency(vc) + x * c1.frequency(vc)

    def kvco(self, vc: ArrayLike, t_k: float) -> np.ndarray:
        """Tuning gain df/dVc in Hz/V (signed) at temperature t_k (K),
        linear in T between the two nearest measured curves."""
        c0, c1, x = self._pair(t_k)
        return (1.0 - x) * c0.kvco(vc) + x * c1.kvco(vc)
