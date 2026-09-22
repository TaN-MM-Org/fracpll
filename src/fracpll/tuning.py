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
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["fit_tuning", "TuningCurve", "TuningFamily"]


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

    def frequency(self, vc):
        """f(Vc) in Hz on the measured range."""
        return self._interp(self._check(vc))

    def kvco(self, vc):
        """Tuning gain df/dVc in Hz/V (signed)."""
        return self._interp.derivative()(self._check(vc))


def fit_tuning(vc, f_hz, reference, label=""):
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


@dataclass(frozen=True)
class TuningFamily:
    """Measured tuning curves at several temperatures.

    curves : mapping T (K) -> TuningCurve.  Interpolation in T is
    linear between measured corners; outside the measured span it
    refuses.
    """
    curves: dict

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

    def frequency(self, vc, t_k):
        """f(Vc) in Hz at temperature t_k (K), linear in T between the
        two nearest measured curves."""
        c0, c1, x = self._pair(t_k)
        return (1.0 - x) * c0.frequency(vc) + x * c1.frequency(vc)

    def kvco(self, vc, t_k):
        """Tuning gain df/dVc in Hz/V (signed) at temperature t_k (K),
        linear in T between the two nearest measured curves."""
        c0, c1, x = self._pair(t_k)
        return (1.0 - x) * c0.kvco(vc) + x * c1.kvco(vc)
