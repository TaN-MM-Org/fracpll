"""Exact log-log integration of phase noise (0.4.0).

`rms_jitter(..., method="loglog")` and `NoiseSpec.phase_variance`
integrate a PSD joined by straight lines on log-log axes exactly.
Anchors: the closed-form integral of a power law (every slope,
including the logarithmic case -1), a dense-grid trapezoid of the
spec's own interpolation, and a hand-summed piecewise power law."""
import numpy as np
import pytest

from fracpll import NoiseSpec, dbc_to_psd, rms_jitter

REF = "placeholder numbers, fracpll test suite 2026-09-30"
# np.trapezoid is NumPy >= 2.0; np.trapz is the same rule before that.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@pytest.mark.parametrize("a", [-3.0, -2.0, -1.0, -1.0 + 1e-13, 0.0, 1.5])
def test_power_law_closed_form(a):
    f = np.array([1e3, 1e4, 1e5, 1e6])         # coarse: decade points
    amp = 1e-3
    s = amp * f ** a
    f0 = 1e9
    eps = a + 1.0
    if abs(eps) < 1e-6:
        # int f^(eps-1) df = sum_k eps^k (ln^(k+1) f2 - ln^(k+1) f1)/(k+1)!
        # (the textbook form below cancels catastrophically here)
        l1, l2 = np.log(f[0]), np.log(f[-1])
        var = amp * ((l2 - l1) + eps * (l2 ** 2 - l1 ** 2) / 2
                     + eps ** 2 * (l2 ** 3 - l1 ** 3) / 6)
    else:
        var = amp / (a + 1) * (f[-1] ** (a + 1) - f[0] ** (a + 1))
    exact = np.sqrt(var) / (2 * np.pi * f0)
    assert abs(rms_jitter(f, s, f0, method="loglog") / exact - 1) < 1e-12


def test_noisespec_variance_is_its_own_exact_integral():
    spec = NoiseSpec((1e4, 1e5, 1e6, 1e7), (-65.0, -92.0, -115.0, -135.0),
                     reference=REF)
    lo, hi = 2e4, 5e6                      # limits between measured points
    var = spec.phase_variance(lo, hi)
    # route 1: dense trapezoid of the spec's interpolation
    fd = np.geomspace(lo, hi, 400001)
    dense = _trapezoid(spec.psd(fd), fd)
    assert abs(var / dense - 1) < 1e-8
    # route 2: the power law of each segment, summed by hand
    fk = np.array([lo, 1e5, 1e6, hi])
    sk = spec.psd(fk)
    tot = 0.0
    for (f1, f2), (s1, s2) in zip(zip(fk[:-1], fk[1:]), zip(sk[:-1], sk[1:])):
        a = np.log(s2 / s1) / np.log(f2 / f1)
        tot += s1 / f1 ** a / (a + 1) * (f2 ** (a + 1) - f1 ** (a + 1))
    assert abs(var / tot - 1) < 1e-12
    # and rms_jitter(loglog) on the measured points is the same integral
    f = np.array([1e4, 1e5, 1e6, 1e7])
    j = rms_jitter(f, dbc_to_psd([-65.0, -92.0, -115.0, -135.0]), 2e9,
                   method="loglog")
    j_spec = np.sqrt(spec.phase_variance(1e4, 1e7)) / (2 * np.pi * 2e9)
    assert abs(j / j_spec - 1) < 1e-12
    # the trapezoid on those four points overestimates (default kept)
    assert rms_jitter(f, dbc_to_psd([-65.0, -92.0, -115.0, -135.0]),
                      2e9) > 2.5 * j


def test_flat_segment_and_refusals():
    f = np.array([1e3, 1e6])
    s = np.array([1e-12, 1e-12])
    assert np.isclose(rms_jitter(f, s, 1e8, method="loglog"),
                      rms_jitter(f, s, 1e8), rtol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        rms_jitter(f, np.array([1e-12, 0.0]), 1e8, method="loglog")
    with pytest.raises(ValueError, match="method"):
        rms_jitter(f, s, 1e8, method="simpson")
    spec = NoiseSpec((1e4, 1e6), (-90.0, -120.0), reference=REF)
    with pytest.raises(ValueError, match="measured range"):
        spec.phase_variance(1e3, 1e5)
    with pytest.raises(ValueError, match="f_lo_hz < f_hi_hz"):
        spec.phase_variance(1e5, 1e5)
