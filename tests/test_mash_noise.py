"""MASH and noise anchors: the exact integer sequence averages to the
programmed fraction EXACTLY (fractions.Fraction over one full period);
the sequence range matches the closed bounds per order; the one-sided
quantization PSD matches the Welch spectrum of the exact sequence for
MASH-2 and MASH-3 (simulation vs formula, two independent paths); the
m = 1 spur refusal fires; jitter closed forms for flat and 1/f^2 PSDs
are reproduced; and NoiseSpec refuses uncited or extrapolated use."""
from fractions import Fraction

import numpy as np
import pytest
from scipy import signal

from fracpll import (NoiseSpec, MASH_RANGE, dbc_to_psd, dsm_phase_psd,
                     mash_sequence, rms_jitter, synthesizer_psd,
                     white_floor, open_loop, loop_filter_impedance)


def test_mean_division_is_exact():
    num, den = 37, 97          # coprime -> period includes all states
    for m in (1, 2, 3):
        d = mash_sequence(num, den, m, den * 4)
        # exact rational mean over an integer number of periods
        assert Fraction(int(d[:den].sum()), den) == Fraction(num, den)
        assert Fraction(int(d.sum()), den * 4) == Fraction(num, den)


def test_range_bounds():
    for m, (lo, hi) in MASH_RANGE.items():
        d = mash_sequence(104857, 1 << 20, m, 1 << 14)
        assert d.min() >= lo and d.max() <= hi
        if m >= 2:
            assert d.min() == lo and d.max() == hi


@pytest.mark.parametrize("m", [2, 3])
def test_psd_formula_matches_exact_sequence(m):
    """The one-sided closed form against the measured Welch PSD of the
    exact accumulator sequence -- the formula is not taken on
    authority."""
    num, den = 104857, 1 << 20
    n = 1 << 18
    d = mash_sequence(num, den, m, n)
    phi = 2.0 * np.pi * np.cumsum(d - num / den)
    f, psd = signal.welch(phi, fs=1.0, nperseg=1 << 12,
                          detrend="linear")
    sel = (f > 0.02) & (f < 0.4)
    theory = dsm_phase_psd(f[sel], 1.0, m)
    ratio = np.median(psd[sel] / theory)
    assert abs(ratio - 1.0) < 0.15


def test_first_order_spur_refusal():
    with pytest.raises(ValueError, match="spurs"):
        dsm_phase_psd(np.array([0.1]), 1.0, 1)
    with pytest.raises(ValueError, match="Nyquist"):
        dsm_phase_psd(np.array([0.6]), 1.0, 2)


def test_jitter_closed_forms():
    f0 = 100e6
    f = np.linspace(1e3, 1e6, 20001)
    # flat PSD: var = S*(f2-f1)
    s = np.full_like(f, 1e-12)
    j = rms_jitter(f, s, f0)
    exact = np.sqrt(1e-12 * (f[-1] - f[0])) / (2 * np.pi * f0)
    assert abs(j / exact - 1.0) < 1e-9
    # 1/f^2 PSD: var = A (1/f1 - 1/f2)
    a = 1e-4
    j2 = rms_jitter(f, a / f ** 2, f0)
    exact2 = np.sqrt(a * (1 / f[0] - 1 / f[-1])) / (2 * np.pi * f0)
    # trapezoid on 20001 points carries ~2e-4 discretization error
    # against the exact 1/f^2 integral
    assert abs(j2 / exact2 - 1.0) < 1e-3


def test_synthesizer_assembly_identity():
    """With only an in-band floor, S_out/N^2/floor equals |L/(1+L)|^2
    computed independently."""
    c1, r2, c2 = 100e-12, 4.7e3, 1.5e-9
    zf = lambda w: loop_filter_impedance(w, c1, [(r2, c2)])
    f = np.geomspace(1e3, 1e7, 500)
    lg = open_loop(2 * np.pi * f, 100e-6, 20e6, 40.0, zf)
    floor = white_floor(-100.0, "example floor, docs demo 2026")
    out = synthesizer_psd(f, lg, 40.0, s_inband=floor.psd(f))
    hl2 = np.abs(lg / (1.0 + lg)) ** 2
    assert np.allclose(out["s_out"],
                       hl2 * 40.0 ** 2 * dbc_to_psd(-100.0),
                       rtol=1e-12)


def test_noisespec_refusals():
    with pytest.raises(ValueError, match="reference"):
        NoiseSpec((1e3, 1e6), (-90.0, -120.0), reference="x")
    spec = NoiseSpec((1e3, 1e6), (-90.0, -120.0),
                     reference="R&S FSWP, lab notebook 2026-09-18")
    with pytest.raises(ValueError, match="measured range"):
        spec.psd(1e7)
    # log-log interpolation hits the endpoints exactly
    assert np.allclose(spec.psd(1e3), dbc_to_psd(-90.0))
    assert np.allclose(spec.psd(1e6), dbc_to_psd(-120.0))
