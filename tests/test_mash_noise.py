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

from fracpll import (NoiseSpec, MASH_RANGE, dbc_to_psd, psd_to_dbc,
                     dsm_phase_psd, closed_loop_lines, mash_line_spectrum,
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
    with pytest.raises(ValueError, match="spur"):
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
    # the delta-sigma path is oscillator-referred: |L/(1+L)|^2, no N^2
    sd = dsm_phase_psd(f, 50e6, 3)
    out2 = synthesizer_psd(f, lg, 40.0, s_dsm=sd)
    assert np.allclose(out2["s_dsm_closed"], hl2 * sd, rtol=1e-12)


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


def test_dbc_convention_is_one_convention():
    """dBc/Hz is L(f) = S_phi(f)/2 (IEEE Std 1139) everywhere: the
    printed form of the MASH noise in W. Rhee's thesis, eq. (3.7),
    L(f) = (2 pi)^2/(12 f_ref) (2 sin(pi f/f_ref))^(2(m-1)), is what
    psd_to_dbc gives for dsm_phase_psd; the conversions invert each
    other; and the dBc of discrete lines uses the same factor."""
    fr = 50e6
    f = np.geomspace(1e4, 2e7, 50)
    for m in (2, 3):
        rhee = (2 * np.pi) ** 2 / (12 * fr) \
            * (2 * np.sin(np.pi * f / fr)) ** (2 * (m - 1))
        assert np.allclose(psd_to_dbc(dsm_phase_psd(f, fr, m)),
                           10 * np.log10(rhee), rtol=0, atol=1e-10)
    x = np.array([-150.0, -100.0, -60.0])
    assert np.allclose(psd_to_dbc(dbc_to_psd(x)), x, rtol=0, atol=1e-12)
    assert np.allclose(dbc_to_psd(-100.0), 2e-10, rtol=1e-12)
    zf = lambda w: loop_filter_impedance(w, 100e-12, [(4.7e3, 1.5e-9)])
    r = mash_line_spectrum(3, 64, 1, fr)
    lg = open_loop(2 * np.pi * r["f_hz"], 100e-6, 20e6, 40.0, zf)
    out = closed_loop_lines(r["f_hz"], r["line_rad2"], lg)
    ok = np.isfinite(out["sideband_dbc"])
    assert ok.any()
    assert np.allclose(out["sideband_dbc"][ok],
                       psd_to_dbc(out["line_rad2"][ok]), rtol=0,
                       atol=1e-12)
