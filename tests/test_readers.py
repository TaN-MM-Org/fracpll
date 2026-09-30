"""Reading measured data files (0.4.0).

Anchors: values written with 17 significant digits come back
bit-for-bit (a float64 round trip), whatever the delimiter, header,
column choice, unit scale or byte-order mark; a loaded trace of an
exact power law integrates to the closed form; and every reading rule
that protects against a misread file refuses with the line number."""
import numpy as np
import pytest

from fracpll import (NoiseSpec, fit_tuning, read_noise_csv,
                     read_tuning_csv)

REF = "synthetic file, fracpll test suite 2026-09-30"
F = np.geomspace(1e3, 1e7, 301) * (1 + 1e-9 * np.arange(301))
L = -60.0 - 20 * np.log10(F / 1e3) + 1e-7 * np.sin(np.arange(301))


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.parametrize("delim", [",", ";", "\t", " "])
def test_round_trip_is_exact(tmp_path, delim):
    p = tmp_path / "trace.txt"
    np.savetxt(p, np.c_[F, L], fmt="%.17g", delimiter=delim,
               header="Phase noise, carrier 2 GHz\nOffset (Hz), L (dBc/Hz)",
               comments="")
    spec = read_noise_csv(p, REF)
    assert np.array_equal(np.array(spec.f_offset_hz), F)
    assert np.array_equal(np.array(spec.dbc_per_hz), L)
    assert spec.label == "trace.txt"
    direct = NoiseSpec(tuple(F), tuple(L), reference=REF)
    x = np.geomspace(2e3, 9e6, 17)
    assert np.array_equal(spec.psd(x), direct.psd(x))


def test_columns_units_bom_and_comments(tmp_path):
    rows = "\n".join(f"{f / 1e3:.17g};{-1.0:.17g};{l:.17g}"
                     for f, l in zip(F, L))
    text = "﻿Offset kHz;raw;smoothed\n# a comment line\n" + rows + "\n\n"
    p = _write(tmp_path, "khz.csv", text)
    spec = read_noise_csv(p, REF, columns=(0, 2), f_scale=1e3, label="x")
    assert np.allclose(spec.f_offset_hz, F, rtol=1e-15, atol=0)
    assert np.array_equal(np.array(spec.dbc_per_hz), L)
    assert spec.label == "x"


def test_loaded_power_law_integrates_to_closed_form(tmp_path):
    # L = -60 - 20 log10(f/1e3): S_phi = 2e-6 (1e3/f)^2, exactly
    f = np.geomspace(1e3, 1e7, 41)
    p = tmp_path / "pl.csv"
    np.savetxt(p, np.c_[f, -60.0 - 20 * np.log10(f / 1e3)], fmt="%.17g",
               delimiter=",")
    spec = read_noise_csv(p, REF)
    exact = 2e-6 * 1e6 * (1 / 1e3 - 1 / 1e7)       # int A f^-2 df
    assert abs(spec.phase_variance(1e3, 1e7) / exact - 1) < 1e-12


def test_tuning_file(tmp_path):
    vc = np.linspace(0.0, 3.0, 13)
    fg = 5.0 - 0.4 * vc - 0.02 * vc ** 2               # GHz, falling
    p = tmp_path / "tune.csv"
    np.savetxt(p, np.c_[vc, fg], fmt="%.17g", delimiter=",",
               header="Vc (V),f (GHz)", comments="")
    curve = read_tuning_csv(p, REF, f_scale=1e9)
    ref = fit_tuning(vc, fg * 1e9, REF)
    v = np.linspace(0.0, 3.0, 29)
    assert np.allclose(curve.frequency(v), ref.frequency(v), rtol=1e-15)
    assert np.allclose(curve.kvco(v), ref.kvco(v), rtol=1e-12)
    # the fold refusal still names its voltage
    fold = fg.copy()
    fold[9:] = fold[9] + 0.01 * np.arange(4)
    np.savetxt(p, np.c_[vc, fold], fmt="%.17g", delimiter=",")
    with pytest.raises(ValueError, match="not monotone"):
        read_tuning_csv(p, REF)


def test_refusals_name_the_line(tmp_path):
    p = _write(tmp_path, "a.csv", "f,L\n1e3,-80\n1e4,-100\nEND OF DATA\n")
    with pytest.raises(ValueError, match="line 4"):
        read_noise_csv(p, REF)
    p = _write(tmp_path, "b.csv", "1e3,-80\n1e4,-100\n1e4,-101\n1e5,-120\n")
    with pytest.raises(ValueError, match="line 3.*strictly increasing"):
        read_noise_csv(p, REF)
    p = _write(tmp_path, "c.csv", "1e3;-80,5\n1e4;-100,5\n")   # decimal comma
    with pytest.raises(ValueError, match="no numeric rows"):
        read_noise_csv(p, REF)
    p = _write(tmp_path, "d.csv", "1e3,-80\n1e4,nan\n")
    with pytest.raises(ValueError, match="line 2.*non-finite"):
        read_noise_csv(p, REF)
    p = _write(tmp_path, "e.csv", "1e3,-80\n1e4,-100\n")
    with pytest.raises(ValueError, match="reference"):
        read_noise_csv(p, "x")
    with pytest.raises(ValueError, match="f_scale"):
        read_noise_csv(p, REF, f_scale=0.0)
