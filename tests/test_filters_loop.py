"""Filter and loop anchors: the general branch impedance equals the
textbook second-order closed form at machine precision (two code
paths); the transfer identity 1/(1+L) + L/(1+L) = 1 holds exactly;
the crossover/phase-margin report matches an independent dense-grid
reading; and every validity refusal fires with its explanation."""
import numpy as np
import pytest

from fracpll import (error_transfer, loop_filter_impedance, open_loop,
                     lowpass_transfer, second_order_impedance,
                     stability)

C1, R2, C2 = 100e-12, 4.7e3, 1.5e-9
W = 2.0 * np.pi * np.geomspace(1e2, 1e8, 300)


def test_two_paths_one_impedance():
    z_gen = loop_filter_impedance(W, C1, [(R2, C2)])
    z_txt = second_order_impedance(W, C1, R2, C2)
    assert np.allclose(z_gen, z_txt, rtol=1e-12, atol=0.0)


def test_third_order_reduces_when_branch_opens():
    """Sending a branch resistance to infinity removes the branch."""
    z2 = loop_filter_impedance(W, C1, [(R2, C2)])
    z3 = loop_filter_impedance(W, C1, [(R2, C2), (1e15, 1e-12)])
    assert np.allclose(z2, z3, rtol=1e-4)


def test_transfer_identity_exact():
    zf = lambda w: loop_filter_impedance(w, C1, [(R2, C2)])
    lg = open_loop(W, 100e-6, 20e6, 40.0, zf)
    ident = error_transfer(lg) + lowpass_transfer(lg)
    assert np.allclose(ident, 1.0, rtol=0.0, atol=1e-12)


def test_stability_matches_dense_grid():
    zf = lambda w: loop_filter_impedance(w, C1, [(R2, C2)])
    rep = stability(100e-6, 20e6, 40.0, zf, f_ref_hz=50e6)
    # independent reading: dense grid, direct |L| and angle
    f = np.geomspace(1e3, 1e7, 400000)
    lg = open_loop(2 * np.pi * f, 100e-6, 20e6, 40.0, zf)
    i = np.argmin(np.abs(np.abs(lg) - 1.0))
    assert abs(rep["f_crossover_hz"] / f[i] - 1.0) < 1e-3
    pm = 180.0 + np.degrees(np.angle(lg[i]))
    assert abs(rep["phase_margin_deg"] - pm) < 0.05
    assert rep["phase_margin_deg"] > 0.0


def test_refusals():
    zf = lambda w: loop_filter_impedance(w, C1, [(R2, C2)])
    with pytest.raises(ValueError, match="cross unity"):
        stability(1e-12, 20e6, 40.0, zf, f_ref_hz=50e6)
    with pytest.raises(ValueError, match="Gardner"):
        # crossover at ~f_ref/5, past the f_ref/10 validity edge
        stability(1.894e-6, 20e6, 2.0, zf, f_ref_hz=1e5)
    with pytest.raises(ValueError, match="shunt"):
        loop_filter_impedance(W, 0.0)
    with pytest.raises(ValueError, match="positive"):
        loop_filter_impedance(np.array([0.0]), C1)
    with pytest.raises(ValueError, match="branch 0"):
        loop_filter_impedance(W, C1, [(-1.0, C2)])
    with pytest.raises(ValueError, match="unstable"):
        # pure integrator (no zero): -180 deg everywhere
        zint = lambda w: loop_filter_impedance(w, C1)
        stability(100e-6, 20e6, 40.0, zint, f_ref_hz=1e9)
