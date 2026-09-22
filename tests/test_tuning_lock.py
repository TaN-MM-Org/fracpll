"""Tuning-curve and lock anchors: the PCHIP gain equals a central
finite difference across the fitted curve (two code paths, one
number); the non-monotone refusal names the offending bias point; the
temperature family interpolates linearly and refuses outside the
measured span; the closed-form static offset is reproduced by the
integrated nonlinear transient (two independent routes); and the
out-of-range lock target is refused rather than integrated forever."""
import numpy as np
import pytest

from fracpll import (TuningFamily, continuous_closed_loop_poles,
                     fit_tuning, loop_filter_impedance, lock_transient,
                     open_loop, stability, static_offset)

REF = "VNA + SMU sweep, lab notebook 2026-09-18"

# a smooth negative-Kvco curve (varactor-tuned ring style), exactly
# representable checks are done against the PCHIP itself, not this form
VC = np.linspace(0.0, 3.0, 13)
F = 5.0e9 - 0.4e9 * VC - 0.02e9 * VC ** 2


def _curve():
    return fit_tuning(VC, F, REF, label="ring A")


def test_kvco_two_paths():
    c = _curve()
    v = np.linspace(0.05, 2.95, 41)
    h = 1e-6
    fd = (c.frequency(v + h) - c.frequency(v - h)) / (2.0 * h)
    assert np.allclose(c.kvco(v), fd, rtol=1e-6)


def test_curve_hits_measured_points_exactly():
    c = _curve()
    assert np.allclose(c.frequency(VC), F, rtol=0.0, atol=1e-6)


def test_nonmonotone_refusal_names_voltage():
    f_fold = F.copy()
    f_fold[9:] = f_fold[9] + np.arange(4) * 1e7   # slope flips at v[9]
    with pytest.raises(ValueError, match="not monotone") as e:
        fit_tuning(VC, f_fold, REF)
    assert f"{VC[9]:.4g}" in str(e.value)


def test_fit_refusals():
    with pytest.raises(ValueError, match="reference"):
        fit_tuning(VC, F, "x")
    with pytest.raises(ValueError, match=">= 4"):
        fit_tuning(VC[:3], F[:3], REF)
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_tuning(VC[::-1], F[::-1], REF)
    c = _curve()
    with pytest.raises(ValueError, match="does not extrapolate"):
        c.frequency(3.5)
    with pytest.raises(ValueError, match="does not extrapolate"):
        c.kvco(-0.1)


def test_family_linear_interpolation_and_refusals():
    c_cold = fit_tuning(VC, F * 1.01, REF, label="233 K")
    c_hot = fit_tuning(VC, F * 0.99, REF, label="398 K")
    fam = TuningFamily({233.0: c_cold, 398.0: c_hot})
    v = 1.5
    # endpoints reproduce the measured corners
    assert np.allclose(fam.frequency(v, 233.0), c_cold.frequency(v))
    assert np.allclose(fam.frequency(v, 398.0), c_hot.frequency(v))
    # midpoint is the arithmetic mean (linear interpolation)
    t_mid = 0.5 * (233.0 + 398.0)
    assert np.allclose(fam.frequency(v, t_mid),
                       0.5 * (c_cold.frequency(v) + c_hot.frequency(v)),
                       rtol=1e-12)
    assert np.allclose(fam.kvco(v, t_mid),
                       0.5 * (c_cold.kvco(v) + c_hot.kvco(v)),
                       rtol=1e-12)
    with pytest.raises(ValueError, match="outside the measured span"):
        fam.frequency(v, 500.0)
    with pytest.raises(ValueError, match=">= 2 measured"):
        TuningFamily({300.0: c_cold})


def test_static_offset_closed_form():
    # linear range: phi_ss ~ 2 pi rho
    assert np.isclose(static_offset(100e-6, i_leak=1e-6),
                      2.0 * np.pi * np.arctanh(0.01))
    # mismatch enters with opposite sign
    assert np.isclose(static_offset(100e-6, i_leak=1e-6, mismatch=0.01),
                      0.0, atol=1e-15)
    with pytest.raises(ValueError, match="saturation"):
        static_offset(100e-6, i_leak=100e-6)
    with pytest.raises(ValueError, match="positive"):
        static_offset(0.0)


def _loop_setup():
    """A loop that settles quickly: measured-style curve, modest N."""
    vc = np.linspace(0.0, 3.0, 13)
    f = 4.8e9 + 0.4e9 * vc            # positive-Kvco for this one
    curve = fit_tuning(vc, f, REF, label="ring B")
    f_ref = 50e6
    n_div = 100.0                     # target 5.0 GHz, mid-range
    return curve, n_div, f_ref


def test_lock_transient_reaches_closed_form_offset():
    curve, n_div, f_ref = _loop_setup()
    i_leak = 2e-6
    res = lock_transient(curve, n_div, f_ref, icp=200e-6,
                         c_shunt=50e-12, branches=[(4.7e3, 1.5e-9)],
                         i_leak=i_leak, settle_tol_rad=0.05)
    assert res["locked"]
    # route 1: the integrated trajectory's final phase
    # route 2: the closed form
    phi_ss = static_offset(200e-6, i_leak=i_leak)
    # (phi_ss = 0.0628 rad here, so the bound must be far tighter than
    # the 0.05 rad settling band to tell the offset from zero)
    assert abs(res["phi_e"][-1] - phi_ss) < 1e-6
    assert np.isclose(res["phi_static"], phi_ss)
    # and the final control voltage puts the oscillator on target
    f_end = float(np.ravel(curve.frequency(res["vc"][-1]))[0])
    assert abs(f_end / (n_div * f_ref) - 1.0) < 1e-9
    assert res["t_settle"] is not None and res["t_settle"] > 0.0


def test_lock_target_out_of_range_refused():
    curve, _, f_ref = _loop_setup()
    with pytest.raises(ValueError, match="cannot reach"):
        lock_transient(curve, 1000.0, f_ref, icp=200e-6,
                       c_shunt=50e-12)


def test_zero_resistance_branch_refused():
    curve, n_div, f_ref = _loop_setup()
    with pytest.raises(ValueError, match="lump"):
        lock_transient(curve, n_div, f_ref, icp=200e-6,
                       c_shunt=50e-12, branches=[(0.0, 1e-9)])


def test_negative_kvco_refused_by_the_averaged_models():
    """The averaged models take the |Kvco| the loop sees; a negative
    value would silently mean positive feedback (L < 0 at DC), so it is
    refused.  The averaged lock model refuses a falling tuning curve
    and points to the edge-level simulator."""
    zf = lambda w: loop_filter_impedance(w, 100e-12, [(4.7e3, 1.5e-9)])
    with pytest.raises(ValueError, match="magnitude"):
        open_loop(np.array([1e5]), 100e-6, -20e6, 40.0, zf)
    with pytest.raises(ValueError, match="magnitude"):
        stability(100e-6, -20e6, 40.0, zf, f_ref_hz=50e6)
    with pytest.raises(ValueError, match="magnitude"):
        continuous_closed_loop_poles(100e-6, -20e6, 40.0, 100e-12,
                                     [(4.7e3, 1.5e-9)])
    with pytest.raises(ValueError, match="pump_polarity=-1"):
        lock_transient(_curve(), 95.0, 50e6, icp=200e-6, c_shunt=50e-12,
                       branches=[(4.7e3, 1.5e-9)])
