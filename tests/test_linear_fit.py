"""Straight-line Kvco fit with uncertainties (0.5.0).

Anchors, none of which uses fit_linear_vco against itself:
* an exact line is recovered, and with equal stated sigmas the slope
  uncertainty equals the closed form sigma / sqrt(sum (x - mean)^2);
* weighted fit and covariance equal NumPy's polyfit (cov="unscaled"),
  and the residual-scatter fit equals SciPy's linregress, two
  independent least-squares codes;
* the chi-square p-value for 2 degrees of freedom equals exp(-chi2/2),
  and the curvature F-test p-value for (1, 2) degrees of freedom
  equals the closed-form Student-t tail 1 - |t|/sqrt(2 + t^2), t^2 = F;
* in a seeded Monte Carlo the scatter of the fitted slope matches the
  reported uncertainty;
* curved data and bad inputs are refused."""
import numpy as np
import pytest
from scipy import stats

from fracpll import LinearVCO, fit_linear_vco, simulate_pll

REF = "synthetic tuning points, fracpll test suite 2026-10-10"


def test_exact_line_and_closed_form_uncertainty():
    vc = np.array([0.1, 0.35, 0.5, 0.8, 1.1, 1.3])
    f = 4.8e9 + 400e6 * vc
    sig = 2e5
    fit = fit_linear_vco(vc, f, REF, f_sigma_hz=sig)
    assert abs(fit["kvco_hz_per_v"] / 400e6 - 1) < 1e-12
    assert abs(fit["f0_hz"] / 4.8e9 - 1) < 1e-14
    sxx = np.sum((vc - vc.mean()) ** 2)
    assert np.isclose(fit["u_kvco_hz_per_v"], sig / np.sqrt(sxx),
                      rtol=1e-12)
    # f at the mean voltage is uncorrelated with the slope: sigma/sqrt(n)
    assert np.isclose(fit["u_f_at_vc_mean_hz"], sig / np.sqrt(vc.size),
                      rtol=1e-12)
    # f0 = f(mean) - Kvco * mean: var = sig^2/n + mean^2 var(Kvco)
    u0 = np.sqrt(sig ** 2 / vc.size + vc.mean() ** 2 * sig ** 2 / sxx)
    assert np.isclose(fit["u_f0_hz"], u0, rtol=1e-12)
    assert fit["chi2"] < 1e-12 and fit["p_value"] > 0.999
    vco = fit["vco"]
    assert isinstance(vco, LinearVCO) and vco.reference == REF


def test_weighted_fit_equals_numpy_polyfit():
    rng = np.random.default_rng(101)
    vc = np.linspace(0.2, 1.4, 9)
    sig = np.linspace(1e5, 4e5, 9)
    f = 5.2e9 - 350e6 * vc + sig * rng.standard_normal(9)   # falling
    fit = fit_linear_vco(vc, f, REF, f_sigma_hz=sig)
    p, cov = np.polyfit(vc, f, 1, w=1 / sig, cov="unscaled")
    assert np.isclose(fit["kvco_hz_per_v"], p[0], rtol=1e-9)
    assert np.isclose(fit["f0_hz"], p[1], rtol=1e-12)
    # polyfit orders (slope, intercept); ours is (f0, Kvco)
    assert np.allclose(fit["cov"], cov[::-1, ::-1], rtol=1e-7)
    assert fit["kvco_hz_per_v"] < 0


def test_residual_fit_equals_scipy_linregress():
    rng = np.random.default_rng(7)
    vc = np.linspace(0.0, 2.0, 12)
    f = 1.99e9 + 20e6 * vc + 3e4 * rng.standard_normal(12)
    fit = fit_linear_vco(vc, f, REF)
    lr = stats.linregress(vc, f)
    assert np.isclose(fit["kvco_hz_per_v"], lr.slope, rtol=1e-10)
    assert np.isclose(fit["f0_hz"], lr.intercept, rtol=1e-12)
    assert np.isclose(fit["u_kvco_hz_per_v"], lr.stderr, rtol=1e-8)
    assert np.isclose(fit["u_f0_hz"], lr.intercept_stderr, rtol=1e-8)
    assert fit["sigma_source"] == "residuals" and fit["dof"] == 10


def test_p_values_equal_closed_forms():
    # chi-square, 4 points -> 2 degrees of freedom: p = exp(-chi2/2)
    vc = np.array([0.0, 1.0, 2.0, 3.0])
    f = 1e9 + 1e7 * vc + np.array([1.0, -2.0, 1.5, -0.5]) * 1e4
    fit = fit_linear_vco(vc, f, REF, f_sigma_hz=1.5e4)
    chi2 = np.sum(fit["residuals_hz"] ** 2) / 1.5e4 ** 2
    assert np.isclose(fit["chi2"], chi2, rtol=1e-12)
    assert np.isclose(fit["p_value"], np.exp(-chi2 / 2), rtol=1e-10)
    # F-test, 5 points -> (1, 2) degrees of freedom: t-tail closed form
    vc = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    f = 1e9 + 1e7 * vc + np.array([2.0, -1.0, -1.5, 0.5, 1.0]) * 1e4
    fit = fit_linear_vco(vc, f, REF)
    # independent F: residual sums of squares from numpy polyfit
    rss1 = np.sum((f - np.polyval(np.polyfit(vc, f, 1), vc)) ** 2)
    rss2 = np.sum((f - np.polyval(np.polyfit(vc, f, 2), vc)) ** 2)
    F = (rss1 - rss2) / (rss2 / 2)
    assert np.isclose(fit["chi2"], F, rtol=1e-6)
    t = np.sqrt(F)
    assert np.isclose(fit["p_value"], 1 - t / np.sqrt(2 + t * t),
                      rtol=1e-6)


def test_seeded_monte_carlo_matches_the_reported_uncertainty():
    rng = np.random.default_rng(20261010)
    vc = np.linspace(0.3, 0.7, 6)
    sig = 1e6
    trials = 3000
    k_stated, u_stated = [], None
    k_resid, u2_resid = [], []
    for _ in range(trials):
        f = 2e9 + 20e6 * (vc - 0.5) + sig * rng.standard_normal(vc.size)
        a = fit_linear_vco(vc, f, REF, f_sigma_hz=sig, p_min=0.0)
        k_stated.append(a["kvco_hz_per_v"])
        u_stated = a["u_kvco_hz_per_v"]
        b = fit_linear_vco(vc, f, REF, p_min=0.0)
        k_resid.append(b["kvco_hz_per_v"])
        u2_resid.append(b["u_kvco_hz_per_v"] ** 2)
    emp = np.std(k_stated, ddof=1)
    # the MC standard deviation has relative error ~1/sqrt(2*3000) = 1.3 %
    assert abs(emp / u_stated - 1) < 0.05
    # residual estimate: unbiased for the variance
    assert abs(np.mean(u2_resid) / np.var(k_resid, ddof=1) - 1) < 0.08
    assert abs(np.mean(k_stated) / 20e6 - 1) < 0.01


def test_curved_points_are_refused_and_a_window_fixes_it():
    vc = np.linspace(0.0, 3.0, 13)
    f = 4.8e9 + 0.4e9 * vc - 0.04e9 * vc ** 2
    with pytest.raises(ValueError, match="chi-square"):
        fit_linear_vco(vc, f, REF, f_sigma_hz=1e5)
    noisy = f + 1e5 * np.sin(7 * vc)
    with pytest.raises(ValueError, match="quadratic term"):
        fit_linear_vco(vc, noisy, REF)
    # a narrow window around 1.5 V: the local slope 0.4e9 - 0.08e9 * 1.5
    # (exact for a parabola: the chord slope over a symmetric window
    # equals the derivative at its centre)
    w = fit_linear_vco(vc, f, REF, f_sigma_hz=3e6, vc_window=(1.25, 1.75))
    assert w["n_points"] == 3
    assert np.isclose(w["kvco_hz_per_v"], 0.4e9 - 0.08e9 * 1.5, rtol=1e-9)


def test_refusals():
    vc = np.array([0.0, 1.0, 2.0, 3.0])
    f = 1e9 + 1e7 * vc
    with pytest.raises(ValueError, match="reference"):
        fit_linear_vco(vc, f, "x")
    with pytest.raises(ValueError, match="exactly on a straight line"):
        fit_linear_vco(vc, f, REF)
    with pytest.raises(ValueError, match=">= 4 points"):
        fit_linear_vco(vc[:3], f[:3] + [1.0, -2.0, 1.0], REF)
    with pytest.raises(ValueError, match=">= 3 points"):
        fit_linear_vco(vc[:2], f[:2], REF, f_sigma_hz=1.0)
    with pytest.raises(ValueError, match="increasing"):
        fit_linear_vco(vc[::-1], f, REF, f_sigma_hz=1.0)
    with pytest.raises(ValueError, match="finite"):
        fit_linear_vco([0.0, 1.0, np.nan, 3.0], f, REF, f_sigma_hz=1.0)
    with pytest.raises(ValueError, match="f_sigma_hz"):
        fit_linear_vco(vc, f, REF, f_sigma_hz=0.0)
    with pytest.raises(ValueError, match="length"):
        fit_linear_vco(vc, f[:3], REF)
    with pytest.raises(ValueError, match="vc_window"):
        fit_linear_vco(vc, f, REF, f_sigma_hz=1.0, vc_window=(2.0, 1.0))


def test_fitted_vco_runs_in_the_simulator():
    """The returned LinearVCO is the simulator's own type: a lock with
    it settles at the fitted lock voltage (N f_ref - f0)/Kvco."""
    vc = np.linspace(0.2, 0.8, 7)
    f = 1.99e9 + 20e6 * vc + np.array([1, -1, 2, 0, -2, 1, -1]) * 1e4
    fit = fit_linear_vco(vc, f, REF, f_sigma_hz=2e4)
    r = simulate_pll(fit["vco"], 100, 20e6, 100e-6, 100e-12,
                     [(4.7e3, 1.5e-9)], n_cycles=20000,
                     phase0_cycles=0.1)
    v_lock = (100 * 20e6 - fit["f0_hz"]) / fit["kvco_hz_per_v"]
    assert abs(r.vc_ref[-1] - v_lock) < 1e-9
