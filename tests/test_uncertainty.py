"""Propagation of input uncertainties to bandwidth and margins (0.5.0).

Anchors, each independent of margin_uncertainty's own arithmetic:
* closed form: for the second-order filter, |L| = 1 at the crossover
  gives d ln f_c / d ln(Icp Kvco) = -1/S with S = d ln|L| / d ln f
  there, and d PM / d ln(Icp Kvco) = (d arg L / d ln f) (-1/S); both
  written out by hand below;
* exact scaling: every C times k, every R divided by k and Icp times k
  leave L unchanged, so the relative sensitivities obey
  s_icp + sum s_C - sum s_R = 0;
* per-cycle model of a fast loop whose phase crossover sits at f_ref/2,
  where G is proportional to Icp: the gain margin moves by exactly
  -20 log10((1 + u)/(1 - u))/2 dB for a step u;
* the per-cycle and averaged sensitivities agree for a slow loop;
* a seeded Monte Carlo agrees with the linear propagation;
* the refusals, including a Monte Carlo failure count reproduced from
  the same seeded draws."""
import numpy as np
import pytest

from fracpll import margin_uncertainty, sampled_margins, stability
from fracpll import loop_filter_impedance

ICP, KV, N, FR = 100e-6, 20e6, 40.0, 50e6
C1, R2, C2 = 100e-12, 4.7e3, 1.5e-9
FAST = dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=50e6,
            c_shunt=2e-12, branches=[(20e3, 200e-12)])


def test_icp_and_kvco_sensitivities_equal_the_closed_form():
    u = 1e-4
    r = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], u_icp=u,
                           u_kvco=u)
    fc = r["nominal"]["f_crossover_hz"]
    w = 2 * np.pi * fc
    t2 = R2 * C2
    tp = R2 * C1 * C2 / (C1 + C2)
    S = (w * t2) ** 2 / (1 + (w * t2) ** 2) - 2 \
        - (w * tp) ** 2 / (1 + (w * tp) ** 2)
    dphi = w * t2 / (1 + (w * t2) ** 2) - w * tp / (1 + (w * tp) ** 2)
    s_fc = fc * (-1.0 / S)                       # Hz per unit ln K
    s_pm = np.degrees(dphi) * (-1.0 / S)         # degrees per unit ln K
    for name in ("icp", "kvco"):
        s = r["sensitivity"][name]
        assert abs(s["f_crossover_hz"] / s_fc - 1) < 1e-6
        assert abs(s["phase_margin_deg"] / s_pm - 1) < 1e-6
    # the nominal phase margin is the closed form atan(w t2) - atan(w tp)
    pm = np.degrees(np.arctan(w * t2) - np.arctan(w * tp))
    assert abs(r["nominal"]["phase_margin_deg"] - pm) < 1e-9
    # two inputs with equal sensitivity and equal u: u_c = sqrt(2) c u
    assert np.isclose(r["u"]["f_crossover_hz"], np.sqrt(2) * s_fc * u,
                      rtol=1e-6)


def test_impedance_scaling_sum_rule():
    u = 1e-4
    br = [(R2, C2), (10e3, 220e-12)]
    r = margin_uncertainty(ICP, KV, N, FR, C1, br, u_icp=u, u_kvco=u,
                           u_c_shunt=u, u_branches=[(u, u), (u, u)])
    s = r["sensitivity"]
    for o in ("f_crossover_hz", "phase_margin_deg"):
        total = s["icp"][o] + s["c_shunt"][o] + s["branch0_c"][o] \
            + s["branch1_c"][o] - s["branch0_r"][o] - s["branch1_r"][o]
        scale = max(abs(v[o]) for v in s.values())
        assert abs(total) < 1e-6 * scale
        # and u^2 is the sum of the squared contributions
        c2 = sum(r["contributions"][k][o] ** 2 for k in s)
        assert np.isclose(r["u"][o] ** 2, c2, rtol=1e-12)


def test_gain_margin_of_a_fast_loop_moves_exactly_with_icp():
    u = 1e-3
    r = margin_uncertainty(**FAST, u_icp=u, model="sampled")
    # the phase crossover stays at f_ref/2 at both steps ...
    for k in (1 + u, 1 - u):
        m = sampled_margins(**{**FAST, "icp": FAST["icp"] * k})
        assert m["f_phase_crossover_hz"] == 0.5 * FAST["f_ref_hz"]
    # ... where G is proportional to Icp, so the step is exact
    exact = -20 / np.log(10) * np.arctanh(u)          # (y+ - y-)/2
    got = r["contributions"]["icp"]["gain_margin_db"]
    assert abs(got / exact - 1) < 1e-9
    m0 = sampled_margins(**FAST)
    assert r["nominal"]["gain_margin_db"] == m0["gain_margin_db"]


def test_per_cycle_and_averaged_sensitivities_agree_for_a_slow_loop():
    kw = dict(u_icp=0.01, u_kvco=0.02, u_c_shunt=0.03,
              u_branches=[(0.01, 0.02)])
    a = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], **kw)
    s = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)],
                           model="sampled", **kw)
    for o in ("f_crossover_hz", "phase_margin_deg"):
        assert abs(s["u"][o] / a["u"][o] - 1) < 2e-3
    assert np.isfinite(s["u"]["gain_margin_db"])


def test_monte_carlo_agrees_with_linear_propagation():
    kw = dict(u_icp=0.02, u_kvco=0.03, u_c_shunt=0.05 / np.sqrt(3),
              u_branches=[(0.01 / np.sqrt(3), 0.05 / np.sqrt(3))])
    r = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], n_mc=1000,
                           seed=20261010, **kw)
    for o in ("f_crossover_hz", "phase_margin_deg"):
        mc = r["mc"][o]
        # MC standard deviation from 1000 draws: ~2.2 % relative error
        assert abs(mc["std"] / r["u"][o] - 1) < 0.1
        assert abs(mc["mean"] - r["nominal"][o]) < 0.3 * r["u"][o]
        assert mc["p2.5"] < r["nominal"][o] < mc["p97.5"]
    # the same seed gives the same draws
    r2 = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], n_mc=1000,
                            seed=20261010, **kw)
    assert r2["mc"] == r["mc"]


def test_curvature_is_reported_and_its_refusal_fires():
    u = 0.05
    r = margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], u_icp=u)
    zf = lambda w: loop_filter_impedance(w, C1, [(R2, C2)])
    y = [stability(ICP * k, KV, N, zf, FR)["phase_margin_deg"]
         for k in (1 + u, 1 - u, 1.0)]
    curv = abs(0.5 * (y[0] + y[1]) - y[2])
    assert np.isclose(r["curvature"]["phase_margin_deg"] * r["u"][
        "phase_margin_deg"], curv, rtol=1e-9)
    with pytest.raises(ValueError, match="too curved"):
        margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], u_icp=u,
                           max_curvature=0.5 * r["curvature"][
                               "phase_margin_deg"])


def test_refusals():
    with pytest.raises(ValueError, match="Gardner"):
        # nominal fast loop refused by the averaged model
        margin_uncertainty(**FAST, u_icp=0.01)
    with pytest.raises(ValueError, match=r"moving icp by \+1"):
        # 2.1 % more Icp makes this loop unstable
        margin_uncertainty(**FAST, u_icp=0.05, model="sampled")
    with pytest.raises(ValueError, match="u_branches has 2 pairs"):
        margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)],
                           u_branches=[(0.1, 0.1), (0.1, 0.1)])
    with pytest.raises(ValueError, match=r"\[0, 0.25\)"):
        margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], u_icp=-0.01)
    with pytest.raises(ValueError, match="model must be"):
        margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], model="exact")
    with pytest.raises(ValueError, match="pump_polarity"):
        margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)],
                           pump_polarity=-1)
    with pytest.raises(ValueError, match="n_mc"):
        margin_uncertainty(ICP, KV, N, FR, C1, [(R2, C2)], n_mc=1)


def test_monte_carlo_failures_are_counted_not_hidden():
    """In the fast loop only Icp varies, and the loop fails exactly
    when Icp grows by more than its gain-margin factor (the phase
    crossover stays at f_ref/2).  Rebuilding the seeded draws gives the
    failure count independently."""
    u, n_mc, seed = 0.015, 400, 11
    gm = sampled_margins(**FAST)["gain_margin_db"]
    draws = 1 + np.random.default_rng(seed).standard_normal((n_mc, 5)) \
        * np.array([u, 0, 0, 0, 0])
    expected = int(np.sum(draws[:, 0] > 10 ** (gm / 20)))
    assert expected > 0
    with pytest.raises(ValueError, match=rf"in {expected} of {n_mc} "):
        margin_uncertainty(**FAST, u_icp=u, model="sampled", n_mc=n_mc,
                           seed=seed)
