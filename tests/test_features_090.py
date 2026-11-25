"""New in 0.9.0, each held against something independent of the code
it tests:

* `sampled_closed_loop_response`: against the averaged
  `closed_loop_response` in a slow loop, against a scan of G/(1+G)
  built from the aliased sum of the averaged loop gain (a separate
  route to G), and against the per-cycle simulation driven by a sine
  wave at the reported -3 dB point and peak (time domain);
* `design_loop_filter` with an R3-C3 section: read back by
  `third_order_impedance` and `stability`, the phase slope at f_c by
  finite differences, the pole time constants and C3/C1 of the parts
  from the polynomial form, and the second-order design as the limit;
* `lock_transient` with an R3-C3 section: the small-signal transient
  against the matrix exponential of the linear system written here
  from the node equations, and the settled state against the closed
  forms;
* `pulse_doublet_vector` with an R3-C3 section: e^{Mt} M B against the
  impulse response of `third_order_impedance` from its poles and
  residues;
* `margin_uncertainty(correlations=...)`: exact cases (Icp and Kvco
  enter the averaged loop only as their product), the impedance-scaling
  sum rule written as a correlation, and the Monte Carlo spread;
* `read_noise_csv` / `read_tuning_csv` with a decimal comma: bit-for-bit
  round trips;
* the measurement plan with method="loglog": the sensitivities against
  finite differences and their sum rule, and the error bar against a
  seeded Monte Carlo."""
import math

import numpy as np
import pytest
from scipy.linalg import expm

from fracpll import (closed_loop_response, design_loop_filter, fit_tuning,
                     jitter_relative_sigma, averages_for_jitter,
                     lock_transient, loop_filter_impedance,
                     margin_uncertainty, open_loop, pulse_doublet_vector,
                     read_noise_csv, read_tuning_csv, rms_jitter,
                     sampled_closed_loop_response, sampled_margins,
                     simulate_sampled, stability,
                     third_order_impedance, dbc_to_psd)
from fracpll.noise import _loglog_integral
from fracpll.plan import _loglog_sensitivities

FR = 50e6
SLOW = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=FR,
            c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)])
# README example 15's loop: crosses over at 7.7 MHz, above f_ref/10
FAST = dict(icp=1e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=FR,
            c_shunt=8.2e-12, branches=[(1.3e3, 110e-12)])
REF = "synthetic data, fracpll test suite"


# --- per-cycle closed-loop response ---------------------------------

def _aliased_h2(f, icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                branches, r3=None, c3=None, M=20000):
    """|G/(1+G)|^2 with G = sum_m L(j(w + m w_ref)), L the averaged
    gain on `third_order_impedance` (or the second-order filter); the
    1/w^2 tail of a second-order filter is summed in closed form, a
    third-order one falls as 1/w^4 and is summed directly."""
    ws = 2 * np.pi * f_ref_hz
    w = 2 * np.pi * f
    wm = w + np.arange(-M, M + 1) * ws
    if r3 is None:
        z = loop_filter_impedance(np.abs(wm), c_shunt, branches)
    else:
        (r2, c2), = branches
        z = third_order_impedance(np.abs(wm), c_shunt, r2, c2, r3, c3)
    z = np.where(wm < 0, np.conj(z), z)
    lv = icp * kvco_hz_per_v * z / (1j * wm * n_div)
    if r3 is None:
        k = icp * kvco_hz_per_v / (n_div * c_shunt)
        g = np.sum(lv + k / wm ** 2) \
            - k / (4 * f_ref_hz ** 2 * np.sin(w / (2 * f_ref_hz)) ** 2)
    else:
        g = np.sum(lv)
    return float(abs(g / (1 + g)) ** 2)


def test_slow_loop_agrees_with_the_averaged_model():
    zf = lambda w: loop_filter_impedance(w, 100e-12, [(4.7e3, 1.5e-9)])
    avg = closed_loop_response(100e-6, 20e6, 40.0, zf, FR)
    got = sampled_closed_loop_response(**SLOW)
    # sampling adds aliases of relative size (f / f_ref)^2 ~ 1e-6 near
    # the loop band, and the per-cycle loop has one extra half period
    # of delay; measured 2.5e-5 relative (f_3db), 2.5e-5 dB and 1.1e-5
    # relative (f_peak)
    assert got["f_3db_hz"] == pytest.approx(avg["f_3db_hz"], rel=1e-4)
    assert got["peaking_db"] == pytest.approx(avg["peaking_db"], abs=1e-4)
    assert got["f_peak_hz"] == pytest.approx(avg["f_peak_hz"], rel=1e-4)
    m = sampled_margins(**SLOW)
    assert got["phase_margin_deg"] == m["phase_margin_deg"]
    assert got["gain_margin_db"] == m["gain_margin_db"]


@pytest.mark.parametrize("loop", [SLOW, FAST,
                                  dict(FAST, r3=10e3, c3=0.82e-12),
                                  dict(SLOW, r3=10e3, c3=10e-12)])
def test_against_the_aliased_sum(loop):
    got = sampled_closed_loop_response(**loop)
    f3, fp = got["f_3db_hz"], got["f_peak_hz"]
    # |H|^2 = 1/2 at the -3 dB point, by the separate route
    # (measured: 1e-14 or less)
    assert abs(_aliased_h2(f3, **loop) - 0.5) < 1e-9
    # the peak value, and it is a maximum: 0.1 % to either side lower
    # (measured: value to 2e-13 dB)
    pk = 10 * math.log10(_aliased_h2(fp, **loop))
    assert abs(pk - got["peaking_db"]) < 1e-8
    for side in (1 - 1e-3, 1 + 1e-3):
        assert 10 * math.log10(_aliased_h2(fp * side, **loop)) < pk


def test_against_the_per_cycle_simulation():
    """A sine wave of divider phase through `simulate_sampled(order=1)`:
    after the start-up the output amplitude over the input amplitude
    is 1/sqrt(2) at the -3 dB point and 10^(peaking/20) at the peak.
    In the fast loop the averaged model refuses."""
    got = sampled_closed_loop_response(**FAST)
    K, keep = 6000, 2000
    k = np.arange(K)
    for f, want in ((got["f_3db_hz"], 1 / math.sqrt(2)),
                    (got["f_peak_hz"], 10 ** (got["peaking_db"] / 20))):
        t = 2 * math.pi * f / FR
        u = 1e-3 * np.cos(t * k)
        psi = simulate_sampled(u_cycles=u, order=1, **FAST)["psi"][-keep:]
        kk = k[-keep:]
        a = np.c_[np.cos(t * kk), np.sin(t * kk)]
        amp = np.hypot(*np.linalg.lstsq(a, psi, rcond=None)[0]) / 1e-3
        # measured: 6e-15 relative
        assert abs(amp / want - 1) < 1e-9


def test_closed_loop_refusals():
    # README example 8: 0.18 dB of gain margin, |H| largest at f_ref/2
    ex8 = dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=FR,
               c_shunt=2e-12, branches=[(20e3, 200e-12)])
    with pytest.raises(ValueError, match="largest at f_ref/2"):
        sampled_closed_loop_response(**ex8)
    with pytest.raises(ValueError, match="low end"):
        sampled_closed_loop_response(**SLOW, f_lo=3e4)   # peak 24.7 kHz
    with pytest.raises(ValueError, match="f_ref/2"):
        sampled_closed_loop_response(**dict(ex8, icp=0.6e-3))
    with pytest.raises(ValueError, match="whole number"):
        sampled_closed_loop_response(**SLOW, n_grid=10.5)


# --- third-order filter design --------------------------------------

DESIGNS = [(40e3, 55.0, 0.1, 0.05), (40e3, 55.0, 0.25, 0.1),
           (100e3, 45.0, 0.5, 0.1), (1e3, 70.0, 0.2, 0.02),
           (2.5e6, 30.0, 0.3, 0.1)]


@pytest.mark.parametrize("fc,pm,r,c", DESIGNS)
def test_third_order_design_reads_back(fc, pm, r, c):
    loop = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0)
    d = design_loop_filter(fc, pm, f_ref_hz=FR, t3_over_t1=r,
                           c3_over_c1=c, **loop)
    c1, (r2, c2), r3, c3 = d["c_shunt"], d["branches"][0], d["r3"], d["c3"]
    zf = lambda w: third_order_impedance(w, c1, r2, c2, r3, c3)
    rep = stability(zfilter=zf, f_ref_hz=FR, **loop)
    # measured: 2e-15 relative and 1.4e-14 degrees
    assert rep["f_crossover_hz"] == pytest.approx(fc, rel=1e-12)
    assert rep["phase_margin_deg"] == pytest.approx(pm, abs=1e-10)
    # the phase peaks at f_c: central difference in ln f, step h,
    # error of order h^2 times the third derivative (measured: 7e-10
    # rad per unit ln f, against a curvature of order 1 rad)
    h = 1e-4
    ph = lambda x: np.angle(open_loop(2 * np.pi * fc * np.exp(x),
                                      zfilter=zf, **loop))[0]
    assert abs((ph(h) - ph(-h)) / (2 * h)) < 1e-7
    assert ph(h) < ph(0.0) and ph(-h) < ph(0.0)
    # the parts: C3/C1, the time constants, from the polynomial form
    # A0 + A1 s + A2 s^2 of `fracpll.filters` (measured 1e-16, 4e-16)
    assert c3 / c1 == pytest.approx(c, rel=1e-14)
    assert r2 * c2 == pytest.approx(d["t2_s"], rel=1e-14)
    a0 = c1 + c2 + c3
    a1 = r2 * c2 * (c1 + c3) + r3 * c3 * (c1 + c2)
    a2 = c1 * c2 * c3 * r2 * r3
    taus = sorted(-1 / np.roots([a2, a1, a0]).real)
    assert taus[0] == pytest.approx(d["t3_s"], rel=1e-9)
    assert taus[1] == pytest.approx(d["t1_s"], rel=1e-9)
    assert d["t3_s"] / d["t1_s"] == pytest.approx(r, rel=1e-14)
    assert min(c1, c2, c3, r2, r3) > 0
    assert d["averaged"] == rep
    pc = sampled_margins(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0,
                         f_ref_hz=FR, c_shunt=c1, branches=[(r2, c2)],
                         r3=r3, c3=c3)
    assert d["per_cycle"]["phase_margin_deg"] == pc["phase_margin_deg"]


def test_third_order_design_tends_to_the_second_order_one():
    loop = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=FR)
    d2 = design_loop_filter(40e3, 55.0, **loop)
    d3 = design_loop_filter(40e3, 55.0, t3_over_t1=1e-7, c3_over_c1=1e-7,
                            **loop)
    # the section's effect is of order T3/T1 and C3/C1 (measured 2e-7)
    for key in ("c1", "r2", "c2", "t1_s", "t2_s"):
        assert d3[key] == pytest.approx(d2[key], rel=1e-5)
    # a slow loop: the per-cycle margin is the target (measured 5e-8)
    d = design_loop_filter(40e3, 55.0, t3_over_t1=0.25, c3_over_c1=0.1,
                           **loop)
    assert abs(d["per_cycle"]["phase_margin_deg"] - 55.0) < 1e-5


def test_third_order_design_refusals():
    loop = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=FR)
    with pytest.raises(ValueError, match="both"):
        design_loop_filter(40e3, 55.0, t3_over_t1=0.2, **loop)
    with pytest.raises(ValueError, match="positive"):
        design_loop_filter(40e3, 55.0, t3_over_t1=0.0, c3_over_c1=0.1,
                           **loop)
    with pytest.raises(ValueError, match="smaller"):
        design_loop_filter(40e3, 55.0, t3_over_t1=1.0, c3_over_c1=0.1,
                           **loop)
    with pytest.raises(ValueError, match="f_ref/10"):
        design_loop_filter(5e6, 55.0, t3_over_t1=0.2, c3_over_c1=0.1,
                           **loop)


# --- lock transient with an R3-C3 section ---------------------------

KVL, NL, ICPL, C0L, RL, C1L = 0.4e9, 100.0, 200e-6, 50e-12, 4.7e3, 1.5e-9
R3L, C3L = 10e3, 5e-12


def _linear_curve():
    vc = np.linspace(0.0, 3.0, 13)
    return fit_tuning(vc, 4.8e9 + KVL * vc, REF)


def test_lock_with_section_small_signal_equals_matrix_exponential():
    curve = _linear_curve()
    v_lock = (NL * FR - 4.8e9) / KVL
    phi0, t_end = 1e-3, 200e-6
    res = lock_transient(curve, NL, FR, ICPL, C0L, [(RL, C1L)],
                         vc0=v_lock, phi0=phi0, t_end=t_end, n_eval=401,
                         settle_tol_rad=1e-5, r3=R3L, c3=C3L)
    # x = [phi, v_node, v_branch, v_c3] - lock; the oscillator reads C3
    g, g3 = 1 / RL, 1 / R3L
    A = np.array([
        [0.0, 0.0, 0.0, -2 * np.pi * KVL / NL],
        [ICPL / (2 * np.pi * C0L), -(g + g3) / C0L, g / C0L, g3 / C0L],
        [0.0, g / C1L, -g / C1L, 0.0],
        [0.0, g3 / C3L, 0.0, -g3 / C3L]])
    x0 = np.array([phi0, 0.0, 0.0, 0.0])
    ref = np.array([(expm(A * t) @ x0)[0] for t in res["t"]])
    vref = np.array([(expm(A * t) @ x0)[3] for t in res["t"]]) + v_lock
    # integrator tolerance rtol 1e-8; tanh(x) = x - x^3/3 at 1e-3 rad
    # (measured: 1.8e-6 of phi0, and 2e-10 V)
    assert np.max(np.abs(res["phi_e"] - ref)) < 1e-5 * phi0
    assert np.max(np.abs(res["vc"] - vref)) < 1e-9
    assert res["locked"]


def test_lock_with_section_settles_on_the_closed_forms():
    curve = fit_tuning([0.0, 1.0, 2.0, 3.0], [4.8e9, 5.2e9, 5.6e9, 6.0e9],
                       REF)
    res = lock_transient(curve, NL, FR, ICPL, C0L, [(RL, C1L)],
                         i_leak=2e-6, r3=R3L, c3=C3L)
    assert res["locked"]
    # 2 pi atanh(I_leak / Icp) and N f_ref (measured: both exact to
    # the last bit)
    assert abs(res["phi_e"][-1] - 2 * np.pi * np.arctanh(0.01)) < 1e-6
    f_end = curve.frequency(res["vc"][-1])[0]
    assert abs(f_end / (NL * FR) - 1) < 1e-9
    with pytest.raises(ValueError, match="both"):
        lock_transient(curve, NL, FR, ICPL, C0L, [(RL, C1L)], r3=R3L)
    with pytest.raises(ValueError, match="r3"):
        lock_transient(curve, NL, FR, ICPL, C0L, [(RL, C1L)], r3=0.0,
                       c3=C3L)


# --- pulse doublet with an R3-C3 section ----------------------------

def test_doublet_with_section_is_the_impulse_response():
    kv, c1, r2, c2, r3, c3 = 20e6, 100e-12, 4.7e3, 1.5e-9, 10e3, 10e-12
    mb = pulse_doublet_vector(kv, c1, [(r2, c2)], r3, c3)
    assert mb[-1] == 0.0                     # no in-pulse ramp on C3
    # build M as `sampled_loop_map` does, from its Phi = expm(M T)
    from fracpll._network import FilterModes, filter_network
    br, out = filter_network(c1, [(r2, c2)], r3, c3)
    a, b, c = FilterModes(c1, br, out).state_matrices()
    M = np.zeros((4, 4))
    M[:3, :3] = a
    M[3, :3] = kv * c
    # Z(s) = (1 + s T2)/(s (A0 + A1 s + A2 s^2)): poles and residues
    a0 = c1 + c2 + c3
    a1 = r2 * c2 * (c1 + c3) + r3 * c3 * (c1 + c2)
    a2 = c1 * c2 * c3 * r2 * r3
    p, q = np.roots([a2, a1, a0])
    t2 = r2 * c2

    def h(t):                                # impulse response (V/C)
        return (1 / a0 + sum((1 + x * t2) / (x * a2 * (x - y))
                             * np.exp(x * t) for x, y in ((p, q), (q, p))))
    for t in (1e-9, 1e-7, 3e-6, 1e-4, 1e-2):
        row = (expm(M * t) @ mb)[3]
        # measured: 1.5e-9 relative (the matrix exponential of M t)
        assert abs(row / (kv * h(t).real) - 1) < 1e-8
    # long after: Kvco / C_total (measured 8e-10)
    assert abs((expm(M * 1.0) @ mb)[3] / (kv / a0) - 1) < 1e-8
    # without the section the vector is the one of 0.8.0
    mb2 = pulse_doublet_vector(kv, c1, [(r2, c2)])
    assert mb2[-1] == kv / c1


# --- correlated inputs in margin_uncertainty -------------------------

def test_correlation_exact_cases():
    """The averaged loop gain depends on Icp and Kvco only through
    their product, so their contributions are equal; r = +1 adds them
    (twice one), r = -1 cancels them, r = 1/2 gives sqrt(3) times one."""
    base = margin_uncertainty(**SLOW, u_icp=0.02, u_kvco=0.02)
    ci = base["contributions"]["icp"]
    ck = base["contributions"]["kvco"]
    for o in ci:
        assert ci[o] == pytest.approx(ck[o], rel=1e-12)   # measured 0
    for r, factor in ((1.0, 2.0), (0.5, math.sqrt(3.0))):
        got = margin_uncertainty(**SLOW, u_icp=0.02, u_kvco=0.02,
                                 correlations={("icp", "kvco"): r})
        for o in ci:
            assert got["u"][o] == pytest.approx(factor * abs(ci[o]),
                                                rel=1e-12)
    # r = -1: the linear result is zero, so the curvature check refuses
    # unless a Monte Carlo run is asked for
    with pytest.raises(ValueError, match="too curved"):
        margin_uncertainty(**SLOW, u_icp=0.02, u_kvco=0.02,
                           correlations={("kvco", "icp"): -1.0})
    got = margin_uncertainty(**SLOW, u_icp=0.02, u_kvco=0.02, n_mc=200,
                             seed=2, correlations={("kvco", "icp"): -1.0})
    for o in ci:
        assert got["u"][o] < 1e-9 * abs(ci[o])
        # the draws move Icp and Kvco in opposite directions, so the
        # product is (1 + z u)(1 - z u) = 1 - z^2 u^2: the output moves
        # by -c z^2 u^2, whose spread is sqrt(2) u |c u| (the standard
        # deviation of z^2 is sqrt(2)).  200 draws give that spread to
        # about 10 % (measured 3 % and 3 %)
        want = math.sqrt(2.0) * 0.02 * abs(ci[o])
        assert got["mc"][o]["std"] == pytest.approx(want, rel=0.25)


def test_correlation_scaling_sum_rule():
    """Every C and Icp times k, every R divided by k, leaves L
    unchanged.  Inputs that move together that way (r = +1 among Icp,
    C1, C2, r = -1 against R2, all with the same u) cancel to first
    order: u_c is far below each contribution.  What is left is of
    second order, from R2 (1 - u) instead of R2 / (1 + u), a relative
    change of u^2 = 1e-4 (measured 1.2e-6 and 4.8e-5 of the largest
    contribution)."""
    names = ["icp", "c_shunt", "branch0_c"]
    corr = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr[(a, b)] = 1.0
        corr[(a, "branch0_r")] = -1.0
    u = 0.01
    got = margin_uncertainty(**SLOW, u_icp=u, u_c_shunt=u,
                             u_branches=[(u, u)], correlations=corr,
                             n_mc=50, seed=3)
    for o in got["u"]:
        big = max(abs(got["contributions"][n][o]) for n in names)
        assert got["u"][o] < 2e-4 * big
    assert got["correlation"].shape == (5, 5)


def test_correlated_monte_carlo_agrees():
    us = dict(u_icp=0.02, u_kvco=0.02, u_c_shunt=0.03,
              u_branches=[(0.01, 0.03)])
    corr = {("icp", "kvco"): 0.8, ("c_shunt", "branch0_c"): 0.6}
    got = margin_uncertainty(**SLOW, n_mc=4000, seed=4, correlations=corr,
                             **us)
    plain = margin_uncertainty(**SLOW, **us)
    for o in got["u"]:
        # Monte Carlo standard error of a standard deviation with 4000
        # draws: about 1.1 % (measured 0.5 % and 2.0 %)
        assert got["mc"][o]["std"] == pytest.approx(got["u"][o], rel=0.05)
    # and the correlation matters here: the crossover's uncertainty
    # changes by far more than the Monte Carlo error (measured +32 %)
    assert abs(got["u"]["f_crossover_hz"]
               / plain["u"]["f_crossover_hz"] - 1) > 0.15


def test_no_correlations_is_unchanged_bit_for_bit():
    kw = dict(u_icp=0.02, u_kvco=0.01, u_c_shunt=0.03,
              u_branches=[(0.01, 0.03)], n_mc=100, seed=5)
    a = margin_uncertainty(**SLOW, **kw)
    for c in (None, {}):
        b = margin_uncertainty(**SLOW, correlations=c, **kw)
        assert a["u"] == b["u"] and a["mc"] == b["mc"]
        assert "correlation" not in b


@pytest.mark.parametrize("corr,match", [
    ({("icp", "nope"): 0.5}, "unknown input"),
    ({("icp", "icp"): 0.5}, "itself"),
    ({("icp", "kvco"): 1.5}, "between -1 and"),
    ({("icp", "kvco"): 0.5, ("kvco", "icp"): 0.5}, "twice"),
    ({("icp", "kvco"): 0.9, ("kvco", "c_shunt"): 0.9,
      ("icp", "c_shunt"): -0.9}, "negative eigenvalue"),
    ({"icp": 0.5}, "pair"),
])
def test_correlation_refusals(corr, match):
    with pytest.raises(ValueError, match=match):
        margin_uncertainty(**SLOW, u_icp=0.02, u_kvco=0.02,
                           u_c_shunt=0.02, correlations=corr)


# --- decimal comma in the readers -----------------------------------

F = np.geomspace(1e3, 1e7, 101) * (1 + 1e-9 * np.arange(101))
L = -60.0 - 20 * np.log10(F / 1e3) + 1e-7 * np.sin(np.arange(101))


@pytest.mark.parametrize("delim,given", [(";", None), ("\t", None),
                                         (" ", None), (";", ";")])
def test_decimal_comma_round_trip(tmp_path, delim, given):
    rows = "\n".join(f"{f:.17g}{delim}{l:.17g}".replace(".", ",")
                     for f, l in zip(F, L))
    p = tmp_path / "eu.csv"
    p.write_text("Frequenz (Hz);Phasenrauschen (dBc/Hz)\n" + rows + "\n",
                 encoding="utf-8")
    spec = read_noise_csv(p, REF, delimiter=given, decimal=",")
    assert np.array_equal(np.array(spec.f_offset_hz), F)
    assert np.array_equal(np.array(spec.dbc_per_hz), L)


def test_decimal_comma_tuning_file_and_refusals(tmp_path):
    vc = np.linspace(0.0, 3.0, 13)
    fg = 5.0 + 0.4 * vc + 0.02 * vc ** 2
    p = tmp_path / "tune.csv"
    p.write_text("Vc;f (GHz)\n" + "\n".join(
        f"{v:.17g};{f:.17g}".replace(".", ",") for v, f in zip(vc, fg)),
        encoding="utf-8")
    curve = read_tuning_csv(p, REF, f_scale=1e9, decimal=",")
    ref = fit_tuning(vc, fg * 1e9, REF)
    assert np.array_equal(curve.f_hz, ref.f_hz)
    assert np.array_equal(curve.vc, ref.vc)
    with pytest.raises(ValueError, match="cannot also separate"):
        read_tuning_csv(p, REF, delimiter=",", decimal=",")
    with pytest.raises(ValueError, match="decimal must be"):
        read_tuning_csv(p, REF, decimal=";")
    q = tmp_path / "mixed.csv"
    q.write_text("1000;-80,5\n10000;-100,25\n1.5e5;-120,0\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="line 3.*point"):
        read_noise_csv(q, REF, decimal=",")
    q.write_text("1000;-80,5\n10000;-1.100,25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2.*thousands"):
        read_noise_csv(q, REF, decimal=",")
    # the same file without decimal=',' finds no numbers, and says why
    q.write_text("1000;-80,5\n10000;-100,25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs decimal=','"):
        read_noise_csv(q, REF)


# --- the measurement plan with the log-log rule -----------------------

FD = np.array([1e4, 1e5, 1e6, 1e7])
SD = dbc_to_psd(np.array([-65.0, -92.0, -115.0, -135.0]))


def test_loglog_sensitivities():
    g = _loglog_sensitivities(FD, SD)
    total = _loglog_integral(FD, SD)
    # scaling every S by k scales the integral by k: sum g = I
    # (measured 2e-16)
    assert abs(g.sum() / total - 1) < 1e-12
    h = 1e-6
    for i in range(FD.size):
        up, dn = SD.copy(), SD.copy()
        up[i] *= math.exp(h)
        dn[i] *= math.exp(-h)
        fd = (_loglog_integral(FD, up) - _loglog_integral(FD, dn)) / (2 * h)
        # central difference error of order h^2 (measured 4e-8; 2e-10
        # for the slope of -1 below)
        assert fd == pytest.approx(g[i], rel=1e-6)
    # a slope of exactly -1 (x = 0) uses the series of E and E'
    f = np.array([1e3, 1e4, 1e5])
    s = np.array([1e-6, 1e-7, 1e-8])
    g = _loglog_sensitivities(f, s)
    assert abs(g.sum() / _loglog_integral(f, s) - 1) < 1e-12
    up = s.copy()
    up[1] *= math.exp(h)
    dn = s.copy()
    dn[1] *= math.exp(-h)
    fd = (_loglog_integral(f, up) - _loglog_integral(f, dn)) / (2 * h)
    assert fd == pytest.approx(g[1], rel=1e-6)


def test_loglog_plan_against_monte_carlo():
    n = 16
    pred = jitter_relative_sigma(FD, SD, n, method="loglog")
    rng = np.random.default_rng(20261125)
    trials = 20000
    # each point: the mean of n exponential readings, a gamma variable
    s_hat = SD[None, :] * rng.gamma(n, 1.0 / n, size=(trials, FD.size))
    j = np.sqrt([_loglog_integral(FD, row) for row in s_hat])
    emp = j.std(ddof=1) / j.mean()
    # Monte Carlo error about 0.5 %, and the first-order rule leaves a
    # few tenths of a percent at 16 averages (measured +0.6 %)
    assert emp == pytest.approx(pred, rel=0.04)
    # the trapezoid rule's plan, on the same four points, is different
    assert abs(jitter_relative_sigma(FD, SD, n) / pred - 1) > 0.1


def test_loglog_plan_inversion_and_dense_grid():
    n, got = averages_for_jitter(0.01, FD, SD, method="loglog")
    assert got <= 0.01 < jitter_relative_sigma(FD, SD, n - 1, "loglog")
    # on a dense grid the two rules integrate alike, and so plan alike
    # (measured 9e-7)
    f = np.geomspace(1e4, 1e7, 800)
    s = dbc_to_psd(np.interp(np.log10(f), np.log10(FD),
                             [-65.0, -92.0, -115.0, -135.0]))
    a = jitter_relative_sigma(f, s, 9)
    b = jitter_relative_sigma(f, s, 9, method="loglog")
    assert b == pytest.approx(a, rel=1e-4)
    assert rms_jitter(f, s, 2e9, method="loglog") > 0
    with pytest.raises(ValueError, match="positive PSD"):
        jitter_relative_sigma(FD, np.r_[SD[:3], 0.0], 4, method="loglog")
    with pytest.raises(ValueError, match="method"):
        averages_for_jitter(0.01, FD, SD, method="simpson")
