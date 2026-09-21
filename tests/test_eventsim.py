"""Edge-level anchors: the event-driven simulator against exact
results it does not contain.

* the periodic steady state against the EXACT charge balance of a
  tri-state PFD (leakage, mismatch, reset delay, dead zone -- five
  cases, both edge orders);
* its small-perturbation trajectory against the exact linearised
  sampled-data map of `fracpll.sampled`, cycle by cycle;
* the exact stability boundary of that map, from both sides, in a
  regime the averaged model refuses;
* the delta-sigma noise it produces against the closed-loop formula
  WITHOUT the N^2 factor fracpll 0.1.0 wrongly applied;
* the exact mean ratio of a fractional-N lock;
* the measured-curve (Gauss-Legendre) phase path against the exact
  linear closed form on an exactly linear curve;
* pull-in through cycle slips, negative-Kvco operation, and the
  refusals."""
import numpy as np
import pytest
from scipy import signal
from scipy.optimize import brentq

from fracpll import (LinearVCO, dsm_phase_psd, fit_tuning,
                     loop_filter_impedance, open_loop, pfd_static_offset,
                     sampled_loop_map, sampled_stability, simulate_pll,
                     stability, static_offset, synthesizer_psd)

REF = "design value, fracpll test suite 2026-09-21"
FR, N = 50e6, 100
C0, BR = 50e-12, [(4.7e3, 1.5e-9)]
VCO = LinearVCO(4.8e9, 400e6, REF)


@pytest.mark.parametrize("kw", [
    dict(i_leak=2e-6, t_reset=100e-12),
    dict(i_leak=0.0, i_dn=180e-6, t_reset=200e-12),        # DN leads
    dict(i_leak=1e-6, i_dn=210e-6, t_reset=150e-12, t_deadzone=60e-12),
    dict(i_leak=1.5e-6, t_reset=20e-12, t_deadzone=80e-12),
    dict(i_leak=-1e-6, t_reset=50e-12),
])
def test_steady_state_equals_exact_charge_balance(kw):
    r = simulate_pll(VCO, N, FR, 200e-6, C0, BR, n_cycles=6000, **kw)
    sim = r.time_error()[-200:].mean()
    exact = pfd_static_offset(FR, 200e-6, **kw)["dt_s"]
    assert abs(sim - exact) < 1e-6 * abs(exact)
    assert r.slips == 0


def test_pfd_offset_refusals_and_first_order_agreement():
    with pytest.raises(ValueError, match="dead zone"):
        pfd_static_offset(FR, 200e-6, t_reset=10e-12, t_deadzone=50e-12)
    with pytest.raises(ValueError, match="reference period"):
        pfd_static_offset(FR, 200e-6, i_leak=250e-6)
    # the averaged tanh detector agrees to first order in rho:
    rho = 0.01
    exact = pfd_static_offset(FR, 200e-6, i_leak=rho * 200e-6)
    avg = static_offset(200e-6, i_leak=rho * 200e-6)
    assert np.isclose(exact["phase_ref_rad"], 2 * np.pi * rho, rtol=1e-12)
    # atanh(rho) - rho = rho^3/3 + ...  -> relative gap rho^2/3
    assert abs(avg / exact["phase_ref_rad"] - 1.0) < 1.01 * rho ** 2 / 3


def test_simulator_follows_exact_sampled_map():
    p0 = 1e-4
    icp = 200e-6
    r = simulate_pll(VCO, N, FR, icp, C0, BR, n_cycles=400,
                     phase0_cycles=p0)
    F = sampled_loop_map(icp, 400e6, N, FR, C0, BR)["F"]
    s = np.zeros(F.shape[0])
    s[-1] = p0
    pred = []
    for _ in range(400):
        pred.append(s[-1])
        s = F @ s
    pred = np.array(pred)
    assert np.max(np.abs(r.psi_ref - pred)) < 1e-5 * np.max(np.abs(pred))


def test_exact_stability_boundary_where_averaged_model_refuses():
    fr, n, kv, c0, br = 50e6, 10, 400e6, 2e-12, [(20e3, 200e-12)]
    g = lambda i: sampled_stability(i, kv, n, fr, c0, br)[
        "spectral_radius"] - 1.0
    icrit = brentq(g, 3e-4, 1e-3, xtol=1e-12)
    vco = LinearVCO(4.0e8, kv, REF)
    zf = lambda w: loop_filter_impedance(w, c0, br)
    below = simulate_pll(vco, n, fr, 0.9 * icrit, c0, br, n_cycles=300,
                         phase0_cycles=1e-4)
    above = simulate_pll(vco, n, fr, 1.1 * icrit, c0, br, n_cycles=300,
                         phase0_cycles=1e-4)
    assert np.abs(below.psi_ref[-50:]).max() < 1e-2 * 1e-4
    assert np.abs(above.psi_ref[-50:]).max() > 1e2 * 1e-4
    for f in (0.9, 1.1):
        with pytest.raises(ValueError, match="sampled_stability"):
            stability(f * icrit, kv, n, zf, f_ref_hz=fr)


def test_delta_sigma_noise_through_the_loop_has_no_n_squared():
    """Three routes to the closed-loop delta-sigma noise: the edge-level
    simulator, the exact linear sampled-data map driven by the SAME
    integer MASH sequence, and the continuous formula |L/(1+L)|^2 S_dsm.
    In the linear band all three agree -- and the 0.1.0 assembly with
    N^2 does not.  (Near the loop bandwidth and below, the edge-level
    run also carries the second-order pulse-width effect that neither
    linear model contains; tests/test_second_order.py explains and
    reproduces it with `simulate_sampled(order=2)`.)"""
    from fracpll import mash_sequence
    from fracpll._network import FilterModes
    fr, n, kv, icp = 50e6, 40, 20e6, 100e-6
    c0, br = 100e-12, [(4.7e3, 1.5e-9)]
    num, den = 104857, 1 << 20
    vco = LinearVCO(n * fr - 0.5 * kv + 0.1 * fr, kv, REF)
    K = (1 << 16) + 2048
    r = simulate_pll(vco, n, fr, icp, c0, br, n_cycles=K,
                     num=num, den=den, mash_order=3)
    # linear sampled map driven by the same divider sequence
    d = mash_sequence(num, den, 3, K + 4)
    cum = np.concatenate(([0.0], np.cumsum(d - num / den)))
    Phi = sampled_loop_map(icp, kv, n, fr, c0, br)["Phi"]
    b = FilterModes(c0, br).state_matrices()[1]
    B = np.r_[b, 0.0]
    s = np.zeros(Phi.shape[0])
    lin = np.empty(K)
    for k in range(K):
        s = Phi @ s
        lin[k] = s[-1]
        s = s + B * (-icp * (s[-1] - cum[k + 1]) / (n * fr))

    def psd(x):
        return signal.welch(2 * np.pi * x[2048:], fs=fr, nperseg=1 << 13,
                            detrend="linear")
    f, p_sim = psd(r.psi_ref)
    _, p_lin = psd(lin)
    zf = lambda w: loop_filter_impedance(w, c0, br)
    band = f > 0
    lg = open_loop(2 * np.pi * f[band], icp, kv, n, zf)
    s_dsm = dsm_phase_psd(f[band], fr, 3)
    model = synthesizer_psd(f[band], lg, n, s_dsm=s_dsm)["s_dsm_closed"]
    fb = f[band]
    # exact linear physics vs the continuous formula, every band (the
    # lowest band is only a few Welch bins on a slope rising as f^4, so
    # its estimate carries a bin-averaging bias; hence its wider bound)
    for a, bb, tol in [(1e4, 3e4, 2.0), (3e4, 1e5, 1.0), (1e5, 3e5, 1.0),
                       (3e5, 1e6, 1.0), (1e6, 3e6, 1.0), (3e6, 1e7, 1.0)]:
        sel = (fb >= a) & (fb < bb)
        gap = 10 * np.log10(np.median(p_lin[band][sel])
                            / np.median(model[sel]))
        assert abs(gap) < tol
    # edge-level simulator vs the formula in the linear band
    sel = (fb >= 1e6) & (fb < 1e7)
    ratio = np.median(p_sim[band][sel] / model[sel])
    assert 0.85 < ratio < 1.2
    # ... and the 0.1.0 assembly (x N^2) is off by N^2 = 1600
    assert ratio * n ** 2 > 1e3
    # the edge-level run is close to linear in the time domain
    assert np.std((r.psi_ref - lin)[2048:]) < 0.2 * np.std(r.psi_ref[2048:])


def test_edge_level_floor_is_converged_not_numerical():
    """The low-offset excess of the edge-level run (the second-order
    pulse-width effect of tests/test_second_order.py) shrinks with the
    PFD pulse width (N and Kvco scaled together keep the linear loop
    identical in cycle units, so the linear band must not move while a
    pulse-width effect must) -- the signature of circuit behaviour of
    the model, not of rounding."""
    fr, icp = 50e6, 100e-6
    c0, br = 100e-12, [(4.7e3, 1.5e-9)]
    lev = {}
    for n, kv in ((40, 20e6), (160, 80e6)):
        vco = LinearVCO(n * fr - 0.5 * kv + 0.1 * fr, kv, REF)
        r = simulate_pll(vco, n, fr, icp, c0, br,
                         n_cycles=(1 << 15) + 2048, num=104857,
                         den=1 << 20, mash_order=3)
        f, p = signal.welch(r.phase_rad()[2048:], fs=fr,
                            nperseg=1 << 12, detrend="linear")
        lo = 10 * np.log10(np.median(p[(f > 1e4) & (f < 3e4)]))
        mid = 10 * np.log10(np.median(p[(f > 1e6) & (f < 3e6)]))
        lev[n] = (lo, mid)
    assert abs(lev[40][1] - lev[160][1]) < 0.5      # linear band fixed
    assert lev[160][0] < lev[40][0] - 6.0          # pulse-width effect


def test_fractional_lock_has_exact_mean_ratio():
    r = simulate_pll(VCO, N, FR, 200e-6, C0, BR, n_cycles=30000,
                     num=37, den=97, mash_order=3)
    psi = r.psi_ref[2000:]
    # psi is measured against the exact rational ratio N + 37/97, so a
    # bounded psi over K cycles pins the mean ratio to |psi|/K
    assert np.ptp(psi) < 0.02
    k = psi.size
    assert abs(psi[-1] - psi[0]) / k < 1e-6


def test_measured_curve_quadrature_equals_exact_linear_path():
    vc = np.linspace(0.0, 3.0, 13)
    curve = fit_tuning(vc, 4.8e9 + 400e6 * vc, REF)   # exactly linear
    kw = dict(n_cycles=300, phase0_cycles=0.2, i_leak=1e-6,
              t_reset=50e-12)
    v_lock = (N * FR - 4.8e9) / 400e6
    a = simulate_pll(VCO, N, FR, 200e-6, C0, BR, **kw)
    b = simulate_pll(curve, N, FR, 200e-6, C0, BR, vc0=v_lock, **kw)
    assert np.max(np.abs(a.t_div - b.t_div)) < 1e-18
    assert np.max(np.abs(a.vc_ref - b.vc_ref)) < 1e-9


def test_pull_in_through_cycle_slips_on_measured_curve():
    vc = np.linspace(0.0, 3.0, 13)
    curve = fit_tuning(vc, 4.8e9 + 0.4e9 * vc - 0.01e9 * vc ** 2, REF)
    r = simulate_pll(curve, N, FR, 20e-6, C0, BR, n_cycles=20000,
                     vc0=0.02)
    assert r.slips > 0                       # it had to slip ...
    e = np.abs(r.nearest_edge_error())
    assert e[-300:].max() < 1e-15            # ... and then locked
    f_end = float(np.ravel(curve.frequency(r.vc_ref[-1]))[0])
    assert abs(f_end / (N * FR) - 1.0) < 1e-9


def test_negative_kvco_and_refusals():
    vneg = LinearVCO(5.2e9, -400e6, REF)
    r = simulate_pll(vneg, N, FR, 200e-6, C0, BR, n_cycles=3000,
                     pump_polarity=-1, phase0_cycles=0.3)
    assert np.abs(r.time_error()[-10:]).max() < 1e-14
    with pytest.raises(ValueError, match="POSITIVE feedback"):
        simulate_pll(vneg, N, FR, 200e-6, C0, BR, n_cycles=10)
    with pytest.raises(ValueError, match="reference"):
        LinearVCO(4.8e9, 400e6, "x")
    vc = np.linspace(0.0, 3.0, 13)
    curve = fit_tuning(vc, 4.8e9 + 0.4e9 * vc, REF)
    with pytest.raises(ValueError, match="measured"):
        # 6.5 GHz target lies above the measured 6.0 GHz top
        simulate_pll(curve, 130, FR, 200e-6, C0, BR, n_cycles=20000,
                     vc0=1.0)
    with pytest.raises(ValueError, match="R must be"):
        simulate_pll(VCO, N, FR, 200e-6, C0, [(0.0, 1e-9)], n_cycles=10)


def test_pump_mismatch_folds_delta_sigma_noise_in_band():
    """The edge-level nonlinearity no averaged model can show: UP/DN
    mismatch folds shaped quantization noise into the loop band, and a
    DC offset current that moves the operating point off the origin
    reduces it (Lin, Ti and Liu, IEEE Trans. Circuits Syst. I 56, 877
    (2009), abstract)."""
    fr, n, kv, icp = 50e6, 40, 20e6, 100e-6
    c0, br = 100e-12, [(4.7e3, 1.5e-9)]
    vco = LinearVCO(n * fr - 0.5 * kv + 0.1 * fr, kv, REF)
    ns = 1 << 15
    lev = {}
    for lab, kw in (("matched", {}), ("mismatch", dict(i_dn=90e-6)),
                    ("offset", dict(i_dn=90e-6, i_leak=3e-6))):
        r = simulate_pll(vco, n, fr, icp, c0, br, n_cycles=ns + 4096,
                         num=104857, den=1 << 20, mash_order=3, **kw)
        f, p = signal.welch(r.phase_rad()[4096:], fs=fr,
                            nperseg=1 << 11, detrend="linear")
        sel = (f > 2e3) & (f < 3e4)
        lev[lab] = 10 * np.log10(np.median(p[sel]))
    assert lev["mismatch"] > lev["matched"] + 20.0
    assert lev["offset"] < lev["mismatch"] - 6.0


def test_first_order_output_spurs_match_exact_lines():
    """Exact MASH-1 lines through |L/(1+L)|^2 vs the spurs the
    edge-level loop actually produces, measured by an exact DFT over
    whole sequence periods after lock -- two independent routes."""
    from fracpll import closed_loop_lines, mash_line_spectrum
    fr, n, kv, icp = 50e6, 40, 20e6, 100e-6
    c0, br = 100e-12, [(4.7e3, 1.5e-9)]
    vco = LinearVCO(n * fr - 0.5 * kv + 0.1 * fr, kv, REF)
    P, reps, skip = 64, 200, 4096
    r = simulate_pll(vco, n, fr, icp, c0, br, n_cycles=P * reps + skip,
                     num=3, den=P, mash_order=1)
    x = 2 * np.pi * r.psi_ref[skip:skip + P * reps]
    c = np.fft.rfft(x - x.mean()) / x.size
    lines = mash_line_spectrum(3, P, 1, fr)
    zf = lambda w: loop_filter_impedance(w, c0, br)
    lg = open_loop(2 * np.pi * lines["f_hz"], icp, kv, n, zf)
    model = closed_loop_lines(lines["f_hz"], lines["line_rad2"], lg)
    for k in (1, 2, 3, 5):
        sim_dbc = 10 * np.log10(np.abs(c[reps * k]) ** 2)
        assert abs(sim_dbc - model["sideband_dbc"][k - 1]) < 0.5
