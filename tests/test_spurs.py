"""Reference spurs of a locked integer-N loop (`reference_spurs`, 0.7.0).

`reference_spurs` works in the frequency domain: the Fourier
coefficients of the pump pulses, times the filter transimpedance at
each harmonic, integrated to oscillator phase.  These tests hold it
against routes that do not use that formula:

* a TIME-DOMAIN periodic steady state built here: the filter's state
  equations (written out below, not taken from the package), stepped
  exactly over each constant-current interval of one period with the
  matrix exponential of the augmented system, closed into a periodic
  orbit, and Fourier-integrated by Gauss-Legendre quadrature (64 nodes
  per interval; the integrand is smooth inside each interval);
* the edge-by-edge simulator `simulate_pll`: its locked timing offset
  and its control voltage at the reference edge, which equals the
  steady-state waveform above at t = 0 (the waveform's mean is fixed
  by the lock: a straight-line oscillator runs at N f_ref on average
  only when the mean control voltage is the lock voltage);
* the impulse limit: when the pulses are very short, the pump current
  is an impulse train of charge I_leak T per period, whose Fourier
  coefficients are all I_leak, so the ripple is |Z(j k w_ref)| I_leak.

Tolerances.  Both the time-domain route and `reference_spurs` are
well-conditioned sums of terms of one sign scale; the periodic orbit
solves (I - Phi) x = g on the filter's decaying modes, whose condition
number here is below 1e3, so 1e-9 of the largest harmonic leaves a
wide margin over rounding.  The simulator is exact between edges and
its run is long enough that the start-up transient has decayed (the
last two recorded voltages agree to rounding).  The observed
differences are given next to each assertion; every tolerance is at
least 50 times the observed difference and above the rounding level
of the quantities compared."""
import math

import numpy as np
import pytest
from scipy.linalg import expm

from fracpll import (LinearVCO, loop_filter_impedance, pfd_static_offset,
                     reference_spurs, sampled_stability, simulate_pll,
                     third_order_impedance)

FR = 50e6
T = 1.0 / FR
# README example 4's loop: Icp 200 uA, Kvco 400 MHz/V, N 100
LOOP = dict(icp=200e-6, kvco_hz_per_v=400e6, n_div=100,
            c_shunt=50e-12, branches=[(4.7e3, 1.5e-9)])


def _pulses_here(dt, icp, i_dn, t_reset, t_dz, pol):
    """The pulses of a tri-state PFD with reset delay and turn-on delay,
    written from the rules: the earlier edge sets its flop, the later
    edge sets the other, both reset t_reset after the later edge, and a
    pump conducts only from t_dz after its flop is set."""
    t_ref, t_div = 0.0, dt
    t_rst = max(t_ref, t_div) + t_reset
    out = []
    for start, cur in ((t_ref + t_dz, pol * icp), (t_div + t_dz,
                                                    -pol * i_dn)):
        if t_rst > start:
            out.append((start, t_rst, cur))
    return out


def _state_model(c_shunt, branches, r3=None, c3=None):
    """C dv/dt = -G v + e0 i for the node voltages [pump node, branch
    capacitors..., (C3)]; returns A = -C^-1 G, b = C^-1 e0 and the
    index of the voltage that drives the oscillator."""
    caps = [c_shunt] + [c for _, c in branches]
    res = [r for r, _ in branches]
    if r3 is not None:
        caps.append(c3)
        res.append(r3)
    n = len(caps)
    g = np.zeros((n, n))
    for k, r in enumerate(res, start=1):
        g[0, 0] += 1.0 / r
        g[k, k] += 1.0 / r
        g[0, k] -= 1.0 / r
        g[k, 0] -= 1.0 / r
    cinv = 1.0 / np.array(caps)
    b = np.zeros(n)
    b[0] = cinv[0]
    return -cinv[:, None] * g, b, (n - 1 if r3 is not None else 0)


def _time_domain(pulses, i_leak, c_shunt, branches, r3=None, c3=None,
                 n_harm=5, v_mean=0.0):
    """Periodic steady state over one period starting at t0 (the first
    pulse start, or 0): returns (V_k for k = 1..n_harm, v(0))."""
    A, b, out = _state_model(c_shunt, branches, r3, c3)
    n = A.shape[0]
    t0 = min([0.0] + [a for a, _, _ in pulses])
    cuts = sorted({t0, t0 + T, *[a for a, _, _ in pulses],
                   *[e for _, e, _ in pulses]})
    segs = []
    for a, e in zip(cuts[:-1], cuts[1:]):
        if e > a:
            cur = sum(c for (p, q, c) in pulses if p <= a < q) - i_leak
            segs.append((a, e, cur))

    def aug(h, cur):
        m = np.zeros((n + 1, n + 1))
        m[:n, :n] = A
        m[:n, n] = b * cur
        return expm(m * h)

    phi, g = np.eye(n), np.zeros(n)
    for a, e, cur in segs:
        E = aug(e - a, cur)
        phi, g = E[:n, :n] @ phi, E[:n, :n] @ g + E[:n, n]
    x0 = np.linalg.lstsq(np.eye(n) - phi, g, rcond=None)[0]
    gx, gw = np.polynomial.legendre.leggauss(64)
    k = np.arange(0, n_harm + 1)
    acc = np.zeros(n_harm + 1, dtype=complex)
    x = x0.copy()
    v_at_0 = None
    for a, e, cur in segs:
        h = e - a
        ts = a + 0.5 * h * (gx + 1.0)
        vs = np.array([(aug(t - a, cur) @ np.r_[x, 1.0])[out]
                       for t in ts])
        acc += 0.5 * h * (np.exp(-2j * np.pi * FR * np.outer(k, ts))
                          @ (gw * vs))
        if a <= 0.0 < e or (v_at_0 is None and a == 0.0):
            v_at_0 = (aug(0.0 - a, cur) @ np.r_[x, 1.0])[out]
        x = (aug(h, cur) @ np.r_[x, 1.0])[:n]
    coef = acc / T
    return coef[1:], v_at_0 - coef[0].real + v_mean


def _complex_ripple(sp, kv):
    """V_k back from the returned line powers: |Phi_k| = |Kv||V_k|/(k f)."""
    k = sp["k"]
    return np.sqrt(sp["line_rad2"] / 2.0) * k * FR / abs(kv)


CASES = {
    "leakage": dict(i_leak=2e-6),
    "mismatch, leakage, reset": dict(i_dn=190e-6, i_leak=2e-6,
                                     t_reset=100e-12),
    "dead zone, reference leads": dict(i_leak=1e-6, t_reset=100e-12,
                                       t_deadzone=50e-12),
    "dead zone, divider leads": dict(i_dn=190e-6, t_reset=100e-12,
                                     t_deadzone=30e-12),
    "reversed pump": dict(i_dn=190e-6, i_leak=1e-6, t_reset=100e-12,
                          pump_polarity=-1),
}


@pytest.mark.parametrize("name", list(CASES))
def test_steady_state_equals_the_edge_simulator_and_time_domain(name):
    kw = CASES[name]
    pol = kw.get("pump_polarity", 1)
    kv = pol * LOOP["kvco_hz_per_v"]
    f0 = 4.8e9 if pol == 1 else 6.0e9
    vco = LinearVCO(f0, kv, "design values for this test")
    run = simulate_pll(vco, n_int=LOOP["n_div"], f_ref_hz=FR,
                       icp=LOOP["icp"], c_shunt=LOOP["c_shunt"],
                       branches=LOOP["branches"], n_cycles=8000, **kw)
    sp = reference_spurs(FR, **{**LOOP, "kvco_hz_per_v": kv}, **kw)
    # 1. the locked offset is the simulator's (observed: 1.2e-10)
    dt_sim = run.time_error()[-1]
    assert sp["dt_s"] == pytest.approx(dt_sim, rel=1e-8, abs=0.0)
    # 2. the harmonics equal the time-domain steady state built from
    #    that offset and the PFD rules written in this file (observed:
    #    2e-13 of the largest harmonic)
    i_dn = kw.get("i_dn", LOOP["icp"])
    rules = (LOOP["icp"], i_dn, kw.get("t_reset", 0.0),
             kw.get("t_deadzone", 0.0), pol)
    v_lock = (LOOP["n_div"] * FR - f0) / kv
    vk, _ = _time_domain(_pulses_here(sp["dt_s"], *rules),
                         kw.get("i_leak", 0.0), LOOP["c_shunt"],
                         LOOP["branches"], v_mean=v_lock)
    scale = np.max(np.abs(vk))
    assert np.max(np.abs(sp["ripple_v"] - 2 * np.abs(vk))) \
        <= 1e-9 * 2 * scale
    # 3. the simulator's control voltage at the reference edge is the
    #    steady-state waveform (with the simulator's own offset) at
    #    t = 0.  Rounding of a control voltage near 0.5 V is ~1e-16 V,
    #    about 1e-12 of this ripple, so both checks allow 1e-8 of it
    #    (observed: 4e-13 for the periodicity, 2e-11 for the value).
    _, v0 = _time_domain(_pulses_here(dt_sim, *rules),
                         kw.get("i_leak", 0.0), LOOP["c_shunt"],
                         LOOP["branches"], v_mean=v_lock)
    assert abs(run.vc_ref[-1] - run.vc_ref[-2]) <= 1e-8 * scale
    assert abs(run.vc_ref[-1] - v0) <= 1e-8 * scale


THIRD = dict(c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)], r3=10e3,
             c3=10e-12)
SLOW = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0)


@pytest.mark.parametrize("filt", [
    dict(c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)]),
    dict(c_shunt=400e-12, branches=[(4.7e3, 1.5e-9), (10e3, 3e-9)]),
    THIRD])
@pytest.mark.parametrize("pfd", [
    dict(i_leak=1e-6),
    dict(i_dn=95e-6, i_leak=0.2e-6, t_reset=200e-12),
    dict(i_dn=105e-6, t_reset=200e-12, t_deadzone=50e-12)])
def test_harmonics_equal_the_time_domain_steady_state(filt, pfd):
    sp = reference_spurs(FR, **SLOW, **filt, **pfd, n_harmonics=5)
    dt = pfd_static_offset(FR, SLOW["icp"], i_dn=pfd.get("i_dn"),
                           i_leak=pfd.get("i_leak", 0.0),
                           t_reset=pfd.get("t_reset", 0.0),
                           t_deadzone=pfd.get("t_deadzone", 0.0))["dt_s"]
    assert sp["dt_s"] == dt
    pulses = _pulses_here(dt, SLOW["icp"], pfd.get("i_dn", SLOW["icp"]),
                          pfd.get("t_reset", 0.0),
                          pfd.get("t_deadzone", 0.0), 1)
    vk, _ = _time_domain(pulses, pfd.get("i_leak", 0.0),
                         filt["c_shunt"], filt["branches"],
                         filt.get("r3"), filt.get("c3"))
    scale = np.max(np.abs(vk))
    assert np.max(np.abs(sp["ripple_v"] - 2 * np.abs(vk))) \
        <= 1e-9 * 2 * scale
    # the spur level is the phase line of that ripple
    want = 20 * np.log10(SLOW["kvco_hz_per_v"] * np.abs(vk)
                         / (sp["k"] * FR))
    assert np.max(np.abs(sp["sideband_dbc"] - want)) < 1e-8


def test_impulse_limit_is_the_leakage_times_the_impedance():
    """Pulses 1e-6 of a period long: |c_k| = I_leak sinc(k dt/T), which
    differs from I_leak by (pi k dt/T)^2/6 < 5e-11 for k <= 5."""
    i_leak = 1e-10
    for filt, z in (
            (dict(c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)]),
             lambda w: loop_filter_impedance(w, 100e-12, [(4.7e3, 1.5e-9)])),
            (THIRD, lambda w: third_order_impedance(w, 100e-12, 4.7e3,
                                                    1.5e-9, 10e3, 10e-12))):
        sp = reference_spurs(FR, **SLOW, **filt, i_leak=i_leak)
        assert sp["dt_s"] / T == pytest.approx(i_leak / SLOW["icp"],
                                               rel=1e-12)
        want = 2 * np.abs(z(2 * np.pi * FR * sp["k"])) * i_leak
        assert np.max(np.abs(sp["ripple_v"] / want - 1)) < 1e-9


def test_exact_scalings_and_no_spur_without_net_current():
    base = dict(c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)], i_leak=1e-6)
    a = reference_spurs(FR, **SLOW, **base)
    # the phase line scales as Kvco^2 (the loop check uses Kvco too, so
    # stay well inside the stable range)
    b = reference_spurs(FR, **{**SLOW, "kvco_hz_per_v": 40e6}, **base)
    assert np.allclose(b["line_rad2"], 4 * a["line_rad2"], rtol=1e-13)
    assert np.allclose(b["sideband_dbc"] - a["sideband_dbc"],
                       20 * np.log10(2.0), rtol=0, atol=1e-12)
    assert np.array_equal(a["ripple_v"], b["ripple_v"])
    # matched pumps, no leakage: the UP and DN currents cancel exactly
    c = reference_spurs(FR, **SLOW, c_shunt=100e-12,
                        branches=[(4.7e3, 1.5e-9)], t_reset=100e-12)
    assert np.all(c["line_rad2"] == 0.0)
    assert np.all(c["sideband_dbc"] == -np.inf)
    # a reversed pump with the currents exchanged is the mirror image
    d = reference_spurs(FR, icp=95e-6, kvco_hz_per_v=20e6, n_div=40.0,
                        i_dn=100e-6, c_shunt=100e-12,
                        branches=[(4.7e3, 1.5e-9)], i_leak=1e-7,
                        t_reset=100e-12)
    e = reference_spurs(FR, icp=100e-6, kvco_hz_per_v=-20e6, n_div=40.0,
                        i_dn=95e-6, c_shunt=100e-12,
                        branches=[(4.7e3, 1.5e-9)], i_leak=1e-7,
                        t_reset=100e-12, pump_polarity=-1)
    assert e["dt_s"] == -d["dt_s"]
    assert np.allclose(e["ripple_v"], d["ripple_v"], rtol=1e-13)


def test_small_angle_rule():
    """dBc only where the phase amplitude is 0.2 rad or less."""
    fast = dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, c_shunt=2e-12,
                branches=[(20e3, 200e-12)])
    sp = reference_spurs(FR, **fast, i_leak=20e-6, n_harmonics=3)
    amp = np.sqrt(2 * sp["line_rad2"])
    assert amp[0] > 0.2 and np.isnan(sp["sideband_dbc"][0])
    ok = amp <= 0.2
    assert np.allclose(sp["sideband_dbc"][ok],
                       10 * np.log10(sp["line_rad2"][ok] / 2), atol=1e-12)


def test_stability_check_uses_the_pulse_the_phase_moves():
    """README example 5's loop has 0.18 dB of gain margin.  With UP at
    0.52 mA and DN at 0.5 mA and a reset delay, the divider edge leads
    and the DN pulse carries the phase error: stable.  With enough
    leakage the reference leads, the UP pulse carries it, and the loop
    with 0.52 mA is unstable: refused."""
    fast = dict(kvco_hz_per_v=400e6, n_div=10, c_shunt=2e-12,
                branches=[(20e3, 200e-12)])
    assert sampled_stability(0.5e-3, f_ref_hz=FR, **fast)["stable"]
    assert not sampled_stability(0.52e-3, f_ref_hz=FR, **fast)["stable"]
    ok = reference_spurs(FR, icp=0.52e-3, i_dn=0.5e-3, t_reset=100e-12,
                         **fast)
    assert ok["dt_s"] < 0.0
    assert ok["check"]["spectral_radius"] == pytest.approx(
        sampled_stability(0.5e-3, f_ref_hz=FR, **fast)["spectral_radius"],
        rel=1e-15)
    with pytest.raises(ValueError, match="unstable"):
        reference_spurs(FR, icp=0.52e-3, i_dn=0.5e-3, t_reset=100e-12,
                        i_leak=5e-6, **fast)


def test_refusals():
    base = dict(c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)])
    with pytest.raises(ValueError, match="dead zone"):
        reference_spurs(FR, **SLOW, **base, t_deadzone=1e-10)
    with pytest.raises(ValueError, match="POSITIVE"):
        reference_spurs(FR, **{**SLOW, "kvco_hz_per_v": -20e6}, **base)
    with pytest.raises(ValueError, match="pump_polarity"):
        reference_spurs(FR, **SLOW, **base, pump_polarity=1.5)
    with pytest.raises(ValueError, match="both r3 and c3"):
        reference_spurs(FR, **SLOW, **base, r3=10e3)
    with pytest.raises(ValueError, match="c3"):
        reference_spurs(FR, **SLOW, **base, r3=10e3, c3=-1e-12)
    for bad in (0, 2.5, -1, True):
        with pytest.raises(ValueError, match="n_harmonics"):
            reference_spurs(FR, **SLOW, **base, n_harmonics=bad)
    with pytest.raises(ValueError, match="n_div"):
        reference_spurs(FR, **{**SLOW, "n_div": float("nan")}, **base)
    with pytest.raises(ValueError, match="kvco"):
        reference_spurs(FR, **{**SLOW, "kvco_hz_per_v": 0.0}, **base)
    with pytest.raises(ValueError, match="finite"):
        reference_spurs(FR, **SLOW, **base, i_leak=float("nan"))
    with pytest.raises(ValueError, match="C must be"):
        reference_spurs(FR, **SLOW, c_shunt=100e-12,
                        branches=[(4.7e3, -1.5e-9)])
    # example 12's refused third-order loop has no steady state
    with pytest.raises(ValueError, match="averaged model cannot confirm"):
        reference_spurs(FR, **SLOW, **base, r3=20e3, c3=1e-9, i_leak=1e-9)
    # the leakage needs more than a full period of pumping
    with pytest.raises(ValueError, match="reference period"):
        reference_spurs(FR, **SLOW, **base, i_leak=150e-6)
