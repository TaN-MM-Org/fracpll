"""Second-order pulse-width anchors.

With a delta-sigma divider the edge-level loop shows in-band noise the
exact LINEAR sampled map misses.  It is explained here, quantitatively
and with no fitted parameter, by the second-order effects of the
finite PFD pulse (chiefly the charge centroid tau/2 away from the
reference edge): the self-consistent order-2 map of `simulate_sampled`
reproduces the edge-level simulator cycle by cycle."""
import numpy as np
import pytest
from scipy import signal
from scipy.linalg import expm

from fracpll import (LinearVCO, pulse_doublet_vector, sampled_loop_map,
                     simulate_pll, simulate_sampled)
from fracpll._network import FilterModes

REF = "design value, fracpll test suite 2026-09-21"
FR, ICP, BR = 50e6, 100e-6, [(4.7e3, 1.5e-9)]
NUM, DEN = 104857, 1 << 20
K = (1 << 16) + 2048
SL = slice(8192, None)


def _bands(x, fr=FR, edges=((1e4, 2e4), (2e4, 5e4))):
    f, p = signal.welch(2 * np.pi * x, fs=fr, nperseg=1 << 14,
                        detrend="linear")
    return np.array([10 * np.log10(np.mean(p[(f >= a) & (f < b)]))
                     for a, b in edges])


def _runs(n, kv, c0, pol=1, f0=None):
    vco = LinearVCO(f0 if f0 is not None else n * FR - 0.5 * kv + 0.1 * FR,
                    kv, REF)
    edge = simulate_pll(vco, n, FR, ICP, c0, BR, n_cycles=K, num=NUM,
                        den=DEN, mash_order=3, pump_polarity=pol)
    kw = dict(n_int=n, num=NUM, den=DEN, mash_order=3, n_cycles=K,
              pump_polarity=pol)
    o1 = simulate_sampled(ICP, kv, n + NUM / DEN, FR, c0, BR, order=1, **kw)
    o2 = simulate_sampled(ICP, kv, n + NUM / DEN, FR, c0, BR, order=2, **kw)
    return edge, o1, o2


@pytest.mark.parametrize("c0", [100e-12, 400e-12])
def test_order2_reproduces_edge_level_loop(c0):
    edge, o1, o2 = _runs(40, 20e6, c0)
    e1 = (edge.psi_ref - o1["psi"])[SL]
    e2 = (edge.psi_ref - o2["psi"])[SL]
    assert e2.std() < 1e-3 * e1.std()          # >1000x closer
    assert np.abs(edge.time_error() - o2["tau"])[SL].max() < 1e-15
    b_edge = _bands(edge.psi_ref[SL])
    assert np.all(np.abs(_bands(o2["psi"][SL]) - b_edge) < 0.5)
    # ... where the linear map is visibly wrong in band
    assert np.any(b_edge - _bands(o1["psi"][SL]) > 3.0)


def test_order1_is_the_linear_sampled_map():
    c0, kv = 100e-12, 20e6
    nd = 40 + NUM / DEN
    o1 = simulate_sampled(ICP, kv, nd, FR, c0, BR, n_int=40, num=NUM,
                          den=DEN, mash_order=3, n_cycles=4000, order=1)
    from fracpll import mash_sequence
    u = np.cumsum(mash_sequence(NUM, DEN, 3, 4004) - NUM / DEN)[:4000]
    F = sampled_loop_map(ICP, kv, nd, FR, c0, BR)
    Phi = F["Phi"]
    b = FilterModes(c0, BR).state_matrices()[1]
    B = np.r_[b, 0.0]
    s = np.zeros(Phi.shape[0])
    ref = np.empty(4000)
    for k in range(4000):
        s = Phi @ s
        ref[k] = s[-1]
        s = s + B * (-ICP * (s[-1] - u[k]) / (nd * FR))
    assert np.allclose(o1["psi"], ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_doublet_vector_ramp_and_lasting_part():
    """Oscillator row of e^{Mt} M B = Kvco x (v_c impulse response):
    Kvco/C_shunt at t = 0+ (the ramp), Kvco/C_total at long times --
    why the in-band effect does not depend on C_shunt."""
    kv = 20e6
    for c0 in (50e-12, 100e-12, 400e-12):
        mb = pulse_doublet_vector(kv, c0, BR)
        assert np.isclose(mb[-1], kv / c0, rtol=1e-12)
        fm = FilterModes(c0, BR)
        a, b, c = fm.state_matrices()
        n = fm.n
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = a
        M[n, :n] = kv * c
        late = (expm(M * 1e-3) @ mb)[-1]
        c_tot = c0 + sum(cc for _, cc in BR)
        assert np.isclose(late, kv / c_tot, rtol=1e-9)


def test_effect_shrinks_with_pulse_width():
    """N and Kvco doubled together: the linear loop is identical in
    oscillator cycles, pulses are half as wide, and the second-order
    source Kvco Icp tau^2 halves -> about -6 dB, in both the edge-level
    run and the order-2 map."""
    lev = {}
    for n, kv in ((40, 20e6), (80, 40e6)):
        edge, o1, o2 = _runs(n, kv, 100e-12)
        lev[n] = (_bands((edge.psi_ref - o1["psi"])[SL]),
                  _bands((o2["psi"] - o1["psi"])[SL]))
    for i in (0, 1):
        drop = lev[40][i] - lev[80][i]
        assert np.all((drop > 4.5) & (drop < 7.5))


def test_negative_kvco_order2():
    edge, o1, o2 = _runs(40, -20e6, 100e-12, pol=-1,
                         f0=40 * FR + 0.5 * 20e6 + 0.1 * FR)
    e1 = (edge.psi_ref - o1["psi"])[SL]
    e2 = (edge.psi_ref - o2["psi"])[SL]
    assert e2.std() < 1e-3 * e1.std()


def test_refusals():
    with pytest.raises(ValueError, match="order"):
        simulate_sampled(ICP, 20e6, 40.1, FR, 100e-12, BR,
                         u_cycles=np.zeros(4), order=3)
    with pytest.raises(ValueError, match="negative"):
        simulate_sampled(ICP, -20e6, 40.1, FR, 100e-12, BR,
                         u_cycles=np.zeros(4))
    with pytest.raises(ValueError, match="u_cycles"):
        simulate_sampled(ICP, 20e6, 40.1, FR, 100e-12, BR)
    with pytest.raises(ValueError, match="mean"):
        simulate_sampled(ICP, 20e6, 40.0, FR, 100e-12, BR, n_int=40,
                         num=NUM, den=DEN, mash_order=3, n_cycles=10)
