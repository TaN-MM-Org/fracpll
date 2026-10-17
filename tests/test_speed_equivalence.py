"""0.6.0 speedups, each held against the 0.5.0 code kept here as the
reference.

* `mash_sequence` and `mash_period`: the accumulators in closed form,
  vectorized.  Integer results, so equal element by element.
* `stability`: Brent's method instead of 80 bisection steps for the
  crossover; the same root to a few units in the last place.
* `simulate_sampled`: the cycle loop on Python floats; the same
  operations except the order of the 3-term sums in Phi s, so equal to
  rounding (asked here: 1e-12 of the largest value)."""
import numpy as np
import pytest
from scipy.linalg import expm

from fracpll import (loop_filter_impedance, mash_period, mash_sequence,
                     open_loop, simulate_sampled, stability)
from fracpll._network import FilterModes


# --- 0.5.0 reference implementations ---------------------------------
def ref_mash_sequence(num, den, m, n_steps):
    acc = [0] * m
    carries = np.zeros((m, n_steps), dtype=np.int64)
    for k in range(n_steps):
        inp = num
        for i in range(m):
            acc[i] += inp
            if acc[i] >= den:
                acc[i] -= den
                carries[i, k] = 1
            inp = acc[i]
    y = carries[0].copy()
    for i in range(1, m):
        d = carries[i].copy()
        for _ in range(i):
            d = np.diff(d, prepend=0)
        y += d
    return y


def ref_mash_period(num, den, m, max_period=1 << 22):
    acc = [0] * m
    for k in range(1, max_period + 1):
        inp = num
        new = []
        for a in acc:
            a = (a + inp) % den
            new.append(a)
            inp = a
        acc = new
        if all(a == 0 for a in acc):
            return k
    return None


def ref_crossover(icp, kv, nd, zf, f_ref, n_grid=4001):
    f = np.geomspace(f_ref * 1e-6, f_ref * 0.5, n_grid)
    mag = np.abs(open_loop(2 * np.pi * f, icp, kv, nd, zf))
    above = mag > 1.0
    i = int(np.flatnonzero(above[:-1] & ~above[1:])[-1])
    a, b = f[i], f[i + 1]
    for _ in range(80):
        m = np.sqrt(a * b)
        if np.abs(open_loop(2 * np.pi * m, icp, kv, nd, zf))[0] > 1.0:
            a = m
        else:
            b = m
    fc = np.sqrt(a * b)
    lg = open_loop(2 * np.pi * fc, icp, kv, nd, zf)[0]
    return fc, 180.0 + np.degrees(np.angle(lg))


def ref_simulate_sampled(icp, kv, nd, f_ref, c_shunt, branches, u, order,
                         pol=1):
    fm = FilterModes(c_shunt, branches)
    a, b, c = fm.state_matrices()
    n = fm.n
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = a
    M[n, :n] = kv * c
    Phi = expm(M / f_ref)
    B = np.r_[b, 0.0]
    MB = M @ B
    c0 = float(c_shunt)
    two = order == 2
    s = np.zeros(n + 1)
    psi = np.empty(u.size)
    taus = np.empty(u.size)
    nf = nd * f_ref
    for k in range(u.size):
        s = Phi @ s
        tau = (u[k] - s[-1]) / nf
        if two:
            dpsi = kv * s[0] * tau
            if tau > 0.0:
                dpsi += kv * pol * icp * tau * tau / (2.0 * c0)
            tau = (u[k] - s[-1] - dpsi) / nf
            v = -(pol * icp * tau * tau / 2.0) * MB
            if tau < 0.0:
                s[-1] += v[-1]
        psi[k] = s[-1]
        taus[k] = tau
        s = s + B * (pol * icp * tau)
        if two:
            if tau < 0.0:
                s[:-1] += v[:-1]
            else:
                s += v
    return psi, taus


# --- tests -----------------------------------------------------------
def test_mash_sequence_equals_the_accumulator_loop():
    rng = np.random.default_rng(11)
    dens = [2, 3, 64, 1000, 2**20, 12345677, 2**43 - 1, 2**44, 2**50 + 3]
    cases = [(int(rng.integers(0, d)), d, m, n)
             for d in dens for m in (1, 2, 3)
             for n in (1, 7, (1 << 16) + 1)]
    cases += [(104857, 2**20, 3, 3 * (1 << 16) + 5), (0, 5, 3, 100),
              (12345, 2**20, 2, (1 << 16) - 1),
              (2**20 - 1, 2**20, 3, 70000)]
    for num, den, m, n in cases:
        assert np.array_equal(mash_sequence(num, den, m, n),
                              ref_mash_sequence(num, den, m, n)), \
            (num, den, m, n)


@pytest.mark.parametrize("num,den,m", [
    (3, 64, 1), (3, 64, 2), (3, 64, 3), (1, 2, 3), (0, 5, 2),
    (5, 4096, 2), (1234, 4097, 3), (12345, 2**17, 3), (7, 2**45, 1)])
def test_mash_period_equals_the_state_loop(num, den, m):
    want = ref_mash_period(num, den, m, max_period=1 << 18)
    if want is None:
        with pytest.raises(ValueError, match="max_period"):
            mash_period(num, den, m, max_period=1 << 18)
    else:
        assert mash_period(num, den, m) == want


def test_stability_root_equals_the_bisection():
    rng = np.random.default_rng(5)
    for _ in range(40):
        c1 = 10 ** rng.uniform(-11, -9)
        r = 10 ** rng.uniform(2.5, 4.5)
        c2 = c1 * 10 ** rng.uniform(0.3, 2)
        icp, kv = 10 ** rng.uniform(-5, -3), 10 ** rng.uniform(6, 9)
        nd = rng.uniform(10, 200)
        zf = lambda w: loop_filter_impedance(w, c1, [(r, c2)])
        fc_old, pm_old = ref_crossover(icp, kv, nd, zf, 50e6)
        rep = stability(icp, kv, nd, zf, 50e6)
        assert rep["f_crossover_hz"] == pytest.approx(fc_old, rel=1e-12)
        assert rep["phase_margin_deg"] == pytest.approx(pm_old, abs=1e-9)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("filt,kv,pol", [
    ((100e-12, [(4.7e3, 1.5e-9)]), 20e6, 1),
    ((400e-12, [(4.7e3, 1.5e-9), (10e3, 3e-9)]), -20e6, -1)])
def test_simulate_sampled_equals_the_array_loop(order, filt, kv, pol):
    num, den = 104857, 2**20
    nd = 40 + num / den
    n = 6000
    d = mash_sequence(num, den, 3, n + 4)
    u = np.cumsum(d - num / den)[:n]
    c0, br = filt
    new = simulate_sampled(100e-6, kv, nd, 50e6, c0, br, n_int=40,
                           num=num, den=den, mash_order=3, n_cycles=n,
                           order=order, pump_polarity=pol)
    psi, tau = ref_simulate_sampled(100e-6, kv, nd, 50e6, c0, br, u,
                                    order, pol)
    assert np.max(np.abs(new["psi"] - psi)) <= 1e-12 * np.max(np.abs(psi))
    assert np.max(np.abs(new["tau"] - tau)) <= 1e-12 * np.max(np.abs(tau))
