"""Lock-transient settling time (0.4.0 fix) and the transient itself.

Before 0.4.0 `t_settle` was rounded up to the next output sample (the
README example reported 31.8 us = two samples of 15.9 us).  It is now
found on the integrator's continuous solution.  Anchors:

* in its small-signal limit (linear tuning curve, start on the lock
  voltage, a small phase step, no leakage) the averaged model is a
  linear system; its trajectory and settling time are held against the
  matrix exponential of that system, built here by hand;
* the settling time no longer depends on the number of output samples,
  and the old rounded value lies within one sample above it."""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from fracpll import fit_tuning, lock_transient

REF = "synthetic tuning curve, fracpll test suite"
FR, N, ICP, C0, R, C1 = 50e6, 100.0, 200e-6, 50e-12, 4.7e3, 1.5e-9
KV = 0.4e9


def _linear_curve():
    vc = np.linspace(0.0, 3.0, 13)
    return fit_tuning(vc, 4.8e9 + KV * vc, REF)   # PCHIP of a line is it


def test_small_signal_transient_equals_matrix_exponential():
    curve = _linear_curve()
    v_lock = (N * FR - 4.8e9) / KV
    phi0, tol, t_end = 1e-3, 1e-5, 200e-6
    res = lock_transient(curve, N, FR, ICP, C0, [(R, C1)], vc0=v_lock,
                         phi0=phi0, t_end=t_end, n_eval=401,
                         settle_tol_rad=tol)
    # x = [phi, v_shunt - v_lock, v_branch - v_lock]; tanh(x) ~ x
    A = np.array([[0.0, -2 * np.pi * KV / N, 0.0],
                  [ICP / (2 * np.pi * C0), -1 / (R * C0), 1 / (R * C0)],
                  [0.0, 1 / (R * C1), -1 / (R * C1)]])
    x0 = np.array([phi0, 0.0, 0.0])
    lin = lambda t: (expm(A * t) @ x0)[0]
    ref = np.array([lin(t) for t in res["t"]])
    assert np.max(np.abs(res["phi_e"] - ref)) < 1e-5 * phi0
    # settling time of the linear system: last exit from the band
    tg = np.linspace(0.0, t_end, 20001)
    step = expm(A * (tg[1] - tg[0]))
    x, g = x0.copy(), np.empty(tg.size)
    for i in range(tg.size):
        g[i] = abs(x[0]) - tol
        x = step @ x
    i = int(np.flatnonzero(g >= 0)[-1])
    t_ref = brentq(lambda t: abs(lin(t)) - tol, tg[i], tg[i + 1],
                   xtol=1e-18)
    assert res["locked"]
    # (limited by the integrator tolerance rtol=1e-8: 5e-6 observed)
    assert abs(res["t_settle"] / t_ref - 1.0) < 3e-5


def test_settling_time_does_not_depend_on_output_sampling():
    curve = fit_tuning([0.0, 1.0, 2.0, 3.0], [4.8e9, 5.2e9, 5.6e9, 6.0e9],
                       REF)
    kw = dict(icp=ICP, c_shunt=C0, branches=[(R, C1)], i_leak=2e-6)
    coarse = lock_transient(curve, N, FR, **kw)                # 2000
    fine = lock_transient(curve, N, FR, n_eval=20000, **kw)
    assert abs(coarse["t_settle"] / fine["t_settle"] - 1.0) < 1e-9
    # the pre-0.4.0 rule: the output sample after the last one outside
    # the band -- at most one sample above the exact crossing
    t, phi = coarse["t"], coarse["phi_e"]
    off = np.flatnonzero(np.abs(phi - coarse["phi_static"]) >= 0.05)
    old = t[off[-1] + 1]
    assert 0.0 <= old - coarse["t_settle"] <= t[1] - t[0]
    # the phase really is at the band edge there.  Checked by a SEPARATE
    # integration that stops exactly at t_settle (its own step sequence,
    # no interpolation or root finding): observed 2e-9 rad from the
    # edge, set by the solver tolerance rtol = 1e-8.  The bound 1e-6 rad
    # is 20x tighter than a 1e-4 relative error in t_settle would give
    # here (2.1e-5 rad); the pre-0.4.0 value sits 0.017 rad inside.
    ts, band = coarse["t_settle"], 0.05
    edge = lambda te: abs(lock_transient(curve, N, FR, t_end=te, n_eval=2,
                                         **kw)["phi_e"][-1]
                          - coarse["phi_static"]) - band
    assert abs(edge(ts)) < 1e-6
    assert edge(ts * (1 - 1e-4)) > 0.0         # just before: outside
    assert edge(old) < -1e-3                   # the old value: inside


def test_new_input_refusals():
    curve = _linear_curve()
    kw = dict(icp=ICP, c_shunt=C0, branches=[(R, C1)])
    with pytest.raises(ValueError, match="does not extrapolate"):
        lock_transient(curve, N, FR, vc0=-5.0, **kw)
    with pytest.raises(ValueError, match="c_shunt"):
        lock_transient(curve, N, FR, icp=ICP, c_shunt=-C0,
                       branches=[(R, C1)])
    with pytest.raises(ValueError, match="branch capacitors"):
        lock_transient(curve, N, FR, icp=ICP, c_shunt=C0,
                       branches=[(R, 0.0)])
    with pytest.raises(ValueError, match="settle_tol_rad"):
        lock_transient(curve, N, FR, settle_tol_rad=0.0, **kw)
    with pytest.raises(ValueError, match="n_eval"):
        lock_transient(curve, N, FR, n_eval=1, **kw)
