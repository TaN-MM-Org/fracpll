"""Regression tests for the 0.6.0 fixes.  Each fails with 0.5.0.

* `stability` reported 180 + arg L as the phase margin with arg L in
  (-180, 180]: a loop gain whose phase had passed -180 degrees came out
  with a margin above 180 degrees and was called stable.
* `simulate_pll` read reference-jitter entry n_cycles + 1 after the
  last edge, so an array of the documented n_cycles + 1 entries raised
  IndexError at the end of the run.
* `TuningFamily` accepted temperature keys that are numbers written as
  text ("300") and then raised KeyError when evaluated.
* Non-finite or non-physical inputs that returned NaN or a result for
  a non-physical circuit."""
import numpy as np
import pytest

from fracpll import (LinearVCO, NoiseSpec, TuningFamily, closed_loop_lines,
                     continuous_closed_loop_poles, fit_tuning, open_loop,
                     sampled_margins, simulate_pll, simulate_sampled,
                     stability, synthesizer_psd, third_order_impedance)

# a third-order filter whose R3-C3 pole (8 kHz) sits below the crossover
THIRD = (100e-12, 4.7e3, 1.5e-9, 20e3, 1e-9)
LOOP = (100e-6, 20e6, 40.0)


def test_phase_past_minus_180_is_refused_as_unstable():
    c1, r2, c2, r3, c3 = THIRD
    icp, kv, nd = LOOP
    zf = lambda w: third_order_impedance(w, *THIRD)
    # independent verdict: the closed-loop characteristic polynomial
    # N s^2 (a2 s^2 + a1 s + a0) + Icp Kvco (1 + s R2 C2) has a root in
    # the right half plane
    a0 = c1 + c2 + c3
    a1 = r2 * c2 * (c1 + c3) + r3 * c3 * (c1 + c2)
    a2 = c1 * c2 * c3 * r2 * r3
    roots = np.roots([nd * a2, nd * a1, nd * a0, icp * kv * r2 * c2,
                      icp * kv])
    assert roots.real.max() > 0.0
    # the crossover (17840 Hz) and the 0.5.0 reading of the margin
    f = np.geomspace(1e3, 1e6, 200001)
    lg = open_loop(2 * np.pi * f, icp, kv, nd, zf)
    i = np.flatnonzero(np.abs(lg) > 1.0)[-1]
    assert f[i] == pytest.approx(17839.6, rel=1e-4)
    old = 180.0 + np.degrees(np.angle(lg[i]))
    assert old == pytest.approx(336.85, abs=0.01)      # 0.5.0 returned it
    with pytest.raises(ValueError, match=r"phase margin -23\.15 deg"):
        stability(icp, kv, nd, zf, f_ref_hz=50e6)


def test_ref_jitter_of_the_documented_length_runs():
    vco = LinearVCO(4.8e9, 400e6, "design values for this test")
    kw = dict(n_int=100, f_ref_hz=50e6, icp=200e-6, c_shunt=50e-12,
              branches=[(4.7e3, 1.5e-9)], n_cycles=200)
    jit = np.random.default_rng(7).normal(0.0, 1e-12, 201)
    a = simulate_pll(vco, ref_jitter_s=jit, **kw)     # 0.5.0: IndexError
    b = simulate_pll(vco, ref_jitter_s=np.r_[jit, 5e-12], **kw)
    assert a.t_ref.size == 200
    assert np.array_equal(a.psi_ref, b.psi_ref)
    assert np.array_equal(a.time_error(), b.time_error())
    # entry k moves the reference edge nominally at k T
    assert np.allclose(a.t_ref, np.arange(1, 201) / 50e6 + jit[1:],
                       rtol=0, atol=1e-21)
    with pytest.raises(ValueError, match="n_cycles \\+ 1"):
        simulate_pll(vco, ref_jitter_s=jit[:200], **kw)


def test_tuning_family_keys_written_as_text():
    ref = "synthetic tuning points, fracpll test suite"
    c0 = fit_tuning([0, 1, 2, 3], [4.8e9, 5.2e9, 5.6e9, 6.0e9], ref)
    c1 = fit_tuning([0, 1, 2, 3], [4.7e9, 5.1e9, 5.5e9, 5.9e9], ref)
    fam = TuningFamily({"300": c0, "350": c1})        # 0.5.0: KeyError
    v = [0.5, 1.7]
    x = (320.0 - 300.0) / 50.0
    want = (1 - x) * c0.frequency(v) + x * c1.frequency(v)
    assert np.array_equal(fam.frequency(v, 320.0), want)
    assert np.array_equal(fam.kvco(v, 320.0),
                          (1 - x) * c0.kvco(v) + x * c1.kvco(v))
    num = TuningFamily({300.0: c0, 350.0: c1})
    assert np.array_equal(num.frequency(v, 320.0), want)
    with pytest.raises(ValueError, match="twice"):
        TuningFamily({"300": c0, 300: c1})
    with pytest.raises(ValueError, match="finite"):
        num.frequency(v, float("nan"))     # 0.5.0: returned NaN


def test_non_finite_and_non_physical_inputs_are_refused():
    nan = float("nan")
    spec = NoiseSpec((1e4, 1e5), (-80.0, -100.0),
                     "synthetic points, fracpll test suite")
    with pytest.raises(ValueError, match="finite"):
        spec.psd([nan])                     # 0.5.0: [nan]
    with pytest.raises(ValueError, match="finite"):
        simulate_sampled(100e-6, 20e6, 40.0, 50e6, 100e-12,
                         [(4.7e3, 1.5e-9)], u_cycles=[0.0, nan, 0.0])
    with pytest.raises(ValueError, match="n_div"):
        synthesizer_psd([1e3], [1 + 0j], nan, s_inband=[1e-12])
    with pytest.raises(ValueError, match="lg"):
        synthesizer_psd([1e3], [nan + 0j], 40.0, s_inband=[1e-12])
    with pytest.raises(ValueError, match="finite"):
        closed_loop_lines([1e3], [nan], [1 + 0j])
    with pytest.raises(ValueError, match="c_shunt"):
        # 0.5.0: the poles of a filter with a negative capacitor
        continuous_closed_loop_poles(100e-6, 20e6, 40, -100e-12,
                                     [(4.7e3, 1.5e-9)])
    with pytest.raises(ValueError, match="icp"):
        continuous_closed_loop_poles(nan, 20e6, 40, 100e-12,
                                     [(4.7e3, 1.5e-9)])
    with pytest.raises(ValueError, match="n_grid"):
        # 0.5.0: "|G| does not fall through 1", a misleading reason
        sampled_margins(100e-6, 20e6, 40.0, 50e6, 100e-12,
                        [(4.7e3, 1.5e-9)], n_grid=1)
    # an R = 0 branch is still a valid (degenerate) averaged filter
    p = continuous_closed_loop_poles(100e-6, 20e6, 40, 100e-12,
                                     [(0.0, 1.5e-9)])
    assert p.size == 2
