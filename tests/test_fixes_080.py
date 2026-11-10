"""Regression tests for the 0.8.0 fixes.  Each fails with 0.7.0.

* Whole-number inputs went through int(), which cuts toward zero
  without a word: `mash_sequence(3.7, 64, ...)` returned the pattern of
  3/64, `simulate_pll(..., n_int=40.5)` ran at N = 40,
  `dsm_phase_psd(..., m=2.5)` returned the second-order spectrum, and
  `jitter_relative_sigma(..., n_avg=2.5)` the error bar of 2 averages.
  The same held for num, den, the MASH order and the cycle counts of
  `simulate_pll` and `simulate_sampled`, and for `mash_period` and
  `mash_line_spectrum`.  A fraction is now refused; a whole number
  written as a float (40.0), a NumPy integer or a string ("64") is
  still accepted, so no valid call changes.
* `simulate_sampled` with a fraction num and mash_order=0 (and no
  n_int) ignored num and returned an integer-N run with no divider
  phase at all (psi identically zero).  `simulate_pll` already
  refused this; now `simulate_sampled` does too.
* `TuningCurve.frequency` and `.kvco` (and so `TuningFamily`) returned
  NaN for a NaN control voltage, which passed the range check.  As for
  `NoiseSpec.psd` since 0.6.0, a non-finite value is now refused."""
import numpy as np
import pytest

from fracpll import (LinearVCO, NoiseSpec, TuningFamily, dsm_phase_psd,
                     fit_tuning, jitter_relative_sigma, mash_line_spectrum,
                     mash_period, mash_sequence, simulate_pll,
                     simulate_sampled)

FR = 50e6
REF = "design values for this test"
SLOW = dict(icp=100e-6, kvco_hz_per_v=20e6, f_ref_hz=FR, c_shunt=100e-12,
            branches=[(4.7e3, 1.5e-9)])
VCO = LinearVCO(1.995e9, 20e6, REF)


def _pll(**kw):
    return simulate_pll(VCO, f_ref_hz=FR, icp=100e-6, c_shunt=100e-12,
                        branches=[(4.7e3, 1.5e-9)], **kw)


@pytest.mark.parametrize("call", [
    lambda: mash_sequence(3.7, 64, 1, 100),
    lambda: mash_sequence(3, 64.5, 1, 100),
    lambda: mash_sequence(3, 64, 2.5, 100),
    lambda: mash_sequence(3, 64, 1, 100.5),
    lambda: mash_period(3.7, 64, 1),
    lambda: mash_period(3, 64, 1, max_period=1e3 + 0.5),
    lambda: mash_line_spectrum(3.7, 64, 1, FR),
    lambda: mash_line_spectrum(3, 64, 1.5, FR),
    lambda: dsm_phase_psd([1e5], FR, 2.5),
    lambda: _pll(n_int=40.5, n_cycles=10),
    lambda: _pll(n_int=40, n_cycles=10, num=3.5, den=64, mash_order=1),
    lambda: _pll(n_int=40, n_cycles=10, num=3, den=64.5, mash_order=1),
    lambda: _pll(n_int=40, n_cycles=10, num=3, den=64, mash_order=1.5),
    lambda: _pll(n_int=40, n_cycles=10.5),
    lambda: _pll(n_int=40, n_cycles=10, quad_nodes=8.5),
    lambda: simulate_sampled(n_div=40 + 3 / 64, **SLOW, n_int=40.5,
                             num=3, den=64, mash_order=1, n_cycles=10),
    lambda: simulate_sampled(n_div=40 + 3 / 64, **SLOW, n_int=40,
                             num=3.5, den=64, mash_order=1, n_cycles=10),
    lambda: simulate_sampled(n_div=40 + 3 / 64, **SLOW, n_int=40, num=3,
                             den=64, mash_order=1.5, n_cycles=10),
    lambda: simulate_sampled(n_div=40.0, **SLOW, n_cycles=10.5),
    lambda: jitter_relative_sigma(np.geomspace(1e4, 1e7, 50),
                                  1e-12 / np.geomspace(1e4, 1e7, 50), 2.5),
])
def test_fractions_are_refused_not_cut(call):
    with pytest.raises(ValueError, match="whole number"):
        call()


def test_whole_numbers_in_other_forms_still_work():
    a = mash_sequence(3, 64, 3, 500)
    for args in ((3.0, 64.0, 3.0, 500.0), (np.int64(3), np.int64(64),
                                           np.int64(3), np.int64(500)),
                 ("3", "64", "3", "500")):
        assert np.array_equal(mash_sequence(*args), a)
    # moduli beyond int64 and beyond a float's 53 bits stay exact
    big = 3 << 70, 64 << 70
    assert np.array_equal(mash_sequence(*big, 3, 500), a)
    assert np.array_equal(mash_sequence(str(big[0]), str(big[1]), 3, 500), a)
    assert mash_period(3.0, 64.0, 2.0) == mash_period(3, 64, 2)
    r0 = _pll(n_int=40, n_cycles=50, num=3, den=64, mash_order=1)
    r1 = _pll(n_int=40.0, n_cycles=50.0, num=3.0, den=64.0,
              mash_order=1.0, quad_nodes=16.0)
    assert np.array_equal(r0.psi_ref, r1.psi_ref)
    f = np.geomspace(1e4, 1e7, 50)
    assert jitter_relative_sigma(f, 1e-12 / f, 4.0) \
        == jitter_relative_sigma(f, 1e-12 / f, 4)
    s0 = simulate_sampled(n_div=40 + 3 / 64, **SLOW, n_int=40, num=3,
                          den=64, mash_order=1, n_cycles=100)
    s1 = simulate_sampled(n_div=40 + 3 / 64, **SLOW, n_int=40.0, num=3.0,
                          den=64.0, mash_order=1.0, n_cycles=100.0)
    assert np.array_equal(s0["psi"], s1["psi"])
    for bad in (float("nan"), float("inf"), "x", None):
        with pytest.raises(ValueError, match="whole number"):
            mash_sequence(3, 64, 1, bad)


def test_fraction_without_a_mash_order_is_refused():
    # 0.7.0 returned psi = 0 exactly (max |psi| = 0.0) for this call
    with pytest.raises(ValueError, match="mash_order 1, 2 or 3"):
        simulate_sampled(n_div=40.125, **SLOW, num=1, den=8, mash_order=0,
                         n_cycles=100)
    # integer-N with num = 0 is unchanged
    r = simulate_sampled(n_div=40.0, **SLOW, num=0, mash_order=0,
                         n_cycles=100)
    assert np.all(r["psi"] == 0.0)


def test_tuning_curve_refuses_a_nan_voltage():
    curve = fit_tuning([0.0, 1.0, 2.0, 3.0], [4.8e9, 5.2e9, 5.6e9, 6.0e9],
                       reference="illustrative tuning points")
    fam = TuningFamily({300.0: curve, 350.0: curve})
    for call in (lambda: curve.frequency(np.nan),
                 lambda: curve.kvco([1.0, np.nan]),
                 lambda: curve.frequency(np.inf),
                 lambda: fam.frequency(np.nan, 300.0),
                 lambda: fam.kvco(np.nan, 320.0)):
        with pytest.raises(ValueError, match="finite"):
            call()
    # finite values are unchanged, and NoiseSpec behaves the same way
    assert curve.frequency(1.5)[0] == 5.4e9
    spec = NoiseSpec((1e4, 1e6), (-80.0, -120.0), reference="test points")
    with pytest.raises(ValueError, match="finite"):
        spec.psd(np.nan)
