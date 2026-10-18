"""Regression tests for the 0.7.0 fixes.  Each fails with 0.6.1.

* `mash_line_spectrum` multiplied the int64 pattern by the modulus in
  NumPy integers; for a modulus of 2^63 or more (valid: `mash_sequence`
  and `mash_period` accept any integer) NumPy 2 raised OverflowError.
  The phase is now summed in Python integers from den = 2^44 on.
  Checked by an exact identity: num/den and (num 2^s)/(den 2^s) give
  the same accumulator states scaled by 2^s, so the same pattern and
  the same lines.
* `pump_polarity` was converted with int(), so 1.5 ran as +1 and -1.2
  as -1 without a word, in `simulate_pll`, `simulate_sampled`,
  `sampled_margins`, `sampled_open_loop` and `margin_uncertainty`.
* `simulate_sampled(order=2.5)` ran as order 2.
* `simulate_pll` with n_cycles of 0 or less returned an empty run, and
  `simulate_sampled` returned empty arrays for 0 and raised NumPy's
  "negative dimensions" error below."""
import numpy as np
import pytest

from fracpll import (LinearVCO, margin_uncertainty, mash_line_spectrum,
                     mash_sequence, sampled_loop_map, sampled_margins,
                     sampled_open_loop, sampled_stability, simulate_pll,
                     simulate_sampled)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("shift", [57, 58, 70])
def test_line_spectrum_of_a_modulus_beyond_int64(m, shift):
    small = mash_line_spectrum(3, 64, m, 50e6)
    big = mash_line_spectrum(3 << shift, 64 << shift, m, 50e6)
    assert np.array_equal(mash_sequence(3 << shift, 64 << shift, m, 300),
                          mash_sequence(3, 64, m, 300))
    assert big["period"] == small["period"]
    # the same integer phase / den, so the same floats
    assert np.array_equal(big["line_rad2"], small["line_rad2"])
    assert big["variance_rad2"] == small["variance_rad2"]


def test_moduli_from_2_44_take_the_python_integer_path():
    """2^44 <= den < 2^63 used to work in int64 (wrapping products that
    cancel in the sum); the Python-integer path gives the same lines."""
    for shift in (39, 50):
        big = mash_line_spectrum(5 << shift, 32 << shift, 3, 50e6)
        assert np.array_equal(big["line_rad2"],
                              mash_line_spectrum(5, 32, 3, 50e6)["line_rad2"])


SLOW = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=50e6,
            c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)])


@pytest.mark.parametrize("pol", [1.5, -1.2, 0.5, 2, "x", None])
def test_pump_polarity_must_be_plus_or_minus_one(pol):
    vco = LinearVCO(1.995e9, 20e6, "design values for this test")
    calls = [
        lambda: simulate_pll(vco, 40, 50e6, 100e-6, 100e-12,
                             [(4.7e3, 1.5e-9)], n_cycles=10,
                             pump_polarity=pol),
        lambda: simulate_sampled(**SLOW, n_cycles=10, pump_polarity=pol),
        lambda: sampled_margins(**SLOW, pump_polarity=pol),
        lambda: sampled_open_loop(1e4, **SLOW, pump_polarity=pol),
        lambda: sampled_loop_map(**SLOW, pump_polarity=pol),
        lambda: sampled_stability(**SLOW, pump_polarity=pol),
        lambda: margin_uncertainty(**SLOW, u_icp=0.01, model="sampled",
                                   pump_polarity=pol),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="pump_polarity must be"):
            call()


def test_whole_number_polarities_are_still_accepted():
    a = sampled_margins(**SLOW)
    for pol in (1.0, np.int64(1)):
        assert sampled_margins(**SLOW, pump_polarity=pol) == a


def test_order_and_cycle_count_checks():
    kw = dict(SLOW, n_div=40 + 1 / 8, n_int=40, num=1, den=8,
              mash_order=2)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        simulate_sampled(**kw, n_cycles=100, order=2.5)
    assert simulate_sampled(**kw, n_cycles=100, order=2.0)["psi"].size \
        == 100
    for n in (0, -3):
        with pytest.raises(ValueError, match="n_cycles must be >= 1"):
            simulate_sampled(**SLOW, n_cycles=n)
        with pytest.raises(ValueError, match="n_cycles must be >= 1"):
            simulate_pll(LinearVCO(1.995e9, 20e6, "design values here"),
                         40, 50e6, 100e-6, 100e-12, [(4.7e3, 1.5e-9)],
                         n_cycles=n)
