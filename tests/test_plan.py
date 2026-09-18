"""Measurement-planning anchors: the closed-form relative scatter of
the jitter estimate is held against a seeded Monte Carlo with
exponentially distributed periodogram bins (the actual one-shot
periodogram statistics) averaged n_avg times; and `averages_for_jitter`
is an exact two-sided inversion -- n meets the target, n-1 fails it."""
import numpy as np
import pytest

from fracpll import averages_for_jitter, jitter_relative_sigma


def _grid():
    f = np.geomspace(1e3, 1e7, 400)
    s = 1e-9 / f ** 2 + 1e-14          # 1/f^2 plus a floor
    return f, s


def test_closed_form_vs_seeded_monte_carlo():
    f, s = _grid()
    n_avg = 16
    pred = jitter_relative_sigma(f, s, n_avg)
    rng = np.random.default_rng(20260918)
    trials = 4000
    # each bin: mean of n_avg exponential draws with mean S_i
    draws = rng.exponential(1.0, size=(trials, n_avg, f.size))
    s_hat = s[None, :] * draws.mean(axis=1)
    # trapezoid-integrate each trial, propagate to J = sqrt(var)
    var = np.trapezoid(s_hat, f, axis=1)
    j = np.sqrt(var)
    emp = j.std(ddof=1) / j.mean()
    # exponential bins have unit relative variance -> matches the
    # e_i/sqrt(n) model; MC error ~ 1/sqrt(2*trials) ~ 1.1 %
    assert abs(emp / pred - 1.0) < 0.06


def test_inversion_is_two_sided():
    f, s = _grid()
    tgt = 0.01
    n, achieved = averages_for_jitter(tgt, f, s)
    assert achieved <= tgt
    if n > 1:
        assert jitter_relative_sigma(f, s, n - 1) > tgt
    # 1/sqrt(n) law: quadrupling n halves the error bar
    assert np.isclose(jitter_relative_sigma(f, s, 4),
                      0.5 * jitter_relative_sigma(f, s, 1))


def test_refusals():
    f, s = _grid()
    with pytest.raises(ValueError, match=">= 1"):
        jitter_relative_sigma(f, s, 0)
    with pytest.raises(ValueError, match="increasing"):
        jitter_relative_sigma(f[::-1], s, 4)
    with pytest.raises(ValueError, match=">= 0"):
        jitter_relative_sigma(f, -s, 4)
    with pytest.raises(ValueError, match="zero"):
        jitter_relative_sigma(f, np.zeros_like(f), 4)
    with pytest.raises(ValueError, match="positive"):
        averages_for_jitter(-0.1, f, s)


def test_metadata():
    import fracpll
    assert fracpll.__version__ == "0.1.0"
    assert len(fracpll.__all__) == len(set(fracpll.__all__))
    for name in fracpll.__all__:
        assert hasattr(fracpll, name)
