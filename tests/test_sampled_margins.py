"""Exact per-cycle loop gain and margins (0.4.0).

`sampled_open_loop` computes G(z) from the one-period matrices; these
tests hold it, and the margins read from it, against routes that do
not use it:

* the Poisson (aliasing) sum of the AVERAGED loop gain,
  sum_m L(j(w + m w_ref)), with its 1/w^2 tail summed in closed form;
* the closed form of a filter with no resistor (a pure capacitor);
* the eigenvalues of the one-period map, rotated by the phase margin or
  scaled by the gain margin, which must land on the unit circle at the
  crossover frequencies;
* the pump-current stability limit found by root-finding on the
  spectral radius;
* the per-cycle simulation driven by a sinusoid and by exact MASH-1
  divider tones (time domain vs frequency domain)."""
import numpy as np
import pytest
from scipy.optimize import brentq

from fracpll import (closed_loop_lines, loop_filter_impedance,
                     mash_line_spectrum, sampled_loop_map, sampled_margins,
                     sampled_open_loop, sampled_stability, simulate_sampled,
                     stability)
from fracpll._network import FilterModes

FR = 50e6
SLOW = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=FR,
            c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)])
# README example 5: the averaged model refuses it (crossover > f_ref/10)
FAST = dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=FR,
            c_shunt=2e-12, branches=[(20e3, 200e-12)])


def _aliased_average(f, icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                     branches, M=20000):
    """sum_m L(j(w + m w_ref)), L the averaged gain Icp Kv Z/(j w N).

    The terms fall as 1/w^2; the asymptote -Icp Kv/(N C0 w^2) is
    summed in closed form, sum_m 1/(x + m)^2 = pi^2 / sin^2(pi x), and
    only the remainder (falling as 1/w^3) is summed directly."""
    ws = 2 * np.pi * f_ref_hz
    w = 2 * np.pi * f
    wm = w + np.arange(-M, M + 1) * ws
    z = loop_filter_impedance(np.abs(wm), c_shunt, branches)
    z = np.where(wm < 0, np.conj(z), z)           # Z(-jw) = conj Z(jw)
    lv = icp * kvco_hz_per_v * z / (1j * wm * n_div)
    k = icp * kvco_hz_per_v / (n_div * c_shunt)
    rest = np.sum(lv - (-k / wm ** 2))
    tail = -k / (4 * f_ref_hz ** 2 * np.sin(w / (2 * f_ref_hz)) ** 2)
    return rest + tail


@pytest.mark.parametrize("loop", [SLOW, FAST])
def test_loop_gain_equals_aliased_averaged_gain(loop):
    for f in (1e4, 4e4, 1e6, 1e7, 2.2e7, FR / 2):
        g = sampled_open_loop(f, **loop)[0]
        ref = _aliased_average(f, **loop)
        assert abs(g / ref - 1.0) < 1e-10
    assert sampled_open_loop(FR / 2, **loop)[0].imag == 0.0


def test_slow_loop_gain_and_margins_reduce_to_the_averaged_model():
    from fracpll import open_loop
    zf = lambda w: loop_filter_impedance(w, SLOW["c_shunt"],
                                         SLOW["branches"])
    lg = open_loop(2 * np.pi * 1e4, 100e-6, 20e6, 40.0, zf)[0]
    assert abs(sampled_open_loop(1e4, **SLOW)[0] / lg - 1.0) < 1e-5
    avg = stability(100e-6, 20e6, 40.0, zf, f_ref_hz=FR)
    m = sampled_margins(**SLOW)
    assert abs(m["f_crossover_hz"] / avg["f_crossover_hz"] - 1.0) < 1e-4
    assert abs(m["phase_margin_deg"] - avg["phase_margin_deg"]) < 0.01
    assert m["stable"]
    # negative Kvco with a reversed pump is the same loop
    neg = dict(SLOW, kvco_hz_per_v=-20e6, pump_polarity=-1)
    m2 = sampled_margins(**neg)
    for key in ("f_crossover_hz", "phase_margin_deg", "gain_margin_db"):
        assert np.isclose(m2[key], m[key], rtol=1e-12)


def test_resistorless_filter_closed_form():
    """C alone: G(e^{j t}) = -Icp Kv T^2 / (4 N C sin^2(t/2)), real and
    negative, so the phase margin is exactly 0, the crossover is
    t_c = 2 asin((T/2) sqrt(Icp Kv/(N C))), and the closed-loop poles
    sit on the unit circle at exp(+-j t_c)."""
    icp, kv, n, c = 100e-6, 20e6, 40.0, 100e-12
    T = 1.0 / FR
    f = np.array([1e3, 1e5, 3e6, 2e7])
    g = sampled_open_loop(f, icp, kv, n, FR, c)
    closed = -icp * kv * T ** 2 / (4 * n * c * np.sin(np.pi * f * T) ** 2)
    assert np.allclose(g, closed, rtol=1e-9, atol=0)
    m = sampled_margins(icp, kv, n, FR, c)
    assert abs(m["phase_margin_deg"]) < 1e-9
    fc = FR / np.pi * np.arcsin(0.5 * T * np.sqrt(icp * kv / (n * c)))
    assert abs(m["f_crossover_hz"] / fc - 1.0) < 1e-10
    z = sampled_stability(icp, kv, n, FR, c)["poles"]
    zc = np.exp(2j * np.pi * fc / FR)
    assert np.min(np.abs(z - zc)) < 1e-9
    assert np.min(np.abs(z - np.conj(zc))) < 1e-9
    # G is real and negative everywhere, so the phase crossover is the
    # crossover itself and the gain margin is 0 dB (the loop is
    # marginal: its poles are on the circle at fc, just checked) --
    # whatever the scan grid.  Before the fix the scan started at the
    # grid point after fc and reported a grid-dependent margin > 0 dB.
    for n_grid in (41, 401, 4001):
        mg = sampled_margins(icp, kv, n, FR, c, n_grid=n_grid)
        assert mg["f_phase_crossover_hz"] == mg["f_crossover_hz"]
        assert abs(mg["f_crossover_hz"] / fc - 1.0) < 1e-10
        assert abs(mg["gain_margin_db"]) < 1e-9


def _map_eigs(loop, factor):
    """Eigenvalues of Phi (I - factor * k B e^T): the one-period map with
    the loop gain multiplied by a (possibly complex) factor."""
    lm = sampled_loop_map(**loop)
    Phi = lm["Phi"]
    n1 = Phi.shape[0]
    b = FilterModes(loop["c_shunt"], loop["branches"]).state_matrices()[1]
    k = loop.get("pump_polarity", 1) * loop["icp"] \
        / (loop["n_div"] * loop["f_ref_hz"])
    kick = np.eye(n1, dtype=complex)
    kick[:n1 - 1, n1 - 1] -= factor * k * b
    return np.linalg.eigvals(Phi @ kick)


@pytest.mark.parametrize("loop", [SLOW, FAST])
def test_margins_match_eigenvalue_routes(loop):
    """1 + G(z) = 0 at the poles.  Rotating G by exp(-j PM) makes the
    loop marginal at the gain crossover, and scaling it by the gain
    margin makes it marginal at the phase crossover -- so an eigenvalue
    of the modified map must sit exactly on the unit circle there."""
    m = sampled_margins(**loop)
    rot = np.exp(-1j * np.radians(m["phase_margin_deg"]))
    z = _map_eigs(loop, rot)
    assert np.min(np.abs(z - np.exp(2j * np.pi * m["f_crossover_hz"]
                                    / FR))) < 1e-8
    gm = 10 ** (m["gain_margin_db"] / 20)
    z = _map_eigs(loop, gm)
    assert np.min(np.abs(z - np.exp(2j * np.pi
                                    * m["f_phase_crossover_hz"] / FR))) \
        < 1e-8
    # the unmodified map is inside the circle, as the flag says
    assert m["stable"] and np.max(np.abs(_map_eigs(loop, 1.0))) < 1.0


def test_gain_margin_is_the_pump_current_stability_limit():
    """In the fast loop, G(z=-1) sets the gain margin; G is proportional
    to Icp, so Icp x 10^(GM/20) must be the current at which the
    spectral radius of the one-period map reaches 1."""
    m = sampled_margins(**FAST)
    assert m["f_phase_crossover_hz"] == FR / 2
    assert 0.0 < m["gain_margin_db"] < 1.0        # stable, but only just
    kw = {k: v for k, v in FAST.items() if k != "icp"}
    rho = lambda i: sampled_stability(i, **kw)["spectral_radius"] - 1.0
    icrit = brentq(rho, 3e-4, 1e-3, xtol=1e-16, rtol=1e-14)
    assert abs(FAST["icp"] * 10 ** (m["gain_margin_db"] / 20)
               / icrit - 1.0) < 1e-9


def test_closed_loop_transfer_in_the_time_domain():
    """A sinusoidal divider phase through the per-cycle simulation:
    after the start-up has decayed, psi_k = Re(A T e^{j t k}) with
    T = G/(1+G) from `sampled_open_loop`."""
    loop = dict(FAST, icp=0.25e-3)
    # slowest pole 0.995 (the branch R C = 200 reference periods): after
    # 12000 cycles the start-up has decayed by 0.995^12000 ~ e^-60
    assert sampled_stability(**loop)["spectral_radius"] < 0.9951
    K, keep = 16000, 4000
    for f in (1e5, 5e6, 1.9e7):
        t = 2 * np.pi * f / FR
        u = 1e-3 * np.cos(t * np.arange(K))
        psi = simulate_sampled(u_cycles=u, order=1, **loop)["psi"]
        g = sampled_open_loop(f, **loop)[0]
        tr = g / (1 + g)
        pred = np.real(1e-3 * tr * np.exp(1j * t * np.arange(K)))
        err = np.max(np.abs(psi[-keep:] - pred[-keep:]))
        assert err < 1e-9 * 1e-3 * max(abs(tr), 1e-3)


def test_fast_loop_spurs_through_the_exact_loop_gain():
    """Exact MASH-1 tones through G/(1+G) (the averaged model refuses
    this loop) against the exact DFT of the per-cycle simulation over
    whole periods of the divider pattern."""
    loop = dict(FAST, icp=0.25e-3)
    P, reps, skip = 64, 40, 10240
    n_mean = 10 + 3 / P
    r = simulate_sampled(loop["icp"], loop["kvco_hz_per_v"], n_mean, FR,
                         loop["c_shunt"], loop["branches"], n_int=10,
                         num=3, den=P, mash_order=1,
                         n_cycles=skip + P * reps, order=1)
    x = 2 * np.pi * r["psi"][skip:]
    c = np.fft.fft(x) / x.size
    lines = mash_line_spectrum(3, P, 1, FR)
    kw = dict(loop, n_div=n_mean)
    g = sampled_open_loop(lines["f_hz"], **kw)
    out = closed_loop_lines(lines["f_hz"], lines["line_rad2"], g)
    k = np.arange(1, P // 2 + 1)
    sim = 2 * np.abs(c[reps * k]) ** 2
    sim[-1] = np.abs(c[reps * (P // 2)]) ** 2          # the Nyquist line
    assert np.allclose(sim, out["line_rad2"], rtol=1e-6,
                       atol=1e-12 * out["line_rad2"].max())
    with pytest.raises(ValueError, match="Gardner"):
        zf = lambda w: loop_filter_impedance(w, loop["c_shunt"],
                                             loop["branches"])
        stability(loop["icp"], loop["kvco_hz_per_v"], n_mean, zf,
                  f_ref_hz=FR)


def test_refusals():
    with pytest.raises(ValueError, match="Nyquist"):
        sampled_open_loop(0.6 * FR, **SLOW)
    with pytest.raises(ValueError, match="Nyquist"):
        sampled_open_loop(0.0, **SLOW)
    with pytest.raises(ValueError, match="f_ref/2"):
        sampled_margins(**dict(FAST, icp=2e-3))
    # |G(-1)| >= 1 is refused as unstable: 5 % above the limit current,
    # the eigenvalues agree
    icrit = FAST["icp"] * 10 ** (sampled_margins(**FAST)["gain_margin_db"]
                                 / 20)
    over = dict(FAST, icp=1.05 * icrit)
    assert sampled_open_loop(FR / 2, **over)[0].real < -1.0
    assert not sampled_stability(**over)["stable"]
    with pytest.raises(ValueError, match="unstable"):
        sampled_margins(**over)
    with pytest.raises(ValueError, match="no crossover"):
        sampled_margins(**dict(SLOW, icp=1e-15))
    with pytest.raises(ValueError, match="positive feedback"):
        sampled_margins(**dict(SLOW, kvco_hz_per_v=-20e6))
