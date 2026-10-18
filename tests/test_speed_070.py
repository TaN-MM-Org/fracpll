"""0.7.0 speedup of `sampled_margins`, held against the 0.6.1 code kept
here as the reference.

0.6.1 rebuilt the one-period loop map (an eigen-decomposition of the
filter and a matrix exponential) for every evaluation of the loop gain
G inside the root finding (about ten per call), and once more for the
final stability check.  0.7.0 builds it once.  Each G value is still
computed by the same operations on the same array shapes, so the
results are expected to be identical; the test nevertheless allows
rounding (1e-12 relative for frequencies and the spectral radius,
1e-9 degrees or dB for the margins), as the release rules ask of any
floating-point comparison between two code paths."""
import numpy as np
import pytest
from scipy.optimize import brentq

from fracpll import (design_loop_filter, margin_uncertainty,
                     sampled_loop_map, sampled_margins, sampled_stability)
from fracpll._network import FilterModes


# --- 0.6.1 reference implementation ----------------------------------
def ref_sampled_open_loop(f_hz, icp, kvco_hz_per_v, n_div, f_ref_hz,
                          c_shunt, branches=(), pump_polarity=1):
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    f_ref = float(f_ref_hz)
    m = sampled_loop_map(icp, kvco_hz_per_v, n_div, f_ref, c_shunt,
                         branches, pump_polarity)
    Phi = m["Phi"]
    n1 = Phi.shape[0]
    b = FilterModes(c_shunt, branches).state_matrices()[1]
    PB = Phi @ np.r_[b, 0.0]
    z = np.exp(2j * np.pi * f / f_ref)
    z[f == 0.5 * f_ref] = -1.0
    a = z[:, None, None] * np.eye(n1)[None, :, :] - Phi[None, :, :]
    rhs = np.broadcast_to(PB.astype(complex), (f.size, n1))[:, :, None]
    x = np.linalg.solve(a, rhs)[:, :, 0]
    k = int(pump_polarity) * float(icp) / (float(n_div) * f_ref)
    return k * x[:, -1]


def ref_sampled_margins(icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt,
                        branches=(), pump_polarity=1, f_lo=None,
                        n_grid=4001):
    f_ref = float(f_ref_hz)
    f_hi = 0.5 * f_ref
    f_lo = f_ref * 1e-6 if f_lo is None else float(f_lo)
    args = (icp, kvco_hz_per_v, n_div, f_ref, c_shunt, branches,
            pump_polarity)
    f = np.geomspace(f_lo, f_hi, int(n_grid))
    f[-1] = f_hi
    g = ref_sampled_open_loop(f, *args)
    mag = np.abs(g)

    def gain(x):
        return ref_sampled_open_loop(min(x, f_hi), *args)[0]

    above = mag > 1.0
    idx = np.flatnonzero(above[:-1] & ~above[1:])
    i = int(idx[-1])
    fc = brentq(lambda x: abs(gain(x)) - 1.0, f[i], f[i + 1],
                xtol=1e-13 * f[i], rtol=1e-15, maxiter=200)
    pm = 180.0 + float(np.degrees(np.angle(gain(fc))))
    if pm > 180.0:
        pm -= 360.0
    fs = np.concatenate(([fc], f[i + 1:]))
    gs = np.concatenate(([gain(fc)], g[i + 1:]))
    im = np.where(np.abs(gs.imag) <= 1e-9 * np.abs(gs), 0.0, gs.imag)
    f_pc = np.nan
    for j in range(fs.size - 1):
        if im[j] == 0.0 and gs.real[j] < 0.0:
            f_pc = float(fs[j])
            break
        if im[j] * im[j + 1] < 0.0:
            r = brentq(lambda x: gain(x).imag, fs[j], fs[j + 1],
                       xtol=1e-13 * fs[j], rtol=1e-15, maxiter=200)
            if gain(r).real < 0.0:
                f_pc = float(r)
                break
    if np.isnan(f_pc) and gs.real[-1] < 0.0:
        f_pc = f_hi
    gm_db = np.inf if np.isnan(f_pc) else \
        float(-20.0 * np.log10(abs(gain(f_pc))))
    st = sampled_stability(*args)
    return {"f_crossover_hz": float(fc), "phase_margin_deg": float(pm),
            "f_phase_crossover_hz": f_pc, "gain_margin_db": gm_db,
            "stable": st["stable"],
            "spectral_radius": st["spectral_radius"]}


def _same(new, old):
    assert new["stable"] == old["stable"]
    for key in ("f_crossover_hz", "spectral_radius"):
        assert new[key] == pytest.approx(old[key], rel=1e-12, abs=0.0)
    assert new["phase_margin_deg"] == pytest.approx(
        old["phase_margin_deg"], rel=0.0, abs=1e-9)
    if np.isfinite(old["gain_margin_db"]):
        assert new["gain_margin_db"] == pytest.approx(
            old["gain_margin_db"], rel=0.0, abs=1e-9)
        assert new["f_phase_crossover_hz"] == pytest.approx(
            old["f_phase_crossover_hz"], rel=1e-12, abs=0.0)
    else:
        assert new["gain_margin_db"] == np.inf
        assert np.isnan(new["f_phase_crossover_hz"])


FR = 50e6
CASES = [
    # README example 1 (slow) and examples 5/8 (fast)
    dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=FR,
         c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)]),
    dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=FR,
         c_shunt=2e-12, branches=[(20e3, 200e-12)]),
    # two branches, negative Kvco with a reversed pump
    dict(icp=100e-6, kvco_hz_per_v=-20e6, n_div=40.0, f_ref_hz=FR,
         c_shunt=400e-12, branches=[(4.7e3, 1.5e-9), (10e3, 3e-9)],
         pump_polarity=-1),
    # a capacitor-only filter: G real everywhere, gain margin 0 dB
    dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=FR,
         c_shunt=1e-9, branches=[]),
]


@pytest.mark.parametrize("loop", CASES)
def test_sampled_margins_equal_the_061_code(loop):
    _same(sampled_margins(**loop), ref_sampled_margins(**loop))


def test_sampled_margins_equal_the_061_code_on_random_loops():
    rng = np.random.default_rng(70)
    n_done = 0
    while n_done < 25:
        c1 = 10 ** rng.uniform(-12, -9)
        r = 10 ** rng.uniform(2.5, 4.5)
        c2 = c1 * 10 ** rng.uniform(0.3, 2)
        loop = dict(icp=10 ** rng.uniform(-5, -3),
                    kvco_hz_per_v=10 ** rng.uniform(6, 9),
                    n_div=rng.uniform(5, 200), f_ref_hz=FR, c_shunt=c1,
                    branches=[(r, c2)])
        try:
            old = ref_sampled_margins(**loop)
        except (ValueError, IndexError):
            continue                       # no crossover: not compared
        _same(sampled_margins(**loop), old)
        n_done += 1


def test_callers_unchanged():
    """`design_loop_filter` and `margin_uncertainty(model="sampled")`
    call `sampled_margins`; their per-cycle results equal what the
    0.6.1 code gives for the same loops."""
    d = design_loop_filter(40e3, 55.0, 100e-6, 20e6, 40.0, FR)
    old = ref_sampled_margins(100e-6, 20e6, 40.0, FR, d["c_shunt"],
                              d["branches"])
    for key in ("f_crossover_hz", "phase_margin_deg", "gain_margin_db",
                "f_phase_crossover_hz", "stable"):
        assert d["per_cycle"][key] == pytest.approx(old[key], rel=1e-12,
                                                    abs=1e-9)
    loop = CASES[1]
    r = margin_uncertainty(**loop, u_icp=1e-3, model="sampled")
    up = ref_sampled_margins(**dict(loop, icp=loop["icp"] * (1 + 1e-3)))
    dn = ref_sampled_margins(**dict(loop, icp=loop["icp"] * (1 - 1e-3)))
    for key in ("f_crossover_hz", "phase_margin_deg", "gain_margin_db"):
        want = 0.5 * (up[key] - dn[key])
        assert r["contributions"]["icp"][key] == pytest.approx(
            want, rel=1e-9, abs=1e-12)
