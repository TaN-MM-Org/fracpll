"""The third-order filter (an R3-C3 section after the second-order
filter) in the per-cycle model and the edge-by-edge simulator (0.8.0).

The section adds one more series branch to the filter network, and the
oscillator reads the voltage on C3 instead of the pump node.  These
tests hold the new results against routes that do not share that code:

* the per-cycle loop gain G against the Poisson (aliasing) sum of the
  AVERAGED loop gain built on `third_order_impedance` (an exact
  identity; its 1/w^3 tail summed in closed form);
* the margins against eigenvalue routes (the loop gain rotated by the
  phase margin, or scaled by the gain margin, puts a pole of the
  one-period map on the unit circle), and the gain margin against the
  pump current at which the spectral radius reaches 1;
* at low bandwidth, the poles of the one-period map against exp(s T)
  of the roots of the third-order characteristic polynomial written
  out here from the polynomial form of the transimpedance;
* the edge-by-edge simulator (exact between edges, its own event
  code) against the per-cycle map after a small nudge, cycle by cycle,
  and on both sides of the stability limit;
* `simulate_sampled(order=2)` against the edge-by-edge simulator with
  a third-order MASH divider;
* the R3 -> 0 limit: C3 in parallel with C1;
* `margin_uncertainty` with the section: the exact scaling of the
  loop (every C times k, every R over k, Icp times k leaves L
  unchanged) gives a sum rule for the sensitivities, and the
  per-cycle and averaged propagations agree in a slow loop.

The observed differences are given next to each assertion."""
import numpy as np
import pytest
from scipy import signal
from scipy.optimize import brentq

from fracpll import (LinearVCO, margin_uncertainty, mash_sequence,
                     open_loop, sampled_loop_map, sampled_margins,
                     sampled_open_loop, sampled_stability, simulate_pll,
                     simulate_sampled, stability, third_order_impedance)

REF = "design value, fracpll test suite 2026-11-10"
FR = 50e6
T = 1.0 / FR
# README example 12: the filter of example 1 plus R3 = 10 kOhm, C3 = 10 pF
SLOW = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=FR,
            c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)], r3=10e3,
            c3=10e-12)
# README examples 5 and 8 (fast: the averaged model refuses it) plus a
# section R3 = 10 kOhm, C3 = 0.2 pF
FAST = dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=FR,
            c_shunt=2e-12, branches=[(20e3, 200e-12)], r3=10e3,
            c3=0.2e-12)


def _z3(loop):
    (r2, c2), = loop["branches"]
    return lambda w: third_order_impedance(w, loop["c_shunt"], r2, c2,
                                           loop["r3"], loop["c3"])


def _aliased_average(f, loop, M=20000):
    """sum_m L(j(w + m w_ref)) with L = Icp Kv Z3/(j w N), Z3 from
    `third_order_impedance`, summed directly over |m| <= M.  For large
    w, Z3 -> R2 C2 / (A2 s^2) with A2 = C1 C2 C3 R2 R3, so L falls as
    1/w^3, an odd function: the terms m and -m cancel to 1/m^4, and
    the terms left out add up to about 4 x K/(w_ref^3 M^3) with
    K = Icp Kv/(N C1 C3 R3), below 1e-13 of G for both loops here
    (the test also compares M with 2M)."""
    ws = 2 * np.pi * FR
    w = 2 * np.pi * f
    wm = w + np.arange(-M, M + 1) * ws
    z = _z3(loop)(np.abs(wm))
    z = np.where(wm < 0, np.conj(z), z)           # Z(-jw) = conj Z(jw)
    lv = loop["icp"] * loop["kvco_hz_per_v"] * z / (1j * wm * loop["n_div"])
    return np.sum(lv)


@pytest.mark.parametrize("loop", [SLOW, FAST])
def test_loop_gain_equals_aliased_averaged_gain(loop):
    # observed: below 1e-13
    for f in (1e4, 4e4, 1e6, 1e7, 2.2e7, FR / 2):
        g = sampled_open_loop(f, **loop)[0]
        ref = _aliased_average(f, loop)
        assert abs(_aliased_average(f, loop, 2 * 20000) / ref - 1) < 1e-12
        assert abs(g / ref - 1.0) < 1e-10
    assert sampled_open_loop(FR / 2, **loop)[0].imag == 0.0


def test_slow_loop_reduces_to_the_averaged_model():
    """Example 12's loop: the per-cycle margins equal what `stability`
    gives with `third_order_impedance` (observed: 8e-9 in bandwidth,
    5e-7 degrees), and so does the loop gain far below f_ref."""
    avg = stability(SLOW["icp"], SLOW["kvco_hz_per_v"], SLOW["n_div"],
                    _z3(SLOW), FR)
    m = sampled_margins(**SLOW)
    assert abs(m["f_crossover_hz"] / avg["f_crossover_hz"] - 1.0) < 1e-6
    assert abs(m["phase_margin_deg"] - avg["phase_margin_deg"]) < 1e-4
    assert m["stable"]
    lg = open_loop(2 * np.pi * 1e4, SLOW["icp"], SLOW["kvco_hz_per_v"],
                   SLOW["n_div"], _z3(SLOW))[0]
    assert abs(sampled_open_loop(1e4, **SLOW)[0] / lg - 1.0) < 1e-5
    # the section matters: without it the margin is 2 degrees higher
    two = {k: v for k, v in SLOW.items() if k not in ("r3", "c3")}
    assert sampled_margins(**two)["phase_margin_deg"] \
        > m["phase_margin_deg"] + 1.5
    # negative Kvco with a reversed pump is the same loop
    neg = dict(SLOW, kvco_hz_per_v=-20e6, pump_polarity=-1)
    m2 = sampled_margins(**neg)
    for key in ("f_crossover_hz", "phase_margin_deg", "gain_margin_db"):
        assert np.isclose(m2[key], m[key], rtol=1e-12)


def _map_eigs(loop, factor):
    """Eigenvalues of Phi (I - factor k B e_psi^T), with B = e_0 / C_shunt
    (the pump charges the pump node) written out here."""
    Phi = sampled_loop_map(**loop)["Phi"]
    n1 = Phi.shape[0]
    b = np.zeros(n1)
    b[0] = 1.0 / loop["c_shunt"]
    k = loop.get("pump_polarity", 1) * loop["icp"] \
        / (loop["n_div"] * loop["f_ref_hz"])
    kick = np.eye(n1, dtype=complex)
    kick[:, n1 - 1] -= factor * k * b
    return np.linalg.eigvals(Phi @ kick)


@pytest.mark.parametrize("loop", [SLOW, FAST])
def test_margins_match_eigenvalue_routes(loop):
    m = sampled_margins(**loop)
    rot = np.exp(-1j * np.radians(m["phase_margin_deg"]))
    z = _map_eigs(loop, rot)
    # observed: below 1e-13
    assert np.min(np.abs(z - np.exp(2j * np.pi * m["f_crossover_hz"]
                                    / FR))) < 1e-8
    gm = 10 ** (m["gain_margin_db"] / 20)
    z = _map_eigs(loop, gm)
    assert np.min(np.abs(z - np.exp(2j * np.pi
                                    * m["f_phase_crossover_hz"] / FR))) \
        < 1e-8
    assert m["stable"] and np.max(np.abs(_map_eigs(loop, 1.0))) < 1.0
    assert sampled_stability(**loop)["spectral_radius"] \
        == m["spectral_radius"]


def test_fast_loop_gain_margin_is_the_stability_limit():
    """The averaged model refuses this loop.  Its phase crossover is at
    f_ref/2, where G is proportional to Icp, so Icp x 10^(GM/20) is the
    current at which the spectral radius reaches 1 (observed: 4e-16)."""
    with pytest.raises(ValueError, match="Gardner"):
        stability(FAST["icp"], FAST["kvco_hz_per_v"], FAST["n_div"],
                  _z3(FAST), FR)
    m = sampled_margins(**FAST)
    assert m["f_phase_crossover_hz"] == FR / 2
    kw = {k: v for k, v in FAST.items() if k != "icp"}
    rho = lambda i: sampled_stability(i, **kw)["spectral_radius"] - 1.0
    icrit = brentq(rho, 0.5e-3, 2e-3, xtol=1e-16, rtol=1e-14)
    assert abs(FAST["icp"] * 10 ** (m["gain_margin_db"] / 20)
               / icrit - 1.0) < 1e-9


def test_poles_reduce_to_the_third_order_characteristic_polynomial():
    """1 + Icp Kv Z3/(s N) = 0 with Z3 = (1 + s R2 C2)/(s (A0 + A1 s +
    A2 s^2)) is s^2 N (A0 + A1 s + A2 s^2) + Icp Kv (1 + s R2 C2) = 0.
    Its roots s give z = exp(s T) (observed: 1e-8 of each pole's
    distance from 1)."""
    c1, (r2, c2), r3, c3 = (SLOW["c_shunt"], SLOW["branches"][0],
                            SLOW["r3"], SLOW["c3"])
    a0 = c1 + c2 + c3
    a1 = r2 * c2 * (c1 + c3) + r3 * c3 * (c1 + c2)
    a2 = c1 * c2 * c3 * r2 * r3
    n, g = SLOW["n_div"], SLOW["icp"] * SLOW["kvco_hz_per_v"]
    char = np.polyadd(n * np.array([a2, a1, a0, 0.0, 0.0]),
                      g * np.array([r2 * c2, 1.0]))
    zc = np.sort_complex(np.exp(np.roots(char) * T))
    z = np.sort_complex(sampled_stability(**SLOW)["poles"])
    assert z.size == zc.size == 4
    assert np.all(np.abs(z - zc) < 1e-4 * np.abs(1.0 - zc))
    # `continuous_closed_loop_poles` builds its polynomial from the
    # admittances of the branches instead (observed: equal to rounding),
    # also for example 12's refused loop, which has a pair of poles in
    # the right half plane
    from fracpll import continuous_closed_loop_poles
    for r3_, c3_ in ((r3, c3), (20e3, 1e-9)):
        b0 = c1 + c2 + c3_
        b1 = r2 * c2 * (c1 + c3_) + r3_ * c3_ * (c1 + c2)
        b2 = c1 * c2 * c3_ * r2 * r3_
        ref = np.sort_complex(np.roots(np.polyadd(
            n * np.array([b2, b1, b0, 0.0, 0.0]),
            g * np.array([r2 * c2, 1.0]))))
        got = np.sort_complex(continuous_closed_loop_poles(
            SLOW["icp"], SLOW["kvco_hz_per_v"], n, c1, [(r2, c2)],
            r3=r3_, c3=c3_))
        assert got.size == 4
        assert np.all(np.abs(got - ref) <= 1e-10 * np.abs(ref))
    assert np.sum(got.real > 0) == 2


@pytest.mark.parametrize("loop, p0, tol", [
    # example 4's loop plus R3 = 10 kOhm, C3 = 5 pF (observed 2.7e-8)
    (dict(icp=200e-6, kvco_hz_per_v=400e6, n_div=100, c_shunt=50e-12,
          branches=[(4.7e3, 1.5e-9)], r3=10e3, c3=5e-12), 1e-4, 1e-6),
    # FAST with half the pump current (observed 9.2e-7, and 9.2e-6 for
    # a nudge of 1e-4: the second-order pulse-width terms, which grow
    # with the nudge)
    (dict(FAST, icp=0.25e-3), 1e-5, 1e-5)])
def test_edge_simulator_follows_the_third_order_map(loop, p0, tol):
    loop = {k: v for k, v in loop.items() if k != "f_ref_hz"}
    kv, n = loop["kvco_hz_per_v"], loop["n_div"]
    vco = LinearVCO(n * FR - 0.5 * kv, kv, REF)
    r = simulate_pll(vco, n, FR, loop["icp"], loop["c_shunt"],
                     loop["branches"], n_cycles=400, phase0_cycles=p0,
                     r3=loop["r3"], c3=loop["c3"])
    F = sampled_loop_map(f_ref_hz=FR, **loop)["F"]
    s = np.zeros(F.shape[0])
    s[-1] = p0
    pred = []
    for _ in range(400):
        pred.append(s[-1])
        s = F @ s
    pred = np.array(pred)
    err = np.max(np.abs(r.psi_ref - pred)) / np.max(np.abs(pred))
    assert err < tol
    # the second-order map (no section) is visibly different
    F2 = sampled_loop_map(loop["icp"], kv, n, FR, loop["c_shunt"],
                          loop["branches"])["F"]
    s = np.zeros(F2.shape[0])
    s[-1] = p0
    pred2 = []
    for _ in range(400):
        pred2.append(s[-1])
        s = F2 @ s
    assert np.max(np.abs(r.psi_ref - np.array(pred2))) \
        > 1e3 * tol * np.max(np.abs(pred))


def test_edge_simulator_on_both_sides_of_the_stability_limit():
    kw = {k: v for k, v in FAST.items() if k != "icp"}
    icrit = brentq(lambda i: sampled_stability(i, **kw)["spectral_radius"]
                   - 1.0, 0.5e-3, 2e-3, xtol=1e-16)
    vco = LinearVCO(4.0e8, FAST["kvco_hz_per_v"], REF)
    runs = [simulate_pll(vco, 10, FR, f * icrit, FAST["c_shunt"],
                         FAST["branches"], n_cycles=300,
                         phase0_cycles=1e-4, r3=FAST["r3"], c3=FAST["c3"])
            for f in (0.9, 1.1)]
    assert np.abs(runs[0].psi_ref[-50:]).max() < 1e-2 * 1e-4
    assert np.abs(runs[1].psi_ref[-50:]).max() > 1e2 * 1e-4


NUM, DEN = 104857, 1 << 20
K = (1 << 16) + 2048
SL = slice(8192, None)


def _bands(x, edges=((1e4, 2e4), (2e4, 5e4))):
    f, p = signal.welch(2 * np.pi * x, fs=FR, nperseg=1 << 14,
                        detrend="linear")
    return np.array([10 * np.log10(np.mean(p[(f >= a) & (f < b)]))
                     for a, b in edges])


@pytest.mark.parametrize("r3, c3", [(10e3, 10e-12), (4.7e3, 33e-12)])
def test_order2_reproduces_the_edge_level_loop_with_the_section(r3, c3):
    """Example 12's loop (and a second section) with a MASH-3 divider.
    With the section the oscillator reads C3, whose voltage a pulse into
    the pump node does not move during the pulse, so the in-pulse ramp
    term of the second-order filter is absent.  Observed: residual 5e-4
    and 4e-4 of the linear model's (1e-2 if the ramp term is wrongly
    kept), pulse widths to 2e-17 s, band levels within 3e-4 dB."""
    loop = {k: v for k, v in SLOW.items() if k not in ("f_ref_hz",
                                                       "n_div")}
    loop.update(r3=r3, c3=c3)
    kv = loop["kvco_hz_per_v"]
    vco = LinearVCO(40 * FR - 0.5 * kv + 0.1 * FR, kv, REF)
    edge = simulate_pll(vco, 40, FR, loop["icp"], loop["c_shunt"],
                        loop["branches"], n_cycles=K, num=NUM, den=DEN,
                        mash_order=3, r3=loop["r3"], c3=loop["c3"])
    kw = dict(loop, n_div=40 + NUM / DEN, f_ref_hz=FR, n_int=40, num=NUM,
              den=DEN, mash_order=3, n_cycles=K)
    o1 = simulate_sampled(order=1, **kw)
    o2 = simulate_sampled(order=2, **kw)
    e1 = (edge.psi_ref - o1["psi"])[SL]
    e2 = (edge.psi_ref - o2["psi"])[SL]
    assert e2.std() < 1e-3 * e1.std()
    assert np.abs(edge.time_error() - o2["tau"])[SL].max() < 1e-15
    b_edge = _bands(edge.psi_ref[SL])
    assert np.all(np.abs(_bands(o2["psi"][SL]) - b_edge) < 0.5)
    assert np.any(b_edge - _bands(o1["psi"][SL]) > 3.0)


def test_order1_with_the_section_is_the_linear_map():
    nd = 40 + NUM / DEN
    loop = dict(SLOW, n_div=nd)
    o1 = simulate_sampled(n_int=40, num=NUM, den=DEN, mash_order=3,
                          n_cycles=4000, order=1, **loop)
    u = np.cumsum(mash_sequence(NUM, DEN, 3, 4004) - NUM / DEN)[:4000]
    Phi = sampled_loop_map(**loop)["Phi"]
    B = np.zeros(Phi.shape[0])
    B[0] = 1.0 / loop["c_shunt"]
    s = np.zeros(Phi.shape[0])
    ref = np.empty(4000)
    for k in range(4000):
        s = Phi @ s
        ref[k] = s[-1]
        s = s + B * (-loop["icp"] * (s[-1] - u[k]) / (nd * FR))
    assert np.allclose(o1["psi"], ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_r3_to_zero_is_c3_in_parallel_with_c1():
    """R3 C3 = 1e-13 s, 5e-6 of a reference period: the margins move
    from those of the filter with C1 + C3 to ground by about that much
    (observed: 5e-10 in bandwidth, 1.4e-6 degrees, 8e-5 dB of gain
    margin)."""
    small = dict(SLOW, r3=1e-2)
    lumped = {k: v for k, v in SLOW.items() if k not in ("r3", "c3")}
    lumped["c_shunt"] = SLOW["c_shunt"] + SLOW["c3"]
    a, b = sampled_margins(**small), sampled_margins(**lumped)
    assert abs(a["f_crossover_hz"] / b["f_crossover_hz"] - 1) < 1e-6
    assert abs(a["phase_margin_deg"] - b["phase_margin_deg"]) < 1e-3
    assert abs(a["gain_margin_db"] - b["gain_margin_db"]) < 1e-3


def test_margin_uncertainty_with_the_section():
    u = dict(u_icp=0.02, u_kvco=0.01, u_c_shunt=0.03,
             u_branches=[(0.01, 0.03)], u_r3=0.01, u_c3=0.03)
    avg = margin_uncertainty(**SLOW, **u)
    pc = margin_uncertainty(**SLOW, **u, model="sampled")
    # the nominal values are those of `stability` with the section
    ref = stability(SLOW["icp"], SLOW["kvco_hz_per_v"], SLOW["n_div"],
                    _z3(SLOW), FR)
    assert avg["nominal"] == ref
    # slow loop: per-cycle and averaged propagation agree (observed 3e-8)
    for o in ("f_crossover_hz", "phase_margin_deg"):
        assert abs(pc["u"][o] / avg["u"][o] - 1) < 2e-3
    assert set(pc["contributions"]) == {"icp", "kvco", "c_shunt",
                                        "branch0_r", "branch0_c", "r3",
                                        "c3"}
    # scaling every C by k, every R by 1/k and Icp by k leaves L, so
    # every output, unchanged: the sensitivities (dy/d ln x) sum to
    # zero with signs + for Icp and the capacitors, - for the resistors.
    # The central differences hold that to second order in the step,
    # so the step is small here (observed: 4e-9 of the largest term)
    e = 1e-4
    small = dict(u_icp=e, u_kvco=e, u_c_shunt=e, u_branches=[(e, e)],
                 u_r3=e, u_c3=e)
    for res in (margin_uncertainty(**SLOW, **small),
                margin_uncertainty(**SLOW, **small, model="sampled")):
        s = res["sensitivity"]
        for o in res["u"]:
            total = (s["icp"][o] + s["c_shunt"][o] + s["branch0_c"][o]
                     + s["c3"][o] - s["branch0_r"][o] - s["r3"][o])
            big = max(abs(s[n][o]) for n in s)
            assert abs(total) < 1e-6 * big


def test_refusals():
    two = {k: v for k, v in SLOW.items() if k not in ("r3", "c3")}
    vco = LinearVCO(1.995e9, 20e6, REF)
    sim = lambda **kw: simulate_pll(vco, 40, FR, 100e-6, 100e-12,
                                    [(4.7e3, 1.5e-9)], n_cycles=10, **kw)
    for call in (sampled_margins, sampled_stability, sampled_loop_map):
        with pytest.raises(ValueError, match="both r3 and c3"):
            call(**two, r3=10e3)
        with pytest.raises(ValueError, match="add it to c_shunt"):
            call(**two, r3=0.0, c3=10e-12)
        with pytest.raises(ValueError, match="c3 must be"):
            call(**two, r3=10e3, c3=-1e-12)
    with pytest.raises(ValueError, match="both r3 and c3"):
        sim(c3=10e-12)
    with pytest.raises(ValueError, match="r3 must be"):
        sim(r3=float("nan"), c3=10e-12)
    with pytest.raises(ValueError, match="both r3 and c3"):
        simulate_sampled(**two, n_cycles=10, r3=10e3)
    with pytest.raises(ValueError, match="need the R3-C3 section"):
        margin_uncertainty(**two, u_r3=0.01)
    with pytest.raises(ValueError, match="both r3 and c3"):
        margin_uncertainty(**two, r3=10e3, u_r3=0.01)
