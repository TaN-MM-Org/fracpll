"""Tutorial: from measured tuning points to a designed loop filter
with error bars on its bandwidth and phase margin.

Steps
1. Fit a straight-line Kvco, with its standard uncertainty, to
   tuning points measured around the lock voltage.
2. Design a second-order loop filter for a 100 kHz loop bandwidth and
   a 55 degree phase margin (`design_loop_filter`).
3. Round the parts to two significant digits (a stand-in for the
   values you can buy) and check the rounded loop, averaged and per
   cycle.
4. Carry the uncertainties of the pump current, Kvco and the parts
   through to the bandwidth and the phase margin (`margin_uncertainty`,
   with a Monte Carlo check).

The "measured" tuning points are synthetic: a straight line plus
Gaussian reading noise of 50 kHz, drawn with a fixed seed, so the
script prints the same numbers every time.  Replace them with yours.

Printed output (fracpll 0.6.0):

    Kvco = 50.06 +- 0.13 MHz/V (chi-square test p = 0.47)
    designed: C1 = 399.8 pF, R2 = 1394 Ohm, C2 = 3.622 nF
    rounded:  C1 = 400 pF, R2 = 1400 Ohm, C2 = 3.6 nF
    rounded loop (averaged):  100.31 kHz, 54.90 degrees
    rounded loop (per cycle): 100.32 kHz, 54.90 degrees, gain margin 66.0 dB
    loop bandwidth 100.3 +- 1.8 kHz
    phase margin 54.9 +- 0.6 degrees
    largest share of the phase margin uncertainty: c_shunt
    Monte Carlo (2000 draws): phase margin 95 % in [53.6, 56.1] degrees
"""
import numpy as np

from fracpll import (design_loop_filter, fit_linear_vco, loop_filter_impedance,
                     margin_uncertainty, sampled_margins, stability)

F_REF = 25e6          # reference (Hz)
N_DIV = 100.0         # divide ratio: 2.5 GHz out
ICP = 1e-3            # pump current (A), calibrated to 2 %

# 1. Synthetic "measured" tuning points around the lock voltage.
rng = np.random.default_rng(2026)
vc = np.linspace(0.8, 1.2, 9)
sigma_f = 50e3                                   # counter reading, 1 sigma
f_meas = 2.45e9 + 50e6 * vc + rng.normal(0.0, sigma_f, vc.size)
fit = fit_linear_vco(vc, f_meas, f_sigma_hz=sigma_f,
                     reference="synthetic tuning points, fracpll tutorial")
kv, u_kv = fit["kvco_hz_per_v"], fit["u_kvco_hz_per_v"]
print(f"Kvco = {kv / 1e6:.2f} +- {u_kv / 1e6:.2f} MHz/V "
      f"(chi-square test p = {fit['p_value']:.2f})")

# 2. Design for 100 kHz and 55 degrees.
d = design_loop_filter(100e3, 55.0, ICP, kv, N_DIV, F_REF)
print(f"designed: C1 = {d['c1'] * 1e12:.1f} pF, R2 = {d['r2']:.0f} Ohm, "
      f"C2 = {d['c2'] * 1e9:.3f} nF")


# 3. Round to two significant digits and check the loop you would build.
def two_digits(x):
    return float(f"{x:.2g}")


c1, r2, c2 = two_digits(d["c1"]), two_digits(d["r2"]), two_digits(d["c2"])
print(f"rounded:  C1 = {c1 * 1e12:.0f} pF, R2 = {r2:.0f} Ohm, "
      f"C2 = {c2 * 1e9:.1f} nF")
zf = lambda w: loop_filter_impedance(w, c1, [(r2, c2)])
avg = stability(ICP, kv, N_DIV, zf, F_REF)
pc = sampled_margins(ICP, kv, N_DIV, F_REF, c1, [(r2, c2)])
print(f"rounded loop (averaged):  {avg['f_crossover_hz'] / 1e3:.2f} kHz, "
      f"{avg['phase_margin_deg']:.2f} degrees")
print(f"rounded loop (per cycle): {pc['f_crossover_hz'] / 1e3:.2f} kHz, "
      f"{pc['phase_margin_deg']:.2f} degrees, "
      f"gain margin {pc['gain_margin_db']:.1f} dB")

# 4. Uncertainties: 2 % pump calibration, Kvco from the fit, +-5 %
#    capacitors and a +-1 % resistor (rectangular: tolerance/sqrt(3)).
r = margin_uncertainty(icp=ICP, kvco_hz_per_v=kv, n_div=N_DIV,
                       f_ref_hz=F_REF, c_shunt=c1, branches=[(r2, c2)],
                       u_icp=0.02, u_kvco=u_kv / kv,
                       u_c_shunt=0.05 / 3 ** 0.5,
                       u_branches=[(0.01 / 3 ** 0.5, 0.05 / 3 ** 0.5)],
                       n_mc=2000, seed=1)
nom, u = r["nominal"], r["u"]
print(f"loop bandwidth {nom['f_crossover_hz'] / 1e3:.1f} +- "
      f"{u['f_crossover_hz'] / 1e3:.1f} kHz")
print(f"phase margin {nom['phase_margin_deg']:.1f} +- "
      f"{u['phase_margin_deg']:.1f} degrees")
big = max(r["contributions"],
          key=lambda k: abs(r["contributions"][k]["phase_margin_deg"]))
print("largest share of the phase margin uncertainty:", big)
mc = r["mc"]["phase_margin_deg"]
print(f"Monte Carlo (2000 draws): phase margin 95 % in "
      f"[{mc['p2.5']:.1f}, {mc['p97.5']:.1f}] degrees")
