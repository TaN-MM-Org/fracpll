"""Tutorial: reference spurs from measured leakage and pump currents,
with error bars, and an R3-C3 section to meet a spur limit.

Steps
1. Average repeated bench readings of the UP and DN pump currents and
   of the leakage current at the control node, each to a mean with its
   standard uncertainty (the standard deviation of the readings over
   the square root of their number).
2. Design a second-order loop filter for 100 kHz and 55 degrees
   (`design_loop_filter`) and round its parts to two digits.
3. Compute the reference spurs of the locked loop (`reference_spurs`)
   and their uncertainty by Monte Carlo (2000 draws).  Each current is
   drawn from the distribution that JCGM 101:2008 (Supplement 1 to the
   GUM, Table 1 and 6.4.9.2) assigns to the mean of n independent
   readings: a t-distribution with n - 1 degrees of freedom, centered
   on the mean and scaled by s/sqrt(n).
4. Add an R3-C3 section after the filter, check the spur against a
   limit of -90 dBc, and see what it costs: phase margin, closed-loop
   bandwidth and peaking (`closed_loop_response`).

The readings are synthetic: fixed true values plus Gaussian reading
noise, drawn with a fixed seed, so the script prints the same numbers
every time.  The reset delay is a design value, not a measurement.
Replace them with yours.

Printed output (fracpll 0.7.0):

    UP current   199.95 +- 0.10 uA (8 readings)
    DN current   195.98 +- 0.08 uA (8 readings)
    leakage      1.861 +- 0.039 uA (12 readings)
    filter: C1 = 640 pF, R2 = 870 Ohm, C2 = 5.8 nF
    locked timing offset: 0.180 ns (reference leads)
    second order: spur at 50 MHz -82.6 +- 0.2 dBc, at 100 MHz -94.7 +- 0.2 dBc
    with R3 = 4.7 kOhm, C3 = 6.8 pF: spur at 50 MHz -102.7 +- 0.2 dBc
    limit -90 dBc at 50 MHz: second order fails, third order passes
                  phase margin  -3 dB bandwidth  peaking
    second order  55.0 degrees  162.1 kHz        2.10 dB
    third order   53.7 degrees  165.1 kHz        2.17 dB
"""
import numpy as np

from fracpll import (closed_loop_response, design_loop_filter,
                     loop_filter_impedance, reference_spurs,
                     third_order_impedance)

F_REF, N_DIV = 50e6, 100.0          # 5 GHz out
KVCO = 400e6                        # Hz/V at the lock voltage
T_RESET = 300e-12                   # PFD reset delay (design value)
LIMIT_DBC = -90.0

# 1. Synthetic "measured" currents: true values plus reading noise.
rng = np.random.default_rng(7)
up = 200e-6 + rng.normal(0.0, 0.4e-6, 8)
dn = 196e-6 + rng.normal(0.0, 0.4e-6, 8)
leak = 2.0e-6 + rng.normal(0.0, 0.15e-6, 12)


def mean_u(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(x.size)


(icp, u_icp), (idn, u_idn), (ilk, u_ilk) = mean_u(up), mean_u(dn), mean_u(leak)
print(f"UP current   {icp * 1e6:.2f} +- {u_icp * 1e6:.2f} uA ({up.size} readings)")
print(f"DN current   {idn * 1e6:.2f} +- {u_idn * 1e6:.2f} uA ({dn.size} readings)")
print(f"leakage      {ilk * 1e6:.3f} +- {u_ilk * 1e6:.3f} uA "
      f"({leak.size} readings)")

# 2. The filter, rounded to two digits.
d = design_loop_filter(100e3, 55.0, icp, KVCO, N_DIV, F_REF)
c1, r2, c2 = (float(f"{x:.2g}") for x in (d["c1"], d["r2"], d["c2"]))
print(f"filter: C1 = {c1 * 1e12:.0f} pF, R2 = {r2:.0f} Ohm, "
      f"C2 = {c2 * 1e9:.1f} nF")

# 3. Spurs and their uncertainty from the measured currents.
R3, C3 = 4.7e3, 6.8e-12


def spurs(i_up, i_dn, i_leak, third=False):
    extra = dict(r3=R3, c3=C3) if third else {}
    return reference_spurs(F_REF, i_up, KVCO, N_DIV, c1, [(r2, c2)],
                           i_dn=i_dn, i_leak=i_leak, t_reset=T_RESET,
                           n_harmonics=2, **extra)


nom = spurs(icp, idn, ilk)
side = "reference leads" if nom["dt_s"] > 0 else "divider leads"
print(f"locked timing offset: {abs(nom['dt_s']) * 1e9:.3f} ns ({side})")
draws = np.column_stack([m + u * rng.standard_t(x.size - 1, 2000)
                         for (m, u), x in (((icp, u_icp), up),
                                           ((idn, u_idn), dn),
                                           ((ilk, u_ilk), leak))])
mc2 = np.array([spurs(*x)["sideband_dbc"] for x in draws])
mc3 = np.array([spurs(*x, third=True)["sideband_dbc"][0] for x in draws])
s2, u2 = mc2.mean(axis=0), mc2.std(axis=0, ddof=1)
s3, u3 = mc3.mean(), mc3.std(ddof=1)
print(f"second order: spur at 50 MHz {s2[0]:.1f} +- {u2[0]:.1f} dBc, "
      f"at 100 MHz {s2[1]:.1f} +- {u2[1]:.1f} dBc")
print(f"with R3 = {R3 / 1e3:.1f} kOhm, C3 = {C3 * 1e12:.1f} pF: "
      f"spur at 50 MHz {s3:.1f} +- {u3:.1f} dBc")

# 4. Against the limit (mean plus two standard uncertainties), and the
#    cost of the extra section in the loop itself.
verdict = ["fails" if s + 2 * u > LIMIT_DBC else "passes"
           for s, u in ((s2[0], u2[0]), (s3, u3))]
print(f"limit {LIMIT_DBC:.0f} dBc at 50 MHz: second order {verdict[0]}, "
      f"third order {verdict[1]}")
z2 = lambda w: loop_filter_impedance(w, c1, [(r2, c2)])
z3 = lambda w: third_order_impedance(w, c1, r2, c2, R3, C3)
print("              phase margin  -3 dB bandwidth  peaking")
for name, zf in (("second order", z2), ("third order ", z3)):
    r = closed_loop_response(icp, KVCO, N_DIV, zf, F_REF)
    print(f"{name}  {r['phase_margin_deg']:.1f} degrees  "
          f"{r['f_3db_hz'] / 1e3:.1f} kHz        {r['peaking_db']:.2f} dB")
