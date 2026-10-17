"""Tutorial: from a measured output phase-noise trace to the RMS
jitter with an error bar, checked against the loop model.

Steps
1. Model the synthesizer: a third-order loop filter (an R3-C3 section
   after the second-order filter, `third_order_impedance`), its
   bandwidth and phase margin, and the output noise from the
   oscillator, the phase detector floor and a third-order delta-sigma
   divider (`synthesizer_psd`).
2. "Measure" the output: here a synthetic trace, the model spectrum
   times the scatter of an analyzer reading averaged 16 times (the
   mean of 16 exponentially distributed periodogram values), drawn
   with a fixed seed and saved as a CSV file of (offset, dBc/Hz) rows.
3. Read the file (`read_noise_csv`), integrate the jitter
   (`rms_jitter`), and give its standard uncertainty from the number of
   averages (`jitter_relative_sigma`).
4. Compare the measured jitter with the model's, in units of that
   uncertainty, and plan how many averages a 0.5 % error bar needs
   (`averages_for_jitter`).

The oscillator points and the detector floor are illustrative values,
not measurements.  With your analyzer file, skip step 2 and pass your
own number of averages.

Printed output (fracpll 0.6.0):

    loop: 39.68 kHz bandwidth, 52.1 degrees phase margin
    read 400 points from 10 kHz to 10 MHz
    measured jitter 10.98 +- 0.12 ps (16 averages)
    model jitter    10.99 ps
    difference      -0.1 standard uncertainties
    averages for a 0.5 % error bar: 79 (gives 0.500 %)
"""
import os
import tempfile

import numpy as np

from fracpll import (NoiseSpec, averages_for_jitter, dsm_phase_psd,
                     jitter_relative_sigma, open_loop, psd_to_dbc,
                     read_noise_csv, rms_jitter, stability, synthesizer_psd,
                     third_order_impedance, white_floor)

F_REF, N_DIV, F_OUT = 50e6, 40.1, 2.005e9
ICP, KVCO = 100e-6, 20e6
N_AVG = 16

# 1. The model.  R3 >= 2 R2 and C3 <= C1/10, the rules of thumb of
#    W. O. Keese, National Semiconductor AN-1001 (1996).
filt = dict(c1=100e-12, r2=4.7e3, c2=1.5e-9, r3=10e3, c3=10e-12)
zf = lambda w: third_order_impedance(w, **filt)
rep = stability(ICP, KVCO, N_DIV, zf, F_REF)
print(f"loop: {rep['f_crossover_hz'] / 1e3:.2f} kHz bandwidth, "
      f"{rep['phase_margin_deg']:.1f} degrees phase margin")

f = np.geomspace(1e4, 1e7, 400)
lg = open_loop(2 * np.pi * f, ICP, KVCO, N_DIV, zf)
vco = NoiseSpec((1e4, 1e5, 1e6, 1e7), (-65.0, -92.0, -115.0, -135.0),
                reference="illustrative VCO noise points, not a measurement")
pfd = white_floor(-102.0, "illustrative PFD+CP floor, not a measurement")
model = synthesizer_psd(f, lg, N_DIV, s_vco=vco.psd(f),
                        s_inband=pfd.psd(f),
                        s_dsm=dsm_phase_psd(f, F_REF, 3))["s_out"]

# 2. A synthetic "measured" trace, written as an analyzer would save it.
rng = np.random.default_rng(17)
measured = model * rng.gamma(N_AVG, 1.0 / N_AVG, f.size)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "output_noise.csv")
    with open(path, "w") as fh:
        fh.write("Offset (Hz),Phase noise (dBc/Hz)\n")
        for fi, li in zip(f, psd_to_dbc(measured)):
            fh.write(f"{fi:.10e},{li:.6f}\n")
    trace = read_noise_csv(path, reference="synthetic trace, fracpll "
                                           "tutorial (16 averages)")

# 3. Jitter and its error bar.
f_m = np.array(trace.f_offset_hz)
s_m = trace.psd(f_m)
print(f"read {f_m.size} points from {f_m[0] / 1e3:.0f} kHz to "
      f"{f_m[-1] / 1e6:.0f} MHz")
jit = rms_jitter(f_m, s_m, F_OUT)
u_jit = jit * jitter_relative_sigma(f_m, s_m, N_AVG)
print(f"measured jitter {jit * 1e12:.2f} +- {u_jit * 1e12:.2f} ps "
      f"({N_AVG} averages)")

# 4. Model check and measurement plan.
jit_model = rms_jitter(f, model, F_OUT)
print(f"model jitter    {jit_model * 1e12:.2f} ps")
print(f"difference      {(jit - jit_model) / u_jit:+.1f} standard "
      "uncertainties")
n, achieved = averages_for_jitter(0.005, f_m, s_m)
print(f"averages for a 0.5 % error bar: {n} (gives {achieved * 100:.3f} %)")
