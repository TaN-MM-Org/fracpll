# fracpll

[![PyPI](https://img.shields.io/pypi/v/fracpll.svg)](https://pypi.org/project/fracpll/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22829473-blue)](https://doi.org/10.5281/zenodo.22829473) [![tests](https://github.com/TaN-MM-Org/fracpll/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/fracpll/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`fracpll` is a Python package for designing and checking a
**fractional-N frequency synthesizer**: the circuit that turns one clean
reference frequency into a precise, adjustable output frequency (for
example 50 MHz in, 2.005 GHz out, a ratio of 40.1). It works from the
numbers you have measured or designed -- your oscillator's tuning curve, your noise
measurements, your pump current, your filter parts -- and answers the
questions every design has to answer:

- Is the loop stable, and with how much safety margin?
- Which filter parts give the loop bandwidth and phase margin I want?
- How much noise does the output have, and how much timing jitter
  does that give?
- Will the loop lock from a cold start, how fast, and how large a
  steady timing offset does leakage leave once it is locked?
- What unwanted tones (spurs) does the fractional divider create?
- How long must the measurement instrument average before the jitter
  number is trustworthy?
- How well are the loop bandwidth and the margins known, given how
  well the pump current, Kvco and the filter parts are known?

It also reads your measured noise traces and tuning curves straight
from text files (CSV and similar), and fits a straight-line Kvco to
measured tuning points with its uncertainty. Two worked tutorials go
from data to a final result with error bars (see [Tutorials](#tutorials)).

The main results are checked by automated tests against independent
calculations (see [How the results are checked](#how-the-results-are-checked)).
One exception is stated there: the lock settling time is an estimate.
When a question falls outside what a model can answer reliably, the
package stops with an error message that says why and what to use
instead, rather than returning a number that looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install](#install)
- [Three ways to model the loop](#three-ways-to-model-the-loop)
- [Examples](#examples) (each with the output it prints)
- [Tutorials](#tutorials)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

A synthesizer of this kind is a feedback loop. Going around the loop:

- **Reference** -- a clean, fixed input frequency (`f_ref`), for
  example from a crystal oscillator.
- **Oscillator (VCO)** -- makes the output. Its frequency is set by a
  control voltage `v_c`. **Kvco** is how many hertz the frequency moves
  per volt. It can be negative (frequency falls as voltage rises). A
  loop built around such an oscillator also reverses the direction of
  its pump current, and the two sign flips cancel. So:
  `fit_tuning` accepts a falling curve; `sampled_loop_map`,
  `sampled_stability`, `simulate_sampled` and `simulate_pll` take the
  negative Kvco together with `pump_polarity=-1`; and the
  averaged-model functions (`open_loop`, `stability`,
  `continuous_closed_loop_poles`, `lock_transient`) need Kvco as a
  positive number and refuse a negative one.
- **Tuning curve** -- the measured relation between control voltage
  and oscillator frequency.
- **Divider** -- divides the output frequency by `N` so it can be
  compared with the reference. In a **fractional-N** synthesizer the
  divider switches among a few nearby whole numbers (from `N-3` to
  `N+4` for a third-order MASH) so that `N` is a fraction on average
  (for example 40.1). The switching pattern comes from a
  **delta-sigma modulator**; the kind used here is called
  **MASH**, and its **order** (1, 2 or 3) says how strongly it pushes
  its switching noise to high frequencies.
- **Phase detector and charge pump (PFD/CP)** -- each reference cycle
  they compare the timing of the reference edge with the divider edge,
  and push a short burst of current (the **pump current**, `Icp`) into
  or out of the loop filter. The burst lasts as long as the timing gap
  (plus a short **reset delay**, if the circuit has one, during which
  both the UP and DN currents flow). The timing gap is called the
  **pulse width** here.
- **Loop filter** -- a few resistors and capacitors that turn those
  current bursts into a smooth control voltage.
- **Loop bandwidth** -- roughly, how fast the loop reacts. Below it the
  loop follows the reference; above it the oscillator runs on its own.
  **Phase margin**, in degrees, measures how far the loop is from
  oscillating: more is safer, and zero or less means it is unstable.
  **Gain margin**, in dB, says how much the loop gain (for example the
  pump current) could grow before the loop becomes unstable.
- **Phase noise** -- small random wobbles in the output timing, given
  as a spectrum over offset frequency. It is quoted in **dBc/Hz**
  (decibels relative to the output, per hertz of bandwidth). Inside the
  package the same noise is held as `S_phi`: the noise power per hertz,
  counted over positive offsets only ("one-sided"), in rad^2/Hz. The
  two are related by the standard rule dBc/Hz = 10 log10(`S_phi`/2)
  (IEEE Std 1139), and `dbc_to_psd` / `psd_to_dbc` convert.
  **Jitter** is the same wobble expressed as a time spread in seconds.
- **Spur** -- an unwanted pure tone next to the output, quoted in
  **dBc**.
- **Leakage** -- a small current that drains the loop filter. The loop
  cancels it by holding a small permanent timing offset.

## Install

```
pip install fracpll
```

It needs Python 3.9 or newer, NumPy 1.22 or newer and SciPy 1.8 or
newer, and nothing else. Units are SI: hertz, volts, amperes, ohms,
farads, seconds. Phases come on three scales:

- **Reference radians**, where 2 pi is one reference period: phase
  errors at the phase detector (`lock_transient`'s `phi_e` and
  `phi_static`, `pfd_static_offset`'s `phase_ref_rad`).
- **Oscillator radians**, where 2 pi is one oscillator period (`N`
  times more radians for the same time shift): the noise spectra,
  spur tones and `SimResult.phase_rad()`.
- **Oscillator cycles**: names containing `cycles` or `psi`.

## Three ways to model the loop

The package offers three models of the same loop. They trade speed
for detail, and each one is checked against the others where they
overlap.

| Model | What it treats as smooth | Good for | Main functions |
|---|---|---|---|
| **Averaged** | The current bursts are averaged over each reference cycle | Stability, phase noise and jitter, lock-in, when the loop bandwidth is well below the reference (below `f_ref/10`) | `stability`, `synthesizer_psd`, `lock_transient` |
| **Per cycle** | Nothing is averaged; one step per reference cycle, small deviations from lock | Stability, phase margin and gain margin at any bandwidth; fast noise runs that include the finite pulse width | `sampled_margins`, `sampled_stability`, `simulate_sampled` |
| **Edge by edge** | Nothing; every reference and divider edge time is solved for (exactly for a straight-line tuning curve, by accurate numerical integration for a measured one) | Large disturbances, pull-in with skipped cycles, pump mismatch, leakage, dead zone, measured tuning curves | `simulate_pll` |

Rule of thumb: start with the averaged model. If it refuses because
the loop is too fast, use the per-cycle model. If you need behavior
far from lock or circuit non-idealities, use the edge-by-edge
simulator.

## Examples

Each example below runs as written, and the output shown is what it
printed with fracpll 0.6.0. The circuit values are illustrative
design values, not a recommended design.

### 1. Is the loop stable?

```python
from fracpll import loop_filter_impedance, stability

# Loop filter: 100 pF to ground, in parallel with 4.7 kOhm in series with 1.5 nF.
zf = lambda w: loop_filter_impedance(w, c_shunt=100e-12,
                                     branches=[(4.7e3, 1.5e-9)])

rep = stability(icp=100e-6,           # charge-pump current: 100 uA
                kvco_hz_per_v=20e6,   # oscillator gain: 20 MHz per volt
                n_div=40.0,           # divide ratio
                zfilter=zf,
                f_ref_hz=50e6)        # reference: 50 MHz
print(f"loop bandwidth {rep['f_crossover_hz']:.0f} Hz, "
      f"phase margin {rep['phase_margin_deg']:.1f} degrees")
```

```
loop bandwidth 40014 Hz, phase margin 54.2 degrees
```

A loop filter is described by one capacitor to ground (`c_shunt`) and
any number of resistor-capacitor pairs in parallel with it
(`branches`, each `(R, C)` in series).

### 2. Output noise and jitter from your measured parts

```python
import numpy as np
from fracpll import (loop_filter_impedance, open_loop, NoiseSpec,
                     white_floor, dsm_phase_psd, synthesizer_psd,
                     rms_jitter, averages_for_jitter)

zf = lambda w: loop_filter_impedance(w, c_shunt=100e-12,
                                     branches=[(4.7e3, 1.5e-9)])
f = np.geomspace(1e4, 1e7, 800)                  # offsets 10 kHz .. 10 MHz
lg = open_loop(2 * np.pi * f, 100e-6, 20e6, 40.0, zf)

# Your measured free-running oscillator noise (dBc/Hz at each offset).
# The numbers here are placeholders: put yours in, and say in
# `reference` where they came from (instrument and date, or a citation).
vco = NoiseSpec((1e4, 1e5, 1e6, 1e7), (-65.0, -92.0, -115.0, -135.0),
                reference="illustrative VCO noise points, not a measurement")
pfd = white_floor(-102.0, "illustrative PFD+CP floor, not a measurement")

out = synthesizer_psd(f, lg, 40.0,
                      s_vco=vco.psd(f),
                      s_inband=pfd.psd(f),
                      s_dsm=dsm_phase_psd(f, 50e6, 3))   # 3rd-order delta-sigma
jit = rms_jitter(f, out["s_out"], 2.0e9)                  # 2 GHz output
print(f"RMS jitter (10 kHz to 10 MHz): {jit * 1e12:.2f} ps")

n_avg, achieved = averages_for_jitter(0.01, f, out["s_out"])
print(f"averages for a 1 % error bar: {n_avg} (gives {achieved * 100:.3f} %)")
```

```
RMS jitter (10 kHz to 10 MHz): 10.82 ps
averages for a 1 % error bar: 9 (gives 0.993 %)
```

The output noise is built from three sources: the oscillator's own
noise (the loop suppresses it at low offsets), the noise floor of the
phase detector and charge pump (the loop passes it at low offsets,
multiplied by `N` squared), and the delta-sigma divider's switching
noise (the loop passes it at low offsets and filters it at high ones).
`out` also holds each of the three separately.

`NoiseSpec` joins your measured points with straight lines on a log-log
plot and refuses offsets outside the measured range. The measurement
plan in the last two lines treats each point of the offset grid `f` as
one analyzer reading averaged `n` times, with independent readings.

### 3. Will it lock, and how fast?

```python
from fracpll import fit_tuning, lock_transient

# Your measured tuning curve: control voltage (V) -> frequency (Hz).
curve = fit_tuning(vc=[0.0, 1.0, 2.0, 3.0],
                   f_hz=[4.8e9, 5.2e9, 5.6e9, 6.0e9],
                   reference="illustrative tuning points, not a measurement")

res = lock_transient(curve, n_div=100.0, f_ref_hz=50e6, icp=200e-6,
                     c_shunt=50e-12, branches=[(4.7e3, 1.5e-9)],
                     i_leak=2e-6)                  # 2 uA leakage
print("locked:", res["locked"])
print(f"settles in {res['t_settle'] * 1e6:.1f} us")
print(f"phase error left by the leakage: {res['phi_static']:.4f} rad")
```

```
locked: True
settles in 29.0 us
phase error left by the leakage: 0.0628 rad
```

The phase error is at the phase detector, in reference radians (2 pi is
one reference period), so 0.0628 rad is 1 % of a 20 ns reference period,
or 0.2 ns.

The run starts from the bottom of the measured voltage range and uses
the averaged model, which is written for a positive Kvco (for a
falling tuning curve it refuses and points to `simulate_pll`). It
replaces the phase detector's exact behavior with a smooth curve
(a tanh), so the settling time is an estimate; `simulate_pll` gives
the edge-accurate transient. The settling time is the last moment the
phase error is more than `settle_tol_rad` (default 0.05 rad) away from
its final value. Since 0.4.0 it is found on the solver's continuous
solution; before, it was rounded up to the next of 2000 output
samples, and this example printed 31.8 us.

The leftover phase error is a closed-form value,
`2 pi atanh(I_leak / Icp)` here (it also includes pump mismatch when
you give one), and the test suite checks that the simulated transient
settles onto it.

### 4. Edge by edge: leakage, reset delay and pump mismatch

```python
from fracpll import LinearVCO, simulate_pll, pfd_static_offset

vco = LinearVCO(f0_hz=4.8e9, kvco_hz_per_v=400e6,
                reference="illustrative design values, not a datasheet")

# 2 uA leakage, 100 ps reset delay, DN current 5 % weaker than UP.
run = simulate_pll(vco, n_int=100, f_ref_hz=50e6, icp=200e-6,
                   c_shunt=50e-12, branches=[(4.7e3, 1.5e-9)],
                   n_cycles=6000, i_leak=2e-6, i_dn=190e-6,
                   t_reset=100e-12)
print(f"simulated timing offset at lock: {run.time_error()[-1] * 1e12:.6f} ps")

exact = pfd_static_offset(50e6, 200e-6, i_dn=190e-6, i_leak=2e-6,
                          t_reset=100e-12)
print(f"exact charge-balance answer:     {exact['dt_s'] * 1e12:.6f} ps")
print("cycle slips:", run.slips)
```

```
simulated timing offset at lock: 195.000000 ps
exact charge-balance answer:     195.000000 ps
cycle slips: 0
```

The timing offset is the divider edge minus the reference edge. Once
locked, the charge the pump adds each cycle must exactly replace what
leakage and mismatch remove; `pfd_static_offset` solves that balance
directly, and the simulator reaches the same value on its own.

### 5. A fast loop the averaged model cannot judge

```python
from fracpll import loop_filter_impedance, stability, sampled_stability

loop = dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=50e6)
zf = lambda w: loop_filter_impedance(w, 2e-12, [(20e3, 200e-12)])

try:
    stability(zfilter=zf, **loop)
except ValueError as err:
    print("averaged model:", err)

st = sampled_stability(c_shunt=2e-12, branches=[(20e3, 200e-12)], **loop)
print("per-cycle model: stable =", st["stable"],
      f"(growth factor per cycle {st['spectral_radius']:.5f}; below 1 means stable)")
```

```
averaged model: crossover 1.566e+07 Hz exceeds f_ref/10 = 5e+06 Hz: the averaged continuous-time model is no longer trustworthy this close to the reference (Gardner, IEEE Trans. Commun. 28, 1849 (1980)); this averaged model refuses here. Use fracpll.sampled_stability for the exact sampled-data answer at any bandwidth, or fracpll.simulate_pll for the edge-level loop
per-cycle model: stable = True (growth factor per cycle 0.99501; below 1 means stable)
```

The averaged model treats the current bursts as a smooth current. That
is only accurate when the loop is much slower than the reference, so
it refuses here. The per-cycle model steps the loop once per reference
cycle exactly and answers at any speed. "Stable" is not the same as
"well designed": example 8 shows that this loop has almost no margin.

### 6. Spurs from a first-order divider

```python
import numpy as np
from fracpll import (mash_line_spectrum, closed_loop_lines,
                     loop_filter_impedance, open_loop)

# A first-order divider with fraction 3/64 repeats every 64 cycles, so its
# error is a set of pure tones (spurs) at multiples of 50 MHz / 64.
lines = mash_line_spectrum(num=3, den=64, m=1, f_ref_hz=50e6)

zf = lambda w: loop_filter_impedance(w, 100e-12, [(4.7e3, 1.5e-9)])
lg = open_loop(2 * np.pi * lines["f_hz"], 100e-6, 20e6, 40.0, zf)
spurs = closed_loop_lines(lines["f_hz"], lines["line_rad2"], lg)

for f, dbc in zip(lines["f_hz"][:3], spurs["sideband_dbc"][:3]):
    print(f"spur at {f / 1e6:.5f} MHz: {dbc:.2f} dBc")
```

```
spur at 0.78125 MHz: -59.19 dBc
spur at 1.56250 MHz: -70.97 dBc
spur at 2.34375 MHz: -52.82 dBc
```

A divider pattern of any MASH order repeats exactly after a fixed
number of cycles, so its error is a set of exact tones.
`mash_line_spectrum` finds that repeat length and every tone (for
repeat lengths up to 2^22 cycles by default; longer ones are refused).
`closed_loop_lines` then passes them through the loop to the output.
A dBc value is only meaningful for a small tone, so it is given only
where the tone's phase swing is 0.2 rad or less and is `NaN` elsewhere.

### 7. Why the noise near the loop bandwidth is higher than a linear model says

```python
import numpy as np
from scipy import signal
from fracpll import simulate_sampled, simulate_pll, LinearVCO

design = dict(icp=100e-6, f_ref_hz=50e6, c_shunt=100e-12,
              branches=[(4.7e3, 1.5e-9)])
frac = dict(n_int=40, num=104857, den=2**20, mash_order=3,
            n_cycles=(1 << 16) + 2048)
n_mean = 40 + 104857 / 2**20          # the average divide ratio

# Fast per-cycle model, first order (linear) and second order.
lin = simulate_sampled(kvco_hz_per_v=20e6, n_div=n_mean, order=1, **design, **frac)
full = simulate_sampled(kvco_hz_per_v=20e6, n_div=n_mean, order=2, **design, **frac)

# The same loop, edge by edge.
vco = LinearVCO(f0_hz=1.995e9, kvco_hz_per_v=20e6,
                reference="design value for this example")
edge = simulate_pll(vco, **design, **frac)

def level_db(psi_cycles, lo=2e4, hi=5e4):
    """Average phase noise between lo and hi (dB rad^2/Hz)."""
    f, p = signal.welch(2 * np.pi * psi_cycles, fs=50e6,
                        nperseg=1 << 14, detrend="linear")
    return 10 * np.log10(p[(f >= lo) & (f < hi)].mean())

keep = slice(8192, None)              # drop the start-up
for name, psi in [("edge by edge", edge.psi_ref),
                  ("per cycle, order 1", lin["psi"]),
                  ("per cycle, order 2", full["psi"])]:
    print(f"{name:20s} {level_db(psi[keep]):7.1f} dB  (20-50 kHz)")
```

```
edge by edge          -145.3 dB  (20-50 kHz)
per cycle, order 1    -160.3 dB  (20-50 kHz)
per cycle, order 2    -145.3 dB  (20-50 kHz)
```

With a delta-sigma divider, the edge-by-edge simulator shows more
noise near the loop bandwidth than a linear model predicts (15 dB more
in this example), even with perfectly matched pumps. The cause is the
width of the current bursts. A burst of width `tau` delivers its charge
centered `tau/2` after the reference edge, not at the edge. The
resulting error grows with `tau` squared, and because the delta-sigma
divider makes `tau` jump around from cycle to cycle, squaring it moves
some of the divider's high-frequency noise down to low frequencies,
where the loop does not filter it. `simulate_sampled(order=2)` includes
this effect with no fitted numbers and matches the edge-by-edge result;
`order=1` is the linear model. (Welch and `signal` above come from
SciPy and are used only to measure the spectra.)

### 8. Phase and gain margin of a fast loop

```python
from fracpll import sampled_margins

loop = dict(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10, f_ref_hz=50e6)
m = sampled_margins(c_shunt=2e-12, branches=[(20e3, 200e-12)], **loop)
print(f"loop bandwidth {m['f_crossover_hz'] / 1e6:.2f} MHz, "
      f"phase margin {m['phase_margin_deg']:.2f} degrees")
print(f"gain margin {m['gain_margin_db']:.2f} dB "
      f"(at {m['f_phase_crossover_hz'] / 1e6:.1f} MHz), stable = {m['stable']}")

slow = sampled_margins(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0,
                       f_ref_hz=50e6, c_shunt=100e-12,
                       branches=[(4.7e3, 1.5e-9)])
print(f"example 1 loop: {slow['f_crossover_hz']:.0f} Hz, "
      f"{slow['phase_margin_deg']:.1f} degrees, "
      f"gain margin {slow['gain_margin_db']:.1f} dB")
```

```
loop bandwidth 22.68 MHz, phase margin 2.06 degrees
gain margin 0.18 dB (at 25.0 MHz), stable = True
example 1 loop: 40015 Hz, 54.2 degrees, gain margin 86.0 dB
```

`sampled_margins` works from the exact per-cycle loop gain
(`sampled_open_loop`), so it answers at any loop speed. The fast loop
of example 5 is stable, but only just: 2 degrees of phase margin, and
a pump current only 2 % (0.18 dB) higher would make it oscillate at
half the reference frequency. For the slow loop of example 1 it agrees
with the averaged model (40014 Hz and 54.2 degrees in example 1). A
sampled loop also has a finite gain margin, which the averaged model
cannot give: its phase always returns to -180 degrees at half the
reference frequency.

### 9. Measured noise from a file, integrated exactly

```python
import numpy as np
from fracpll import read_noise_csv, rms_jitter, dbc_to_psd

# A small file of the kind an analyzer or a bench script saves:
# a header line, then one "offset, dBc/Hz" row per point.
with open("vco_noise.csv", "w") as fh:
    fh.write("Offset (Hz),Phase noise (dBc/Hz)\n"
             "1e4,-65\n1e5,-92\n1e6,-115\n1e7,-135\n")

vco = read_noise_csv("vco_noise.csv",
                     reference="illustrative decade points, not a measurement")
var = vco.phase_variance(1e4, 1e7)          # rad^2, exact between points
print(f"jitter of these points at 2 GHz: "
      f"{np.sqrt(var) / (2 * np.pi * 2e9) * 1e12:.3f} ps")

f = np.array(vco.f_offset_hz)
s = dbc_to_psd(np.array(vco.dbc_per_hz))
print(f"same four points, log-log rule: "
      f"{rms_jitter(f, s, 2e9, method='loglog') * 1e12:.3f} ps")
print(f"same four points, trapezoid:    {rms_jitter(f, s, 2e9) * 1e12:.3f} ps")
```

```
jitter of these points at 2 GHz: 4.869 ps
same four points, log-log rule: 4.869 ps
same four points, trapezoid:    13.579 ps
```

`read_noise_csv` skips the header line, reads the rows and builds a
`NoiseSpec` (the `reference` is still required). `phase_variance`
integrates the noise exactly as the spec draws it: straight lines
between the points on a log-log plot. Given only a few points far
apart, the ordinary trapezoid rule (straight lines on linear axes,
the default of `rms_jitter`) overstates the jitter badly, here by a
factor of 2.8; `method="loglog"` avoids that. On a dense grid the
two rules agree: on the 800 points of example 2 they differ by about
1 part in 10^5.
`read_tuning_csv` does the same for a (control voltage, frequency)
file and returns a fitted tuning curve.

### 10. From measured tuning points to a phase margin with an error bar

```python
from fracpll import fit_linear_vco, margin_uncertainty

# Measured tuning points around the lock voltage: control voltage (V)
# and frequency counter reading (Hz), each reading good to 0.1 MHz.
vc = [0.30, 0.40, 0.50, 0.60, 0.70]
f_hz = [1.99607e9, 1.99797e9, 2.00010e9, 2.00197e9, 2.00406e9]
fit = fit_linear_vco(vc, f_hz, f_sigma_hz=0.1e6,
                     reference="illustrative tuning points, not a measurement")
kv, u_kv = fit["kvco_hz_per_v"], fit["u_kvco_hz_per_v"]
print(f"Kvco = {kv / 1e6:.2f} +- {u_kv / 1e6:.2f} MHz/V "
      f"(straight-line test: p = {fit['p_value']:.2f})")

# Carry that, the pump-current calibration and the part tolerances
# through to the loop of example 1.
r = margin_uncertainty(icp=100e-6, kvco_hz_per_v=kv, n_div=40.0,
                       f_ref_hz=50e6, c_shunt=100e-12,
                       branches=[(4.7e3, 1.5e-9)],
                       u_icp=0.02,                     # 2 % calibration
                       u_kvco=u_kv / kv,               # from the fit
                       u_c_shunt=0.05 / 3 ** 0.5,      # +-5 % capacitor
                       u_branches=[(0.01 / 3 ** 0.5,   # +-1 % resistor
                                    0.05 / 3 ** 0.5)], # +-5 % capacitor
                       n_mc=2000, seed=1)
fc, pm = r["nominal"]["f_crossover_hz"], r["nominal"]["phase_margin_deg"]
print(f"loop bandwidth {fc:.0f} +- {r['u']['f_crossover_hz']:.0f} Hz")
print(f"phase margin {pm:.1f} +- {r['u']['phase_margin_deg']:.1f} degrees")
big = max(r["contributions"],
          key=lambda k: abs(r["contributions"][k]["phase_margin_deg"]))
print("largest share of the phase margin uncertainty:", big)
mc = r["mc"]["phase_margin_deg"]
print(f"Monte Carlo: {mc['std']:.1f} degrees, 95 % of draws in "
      f"[{mc['p2.5']:.1f}, {mc['p97.5']:.1f}]")
```

```
Kvco = 19.98 +- 0.32 MHz/V (straight-line test: p = 0.69)
loop bandwidth 39983 +- 844 Hz
phase margin 54.2 +- 0.8 degrees
largest share of the phase margin uncertainty: branch0_c
Monte Carlo: 0.8 degrees, 95 % of draws in [52.6, 55.7]
```

`fit_linear_vco` fits the straight line f = f0 + Kvco Vc by least
squares and gives Kvco with its standard uncertainty (the `+-` is one
standard deviation). It also tests whether a straight line describes
the points: with stated reading uncertainties (`f_sigma_hz`) by a
chi-square test, and without them by testing for a curved (quadratic)
term. If the test fails (p below 0.001 by default), it refuses and
suggests fitting a narrower voltage window (`vc_window`) or using
`fit_tuning`. Without `f_sigma_hz`, the uncertainty comes from the
scatter of the points about the line, and with only a few points it
is itself uncertain (`fit["dof"]` gives its degrees of freedom).
`fit["vco"]` is a `LinearVCO` ready for `simulate_pll`.

`margin_uncertainty` moves each input by one standard uncertainty in
each direction and combines the effects by the standard law of
propagation of uncertainty (the GUM, JCGM 100:2008). Each `u_` value
is a relative standard uncertainty: 0.02 means 2 %. A tolerance of
+-5 % with nothing else known about it counts as a rectangular
distribution, whose standard uncertainty is 5 %/sqrt(3) (GUM 4.3.7),
as written above. `r["contributions"]` shows how much each input adds
(`branch0_c` is the capacitor of the first branch). With `n_mc` it
also draws the inputs at random (the Monte Carlo method of JCGM
101:2008) as a check. It refuses when one standard uncertainty in
any input can make the loop unstable or take it outside the model,
and when the result is too curved for the straight-line method (then
it points to `n_mc`). With `model="sampled"` it uses the per-cycle
model, at any loop bandwidth, and adds the gain margin.

### 11. Design the filter for a bandwidth and a phase margin

```python
from fracpll import design_loop_filter

d = design_loop_filter(f_c_hz=40e3, phase_margin_deg=55.0, icp=100e-6,
                       kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=50e6)
print(f"C1 = {d['c1'] * 1e12:.1f} pF, R2 = {d['r2']:.0f} Ohm, "
      f"C2 = {d['c2'] * 1e9:.4f} nF")
a, p = d["averaged"], d["per_cycle"]
print(f"averaged model:  {a['f_crossover_hz']:.1f} Hz, "
      f"{a['phase_margin_deg']:.3f} degrees")
print(f"per-cycle model: {p['f_crossover_hz']:.1f} Hz, "
      f"{p['phase_margin_deg']:.3f} degrees, "
      f"gain margin {p['gain_margin_db']:.1f} dB")
```

```
C1 = 249.6 pF, R2 = 5581 Ohm, C2 = 2.2610 nF
averaged model:  40000.0 Hz, 55.000 degrees
per-cycle model: 40000.1 Hz, 55.000 degrees, gain margin 94.0 dB
```

`design_loop_filter` is the reverse of `stability`. Give it the loop
bandwidth and phase margin you want, and your pump current, Kvco and
`N`, and it returns the second-order filter (`c_shunt` = C1 and
`branches` = [(R2, C2)]) that meets both exactly in the averaged
model. It places the peak of the loop phase at the bandwidth, the
classic method of W. O. Keese (National Semiconductor application
note AN-1001, 1996). It also reads the result back with `stability`
and with the exact per-cycle model (`sampled_margins`). The per-cycle
margin moves away from the target as the bandwidth approaches the
reference: the same loop designed for 4.99 MHz at 55 degrees has a
per-cycle phase margin of 50.1 degrees. The design
refuses a bandwidth of `f_ref/10` or more, and a phase margin outside
0 to 90 degrees. Round the values to parts you can buy and check
the rounded filter again (the first tutorial below does this).

### 12. A third-order filter

```python
import numpy as np
from fracpll import loop_filter_impedance, stability, third_order_impedance

loop = dict(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0, f_ref_hz=50e6)

# The filter of example 1, then R3 = 10 kOhm in series to C3 = 10 pF.
zf3 = lambda w: third_order_impedance(w, c1=100e-12, r2=4.7e3, c2=1.5e-9,
                                      r3=10e3, c3=10e-12)
rep = stability(zfilter=zf3, **loop)
print(f"third order: {rep['f_crossover_hz']:.0f} Hz, "
      f"{rep['phase_margin_deg']:.1f} degrees")

# What the R3-C3 section buys: less ripple on v_c at the reference.
w_ref = 2 * np.pi * 50e6
z2 = loop_filter_impedance(w_ref, 100e-12, [(4.7e3, 1.5e-9)])[0]
z3 = zf3(w_ref)[0]
print(f"extra attenuation at 50 MHz: {20 * np.log10(abs(z2 / z3)):.1f} dB")

# Too much: R3 = 20 kOhm, C3 = 1 nF puts an 8 kHz pole below the crossover.
zbad = lambda w: third_order_impedance(w, 100e-12, 4.7e3, 1.5e-9, 20e3, 1e-9)
try:
    stability(zfilter=zbad, **loop)
except ValueError as err:
    print("refused:", str(err).split(":")[0])
```

```
third order: 39762 Hz, 52.1 degrees
extra attenuation at 50 MHz: 29.9 dB
refused: phase margin -23.15 deg at crossover 1.784e+04 Hz
```

Many designs add a resistor R3 and a capacitor C3 after the
second-order filter, to filter the reference frequency out of the
control voltage. `third_order_impedance` gives what the loop needs:
the control voltage at C3 per unit of pump current. Use it in the
averaged model (`stability`, `open_loop`, `synthesizer_psd`). Keese's
note suggests C3 at most C1/10 and R3 at least twice R2, so that the
new pole does not interact with the main poles of the filter. The last case breaks
that rule. Its loop phase passes -180 degrees, and the loop is
unstable. Before 0.6.0, `stability` reported it as stable with a
"phase margin" of 336.85 degrees (see the 0.6.0 corrections below).

## Tutorials

Two scripts in [`examples/`](examples) go from data to a final result
with uncertainties. Each runs in a few seconds, uses only `fracpll`
(and NumPy), makes its "measured" data with a fixed random seed, and
prints the output shown here (a test checks that it still does).

**`examples/tutorial_design_and_margins.py`**: tuning points measured
around the lock voltage give Kvco with its uncertainty
(`fit_linear_vco`); `design_loop_filter` designs the filter for
100 kHz and 55 degrees; the parts are rounded to two digits and the
rounded loop is checked in both models; and `margin_uncertainty`
carries the pump current calibration, the Kvco uncertainty and the
part tolerances to the bandwidth and the phase margin.

```
Kvco = 50.06 +- 0.13 MHz/V (chi-square test p = 0.47)
designed: C1 = 399.8 pF, R2 = 1394 Ohm, C2 = 3.622 nF
rounded:  C1 = 400 pF, R2 = 1400 Ohm, C2 = 3.6 nF
rounded loop (averaged):  100.31 kHz, 54.90 degrees
rounded loop (per cycle): 100.32 kHz, 54.90 degrees, gain margin 66.0 dB
loop bandwidth 100.3 +- 1.8 kHz
phase margin 54.9 +- 0.6 degrees
largest share of the phase margin uncertainty: c_shunt
Monte Carlo (2000 draws): phase margin 95 % in [53.6, 56.1] degrees
```

**`examples/tutorial_jitter_from_trace.py`**: a third-order loop is
modeled; an output phase-noise trace, averaged 16 times on the
analyzer, is read from a CSV file (`read_noise_csv`); its RMS jitter
gets a standard uncertainty from the number of averages
(`jitter_relative_sigma`); it is compared with the model; and
`averages_for_jitter` plans the averages for a 0.5 % error bar. The
trace is synthetic: the model spectrum times the scatter of a reading
averaged 16 times.

```
loop: 39.68 kHz bandwidth, 52.1 degrees phase margin
read 400 points from 10 kHz to 10 MHz
measured jitter 10.98 +- 0.12 ps (16 averages)
model jitter    10.99 ps
difference      -0.1 standard uncertainties
averages for a 0.5 % error bar: 79 (gives 0.500 %)
```

## What is in the package

**Loop and stability** (averaged model)

- `loop_filter_impedance(w, c_shunt, branches)` -- the impedance of
  a passive loop filter: one capacitor to ground plus any number of
  series resistor-capacitor branches. `second_order_impedance` is the
  textbook two-capacitor, one-resistor formula, kept as an independent
  check.
- `open_loop(w, icp, kvco_hz_per_v, n_div, zfilter)` -- the loop gain
  `L = Icp * Kvco * Z / (j w N)`; `error_transfer` and
  `lowpass_transfer` give `1/(1+L)` and `L/(1+L)`.
- `stability(...)` -- loop bandwidth and phase margin.
- `third_order_impedance(w, c1, r2, c2, r3, c3)` -- the third-order
  filter (an R3-C3 section after the second-order filter), as the
  control voltage per unit of pump current, for the averaged model
  (example 12).

**Designing the filter**

- `design_loop_filter(f_c_hz, phase_margin_deg, icp, kvco_hz_per_v,
  n_div, f_ref_hz)` -- the second-order filter for a target loop
  bandwidth and phase margin, read back in the averaged and per-cycle
  models (example 11).

**Delta-sigma divider**

- `mash_sequence(num, den, m, n_steps)` -- the exact divider pattern
  of an order-`m` MASH, computed with whole-number arithmetic, so the
  average fraction is exactly `num/den`.
- `dsm_phase_psd(f_hz, f_ref_hz, m)` -- the smooth noise spectrum of that
  pattern (orders 2 and 3; order 1 makes tones, not smooth noise, and
  is refused with a pointer to the next item).
- `mash_period`, `mash_line_spectrum` -- the exact repeat length and
  exact tones of any order-1, 2 or 3 pattern.
- `MASH_RANGE` -- the smallest and largest divider step each order
  can produce.

**Noise and jitter**

- `NoiseSpec`, `white_floor` -- your measured noise, each with a
  required `reference` saying where it came from.
- `synthesizer_psd` -- the output noise from the three sources.
- `closed_loop_lines` -- divider tones as output spurs.
- `rms_jitter` -- jitter in seconds from a noise spectrum;
  `method="loglog"` integrates exactly between points joined by
  straight lines on a log-log plot (use it for a few measured points).
- `NoiseSpec.phase_variance(f_lo, f_hi)` -- the exact integrated
  phase noise (rad^2) of a measured spec, with no grid.
- `dbc_to_psd`, `psd_to_dbc` -- conversions between dBc/Hz and
  rad^2/Hz, using dBc/Hz = 10 log10(`S_phi`/2).

**Tuning curves and lock**

- `fit_tuning`, `TuningCurve`, `TuningFamily` -- a smooth,
  never-folding curve through your measured (voltage, frequency)
  points, its slope (Kvco), and straight-line blending between curves
  measured at different temperatures. Negative Kvco is supported.
- `fit_linear_vco` -- a straight-line Kvco (and f0) from measured
  tuning points, with standard uncertainties, a test that a straight
  line describes the points, and a ready `LinearVCO` (example 10).
- `lock_transient`, `static_offset` -- lock-in from a cold start on
  the measured curve, and the phase error that leakage and mismatch
  leave behind.

**Uncertainty of the loop**

- `margin_uncertainty` -- the standard uncertainty of the loop
  bandwidth and phase margin (and of the gain margin, with
  `model="sampled"`) from the uncertainties of the pump current, Kvco
  and each filter part, with each input's share and an optional Monte
  Carlo check (example 10).

**Reading measured data**

- `read_noise_csv`, `read_tuning_csv` -- a `NoiseSpec` or a fitted
  tuning curve from a text file (comma, semicolon, tab or space
  separated; header lines skipped; choose the columns and the
  frequency unit). Text inside the data, repeated or unsorted
  points and missing values are refused with the line number.

**Measurement planning**

- `jitter_relative_sigma`, `averages_for_jitter` -- how uncertain a
  measured jitter value is after `n` averages, and how many averages a
  target uncertainty needs.

**Per-cycle model** (any loop bandwidth)

- `sampled_loop_map`, `sampled_stability` -- the exact one-cycle
  step of the loop for small deviations from lock, and stability from
  it.
- `sampled_open_loop`, `sampled_margins` -- the exact per-cycle loop
  gain, and the phase margin and gain margin read from it, at any
  loop bandwidth.
- `continuous_closed_loop_poles` -- the averaged model's poles, for
  comparison.
- `simulate_sampled(..., order=1 or 2)` -- a fast cycle-by-cycle
  noise run; `order=2` adds the pulse-width effect of example 7.
- `pulse_doublet_vector` -- the building block of that effect.

**Edge-by-edge simulator**

- `simulate_pll(vco, ...)` with `LinearVCO` or a measured
  `TuningCurve` -- supports a phase detector reset delay, a dead zone,
  unequal UP and DN currents, leakage, negative Kvco, fractional
  division, optional reference timing noise, and counts skipped cycles
  (cycle slips).
- `SimResult` -- the recorded edge times, control voltage and phase;
  `time_error()`, `nearest_edge_error()` and `phase_rad()` read them.
- `pfd_static_offset` -- the exact locked timing offset from charge
  balance.

Each function's docstring (`help(fracpll.stability)`, for example)
gives its inputs, units and conventions.

## When it refuses, and why

`fracpll` raises an error instead of guessing when:

- the averaged model's loop bandwidth passes `f_ref/10`, where that
  model stops being accurate (Gardner, IEEE Trans. Commun. 28, 1849
  (1980)); the message points to the per-cycle model;
- the loop gain does not fall through 1 within the scanned band
  (by default `f_ref/10^6` to `f_ref/2`), or the phase margin is zero
  or negative (`stability`); for `sampled_margins`, when the loop gain
  is still 1 or more at `f_ref/2` (such a loop is always unstable);
- a negative Kvco is given to an averaged-model function (pass its
  size; the sign belongs in `pump_polarity` for the other models);
- smooth noise is asked of a first-order divider (it makes tones; use
  `mash_line_spectrum`);
- a noise value or tuning-curve value is asked outside the measured
  range;
- a measured tuning curve folds back (the error names the voltage
  where it happens);
- the lock target frequency, or the starting control voltage of
  `lock_transient`, is outside the oscillator's measured range;
- leakage or mismatch is too large for the pump to cancel;
- the phase detector has no locked state (for example inside a dead
  zone with nothing to push it out);
- a measured value is given without saying where it came from;
- a data file has text inside its data, repeated or unsorted points,
  or missing values (the message gives the line number);
- a delta-sigma pattern would ask the divider for fewer than one
  oscillator cycle (`simulate_pll` with a small `n_int`);
- a straight line does not describe the tuning points given to
  `fit_linear_vco` (the message gives the test result), or points lie
  exactly on a line with no stated uncertainty;
- a filter design is asked for a bandwidth of `f_ref/10` or more, or
  for a phase margin outside 0 to 90 degrees (`design_loop_filter`);
- one standard uncertainty in any input of `margin_uncertainty` can
  make the loop unstable or take it outside the model, the result is
  too curved for the straight-line method, or some Monte Carlo draws
  fail (the message gives how many);
- a number is NaN or infinite where it would otherwise give a NaN
  result, a NumPy error or (in `simulate_pll`, for `f_ref_hz`) a run
  that never finishes: the inputs of `simulate_pll`, the spectra
  given to `rms_jitter`, `psd_to_dbc` and the measurement planning
  functions, the loop values of the per-cycle model, and (since
  0.6.0) the offsets given to `NoiseSpec.psd`, the divider phase
  given to `simulate_sampled`, the loop gain given to
  `synthesizer_psd` and `closed_loop_lines`, `N` given to
  `synthesizer_psd`, and the temperature given to `TuningFamily`;
- a filter part is zero, negative or not a number in
  `continuous_closed_loop_poles` (as in every other model).

## How the results are checked

160 automated tests run on every change, on Python 3.9 to 3.14,
and once more on Python 3.9 with the oldest NumPy (1.22.0) and SciPy
(1.8.0) the package allows. Each numerical check compares the package
with something independent of it: an exact formula, a second
calculation done a different way, or a published result; none
compares against a number stored from an earlier run. The rest check
that the refusals fire, that the tutorials print what this README
shows (a check of the documentation, not of the physics), plus one
check of the version number and the list of exported names. The lock settling
time is checked against an exact solution only in the small-signal
limit (below); for a large start-up transient such as example 3 it
remains the tanh model's estimate. The main checks:

**Averaged model**

- The general filter formula equals the textbook second-order formula
  to 1 part in 10^12.
- The reported bandwidth and phase margin agree with a direct search
  over 400 000 frequencies.
- The divider pattern's average fraction is checked with exact
  fractions.
- The smooth delta-sigma noise formula agrees within 15 % with the
  measured spectrum of the actual divider pattern.
- The jitter calculation matches exact results for flat and
  falling-slope noise. The log-log rule matches the exact integral of
  a power law, for slopes from -3 to +1.5 including -1 (a
  logarithm), to 1 part in 10^12, and a dense-grid integration of a
  measured spec to 1 part in 10^8.
- The measurement-planning formula agrees within 6 % with 4000
  simulated measurements.
- The lock transient settles onto its closed-form phase error to
  10^-6 rad, with the oscillator on `N * f_ref` to 1 part in 10^9.
- For a small phase step from lock on a straight-line tuning curve,
  where the averaged model is linear, the transient matches the
  matrix-exponential solution of that linear system to 10^-5 of the
  step, and the settling time to 3 parts in 10^5. The settling time
  no longer changes with the number of output samples (to 1 part in
  10^9).
- The dBc/Hz conversion reproduces the printed form of the delta-sigma
  noise formula (eq. 3.7 of W. Rhee's thesis, University of Illinois,
  2001), and spur dBc values use the same rule.

**Reading data files**

- Numbers written to a file with 17 significant digits read back
  bit for bit, with any of the four separators, a header, a byte-order
  mark, a chosen column or a unit scale; a file of an exact power law
  integrates to the closed form to 1 part in 10^12; each refusal is
  checked to name the right line.

**Divider tones**

- For a first-order divider, the exact tones match a known closed form
  to 1 part in 10^12, and they add up to the total variance.
- For long patterns, the exact tones agree with the smooth formula
  within 10 % in every band.

**Edge-by-edge simulator**

- The locked timing offset matches the exact charge balance to 1 part
  in 10^6, in five cases covering leakage, mismatch, reset delay and
  dead zone.
- After a small nudge, it follows the exact per-cycle model cycle by
  cycle, to 1 part in 10^5.
- With the pump current 10 % below the stability limit that the
  per-cycle model predicts, the simulated loop settles; 10 % above, it
  oscillates with growing size. The averaged model refuses this case.
- The measured-curve path matches the exact straight-line path to
  10^-18 s.
- A fractional lock keeps the average ratio `N + num/den`, drifting
  by less than 10^-6 cycles per reference cycle.
- It locks after skipping cycles, and it works with negative Kvco.
- In a fractional-N pull-in with 60 skipped cycles, every divider
  count equals a separately generated MASH pattern, to the last edge,
  and the run ends locked on the fractional ratio.
- Its delta-sigma noise at high offsets matches the averaged formula
  (ratio between 0.85 and 1.2), and would be off by `N` squared (1600)
  with the error that version 0.1.0 had (the test asserts more than
  1000; see below).
- Its first-order spurs match the exact tones through the loop within
  0.5 dB.
- Unequal UP and DN currents raise the low-offset noise by more than
  20 dB, and a small deliberate offset current lowers it by more than
  6 dB. This known effect is checked here qualitatively (T.-H. Lin,
  C.-L. Ti and Y.-H. Liu, IEEE Trans. Circuits Syst. I 56, 877
  (2009)).

**Per-cycle model and pulse-width effect**

- The per-cycle loop gain equals the averaged loop gain summed over
  all its aliases (shifts by multiples of `f_ref`, an exact identity)
  to 1 part in 10^10, for the slow loop of example 1 and the fast
  loop of example 5, and, for a filter that is a single capacitor, a
  closed form to 1 part in 10^9; for that filter the phase margin and
  the gain margin are both 0 whatever the scan grid.
- The margins are checked by eigenvalues: shifting the phase of the
  loop gain by the phase margin, or scaling it by the gain margin,
  puts a pole of
  the per-cycle map on the unit circle at the matching frequency (to
  10^-8). In the fast loop, the pump current times the gain margin
  equals the current at which the loop becomes unstable (found
  separately from the poles) to 1 part in 10^9. For the slow loop the
  margins agree with the averaged model (bandwidth to 10^-4, phase
  margin to 0.01 degrees).
- Driven by a sine wave, or by the exact tones of a first-order
  divider in a fast loop the averaged model refuses, the per-cycle
  simulation gives the output that the per-cycle loop gain predicts
  (to 1 part in 10^9 and 10^6 respectively).

- For a slow loop, the per-cycle and averaged models predict the same
  settling behavior: their poles match after the conversion
  `z = exp(s T)` with `T = 1/f_ref`, to 0.1 % of each pole's distance
  from 1.
- `simulate_sampled(order=2)` matches the edge-by-edge simulator,
  cycle by cycle, more than 1000 times more closely than the linear
  model. For two values of the capacitor to ground, `c_shunt` (100 pF
  and 400 pF), its pulse widths also agree to 10^-15 s and its
  low-offset noise within 0.5 dB. With negative Kvco, the 1000-times
  match is checked.
- `order=1` equals the linear per-cycle model to 1 part in 10^12.
- Making the pulses half as wide in the same loop lowers the effect by
  about 6 dB (the test allows 4.5 to 7.5 dB), in both the simulator
  and `order=2`.

**Kvco fit and uncertainty of the loop**

- `fit_linear_vco` recovers an exact line; with equal stated
  uncertainties its Kvco uncertainty equals the closed form
  sigma/sqrt(sum (Vc - mean)^2) to 1 part in 10^12. Its weighted fit
  and covariance equal NumPy's `polyfit`, and its fit from the scatter
  equals SciPy's `linregress`, two independent least-squares codes.
- Its test p-values equal closed forms: exp(-chi2/2) for 2 degrees of
  freedom, and 1 - |t|/sqrt(2 + t^2) (Student's t with 2 degrees of
  freedom) for the curvature test on 5 points. In 3000 simulated
  measurements, the scatter of the fitted Kvco matches the reported
  uncertainty within 5 %. Curved points are refused.
- For the second-order filter, the sensitivities of the loop
  bandwidth and phase margin to the pump current and Kvco equal
  closed forms derived from |L| = 1 at the crossover, to 1 part in
  10^6. Scaling every capacitor and the pump current up by the same
  factor, and every resistor down by it, leaves the loop unchanged;
  the sensitivities obey the sum rule this implies, to 10^-6.
- In the fast loop of example 8, the gain margin moves exactly with
  the pump current (to 1 part in 10^9), because the loop gain at half
  the reference frequency is proportional to it. In a slow loop the
  per-cycle and averaged uncertainties agree within 0.2 %. A Monte
  Carlo run of 1000 draws agrees with the linear propagation within
  10 %, and the number of failed draws it reports equals a count made
  independently from the same random draws.

**Filter design and the third-order filter** (new in 0.6.0)

- `design_loop_filter` gives the same parts, to 1 part in 10^12, as
  a separate derivation in the test (the phase peak written in its
  half-angle form, and |L| = 1 solved on the textbook second-order
  impedance). On that impedance the designed loop has |L| = 1 and
  the target phase margin at the target bandwidth (to 10^-10
  degrees), with the phase at its peak there. `stability` reads the
  designed loop back to 1 part in 10^9 and 10^-9 degrees, for five
  designs from 1 kHz to 2.5 MHz and from 10 to 89 degrees.
- `third_order_impedance` equals the polynomial form found by nodal
  analysis (1 part in 10^11) and a direct solve of the three-node
  circuit (10^9), and with R3 = 0 it is the second-order filter with
  C1 + C3 to ground (10^12). For the loop of example 12, the margin
  from `stability` equals the polynomial form's phase at the
  crossover, and the poles of the closed loop (roots of its
  characteristic polynomial) all lie in the left half plane. For the
  refused loop, one pole lies in the right half plane.

**Faster code, same results** (new in 0.6.0)

- The 0.5.0 code is kept in the tests as the reference. The divider
  patterns and repeat lengths equal it exactly (whole numbers), the
  crossover of `stability` to 1 part in 10^12 and its phase margin to
  10^-9 degrees, and `simulate_sampled` to 10^-12 of the largest
  value.

**Fixes in this version**

- Each fix listed under 0.6.0 below has a test that fails with the
  old code and passes with the new.

## Corrections in earlier versions

**0.2.0 fixed a noise error in 0.1.0.** Version 0.1.0 multiplied the
delta-sigma divider noise by `N` squared in `synthesizer_psd`. That was
wrong. The divider noise formula already describes the error at the
oscillator, so it reaches the output through the loop alone, with no
factor of `N` squared. This agrees with eq. (3.7) of W. Rhee's
thesis (University of Illinois, 2001) and with the edge-by-edge
simulator. The error overstated delta-sigma noise by `N` squared (32 dB
at `N = 40`). The phase detector and charge pump floor does keep its
`N` squared. If you used 0.1.0 with a delta-sigma term, please re-run
that analysis.

**0.3.0 explained an effect that 0.2.0 only reported.** 0.2.0 noted
extra low-offset noise in the edge-by-edge simulator without a cause.
0.3.0 identifies the cause (example 7).

**0.3.1 fixed three problems found in a review of 0.3.0.**

- `dbc_to_psd` turned dBc/Hz into rad^2/Hz without the standard factor
  of 2, while the delta-sigma formula and the spur dBc values used it.
  Measured oscillator noise and noise floors were therefore counted
  3 dB too low next to the delta-sigma noise, and jitter computed from
  them came out too small by a factor of about 1.41 (example 2 printed
  7.65 ps before the fix and 10.82 ps after). If you converted measured
  dBc/Hz values with 0.3.0 or earlier, please re-run those results.
- The averaged-model functions accepted a negative Kvco and returned
  results for a loop with positive feedback (for example a phase
  margin above 180 degrees), and `lock_transient` returned an
  unlocked run for a falling tuning curve. Both are now refused with
  a pointer to the right tool.
- `rms_jitter` used a NumPy function that only exists from NumPy 2.0,
  although the package allows NumPy 1.22. It now works on both, and
  NumPy 1.22.0 with SciPy 1.8.0 is tested.

**0.4.0 fixed the lock settling time and tightened input checks.**

- `lock_transient` rounded its settling time up to the next output
  sample (by default 2000 samples over about 32 ms, so 15.9 us
  steps). Example 3 printed 31.8 us, exactly two samples; the settling
  time is 29.0 us. It is now found on the solver's continuous
  solution, whatever `n_eval` is. The transient itself (`t`, `phi_e`,
  `vc`, `locked`) is unchanged, bit for bit.
- `lock_transient` accepted a starting control voltage outside the
  measured tuning curve (holding the oscillator at the curve's edge
  while the filter voltage came back into range) and a negative
  capacitor (returning a meaningless run); `pfd_static_offset`
  returned NaN for a NaN input and raised a division-by-zero error for
  `f_ref_hz = 0`; `stability` scanned a reversed band and gave a
  misleading message. All are now refused with a clear message.

**0.5.0 fixed three silent errors and tightened input checks.**

- `simulate_pll` made the delta-sigma divider pattern only 8 entries
  longer than the number of reference cycles. A pull-in from above
  makes one extra divider edge per skipped cycle, so after more than
  8 skipped cycles the last divider edges silently used the integer
  ratio `N` instead of the pattern, and the end of the record (where
  you read the locked state) was pulled toward `N` times the
  reference. In a test run with 60 skipped cycles, the last 52 counts
  were wrong, and the control voltage ended at 0.4754 V instead of
  0.4998 V (the lock voltage is 0.5000 V). The pattern now grows as
  needed. Runs with 8 or fewer skipped cycles are unchanged, bit for
  bit.
- `simulate_pll` accepted an integer ratio so small that the pattern
  asked the divider for 0 or a negative number of oscillator cycles
  (for example `n_int=1` with a third-order MASH gave counts of -1 and
  reported 63 skipped cycles). This is now refused.
- `averages_for_jitter` could return one average too many. Asked for
  exactly the error bar that 8 averages give (here 1.888 %), it
  rounded 8 up to just above 8 and answered 9. It now returns the
  smallest number that meets the target, as documented.
- `simulate_pll` with `f_ref_hz` set to NaN never finished, and NaN
  pump currents, spectra or loop values gave NaN results or a NumPy
  error in several functions. These are now refused with a message
  that names the value.

**0.6.0 fixed a wrong stability verdict and three crashes or silent
NaNs.**

- `stability` reported the phase margin as 180 degrees plus the angle
  of the loop gain, an angle that NumPy gives between -180 and 180
  degrees. When the loop phase had passed -180 degrees, the angle
  wrapped around and the "margin" came out above 180 degrees, so an
  unstable loop was reported as stable. This cannot happen with the
  filters of `loop_filter_impedance`, whose phase stays above -180
  degrees, but it can with any other filter you pass, such as a
  third-order filter. In example 12's refused loop, 0.5.0 returned a
  phase margin of 336.85 degrees; the margin is -23.15 degrees, and
  the loop has a pole in the right half plane. The margin is now
  given between -180 and 180 degrees, as `sampled_margins` already
  did, and this loop is refused as unstable. No result for the
  filters of `loop_filter_impedance` changes.
- `simulate_pll` with reference timing noise (`ref_jitter_s`) of
  exactly `n_cycles + 1` entries, the length the check asks for,
  crashed with an IndexError at the end of the run. It read one entry
  past the last reference edge, which the run never uses. It now runs,
  and gives the same result, bit for bit, as with a longer array.
- `TuningFamily` accepted temperatures written as text (`"300"`, as
  read from a file header) and then failed with a KeyError when
  evaluated. They are now read as numbers. A temperature given twice
  is refused, and a NaN temperature, which returned NaN, is refused.
- `NoiseSpec.psd`, `simulate_sampled`, `synthesizer_psd` and
  `closed_loop_lines` returned NaN for a NaN input, and
  `continuous_closed_loop_poles` returned the poles of a filter with a
  negative capacitor. `sampled_margins` with `n_grid=1` refused with
  the misleading reason that the loop gain does not cross 1. All are
  now refused with a clear message.

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- The averaged model is valid only for loops much slower than the
  reference. It says so and refuses beyond `f_ref/10`; use
  `sampled_margins` for stability margins of faster loops.
- `lock_transient` replaces the phase detector with a smooth tanh
  curve, so for a large start-up transient its settling time is only
  an estimate of the real loop's; the tests check it against an exact
  solution only in the small-signal limit. Use `simulate_pll` for an
  edge-accurate transient.
- The per-cycle model covers small deviations from lock, with equal
  UP and DN currents, no leakage, no dead zone and a constant Kvco.
  `order=2` keeps effects up to the square of the pulse width and
  drops smaller ones (cube and higher); in the tested designs these
  are 60 dB or more below the second-order effect.
  `sampled_open_loop` and `sampled_margins` use the linear
  (first-order) per-cycle model; the pulse-width effect of
  `order=2` is not part of the margins.
- In the edge-by-edge simulator, the pump switches instantly apart
  from the stated delays. The dead zone is modeled as a turn-on delay,
  not as transistor physics. The only noise the simulator itself
  injects is the delta-sigma divider's and any reference timing noise
  you supply; other noise sources are added in the frequency domain
  with `synthesizer_psd`.
- The smooth delta-sigma noise formula is an approximation: it
  ignores that the divider pattern repeats exactly. The tests confirm
  it within 10 % for long repeat lengths; for short ones, use
  `mash_line_spectrum` for the exact tones.
- `rms_jitter` keeps the trapezoid rule as its default, so existing
  results do not change; on a few widely spaced points it overstates
  a falling spectrum, and `method="loglog"` is the right choice there.
  The measurement plan (`jitter_relative_sigma`,
  `averages_for_jitter`) uses the trapezoid weights, which suits a
  dense analyzer trace.
- The file readers do not know any instrument's file format; you
  choose the columns and the frequency unit. A decimal comma (`1,5`)
  is not supported.
- `fit_linear_vco` fits a straight line. Over a wide voltage range a
  real tuning curve bends, and the test then refuses; fit a window
  around the operating point. Its uncertainties assume independent
  readings with the stated (or a common) standard deviation.
- `design_loop_filter` designs only the second-order filter, and
  only in the averaged model; its result is exact for the values it
  returns, not for the rounded parts you buy. `third_order_impedance`
  works only in the averaged model: the per-cycle model, the
  edge-by-edge simulator and `margin_uncertainty` take only a
  capacitor to ground with series resistor-capacitor branches.
- `stability` gives the phase margin between -180 and 180 degrees. A
  loop whose phase at the crossover has fallen to -360 degrees or
  below (it takes more extra poles than the third-order filter has)
  would be misread; no filter in this package reaches that.
- `margin_uncertainty` treats the inputs as independent (uncorrelated)
  and takes `f_ref` and `N` as exact. It is a first-order method;
  the curvature check and the optional Monte Carlo say when that is
  not enough. It knows only the uncertainties you give it, and the
  model it calls (`stability` or `sampled_margins`) keeps its own
  limits.
- No transistor models, device physics or foundry data ship with the
  package. Your measured tuning curves and noise carry the technology,
  each with a required `reference` field.

## Where it comes from

The methods were first written for a study of a single-chip
GaN-on-SOI transistor-varactor PLL, where the oscillator has negative
Kvco and its tuning curve comes from measurements rather than a
formula. The parts that apply to any technology that uses a
charge-pump loop (CMOS, SiGe, GaN, or board level) are here. The
study's machine-learning and optimization parts stay with the study,
and its foundry design-kit files are licensed material that is not
included.

## Citing, support and license

If `fracpll` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22829473](https://doi.org/10.5281/zenodo.22829473),
which always points to the latest release; each release also has its
own DOI on Zenodo. [CITATION.cff](CITATION.cff) has the details.

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and decides on scope and releases. Questions and bug
reports are welcome in the
[issue tracker](https://github.com/TaN-MM-Org/fracpll/issues). A
docstring that leaves a unit or convention unclear counts as a
documentation bug. Changes that touch the physics come with a test,
and claims come with a source; see [CONTRIBUTING.md](CONTRIBUTING.md).
Until version 1.0 the programming interface may still change between
minor versions; any such change is listed in the release notes.

Licensed under Apache-2.0.
