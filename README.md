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
- How much noise does the output have, and how much timing jitter
  does that give?
- Will the loop lock from a cold start, how fast, and how large a
  steady timing offset does leakage leave once it is locked?
- What unwanted tones (spurs) does the fractional divider create?
- How long must the measurement instrument average before the jitter
  number is trustworthy?

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
| **Per cycle** | Nothing is averaged; one step per reference cycle, small deviations from lock | Stability at any bandwidth; fast noise runs that include the finite pulse width | `sampled_stability`, `simulate_sampled` |
| **Edge by edge** | Nothing; every reference and divider edge time is solved for (exactly for a straight-line tuning curve, by accurate numerical integration for a measured one) | Large disturbances, pull-in with skipped cycles, pump mismatch, leakage, dead zone, measured tuning curves | `simulate_pll` |

Rule of thumb: start with the averaged model. If it refuses because
the loop is too fast, use the per-cycle model. If you need behaviour
far from lock or circuit non-idealities, use the edge-by-edge
simulator.

## Examples

Each example below runs as written, and the output shown is what it
printed with fracpll 0.3.1. The circuit values are illustrative
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
# The numbers here are placeholders; the reference text is required.
vco = NoiseSpec((1e4, 1e5, 1e6, 1e7), (-65.0, -92.0, -115.0, -135.0),
                reference="R&S FSWP, open-loop VCO, notebook 2026-09-18")
pfd = white_floor(-102.0, "PFD+CP floor, notebook 2026-09-18")

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
                   reference="VNA + SMU sweep, notebook 2026-09-18")

res = lock_transient(curve, n_div=100.0, f_ref_hz=50e6, icp=200e-6,
                     c_shunt=50e-12, branches=[(4.7e3, 1.5e-9)],
                     i_leak=2e-6)                  # 2 uA leakage
print("locked:", res["locked"])
print(f"settles in {res['t_settle'] * 1e6:.1f} us")
print(f"phase error left by the leakage: {res['phi_static']:.4f} rad")
```

```
locked: True
settles in 31.8 us
phase error left by the leakage: 0.0628 rad
```

The phase error is at the phase detector, in reference radians (2 pi is
one reference period), so 0.0628 rad is 1 % of a 20 ns reference period,
or 0.2 ns.

The run starts from the bottom of the measured voltage range and uses
the averaged model, which is written for a positive Kvco (for a
falling tuning curve it refuses and points to `simulate_pll`). It
replaces the phase detector's exact behaviour with a smooth curve
(a tanh), so the settling time is an estimate; `simulate_pll` gives
the edge-accurate transient. The leftover phase error is a closed-form value,
`2 pi atanh(I_leak / Icp)` here (it also includes pump mismatch when
you give one), and the test suite checks that the simulated transient
settles onto it.

### 4. Edge by edge: leakage, reset delay and pump mismatch

```python
from fracpll import LinearVCO, simulate_pll, pfd_static_offset

vco = LinearVCO(f0_hz=4.8e9, kvco_hz_per_v=400e6,
                reference="VCO datasheet rev B, Kvco at 25 C")

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
cycle exactly and answers at any speed.

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
centred `tau/2` after the reference edge, not at the edge. The
resulting error grows with `tau` squared, and because the delta-sigma
divider makes `tau` jump around from cycle to cycle, squaring it moves
some of the divider's high-frequency noise down to low frequencies,
where the loop does not filter it. `simulate_sampled(order=2)` includes
this effect with no fitted numbers and matches the edge-by-edge result;
`order=1` is the linear model. (Welch and `signal` above come from
SciPy and are used only to measure the spectra.)

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
- `rms_jitter` -- jitter in seconds from a noise spectrum.
- `dbc_to_psd`, `psd_to_dbc` -- conversions between dBc/Hz and
  rad^2/Hz, using dBc/Hz = 10 log10(`S_phi`/2).

**Tuning curves and lock**

- `fit_tuning`, `TuningCurve`, `TuningFamily` -- a smooth,
  never-folding curve through your measured (voltage, frequency)
  points, its slope (Kvco), and straight-line blending between curves
  measured at different temperatures. Negative Kvco is supported.
- `lock_transient`, `static_offset` -- lock-in from a cold start on
  the measured curve, and the phase error that leakage and mismatch
  leave behind.

**Measurement planning**

- `jitter_relative_sigma`, `averages_for_jitter` -- how uncertain a
  measured jitter value is after `n` averages, and how many averages a
  target uncertainty needs.

**Per-cycle model** (any loop bandwidth)

- `sampled_loop_map`, `sampled_stability` -- the exact one-cycle
  step of the loop for small deviations from lock, and stability from
  it.
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
  or negative;
- a negative Kvco is given to an averaged-model function (pass its
  size; the sign belongs in `pump_polarity` for the other models);
- smooth noise is asked of a first-order divider (it makes tones; use
  `mash_line_spectrum`);
- a noise value or tuning-curve value is asked outside the measured
  range;
- a measured tuning curve folds back (the error names the voltage
  where it happens);
- the lock target frequency is outside the oscillator's measured
  range;
- leakage or mismatch is too large for the pump to cancel;
- the phase detector has no locked state (for example inside a dead
  zone with nothing to push it out);
- a measured value is given without saying where it came from.

## How the results are checked

60 automated tests run on every change, on Python 3.9 to 3.14, and
once more on Python 3.9 with the oldest NumPy (1.22.0) and SciPy
(1.8.0) the package allows. Each numerical check compares the package with something independent of
it: an exact formula, a second calculation done a different way, or a
published result; none compares against a number stored from an
earlier run. The rest check that the refusals fire, plus one check of
the version number and the list of exported names. The lock settling
time of example 3 is only checked to be positive; treat it as an
estimate. The main checks:

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
  falling-slope noise.
- The measurement-planning formula agrees within 6 % with 4000
  simulated measurements.
- The lock transient settles onto its closed-form phase error to
  10^-6 rad, with the oscillator on `N * f_ref` to 1 part in 10^9.
- The dBc/Hz conversion reproduces the printed form of the delta-sigma
  noise formula (eq. 3.7 of W. Rhee's thesis, University of Illinois,
  2001), and spur dBc values use the same rule.

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

- For a slow loop, the per-cycle and averaged models predict the same
  settling behaviour: their poles match after the conversion
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

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- The averaged model is valid only for loops much slower than the
  reference. It says so and refuses beyond `f_ref/10`.
- `lock_transient` replaces the phase detector with a smooth tanh
  curve, so its settling time is an estimate. Use `simulate_pll` for
  an edge-accurate transient.
- The per-cycle model covers small deviations from lock, with equal
  UP and DN currents, no leakage, no dead zone and a constant Kvco.
  `order=2` keeps effects up to the square of the pulse width and
  drops smaller ones (cube and higher); in the tested designs these
  are 60 dB or more below the second-order effect.
- In the edge-by-edge simulator, the pump switches instantly apart
  from the stated delays. The dead zone is modelled as a turn-on delay,
  not as transistor physics. The only noise the simulator itself
  injects is the delta-sigma divider's and any reference timing noise
  you supply; other noise sources are added in the frequency domain
  with `synthesizer_psd`.
- The smooth delta-sigma noise formula is an approximation: it
  ignores that the divider pattern repeats exactly. The tests confirm
  it within 10 % for long repeat lengths; for short ones, use
  `mash_line_spectrum` for the exact tones.
- No transistor models, device physics or foundry data ship with the
  package. Your measured tuning curves and noise carry the technology,
  each with a required `reference` field.

## Where it comes from

The methods were first written for a study of a single-chip
GaN-on-SOI transistor-varactor PLL, where the oscillator has negative
Kvco and its tuning curve comes from measurements rather than a
formula. The parts that apply to any technology that uses a
charge-pump loop (CMOS, SiGe, GaN, or board level) are here. The
study's machine-learning and optimisation parts stay with the study,
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
