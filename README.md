# fracpll

[![PyPI](https://img.shields.io/pypi/v/fracpll.svg)](https://pypi.org/project/fracpll/) [![tests](https://github.com/TaN-MM-Org/fracpll/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/fracpll/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Designing a charge-pump fractional-N synthesizer means answering the
same questions in every technology: is the loop stable, and by how
much? What does the delta-sigma modulator add to the noise? What does
the closed loop do to YOUR measured oscillator noise, and what jitter
comes out? Will it lock, from where, how fast -- and where does
leakage park the static phase? How long must the analyzer average
before the jitter number has the error bar you need? `fracpll`
answers all of these from measured pieces -- your tuning curves, your
noise points, your pump current -- and refuses, with an explanation,
wherever its own averaged model stops being trustworthy.

The machinery is distilled from a monolithic GaN-on-SOI HEMT-varactor
PLL co-design study, where the oscillator has negative Kvco, the
tuning curve comes from measurements rather than a formula, and the
loop is judged against the same closed forms this package tests
itself with. The physics-informed-ML and adjoint layers of that study
stay with the study; what generalizes -- the loop algebra, the exact
MASH machinery, the noise assembly, the measured-curve discipline --
is here, for any technology (CMOS, SiGe, GaN, board-level) that fits
a charge-pump loop.

## Install

```
pip install fracpll        # NumPy + SciPy
```

## The loop in one example

```python
import numpy as np
from fracpll import (loop_filter_impedance, stability, fit_tuning,
                     mash_sequence, dsm_phase_psd, NoiseSpec,
                     white_floor, open_loop, synthesizer_psd,
                     rms_jitter, lock_transient, averages_for_jitter)

# 1. Loop stability from the schematic values (refuses past
#    f_ref/10, where the averaged model itself stops holding).
zf = lambda w: loop_filter_impedance(w, c_shunt=100e-12,
                                     branches=[(4.7e3, 1.5e-9)])
rep = stability(icp=100e-6, kvco_hz_per_v=20e6, n_div=40.0,
                zfilter=zf, f_ref_hz=50e6)
print(rep["f_crossover_hz"], rep["phase_margin_deg"])

# 2. The MASH sequence is computed EXACTLY (integer accumulators),
#    and its phase PSD is a closed form the tests hold against the
#    Welch spectrum of that exact sequence.
d = mash_sequence(num=104857, den=1 << 20, m=3, n_steps=1 << 16)

# 3. Closed-loop noise from YOUR measured pieces. No oscillator
#    noise ships with this package: a NoiseSpec carries your points
#    and their provenance, and refuses outside the measured range.
svco = NoiseSpec((1e4, 1e5, 1e6, 1e7), (-65.0, -92.0, -115.0, -135.0),
                 reference="R&S FSWP, open-loop VCO, notebook 2026-09-18")
f = np.geomspace(1e4, 1e7, 800)
lg = open_loop(2 * np.pi * f, 100e-6, 20e6, 40.0, zf)
out = synthesizer_psd(f, lg, 40.0, s_vco=svco.psd(f),
                      s_inband=white_floor(-102.0,
                          "PFD+CP floor, notebook 2026-09-18").psd(f),
                      s_dsm=dsm_phase_psd(f, 50e6, 3))
print(rms_jitter(f, out["s_out"], 2.0e9), "s RMS")

# 4. Lock, from a cold start, on the MEASURED tuning curve --
#    including where leakage parks the static phase (a closed form
#    the transient is tested against).
curve = fit_tuning(vc=[0.0, 1.0, 2.0, 3.0],
                   f_hz=[4.8e9, 5.2e9, 5.6e9, 6.0e9],
                   reference="VNA + SMU sweep, notebook 2026-09-18")
res = lock_transient(curve, n_div=100.0, f_ref_hz=50e6, icp=200e-6,
                     c_shunt=50e-12, branches=[(4.7e3, 1.5e-9)],
                     i_leak=2e-6)
print(res["locked"], res["t_settle"], res["phi_static"])

# 5. Price the measurement: averages needed for a 1 % jitter error
#    bar (an exact 1/sqrt(n) inversion, held against seeded MC).
n_avg, achieved = averages_for_jitter(0.01, f, out["s_out"])
```

## What is inside

- **`loop_filter_impedance` / `open_loop` / `stability`**: the
  general passive loop-filter one-port, the charge-pump loop gain
  L = Icp Kvco Z / (jw N), and a crossover/phase-margin report that
  refuses when there is no crossover, when the margin is not
  positive, and when the crossover passes f_ref/10 -- the classical
  validity edge of the averaged continuous-time model (Gardner,
  IEEE Trans. Commun. 28, 1849 (1980)).
- **`mash_sequence` / `dsm_phase_psd`**: MASH 1-1(-1) delta-sigma
  division sequences computed with exact integer accumulators (the
  mean division offset is EXACTLY num/den, asserted with rational
  arithmetic), and the one-sided quantization phase PSD -- a closed
  form validated in the tests against the Welch spectrum of the
  exact sequence, not taken from authority. First-order modulation
  is refused: it produces discrete spurs, not the smooth PSD.
- **`NoiseSpec` / `white_floor` / `synthesizer_psd` / `rms_jitter`**:
  measured phase-noise points with mandatory provenance, log-log
  interpolation that refuses to extrapolate, the three-path
  closed-loop noise assembly, and jitter with the one-sided
  convention stated and tested.
- **`fit_tuning` / `TuningCurve` / `TuningFamily`**: monotone
  (PCHIP) tuning curves through YOUR measured (Vc, f) points --
  negative-Kvco oscillators are first-class -- with the gain from
  the analytic derivative, linear interpolation between measured
  temperatures, and refusals outside the measured voltage and
  temperature ranges. A folded (non-monotone) measurement is
  refused with the offending bias voltage named.
- **`lock_transient` / `static_offset`**: the averaged nonlinear
  lock transient (saturating tanh detector, leakage, mismatch) on
  the measured curve, and the closed-form static phase offset
  2 pi atanh(I_leak/Icp - mismatch) it must settle to -- two
  independent routes, one number, asserted in the tests. A lock
  target outside the measured tuning range is refused, not
  integrated forever.
- **`jitter_relative_sigma` / `averages_for_jitter`**: the closed
  form for the error bar of an averaged jitter measurement, and its
  exact inversion into instrument time.

## Refusals, not guesses

A loop whose crossover violates its own model's validity, a
first-order modulator's spurs, an extrapolated noise slope, a folded
tuning curve, a lock target the oscillator cannot reach, leakage the
pump cannot cancel at any phase: each is refused with the reason and,
where one exists, the remedy.

## How it is checked

26 tests (Python 3.9-3.14, run in CI on every push), every claim
pinned to a closed form, an exact identity, published guidance, or
two independent code paths -- never a stored number. Highlights: the
general filter impedance equals the textbook second-order closed form
at machine precision; 1/(1+L) + L/(1+L) = 1 exactly; the
crossover/margin report matches an independent 400 000-point dense
grid; the MASH mean is asserted with exact rational arithmetic and
the sequence ranges hit their closed bounds; the quantization PSD
formula matches the Welch spectrum of the exact accumulator sequence
for MASH-2 and MASH-3; jitter reproduces the flat and 1/f^2 closed
forms; the PCHIP gain equals a central finite difference; the
integrated nonlinear transient settles to the closed-form static
offset and puts the oscillator exactly on N f_ref; and the
measurement-planning error bar matches a seeded Monte Carlo with
exponentially distributed periodogram bins.

## Honest limits

Deliberate scope, designed out with reasons: everything here is the
AVERAGED model -- per-edge PFD behavior (dead zone, cycle-slip
granularity, reset overlap) is below its resolution, and the
stability report refuses rather than pretending otherwise past
f_ref/10. The quantization PSD is the standard high-rate additive
model, validated against the exact sequence in-band and refused at
first order where it fails. No device physics, no transistor models
and no PDK data ship with this package: the source study's foundry
PDK files are licensed material and are not redistributed -- your
measured tuning curves and noise points carry the technology, each
behind a mandatory `reference` field, the same rule every package in
this organization applies.

## Support and governance

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches the physics arrives with a
test, and a claim arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/fracpll/issues). Usage
questions are welcome alongside bug reports; a docstring that left a
unit or a convention unclear is treated as a documentation bug, not
user error. While the version is below 1.0 the API may still move
between minor versions; such changes are called out in the release
notes.

## License

Apache-2.0.
