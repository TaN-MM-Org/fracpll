# fracpll

[![PyPI](https://img.shields.io/pypi/v/fracpll.svg)](https://pypi.org/project/fracpll/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22829473-blue)](https://doi.org/10.5281/zenodo.22829473) [![tests](https://github.com/TaN-MM-Org/fracpll/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/fracpll/actions)
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

Version 0.2 goes past the averaged model. An **edge-accurate
event-driven simulator** runs the loop reference edge by reference
edge: a tri-state PFD with reset delay, dead zone, UP/DN mismatch and
leakage, the loop filter integrated in closed form between edges (no
time step), divider edges found by root-finding on the exact
oscillator phase, measured tuning curves, and MASH-driven fractional
division. It resolves cycle slips during pull-in, the edge-level static
offset, loops whose bandwidth approaches the reference, and the
nonlinear folding of delta-sigma noise that no averaged model can
show. An **exact sampled-data stability map** answers the stability
question at any bandwidth, where the averaged model has to refuse,
and an **exact MASH line spectrum** replaces the refusal at first
order with the actual spur lines.

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

## Beyond the averaged model (new in 0.2)

```python
import numpy as np
from fracpll import (LinearVCO, simulate_pll, pfd_static_offset,
                     sampled_stability, mash_line_spectrum)

vco = LinearVCO(f0_hz=4.8e9, kvco_hz_per_v=400e6,
                reference="VCO datasheet rev B, Kvco at 25 C")

# Edge by edge: 2 uA leakage, 100 ps reset delay, 5 % pump mismatch.
run = simulate_pll(vco, n_int=100, f_ref_hz=50e6, icp=200e-6,
                   c_shunt=50e-12, branches=[(4.7e3, 1.5e-9)],
                   n_cycles=6000, i_leak=2e-6, i_dn=190e-6,
                   t_reset=100e-12)
print(run.time_error()[-1], "s divider-minus-reference at lock")
# ... which is the exact charge balance of the tri-state PFD:
print(pfd_static_offset(50e6, 200e-6, i_dn=190e-6, i_leak=2e-6,
                        t_reset=100e-12)["dt_s"])

# Stability at ANY bandwidth (the averaged model refuses past f_ref/10):
st = sampled_stability(icp=0.5e-3, kvco_hz_per_v=400e6, n_div=10,
                       f_ref_hz=50e6, c_shunt=2e-12,
                       branches=[(20e3, 200e-12)])
print(st["stable"], st["spectral_radius"])

# First-order MASH: the exact spur lines instead of a refusal, and
# where they land at the output after the closed loop (dBc is reported
# only where the small-angle approximation holds).
from fracpll import closed_loop_lines, loop_filter_impedance, open_loop
lines = mash_line_spectrum(num=3, den=64, m=1, f_ref_hz=50e6)
zf = lambda w: loop_filter_impedance(w, 100e-12, [(4.7e3, 1.5e-9)])
lg = open_loop(2 * np.pi * lines["f_hz"], 100e-6, 20e6, 40.0, zf)
spurs = closed_loop_lines(lines["f_hz"], lines["line_rad2"], lg)
print(lines["f_hz"][:3], spurs["sideband_dbc"][:3])

# Fractional-N noise including the second-order pulse-width effect,
# per cycle, without the edge-level simulator (new in 0.3):
from fracpll import simulate_sampled
fast = simulate_sampled(icp=100e-6, kvco_hz_per_v=20e6,
                        n_div=40 + 104857 / 2**20, f_ref_hz=50e6,
                        c_shunt=100e-12, branches=[(4.7e3, 1.5e-9)],
                        n_int=40, num=104857, den=2**20, mash_order=3,
                        n_cycles=1 << 15, order=2)
print(fast["psi"].std(), "cycles rms excess phase")
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
  exact sequence, not taken from authority, and oscillator-referred
  (no N^2 in the closed loop). First-order modulation produces
  discrete spurs, not a smooth PSD: `dsm_phase_psd` refuses it and
  `mash_line_spectrum` gives the exact lines.
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
- **`simulate_pll` / `LinearVCO`** (new in 0.2): the edge-accurate
  event-driven loop -- tri-state PFD with reset delay, dead zone
  (a stated behavioural turn-on-delay model), UP/DN mismatch,
  leakage and pump polarity; the filter in exact modal closed form
  between edges; a linear oscillator with exact phase, or a measured
  tuning curve with Gauss-Legendre phase; MASH-driven fractional
  division; cycle-slip counting; optional reference jitter.
- **`sampled_loop_map` / `sampled_stability`** (new in 0.2): the
  exact linearised one-period map of the charge-pump loop for any
  passive filter and any bandwidth (the impulse-invariant model of
  Gardner 1980, built numerically), with its poles.
- **`simulate_sampled` / `pulse_doublet_vector`** (new in 0.3): a fast
  per-cycle loop, exact to first order (`order=1`) or including every
  second-order effect of the finite PFD pulse (`order=2`): the charge
  centroid tau/2 from the reference edge, and the divider firing at the
  oscillator's actual phase. With a delta-sigma divider, order 2
  reproduces the in-band noise of the edge-level simulator that the
  linear map misses (residual 60 dB or more below the effect), about
  five times faster.
- **`pfd_static_offset`** (new in 0.2): the exact steady-state
  offset of a tri-state PFD from charge balance, with mismatch,
  reset delay and dead zone, and refusals where no operating point
  exists (inside a dead zone with nothing to bias out of it).
- **`mash_period` / `mash_line_spectrum` / `closed_loop_lines`**
  (new in 0.2): the exact period and the exact line spectrum of any
  MASH-1/2/3 sequence, and those lines through the closed loop as
  output spurs in dBc.

## Refusals, not guesses

A loop whose crossover violates its own model's validity, a
first-order modulator's spurs, an extrapolated noise slope, a folded
tuning curve, a lock target the oscillator cannot reach, leakage the
pump cannot cancel at any phase: each is refused with the reason and,
where one exists, the remedy.

## How it is checked

57 tests (Python 3.9-3.14, run in CI on every push), every claim
pinned to a closed form, an exact identity, published guidance, or
two independent code paths -- never a stored number. Highlights: the
general filter impedance equals the textbook second-order closed form
at machine precision; 1/(1+L) + L/(1+L) = 1 exactly; the
crossover/margin report matches an independent 400 000-point dense
grid; the MASH mean is asserted with exact rational arithmetic; the
quantization PSD formula matches the Welch spectrum of the exact
sequence AND, band by band within 10 %, the exact line spectrum of a
long-period sequence; the exact first-order lines equal the closed form
of the sawtooth and sum to the variance (Parseval); jitter reproduces
the flat and 1/f^2 closed forms; the PCHIP gain equals a central finite
difference; the measurement-planning error bar matches a seeded Monte
Carlo.

For the edge-level simulator: its steady state equals the exact
charge balance of the tri-state PFD to 1e-6 in five cases (leakage,
mismatch, reset delay, dead zone, both edge orders); its
small-perturbation trajectory follows the exact sampled-data map cycle
by cycle to 1e-5; it decays just below and grows just above the exact
stability boundary, where the averaged model refuses; at low bandwidth
the sampled poles equal exp(sT) of the averaged model's poles; its
measured-curve (quadrature) path equals the exact linear path on a
linear curve to 1e-18 s; its fractional-N lock holds the exact mean
ratio N + num/den; it pulls in through cycle slips; and its delta-sigma
phase noise matches |L/(1+L)|^2 S_dsm in the linear band, as does the
exact linear sampled map driven by the same integer sequence in every
band; and the first-order spurs it produces at the output equal the
exact MASH-1 lines through the closed loop to within 0.5 dB. The
second-order map reproduces the edge-level run: per-cycle pulse widths
to 1e-15 s, time-domain residual more than 1000 times below the linear
map's, every in-band level within 0.5 dB, at two shunt capacitances and
for negative Kvco. The pulse-width effect halves (-6 dB) when the
pulses are made half as wide at the same loop.

## A correction in 0.2.0

fracpll 0.1.0 multiplied the delta-sigma noise by N^2 in
`synthesizer_psd`. That is wrong: the MASH phase PSD is already
oscillator-referred (a divider count error is an oscillator cycle),
as eq. (3.7) of W. Rhee's thesis (University of Illinois, 2001) and the
derivation in `fracpll.mash` show. It overstated delta-sigma noise by
N^2 (32 dB at N = 40). The edge-level simulator measures the corrected
result, and the tests hold it. The PFD-referred in-band floor keeps its
N^2. If you used 0.1.0 with a delta-sigma term, re-run that analysis.

## Honest limits

Deliberate scope, designed out with reasons. The averaged model
(`loop`, `lock`, `noise`) is what it says; where it stops, the
edge-level simulator and the sampled-data map take over, and the
averaged stability report points to them rather than pretending. The
simulator's pump switches instantaneously apart from the stated delays,
its dead zone is a behavioural turn-on-delay model rather than a device
model, and it injects only delta-sigma and user-supplied reference
jitter (other noise enters through `noise` in the frequency domain).
The sampled-data map is small-signal, with matched pumps and no dead
zone; large-signal behaviour belongs to the simulator. The additive
delta-sigma PSD is a continuum model: for short periods the exact line
spectrum can differ from it by tens of percent at low offsets, and
`mash_line_spectrum` shows by how much. With a delta-sigma divider the
edge-level run shows an extra in-band floor, even with matched pumps,
that no linear model contains. Version 0.3 explains it and computes
it: it is the second-order effect of the finite PFD pulse (the charge
centroid sits tau/2 from the reference edge, and tau fluctuates with
the delta-sigma phase), reproduced cycle by cycle by
`simulate_sampled(order=2)`. No device physics, no transistor
models and no PDK data ship with this package: the source study's
foundry PDK files are licensed material and are not redistributed.
Your measured tuning curves and noise points carry the technology, each
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

Apache-2.0. Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22829473](https://doi.org/10.5281/zenodo.22829473),
which always resolves to the latest version.
