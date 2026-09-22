# Changelog

Every claim added in any release is pinned by a test against a closed
form, an exact identity, two independent code paths, or seeded
simulation against an exact formula; the release notes on GitHub
carry the full anchor lists.

## v0.3.1 - 2026-09-22

Three code fixes found in a review of 0.3.0, two corrections to
earlier notes, one missing test added, and a plain-language README.

### Fixed
- `dbc_to_psd` / `psd_to_dbc` used S_phi = 10^(L/10). The standard
  definition is L(f) = S_phi(f)/2 (IEEE Std 1139), and the rest of the
  package already used it: `dsm_phase_psd` (W. Rhee's eq. 3.7 is
  printed as L(f)) and the dBc of `mash_line_spectrum` /
  `closed_loop_lines`. Measured oscillator noise and `white_floor`
  floors were therefore 3 dB low next to the delta-sigma term, and
  jitter from them too small by sqrt(2) (the README example: 7.65 ps
  before, 10.82 ps after). Now S_phi = 2 x 10^(L/10). Re-run any
  result that converted measured dBc/Hz with 0.3.0 or earlier.
- `open_loop`, `stability` and `continuous_closed_loop_poles` accepted
  a negative Kvco and returned results for a positive-feedback loop
  (e.g. a 234 degree "phase margin"). They now refuse it: pass |Kvco|,
  which is what a loop with a reversed pump sees. `lock_transient`
  returned an unlocked run for a falling (negative-Kvco) tuning curve;
  it now refuses and points to `simulate_pll(..., pump_polarity=-1)`.
- `rms_jitter` called `np.trapezoid`, which exists only from NumPy
  2.0, while the package declares NumPy >= 1.22. It now falls back to
  `np.trapz`. CI adds Python 3.10 (missing from the matrix before)
  and a job on the oldest allowed NumPy 1.22.0 and SciPy 1.8.0.
- The v0.2.0 notes and the `fracpll.sampled` docstring listed "sampled
  poles vs exp(sT) of the averaged poles" as a test. It had only been
  run as a separate script. It is now a test.
- Wording: the v0.3.0 notes below (and the `fracpll.eventsim`
  docstring) implied the 1e-15 s pulse-width and 0.5 dB in-band checks
  also ran with negative Kvco. For negative Kvco the test checks only
  the more-than-1000x time-domain match.

### Tests (60 total)
- Sampled poles vs exp(sT) of the averaged poles, to 1e-3 of their
  distance from z = 1.
- One dBc convention: psd_to_dbc of `dsm_phase_psd` equals Rhee's
  printed L(f) to 1e-10 dB, the conversions invert each other, and the
  line dBc uses the same rule.
- The negative-Kvco refusals.
- The lock-transient test is tightened from 0.05 rad to 1e-6 rad on
  the static offset (0.0628 rad), and from 1e-3 to 1e-9 on the final
  frequency.

### Changed
- README rewritten in plain language: a short guide to the terms, a
  table of the three models and when to use each, seven runnable
  examples each with the output it prints (checked by running them),
  the refusals, and what each test compares against.
- Docstrings added to `LinearVCO.frequency` and
  `TuningFamily.frequency` / `.kvco`; units added to `static_offset`
  and `jitter_relative_sigma`.

## v0.3.0 - 2026-09-21

The in-band effect that 0.2.0 reported without a mechanism is now
explained, computed and tested.

### Explained
- With a delta-sigma divider, the edge-level loop carries in-band noise
  that the exact linear sampled map misses, even with matched pumps.
  It is the second-order effect of the finite PFD pulse. A pulse is a
  rectangle of charge q = pol Icp tau whose centroid sits tau/2 from
  the reference edge. To second order that adds
  -(pol Icp tau^2/2) M B to the loop state. Because the delta-sigma
  divider makes tau fluctuate, tau^2 folds shaped noise into the band.
- The in-pulse ramp (Kvco Icp tau^2/(2 C_shunt)) and the filter's
  charge redistribution nearly cancel. The lasting part is
  Kvco Icp tau^2/(2 C_total), which is why the effect does not depend
  on C_shunt.
- How it was found: no fitted parameters. The two halves of the pulse
  correction are each 20 dB off alone and correct together. The
  second-order divider-timing terms are negligible.
- 0.2.0's docs described the effect as a flat in-band floor. With finer
  spectral resolution it peaks around the loop bandwidth and falls away
  on both sides. The flat appearance came from coarse Welch bins.

### Added
- `sampled.simulate_sampled(order=1|2)`: a fast per-cycle loop. Order 2
  includes every second-order pulse-width effect and is computed
  self-consistently.
- `sampled.pulse_doublet_vector`.

### Anchors (7 new tests, 57 total)
- Order 2 vs the edge-level simulator: per-cycle tau to 1e-15 s,
  residual more than 1000x below the linear map's, and in-band levels
  within 0.5 dB, at C_shunt 100 and 400 pF and with negative Kvco.
- Order 1 equals the linear sampled map exactly.
- The doublet's oscillator row equals Kvco/C_shunt at t = 0+ and
  Kvco/C_total at long times.
- The effect drops about 6 dB when the pulse width halves at the same
  loop, in both the edge-level run and the order-2 map.

## v0.2.0 - 2026-09-21

Beyond the averaged model, and one correction.

### Fixed
- `synthesizer_psd` multiplied the delta-sigma PSD by N^2. The MASH
  phase PSD is already oscillator-referred, so it reaches the output
  through |L/(1+L)|^2 alone. 0.1.0 overstated delta-sigma noise by N^2
  (32 dB at N = 40). Confirmed three ways: the derivation, eq. (3.7) of
  W. Rhee's thesis (University of Illinois, 2001), and the new
  edge-level simulator. The PFD-referred in-band floor keeps its N^2.
- The `fracpll.lock` docstring claimed a test against a frequency-domain
  step-response synthesis that did not exist. The claim is removed.

### Added
- `eventsim.simulate_pll` / `LinearVCO`: an edge-accurate event-driven
  loop. It has a tri-state PFD with reset delay, a dead zone (a stated
  behavioural turn-on-delay model), UP/DN mismatch, leakage and pump
  polarity. The filter is solved in exact modal closed form between
  edges, with time tracked relative to the current reference edge
  (~1e-24 s resolution on long runs). Oscillator phase is exact for a
  linear VCO and uses Gauss-Legendre quadrature for a measured curve.
  Also: MASH-driven fractional division, cycle-slip counting and
  optional reference jitter.
- `sampled.sampled_loop_map` / `sampled_stability` /
  `continuous_closed_loop_poles`: the exact linearised sampled-data
  loop for any passive filter and any bandwidth. The averaged
  stability refusal now points to it.
- `pfd.pfd_static_offset`: the exact tri-state PFD steady state from
  charge balance, with refusals.
- `mash.mash_period` / `mash_line_spectrum`, `noise.closed_loop_lines`:
  the exact period and line spectrum of any MASH-1/2/3 sequence, and
  its lines at the output. dBc is reported only where the small-angle
  approximation holds.
- `dsm_phase_psd` now cites the printed form of its formula and states
  its oscillator referral. Its m = 1 refusal points to the exact lines.

### Anchors (24 new tests, 50 total)
- Steady state vs exact charge balance: 5 cases, to 1e-6.
- Simulator vs sampled map, cycle by cycle, to 1e-5.
- Exact stability boundary, from both sides, where the averaged model
  refuses.
- Sampled poles vs exp(sT) of the averaged poles.
- Quadrature path vs exact linear path, to 1e-18 s.
- Exact fractional mean ratio.
- Pull-in through cycle slips, and negative Kvco.
- Delta-sigma noise through the loop without N^2: three routes.
- Mismatch noise folding and its offset-current cure, qualitatively
  (Lin, Ti and Liu, IEEE Trans. Circuits Syst. I 56, 877 (2009)).
- MASH-1 output spurs, edge-level vs exact lines, within 0.5 dB.
- Exact MASH-1 lines vs the sawtooth closed form, and Parseval.
- Long-period exact lines vs the additive formula, within 10 % band
  by band.

### Reported, not claimed
- With a delta-sigma divider, the edge-level loop shows an extra
  in-band floor even with matched pumps. It is converged in time
  resolution, independent of the Welch window, and shrinks with pulse
  width (asserted). Its mechanism is not identified here.

## v0.1.0 - 2026-09-18

First release: the charge-pump fractional-N loop, noise, delta-sigma,
tuning-curve, lock and measurement-planning machinery distilled from
a monolithic GaN-on-SOI HEMT-varactor PLL co-design study,
generalized to any technology that fits a charge-pump loop.

- `filters`: the general passive loop-filter one-port
  (shunt C + series-RC branches) and the textbook second-order
  closed form kept as an independent test path.
- `loop`: L = Icp Kvco Z / (jw N), the exact complementary
  transfers, and a crossover/phase-margin report with validity
  refusals (no crossover, non-positive margin, crossover past
  f_ref/10 -- Gardner, IEEE Trans. Commun. 28, 1849 (1980)).
- `mash`: exact integer-accumulator MASH 1/2/3 division sequences
  (mean offset EXACTLY num/den, asserted with rational arithmetic;
  closed per-order ranges) and the one-sided quantization phase PSD,
  validated against the Welch spectrum of the exact sequence and
  refused at first order (discrete spurs).
- `noise`: measured `NoiseSpec` points with mandatory provenance and
  no extrapolation, the three-path closed-loop noise assembly, and
  one-sided RMS jitter held against flat and 1/f^2 closed forms.
- `tuning`: monotone PCHIP tuning curves through measured (Vc, f)
  points (negative Kvco first-class), analytic-derivative gain,
  linear temperature interpolation between measured corners, and
  refusals for folded curves (offending bias named) and any
  extrapolation.
- `lock`: the averaged nonlinear lock transient (tanh detector,
  leakage, mismatch) on the measured curve, tested to settle to the
  closed-form static offset 2 pi atanh(I_leak/Icp - mismatch) and
  onto N f_ref exactly; refusals for unreachable targets and
  uncancelable leakage.
- `plan`: the closed-form error bar of an averaged jitter
  measurement and its exact 1/sqrt(n) inversion, held against a
  seeded Monte Carlo with exponential periodogram statistics.
- 26 tests, Python 3.9-3.14.
