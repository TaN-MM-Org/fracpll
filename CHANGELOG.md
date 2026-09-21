# Changelog

Every claim added in any release is pinned by a test against a closed
form, an exact identity, two independent code paths, or seeded
simulation against an exact formula; the release notes on GitHub
carry the full anchor lists.

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
