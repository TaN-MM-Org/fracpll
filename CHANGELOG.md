# Changelog

Every claim added in any release is pinned by a test against a closed
form, an exact identity, two independent code paths, or seeded
simulation against an exact formula; the release notes on GitHub
carry the full anchor lists.

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
