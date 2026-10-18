# Changelog

Every claim added in any release is pinned by a test against a closed
form, an exact identity, two independent code paths, or seeded
simulation against an exact formula; the release notes on GitHub
carry the full anchor lists.

## 0.7.0 (2026-10-18)

Reference spurs of a locked integer-N loop, the closed-loop bandwidth
and peaking, a fix to a crash, stricter input checks, a faster
per-cycle margin calculation, and a third worked tutorial.

### Added
- `reference_spurs(f_ref_hz, icp, kvco_hz_per_v, n_div, c_shunt,
  branches=(), i_dn=None, i_leak=0.0, t_reset=0.0, t_deadzone=0.0,
  pump_polarity=1, r3=None, c3=None, n_harmonics=5)` (new module
  `fracpll.spurs`): the spurs at k f_ref of a locked integer-N loop
  from leakage, UP/DN mismatch, reset delay and dead zone, with an
  optional R3-C3 section. The locked offset comes from the exact
  charge balance (`pfd_static_offset`), the pulses from the tri-state
  detector's rules, and the control-voltage ripple from the filter's
  transimpedance at each harmonic times the pulses' Fourier
  coefficients; the oscillator phase line is |Kvco| V_k/(k f_ref).
  No small-pulse approximation. Returns the dBc (small-angle rule of
  the package; -inf for an exactly zero line), the line powers, the
  ripple amplitudes and the offset. Refused: a loop that the per-cycle
  model finds unstable (checked with the current of the pump whose
  pulse the phase error moves), with an R3-C3 section a loop that
  `stability` refuses, no locked operating point, positive feedback,
  r3 without c3, and non-finite or non-physical inputs.
- `closed_loop_response(icp, kvco_hz_per_v, n_div, zfilter,
  f_ref_hz, f_lo=None, n_grid=4001)` (in `fracpll.loop`): the -3 dB
  bandwidth and the peaking of L/(1+L) in the averaged model, with the
  crossover and phase margin. Refused: what `stability` refuses, no
  peak inside the band (a type-1 loop), and a -3 dB point above
  f_ref/10.
- `examples/tutorial_reference_spurs.py` (below), README examples 13
  (closed-loop bandwidth and peaking) and 14 (reference spurs).

### Fixed
- `mash_line_spectrum` raised OverflowError ("Python int too large to
  convert to C long", NumPy 2.5.3) for a modulus den >= 2^63, which
  `mash_sequence` and `mash_period` accept: the int64 pattern was
  multiplied by den in NumPy integers. From den = 2^44 on the phase is
  now summed in Python integers. Test case 3 x 2^57 / 2^63, MASH-3:
  before, OverflowError; after, period 128 and lines equal element by
  element to those of 3/64 (the same pattern; also for shifts 58 and
  70 and orders 1 and 2). Under NumPy 1.22.0 the old code ran and
  gave the same lines (checked), so the crash was NumPy 2 only.
- `pump_polarity` was converted with int(): 1.5 ran as +1 and -1.2 as
  -1 without a word in `simulate_pll`, `simulate_sampled`,
  `sampled_margins`, `sampled_open_loop` and `margin_uncertainty`
  (before, `simulate_pll(..., pump_polarity=1.5)` returned the +1 run,
  equal element by element; `margin_uncertainty(..., model="sampled",
  pump_polarity=1.5)` returned uncertainties). Now anything but +1 or
  -1 raises ValueError.
- `simulate_sampled(order=2.5)` ran as order 2; now refused.
  `simulate_pll` with n_cycles 0 or -5 returned an empty run, and
  `simulate_sampled` with n_cycles 0 returned empty arrays and with -3
  raised NumPy's "negative dimensions are not allowed"; both now raise
  "n_cycles must be >= 1".

### Performance
Measured on the shared 2-CPU test machine with OMP_NUM_THREADS=1, as
CPU time (process time; wall time varied too much with the machine
load), median of 25 runs (3 for the Monte Carlo), in 5 rounds that
alternate 0.6.1 and 0.7.0; given as the median ratio and the range
over the rounds.
- `sampled_margins`: the loop map (an eigen-decomposition and a matrix
  exponential) is built once per call instead of once per evaluation
  of the loop gain inside the root finding (about ten) plus once for
  the stability check. Each loop-gain value is computed by the same
  operations as before. README example 8 (two calls): 1.6x (1.2x to
  1.7x); `margin_uncertainty(model="sampled")` with 200 Monte Carlo
  draws: 1.6x (1.6x to 1.8x); `design_loop_filter` (README example
  11): 1.4x (0.9x to 2.0x). Against the 0.6.1 code kept in the tests:
  crossovers and spectral radius to 1e-12 relative, margins to 1e-9
  degrees and dB, on 4 fixed and 25 random loops; on this machine all
  results were identical on 300 loops.

### Changed
- Results: no README example output changes (all twelve earlier
  examples run and compared; the header now says 0.7.0). No result of
  a valid call changes. Inputs listed under Fixed now raise
  ValueError.
- README: two questions in the introduction, examples 13 and 14, the
  third tutorial, new entries in the function list, refusals, checks,
  corrections and limits; 226 tests.
- Packaging: a new `MANIFEST.in` puts `examples/` into the source
  distribution, so the tutorial tests run from it instead of being
  skipped (checked: all 226 tests pass from the unpacked sdist).

### Tutorials
- `examples/tutorial_reference_spurs.py`: synthetic bench readings
  (seeded) of the UP and DN currents and of the leakage, averaged to
  means with standard uncertainties; a filter designed for 100 kHz and
  55 degrees and rounded; the 50 MHz spur -82.6 +- 0.2 dBc with a
  2000-draw Monte Carlo that draws each mean from the t-distribution
  JCGM 101:2008 (Table 1, 6.4.9.2) assigns to the mean of n readings;
  an R3-C3 section (4.7 kOhm, 6.8 pF) brings it to -102.7 +- 0.2 dBc,
  below a -90 dBc limit, for 1.3 degrees of phase margin and 0.07 dB
  more peaking. About 5 s of CPU time.
- `tests/test_tutorials.py` now also runs it, and checks two of its
  printed numbers against independent values: the drop from the R3-C3
  section against the ratio of the two filter transimpedances at
  50 MHz (0.1 dB), and the spur's spread against 20 log10(e) times the
  relative uncertainty of the leakage, scaled for the t-distribution
  (30 %).

### Tests (226 total, 66 new)
- Reference spurs against a time-domain periodic steady state built in
  the test (matrix exponentials of the circuit equations over each
  constant-current piece, Gauss-Legendre Fourier integrals) to 1e-9 of
  the largest harmonic and 1e-8 dB, for second- and third-order
  filters with leakage, mismatch, reset delay and dead zone; against
  the edge-by-edge simulator (locked offset to 1e-8 relative, control
  voltage at the reference edge to 1e-8 of the ripple) in five cases
  including a dead zone with either edge first and a reversed pump;
  the short-pulse limit |Z(j k w_ref)| I_leak (1e-9); exact Kvco^2
  scaling, zero spur for matched pumps without leakage, and the
  mirror symmetry of a reversed pump; the stability check's choice of
  pump current; refusals.
- Closed-loop response against the closed forms of the second-order
  type-2 loop derived in the test (-3 dB frequency 1e-12 relative,
  peaking 1e-9 dB, peak position 2e-6, the last set by the flat
  maximum), 18 loops; against a million-point scan with a separate
  impedance code path for the filters of examples 1 and 12; refusals.
- A regression test for each fix above (each fails with 0.6.1).
- The speedup held against the 0.6.1 code kept in the tests.

## 0.6.1 (2026-10-18)

Documentation only. No result of the package changes.

### Changed

- The study fracpll comes from has a new manuscript: T. M. Mahim,
  M. M. Rahman and N. Chowdhury, "Device-physics design rules for
  wide-temperature enhancement-mode gallium nitride integrated
  circuits, revealed by transistor-anchored machine learning",
  manuscript under review (2026); code and circuit-level results at
  doi:10.5281/zenodo.23284701. The README ("Where it comes from") and
  `CITATION.cff` (`references`) now cite it, and the package docstring
  names it.
- Statements about that study now match its manuscript and released
  code. The README said the study's oscillator has negative Kvco and a
  measured tuning curve; the study's tuning curves come from
  transistor-level simulation, and neither the manuscript nor its
  released code states a negative Kvco, so that remark is gone (also
  from the `open_loop` and `fit_tuning` docstrings). The
  `fracpll.noise` docstring said the noise assembly is "exactly as" in
  the study; the study's loop model multiplies the delta-sigma noise by
  N^2, which fracpll removed in 0.2.0, so it now says so. The
  `lock_transient` docstring no longer attributes its default start
  voltage to the study.

## 0.6.0 (2026-10-17)

Filter design for a target bandwidth and phase margin, the third-order
loop filter in the averaged model, a fix to a wrong stability verdict,
faster core routines, and two worked tutorials.

### Added
- `design_loop_filter(f_c_hz, phase_margin_deg, icp, kvco_hz_per_v,
  n_div, f_ref_hz)` (new module `fracpll.design`): the second-order
  passive filter (C1, R2, C2) whose averaged loop has exactly the
  given unity-gain bandwidth and phase margin, with the loop phase at
  its peak at the bandwidth. W. O. Keese, "An Analysis and
  Performance Evaluation of a Passive Filter Design Technique for
  Charge Pump Phase-Locked Loops", National Semiconductor Application
  Note AN-1001 (May 1996), eqs. 8-12. It returns `c_shunt` and
  `branches` ready for every model, the time constants, the read-back
  by `stability`, and the exact per-cycle margins (`sampled_margins`).
  Refused: a bandwidth at or above f_ref/10 (the averaged model's
  limit), a phase margin outside (0, 90) degrees, and non-finite or
  non-positive inputs.
- `third_order_impedance(w, c1, r2, c2, r3, c3)`: the transimpedance
  (control voltage at C3 per pump current) of the third-order filter
  of AN-1001, C1 || (R2 + C2) followed by R3 in series to C3,
  Z = (1 + s R2 C2) / (s (A0 + A1 s + A2 s^2)) with A0 = C1 + C2 + C3,
  A1 = R2 C2 (C1 + C3) + R3 C3 (C1 + C2), A2 = C1 C2 C3 R2 R3 (nodal
  analysis). For the averaged model (`open_loop`, `stability`,
  `synthesizer_psd`); the per-cycle and edge models do not take it.
- `examples/` with two tutorials (below), and README examples 11
  (filter design) and 12 (third-order filter).

### Fixed
- `stability` returned 180 + arg L with arg L in (-180, 180] (NumPy's
  angle), so a loop whose phase had passed -180 degrees got a "phase
  margin" above 180 degrees and was reported as stable. Not reachable
  with `loop_filter_impedance` filters (their loop phase stays above
  -180 degrees), but reachable with any other `zfilter`. Test case:
  third-order filter C1 = 100 pF, R2 = 4.7 kOhm, C2 = 1.5 nF,
  R3 = 20 kOhm, C3 = 1 nF, Icp = 100 uA, Kvco = 20 MHz/V, N = 40,
  50 MHz: before, f_c = 17839.6 Hz and phase margin 336.85 degrees;
  after, the margin is -23.15 degrees and the loop is refused as
  unstable (its closed-loop characteristic polynomial has a root in
  the right half plane, checked in the test). The margin is now
  reported in (-180, 180], the convention of `sampled_margins`.
- `simulate_pll` with `ref_jitter_s` of exactly `n_cycles + 1`
  entries (the length the input check asks for) raised IndexError
  after the last reference edge: entry `n_cycles + 1` was read for an
  edge the run never reaches. Before, IndexError ("index 201 is out
  of bounds for axis 0 with size 201" for 200 cycles); after, the run
  completes and equals, bit for bit, the run with one extra entry.
  The docstring now says which entry moves which edge (entry k moves
  the edge at k / f_ref; entry 0 is unused).
- `TuningFamily` accepted temperature keys that are numbers written as
  text (`{"300": c0, "350": c1}`) and then raised KeyError on every
  evaluation. They are now read as numbers; the result equals the
  family with float keys and the linear interpolation formula (checked
  bit for bit). A temperature given twice is refused.
- Inputs that gave NaN or a non-physical result now raise ValueError:
  a NaN temperature in `TuningFamily.frequency`/`kvco` (returned NaN),
  a NaN offset in `NoiseSpec.psd` (returned NaN), a NaN in
  `simulate_sampled(u_cycles=...)` (NaN from that cycle on), a NaN
  `n_div` or loop gain in `synthesizer_psd` and a NaN in
  `closed_loop_lines` (returned NaN), a negative capacitor in
  `continuous_closed_loop_poles` (returned the poles of the
  non-physical filter, one at +2.22e6 rad/s) and a NaN `icp` there
  (NumPy's LinAlgError), and `sampled_margins(n_grid=1)` (refused with
  the misleading reason that |G| does not fall through 1).

### Performance
Measured on the shared 2-CPU test machine with OMP_NUM_THREADS=1: for
each routine the median of 5 to 25 runs, repeated in 3 rounds that
alternate 0.5.0 and 0.6.0; given as the median ratio and the range
over the rounds (the machine load varied).
- `mash_sequence`: the accumulators in closed form, vectorized in
  blocks of 65536 steps (exact int64 below den = 2^44; above that the
  integer loop of 0.5.0 runs). 2^16 + 2048 MASH-3 steps: 5.9x
  (4.4x to 11.5x). Equal element by element to the 0.5.0 loop.
- `mash_period`: the same closed form, searched block by block.
  104857/2^20, MASH-3 (period 2^21): 37x (18x to 61x), 2.3 s to
  0.06 s. Equal to the 0.5.0 state loop. `mash_line_spectrum` gains
  through both.
- `stability`: Brent's method on ln|L| against ln f instead of 80
  geometric bisection steps (5 or 6 filter evaluations instead of
  80, on 40 random loops). README example 1: 11.5x (7.2x to 15.8x); `margin_uncertainty`
  with 2000 Monte Carlo draws (README example 10): 5.9x (4.6x to
  7.1x). Against the bisection, kept in the tests, on 40 random loops:
  crossover to 1e-12 relative and phase margin to 1e-9 degrees
  (observed 3.8e-15 and 8.5e-14 on 300 loops).
- `simulate_sampled`: the cycle loop on Python floats instead of
  small NumPy arrays; the same operations in the same order except the
  summation order inside Phi s. README example 7 (order 2, 67584
  cycles): 3.2x (2.8x to 5.7x). Against the 0.5.0 loop, kept in the
  tests: psi and tau to 1e-12 of their largest value (observed up to
  1.9e-13), orders 1 and 2, one- and two-branch filters, both pump
  polarities.

### Changed
- Results: no README example output changes (all twelve examples
  run and compared). `stability`'s crossover and margin change only at
  the level stated under Performance. Inputs listed under Fixed now
  raise ValueError.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.
- README: the design question in the introduction, examples 11 and
  12, a Tutorials section, new entries in the function list,
  refusals, checks, corrections and limits, 160 tests.

### Tutorials
- `examples/tutorial_design_and_margins.py`: synthetic tuning points
  (seeded) to Kvco with its uncertainty, a filter designed for 100 kHz
  and 55 degrees, parts rounded to two digits and rechecked in both
  models, and the bandwidth (100.3 +- 1.8 kHz) and phase margin
  (54.9 +- 0.6 degrees) with a 2000-draw Monte Carlo check. About
  2 s.
- `examples/tutorial_jitter_from_trace.py`: a third-order loop, a
  synthetic output trace averaged 16 times read from CSV, the RMS
  jitter 10.98 +- 0.12 ps against the model's 10.99 ps, and the
  averages for a 0.5 % error bar (79). About 1 s.
- The printed output of both is in the README and in each script's
  docstring; `tests/test_tutorials.py` runs them and compares.

### Tests (160 total, 38 new)
- Filter design: equal to a separate derivation in the test (half-angle
  form of the phase peak, |L| = 1 on `second_order_impedance`) to
  1e-12; |L| = 1 and the target margin to 1e-10 degrees at the target
  bandwidth, with the phase at its peak; read back by `stability` to
  1e-9 (five designs, 1 kHz to 2.5 MHz, 10 to 89 degrees); per-cycle
  margin equal to the target for a slow loop (1e-6 degrees); refusals.
- Third-order filter: equal to the polynomial form (1e-11), a
  three-node nodal-matrix solve (1e-9) and the R3 = 0 limit (1e-12);
  its loop margin equals the polynomial's phase at the crossover, and
  its closed-loop poles lie in the left half plane; refusals.
- A regression test for each fix above (each fails with 0.5.0).
- Speedups held against the 0.5.0 code kept in the tests.
- The tutorials print their documented output.

## 0.5.0 (2026-10-10)

A Kvco fit with uncertainties, propagation of measured uncertainties
to the loop bandwidth and margins, and three fixes to silent errors.

### Added
- `fit_linear_vco(vc, f_hz, reference, f_sigma_hz=None,
  vc_window=None, p_min=1e-3)`: straight-line least-squares fit
  f = f0 + Kvco Vc of measured tuning points, weighted by 1/sigma^2
  when reading uncertainties are stated, otherwise with uncertainties
  from the residual scatter (n - 2 degrees of freedom). Returns Kvco,
  f0, their standard uncertainties and covariance, the fitted
  frequency at the mean voltage (uncorrelated with the slope), the
  residuals, a test of the line and a ready `LinearVCO`. The line is
  tested by chi-square (stated sigmas) or by an F-test for a quadratic
  term (no sigmas), and refused below `p_min`. Also refused: fewer
  than 3 points (4 without sigmas), non-finite or non-increasing data,
  non-positive sigmas, and exactly collinear points with no stated
  sigma. Standard least squares; the docstring cites Bevington and
  Robinson, "Data Reduction and Error Analysis for the Physical
  Sciences", 3rd ed. (2003), chapters 6 and 11.
- `margin_uncertainty(...)` (new module `fracpll.uncertainty`): the
  standard uncertainty of the loop bandwidth and phase margin
  (`model="averaged"`, via `stability`) or also of the gain margin
  (`model="sampled"`, via `sampled_margins`, any bandwidth), from
  relative standard uncertainties of Icp, Kvco, `c_shunt` and each
  branch R and C. Law of propagation of uncertainty for uncorrelated
  inputs (JCGM 100:2008, eq. (10)), with each sensitivity taken as a
  central difference over +-1 standard uncertainty; each input's
  signed contribution is returned. Optional seeded Monte Carlo
  (JCGM 101:2008). Refuses when +-1 standard uncertainty of any input
  takes the loop out of the model, when the curvature over that step
  exceeds `max_curvature` (default 10 %) of the combined uncertainty
  and no Monte Carlo is asked for, and when any Monte Carlo draw fails
  (the message gives the failed fraction).
- Type hints on every public function and method.

### Fixed
- `simulate_pll`: the delta-sigma divider pattern was generated for
  `n_cycles + 8` divider edges, and later edges fell back to the
  integer ratio N without warning. A pull-in from above makes one
  extra divider edge per cycle slip, so after more than 8 slips the
  end of the record was wrong. Test case (N = 40 + 104857/2^20,
  MASH-3, 50 MHz, start 3 V above lock, 12000 cycles, 60 slips):
  before, the last 52 counts used N (35 differed from the pattern),
  the excess phase drifted by -2.4e-3 cycles per cycle over the last
  50 cycles and v_c ended at 0.47537 V; after, all 12059 counts equal
  an independently generated MASH-3 pattern, the drift is 1.5e-6 cycles
  per cycle and v_c ends at 0.49981 V (lock voltage 0.50000 V). The
  pattern is now regrown from the zero state when needed; runs with 8
  or fewer slips are bit-identical to 0.4.0 (checked on three runs).
- `simulate_pll` accepted a MASH pattern that asks the divider for
  fewer than one oscillator cycle. n_int = 1, MASH-3, fraction 1/4:
  before, counts down to -1 and 63 reported cycle slips; n_int = 2:
  counts of 0. Now a ValueError names the count and the edge.
- `averages_for_jitter` returned n + 1 when the target equaled the
  error bar of exactly n averages: ceil((sigma_1/target)^2) rounded
  a ratio of 8 (plus one unit in the last place) up to 9. On the test
  grid, target 0.01888422011962381 (the error bar of 8 averages):
  before n = 9 (achieved 0.0178042), after n = 8 (achieved
  0.0188842). For targets 1 to 59 averages, 16 of 59 were off by one
  before. The estimate is now corrected against
  `jitter_relative_sigma` itself.
- Input checks. `simulate_pll` with `f_ref_hz = NaN` never returned
  (no event time compared true); NaN `icp` returned a run with no
  divider edges and NaN v_c. `rms_jitter`, `psd_to_dbc` and
  `jitter_relative_sigma` returned NaN for a NaN in the spectrum or
  grid. `sampled_stability` with NaN `icp` raised NumPy's LinAlgError.
  `simulate_sampled` ran with a negative `icp`, and with
  `pump_polarity=2` silently doubled the loop gain. `stability` with
  `n_grid=1` said that |L| does not cross unity. All now raise
  ValueError with the offending values in the message.

### Changed
- Inputs listed under Fixed that used to return a result (or hang)
  now raise ValueError. No result of a valid call changes: the README
  examples print the same values, and a bit-for-bit comparison with
  0.4.0 (three `simulate_pll` runs, `simulate_sampled`,
  `sampled_margins`, `stability`, `rms_jitter`, `averages_for_jitter`
  at five targets) found no difference.
- README: example 10 (measured tuning points to a phase margin with an
  error bar), new entries in the function list, refusals, checks and
  limits, the 0.5.0 corrections, 122 tests; American spelling in the
  prose that was touched.

### Tests (122 total, 32 new)
- Regression tests for each fix above; each fails with 0.4.0 (the
  NaN `f_ref_hz` case hangs with 0.4.0).
- `fit_linear_vco`: closed-form uncertainties of an exact line; equal
  to `numpy.polyfit(cov="unscaled")` (weighted) and
  `scipy.stats.linregress` (residual scatter); chi-square p-value vs
  exp(-chi2/2) for 2 degrees of freedom; F-test vs the Student-t
  closed form 1 - |t|/sqrt(2 + t^2) for (1, 2) degrees of freedom;
  3000-trial seeded Monte Carlo of the slope scatter within 5 %;
  window fit of a parabola equal to its derivative at the window
  center; refusals; the fitted `LinearVCO` locks `simulate_pll` at the
  fitted lock voltage to 1e-9 V.
- `margin_uncertainty`: Icp and Kvco sensitivities vs the closed form
  from |L| = 1 (S = d ln|L|/d ln f) to 1e-6; the impedance-scaling sum
  rule s_Icp + sum s_C - sum s_R = 0 to 1e-6; the fast-loop gain
  margin step equal to -20 log10(e) atanh(u) dB to 1e-9; per-cycle vs
  averaged uncertainties within 0.2 % for the slow loop; 1000-draw
  Monte Carlo within 10 % of the linear result; the curvature value
  vs direct `stability` calls; the failed-draw count reproduced from
  the same seeded draws; refusals.

## v0.4.0 - 2026-09-30

Stability margins at any loop bandwidth, exact integration of measured
phase noise, readers for measured data files, and a fix to the lock
settling time.

### Added
- `sampled_open_loop`: the exact per-cycle open-loop gain
  G(z) = (pol Icp/(N f_ref)) e_psi^T (zI - Phi)^-1 Phi B of the linear
  sampled map, on the unit circle for 0 < f <= f_ref/2. The closed
  loop from divider phase to sampled oscillator phase is G/(1+G).
- `sampled_margins`: phase margin, gain margin, both crossover
  frequencies, and the eigenvalue stability flag, at any bandwidth
  (the averaged `stability` refuses beyond f_ref/10). A sampled loop
  has a finite gain margin: G is real at z = -1. It refuses when
  |G| >= 1 at f_ref/2. README example 5's loop, reported as stable,
  has 2.06 degrees of phase margin and 0.18 dB of gain margin.
- `rms_jitter(..., method="loglog")`: exact integration between
  points joined by straight lines on log-log axes (a power law per
  segment). On four decade points (10 kHz to 10 MHz, -65/-92/-115/
  -135 dBc/Hz, 2 GHz) it gives 4.869 ps where the trapezoid gives
  13.579 ps. The default stays "trapezoid".
- `NoiseSpec.phase_variance(f_lo, f_hi)`: the exact integral (rad^2)
  of a spec's own log-log interpolation, with no grid.
- `read_noise_csv`, `read_tuning_csv` (new module `fracpll.readers`):
  a `NoiseSpec` or a fitted `TuningCurve` from a delimited text file.
  Separator detected per line (; , tab or whitespace), header lines
  skipped, columns and frequency unit selectable, UTF-8 byte-order
  mark accepted. Text inside the data, non-increasing x, and
  non-finite values are refused with the line number. No instrument
  format is assumed; a decimal comma is not supported.

### Fixed
- `lock_transient`'s `t_settle` was rounded up to the next output
  sample: the last sample outside the band plus one. With the default
  2000 samples over t_end = 31.8 ms (15.9 us apart), the README
  example reported 31.8 us, exactly two samples. It is now the root of
  |phi(t) - phi_static| = settle_tol_rad on the solver's dense
  output (scanning each solver step at its ends and 8 interior
  points, then Brent's method), independent of `n_eval`.
- `lock_transient` accepted `vc0` outside the measured tuning range
  (the curve does not extrapolate; the run held the oscillator at the
  edge frequency), a non-positive `c_shunt` or branch capacitor, a
  non-positive `settle_tol_rad` or `t_end`, `n_eval < 2`, a
  non-finite or non-positive `f_ref_hz`, and `n_div < 1`. These are
  now refused.
- `lock_transient`'s docstring said the default `t_end` was "400
  reference-scaled loop time constants"; the code has always used
  1e5 N / (2 pi f_ref) s (31.8 ms for N = 100 at 50 MHz). The
  docstring now states the formula; the default is unchanged.
- `pfd_static_offset` returned NaN for NaN inputs and raised
  ZeroDivisionError for `f_ref_hz = 0` (the division ran before the
  check); both now raise ValueError.
- `stability` with `f_lo >= f_hi` (or `f_lo <= 0`) scanned the band
  backwards and refused with a misleading "crosses unity only upward"
  message; it now says the band is invalid.

### Behaviour changes
- `lock_transient(...)["t_settle"]` changes. README example 3:
  31.847 us before, 28.985 us after (printed 31.8 -> 29.0 us). The
  pre-0.4.0 value is always at most one output sample above the new
  one (asserted). Everything else it returns (`t`, `phi_e`, `vc`,
  `locked`, `phi_static`) is bit-identical to 0.3.1 for the same
  inputs (checked by hand on the README example, not by a test).
- Inputs listed under Fixed that used to return a result now raise
  ValueError.

### Tests (90 total, 30 new)
- `sampled_open_loop` vs the aliased averaged gain
  sum_m L(j(w + m 2 pi f_ref)) (the 1/w^2 tail summed in closed form)
  to 1e-10 relative, slow and fast loop, including z = -1; vs the
  closed form -Icp Kv T^2 / (4 N C sin^2(pi f T)) of a capacitor-only
  filter to 1e-9, whose phase margin is 0 (to 1e-9 degrees) and whose
  poles sit at exp(+-j 2 pi f_c T) (to 1e-9). For that filter G is
  real and negative at every frequency, so the phase crossover is the
  crossover itself and the gain margin 0 dB (to 1e-9 dB), checked with
  41, 401 and 4001 scan points. (During review, the scan was found to
  start at the grid point after the crossover, which here gave a
  grid-dependent margin; it now starts at the crossover. In a scan of
  several hundred one- and two-branch loops, none had a phase
  crossover other than f_ref/2 or this degenerate one.)
- Margins by eigenvalues: rotating the loop gain by the phase margin,
  or scaling it by the gain margin, puts an eigenvalue of the modified
  one-period map at exp(j 2 pi f T) of the matching crossover, to
  1e-8. Icp x gain margin equals the Icp at which the spectral radius
  reaches 1 (found by root finding on the eigenvalues) to 1e-9. Slow
  loop: crossover within 1e-4 and phase margin within 0.01 degrees of
  the averaged `stability`; negative Kvco with pump_polarity=-1 gives
  the same margins to 1e-12.
- Time domain: a sinusoidal divider phase through
  `simulate_sampled(order=1)` gives Re(A G/(1+G) e^{j theta k}) to
  1e-9; MASH-1 tones through `closed_loop_lines` with the per-cycle
  gain equal the exact DFT of the per-cycle simulation to 1e-6, in a
  fast loop `stability` refuses.
- Lock: in the small-signal limit (linear curve, start at the lock
  voltage, 1e-3 rad step, no leakage) the trajectory equals expm(A t)
  of the hand-built linear system to 1e-5 of the step (1.7e-6
  observed) and `t_settle` its exact band exit to 3e-5 (5e-6
  observed). `t_settle` with 2000 and 20000 samples agrees to 1e-9,
  and a separate integration stopped exactly at `t_settle` puts the
  phase on the band edge to 1e-6 rad (2e-9 observed; a 1e-4 relative
  error in `t_settle` would give 2.1e-5 rad), outside it 1e-4 earlier,
  and 0.017 rad inside it at the pre-0.4.0 value.
- Log-log integration vs the closed-form power-law integral for
  slopes -3, -2, -1, -1+1e-13 (against a Taylor series, since the
  textbook form cancels there), 0 and 1.5, to 1e-12;
  `NoiseSpec.phase_variance` vs a 400001-point trapezoid of the spec
  to 1e-8 and vs a hand-summed piecewise power law to 1e-12.
- Readers: 17-digit values read back bit for bit with each separator,
  header, byte-order mark, column choice and unit scale; a loaded
  power-law trace integrates to the closed form to 1e-12; refusals
  name the line.
- Input refusals listed under Fixed.
- The full suite also passes with NumPy 1.22.0 and SciPy 1.8.0
  (run locally on Python 3.10; CI runs this job on Python 3.9).

### Changed
- README: examples 8 (margins of a fast loop) and 9 (noise from a
  file, integrated exactly), example 3's output, the list of checks,
  refusals and limits; 90 tests.
- Honest labels: README examples 2, 3 and 4 gave their illustrative
  numbers `reference` strings naming an instrument and a notebook date
  ("R&S FSWP, open-loop VCO, notebook 2026-09-18", "PFD+CP floor,
  notebook 2026-09-18", "VNA + SMU sweep, notebook 2026-09-18") or a
  datasheet ("VCO datasheet rev B, Kvco at 25 C"). Those numbers were
  never measured; the strings now say "illustrative ..., not a
  measurement" (or "not a datasheet"). Test fixtures that did the same
  now say "synthetic ..., fracpll test suite". No printed result
  changes.

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
