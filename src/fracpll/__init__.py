"""fracpll: charge-pump fractional-N PLL design from measured pieces.

The loop, noise, delta-sigma, tuning-curve and lock machinery of a
(fractional-)N synthesizer, distilled from the loop model of the monolithic GaN-on-SOI
PLL study (T. M. Mahim, M. M. Rahman and N. Chowdhury; see the README
for the citation) into a NumPy/SciPy engine that runs on YOUR
measured tuning curves and noise numbers -- and refuses, with an
explanation, wherever its own model stops being trustworthy.  Where
the averaged model stops, version 0.2 goes on: an edge-accurate
event-driven simulator (`simulate_pll`), the exact sampled-data
stability map (`sampled_stability`), the exact tri-state-PFD static
offset (`pfd_static_offset`) and the exact MASH line spectrum
(`mash_line_spectrum`).  Version 0.4 adds the exact per-cycle loop
gain with phase and gain margins at any bandwidth
(`sampled_open_loop`, `sampled_margins`), exact log-log integration of
measured phase noise (`rms_jitter(method="loglog")`,
`NoiseSpec.phase_variance`), readers for measured data files
(`read_noise_csv`, `read_tuning_csv`), and a lock settling time that
no longer depends on the output sampling.  Version 0.5 adds a
straight-line Kvco fit of measured tuning points with its standard
uncertainty and a test of the line (`fit_linear_vco`), and the
propagation of measured input uncertainties (pump current, Kvco,
filter parts) to the loop bandwidth and margins
(`margin_uncertainty`).  Version 0.6 adds the design of a
second-order loop filter for a target bandwidth and phase margin
(`design_loop_filter`) and the third-order filter with an R3-C3
section (`third_order_impedance`) for the averaged model.
"""

from .design import design_loop_filter
from .filters import (loop_filter_impedance, second_order_impedance,
                      third_order_impedance)
from .lock import lock_transient, static_offset
from .loop import (error_transfer, lowpass_transfer, open_loop,
                   stability)
from .eventsim import LinearVCO, SimResult, simulate_pll
from .mash import (MASH_RANGE, dsm_phase_psd, mash_line_spectrum,
                   mash_period, mash_sequence)
from .noise import (NoiseSpec, closed_loop_lines, dbc_to_psd,
                    psd_to_dbc, rms_jitter, synthesizer_psd, white_floor)
from .pfd import pfd_static_offset
from .sampled import (continuous_closed_loop_poles, pulse_doublet_vector,
                      sampled_loop_map, sampled_margins, sampled_open_loop,
                      sampled_stability, simulate_sampled)
from .plan import averages_for_jitter, jitter_relative_sigma
from .readers import read_noise_csv, read_tuning_csv
from .tuning import TuningCurve, TuningFamily, fit_linear_vco, fit_tuning
from .uncertainty import margin_uncertainty

__version__ = "0.6.1"

__all__ = [
    "loop_filter_impedance", "second_order_impedance",
    "third_order_impedance", "design_loop_filter",
    "open_loop", "error_transfer", "lowpass_transfer", "stability",
    "mash_sequence", "dsm_phase_psd", "MASH_RANGE",
    "mash_period", "mash_line_spectrum",
    "NoiseSpec", "white_floor", "synthesizer_psd", "rms_jitter",
    "dbc_to_psd", "psd_to_dbc", "closed_loop_lines",
    "LinearVCO", "SimResult", "simulate_pll", "pfd_static_offset",
    "sampled_loop_map", "sampled_stability",
    "sampled_open_loop", "sampled_margins",
    "continuous_closed_loop_poles", "simulate_sampled",
    "pulse_doublet_vector",
    "fit_tuning", "fit_linear_vco", "TuningCurve", "TuningFamily",
    "lock_transient", "static_offset",
    "jitter_relative_sigma", "averages_for_jitter",
    "read_noise_csv", "read_tuning_csv",
    "margin_uncertainty",
    "__version__",
]
