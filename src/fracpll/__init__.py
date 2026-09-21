"""fracpll: charge-pump fractional-N PLL design from measured pieces.

The loop, noise, delta-sigma, tuning-curve and lock machinery of a
(fractional-)N synthesizer, distilled from the monolithic GaN-on-SOI
PLL co-design study into a NumPy/SciPy engine that runs on YOUR
measured tuning curves and noise numbers -- and refuses, with an
explanation, wherever its own model stops being trustworthy.  Where
the averaged model stops, version 0.2 goes on: an edge-accurate
event-driven simulator (`simulate_pll`), the exact sampled-data
stability map (`sampled_stability`), the exact tri-state-PFD static
offset (`pfd_static_offset`) and the exact MASH line spectrum
(`mash_line_spectrum`).
"""

from .filters import loop_filter_impedance, second_order_impedance
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
                      sampled_loop_map, sampled_stability,
                      simulate_sampled)
from .plan import averages_for_jitter, jitter_relative_sigma
from .tuning import TuningCurve, TuningFamily, fit_tuning

__version__ = "0.3.0"

__all__ = [
    "loop_filter_impedance", "second_order_impedance",
    "open_loop", "error_transfer", "lowpass_transfer", "stability",
    "mash_sequence", "dsm_phase_psd", "MASH_RANGE",
    "mash_period", "mash_line_spectrum",
    "NoiseSpec", "white_floor", "synthesizer_psd", "rms_jitter",
    "dbc_to_psd", "psd_to_dbc", "closed_loop_lines",
    "LinearVCO", "SimResult", "simulate_pll", "pfd_static_offset",
    "sampled_loop_map", "sampled_stability",
    "continuous_closed_loop_poles", "simulate_sampled",
    "pulse_doublet_vector",
    "fit_tuning", "TuningCurve", "TuningFamily",
    "lock_transient", "static_offset",
    "jitter_relative_sigma", "averages_for_jitter",
    "__version__",
]
