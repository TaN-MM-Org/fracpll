"""fracpll: charge-pump fractional-N PLL design from measured pieces.

The loop, noise, delta-sigma, tuning-curve and lock machinery of a
(fractional-)N synthesizer, distilled from the monolithic GaN-on-SOI
PLL co-design study into a NumPy/SciPy engine that runs on YOUR
measured tuning curves and noise numbers -- and refuses, with an
explanation, wherever its own model stops being trustworthy.
"""

from .filters import loop_filter_impedance, second_order_impedance
from .lock import lock_transient, static_offset
from .loop import (error_transfer, lowpass_transfer, open_loop,
                   stability)
from .mash import MASH_RANGE, dsm_phase_psd, mash_sequence
from .noise import (NoiseSpec, dbc_to_psd, psd_to_dbc, rms_jitter,
                    synthesizer_psd, white_floor)
from .plan import averages_for_jitter, jitter_relative_sigma
from .tuning import TuningCurve, TuningFamily, fit_tuning

__version__ = "0.1.0"

__all__ = [
    "loop_filter_impedance", "second_order_impedance",
    "open_loop", "error_transfer", "lowpass_transfer", "stability",
    "mash_sequence", "dsm_phase_psd", "MASH_RANGE",
    "NoiseSpec", "white_floor", "synthesizer_psd", "rms_jitter",
    "dbc_to_psd", "psd_to_dbc",
    "fit_tuning", "TuningCurve", "TuningFamily",
    "lock_transient", "static_offset",
    "jitter_relative_sigma", "averages_for_jitter",
    "__version__",
]
