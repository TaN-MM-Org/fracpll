"""Closed-loop phase noise and jitter, from YOUR measured pieces.

The synthesizer output phase noise assembles three paths through the
loop transfers, exactly as in the GaN-on-SOI co-design study this
package is distilled from:

    S_out(f) = |1/(1+L)|^2 S_vco(f)
             + |L/(1+L)|^2 N^2 ( S_inband(f) + S_dsm(f) ),

with S_vco the open-loop oscillator noise, S_inband the in-band
reference/PFD/CP floor referred to the divider input (hence the N^2),
and S_dsm the MASH quantization noise of `fracpll.mash`.  All PSDs in
this package are ONE-SIDED, in rad^2/Hz, and RMS jitter is

    sigma_t = sqrt( integral S_out df ) / (2 pi f0)      [seconds]

-- integrate once; the one/two-sided factor already lives in the PSDs.

House rule: no oscillator noise numbers ship with this package.  A
`NoiseSpec` is built from YOUR measured (offset, dBc/Hz) points and a
mandatory literature-or-lab reference for where they came from; it
interpolates log-log between the points and REFUSES outside the
measured range rather than extrapolating a slope it was never shown.
`white_floor` is the one deliberate modeling shortcut -- a flat floor
from a single measured number -- and says so in its record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .loop import error_transfer, lowpass_transfer

__all__ = ["NoiseSpec", "white_floor", "synthesizer_psd", "rms_jitter",
           "dbc_to_psd", "psd_to_dbc"]


def dbc_to_psd(dbc_per_hz):
    """dBc/Hz -> one-sided rad^2/Hz (small-angle: L(f) = S_phi/1)."""
    return 10.0 ** (np.asarray(dbc_per_hz, dtype=float) / 10.0)


def psd_to_dbc(s_phi):
    """One-sided rad^2/Hz -> dBc/Hz."""
    s = np.asarray(s_phi, dtype=float)
    if np.any(s <= 0.0):
        raise ValueError("PSD values must be positive")
    return 10.0 * np.log10(s)


@dataclass(frozen=True)
class NoiseSpec:
    """Measured phase-noise points with provenance.

    f_offset_hz : measured offsets (Hz), strictly increasing, >= 2 pts.
    dbc_per_hz : measured single-sideband noise at those offsets.
    reference : where the numbers come from -- an instrument and date,
        or a citation.  Mandatory, >= 8 characters; this package does
        not ship uncited noise.
    """
    f_offset_hz: tuple
    dbc_per_hz: tuple
    reference: str
    label: str = ""
    _logf: np.ndarray = field(init=False, repr=False, compare=False,
                              default=None)
    _ldb: np.ndarray = field(init=False, repr=False, compare=False,
                             default=None)

    def __post_init__(self):
        f = np.asarray(self.f_offset_hz, dtype=float)
        d = np.asarray(self.dbc_per_hz, dtype=float)
        if not isinstance(self.reference, str) \
                or len(self.reference.strip()) < 8:
            raise ValueError("a NoiseSpec requires a real reference "
                             "(instrument+date or citation, >= 8 "
                             "characters); this package does not ship "
                             "uncited noise")
        if f.size < 2 or f.size != d.size:
            raise ValueError("need >= 2 (offset, dBc/Hz) points of "
                             "equal length")
        if not np.all(np.isfinite(f)) or np.any(f <= 0.0) \
                or np.any(np.diff(f) <= 0.0):
            raise ValueError("offsets must be finite, positive and "
                             "strictly increasing")
        if not np.all(np.isfinite(d)):
            raise ValueError("dBc/Hz values must be finite")
        object.__setattr__(self, "_logf", np.log10(f))
        object.__setattr__(self, "_ldb", d)

    def psd(self, f_hz):
        """One-sided S_phi (rad^2/Hz), log-log interpolated.

        Refuses outside the measured offset range: this spec was never
        shown a slope there, and will not invent one.
        """
        f = np.atleast_1d(np.asarray(f_hz, dtype=float))
        lo, hi = 10.0 ** self._logf[0], 10.0 ** self._logf[-1]
        if np.any(f < lo * (1 - 1e-12)) or np.any(f > hi * (1 + 1e-12)):
            raise ValueError(
                f"offset outside the measured range [{lo:.4g}, "
                f"{hi:.4g}] Hz of this NoiseSpec ({self.reference}); "
                "measure there, or add points, instead of "
                "extrapolating")
        db = np.interp(np.log10(f), self._logf, self._ldb)
        return dbc_to_psd(db)


def white_floor(dbc_per_hz, reference, f_lo_hz=1.0, f_hi_hz=1e9):
    """A flat noise floor from one measured number.

    This is a deliberate modeling assumption (white over the band),
    recorded as such in the returned spec's label.
    """
    return NoiseSpec((float(f_lo_hz), float(f_hi_hz)),
                     (float(dbc_per_hz), float(dbc_per_hz)),
                     reference=reference,
                     label="white-floor assumption")


def synthesizer_psd(f_hz, lg, n_div, s_vco=None, s_inband=None,
                    s_dsm=None):
    """Assemble the output phase-noise PSD from its pieces.

    f_hz : offset grid (Hz).
    lg : open-loop gain L(j 2 pi f) on that grid (`fracpll.loop`).
    n_div : mean division ratio (multiplies the input-referred paths
        by N^2).
    s_vco, s_inband, s_dsm : one-sided rad^2/Hz arrays on the grid
        (None = that path absent).

    Returns dict(s_out, s_vco_closed, s_inband_closed, s_dsm_closed).
    """
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    lg = np.asarray(lg, dtype=complex)
    if lg.shape != f.shape:
        raise ValueError("lg must be evaluated on the same grid as f")
    n2 = float(n_div) ** 2
    he2 = np.abs(error_transfer(lg)) ** 2
    hl2 = np.abs(lowpass_transfer(lg)) ** 2
    zero = np.zeros_like(f)
    v = zero if s_vco is None else np.asarray(s_vco, dtype=float)
    ib = zero if s_inband is None else np.asarray(s_inband, dtype=float)
    dq = zero if s_dsm is None else np.asarray(s_dsm, dtype=float)
    for name, arr in (("s_vco", v), ("s_inband", ib), ("s_dsm", dq)):
        if arr.shape != f.shape or np.any(arr < 0.0) \
                or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite, >= 0, on the "
                             "same grid as f")
    out = {"s_vco_closed": he2 * v,
           "s_inband_closed": hl2 * n2 * ib,
           "s_dsm_closed": hl2 * n2 * dq}
    out["s_out"] = (out["s_vco_closed"] + out["s_inband_closed"]
                    + out["s_dsm_closed"])
    return out


def rms_jitter(f_hz, s_out, f0_hz):
    """RMS absolute jitter (s): sqrt(int S df) / (2 pi f0).

    One-sided convention -- the integral is taken ONCE.  The tests
    hold this against the closed forms for a flat and a 1/f^2 PSD.
    """
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    s = np.atleast_1d(np.asarray(s_out, dtype=float))
    f0 = float(f0_hz)
    if f.size < 2 or np.any(np.diff(f) <= 0.0):
        raise ValueError("need an increasing offset grid with >= 2 "
                         "points")
    if s.shape != f.shape or np.any(s < 0.0):
        raise ValueError("s_out must be >= 0 on the same grid as f")
    if not (np.isfinite(f0) and f0 > 0.0):
        raise ValueError("f0_hz must be finite and positive")
    var = np.trapezoid(s, f)
    return float(np.sqrt(var) / (2.0 * np.pi * f0))
