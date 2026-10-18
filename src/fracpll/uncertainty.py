"""How well are the loop bandwidth and the margins known?

The pump current is measured with some calibration uncertainty, Kvco
comes from a fit of a measured tuning curve (`fit_linear_vco` gives its
standard uncertainty), and every resistor and capacitor of the loop
filter has a tolerance.  `margin_uncertainty` carries those input
uncertainties through to the loop bandwidth, the phase margin and (for
the per-cycle model) the gain margin.

Method: the law of propagation of uncertainty for uncorrelated inputs,
eq. (10) of JCGM 100:2008, "Evaluation of measurement data - Guide to
the expression of uncertainty in measurement" (the GUM),

    u_c(y)^2 = sum_i (c_i u_i)^2,

where u_i is the relative standard uncertainty of input x_i and
c_i = dy/d(ln x_i) its sensitivity coefficient.  Here c_i u_i is
evaluated as the central difference (y(x_i (1 + u_i)) - y(x_i (1 - u_i)))/2,
that is, by moving each input by one standard uncertainty either way.
The same two runs also measure how curved y is over that step,
(y(+) + y(-) - 2 y(0))/2; when that is not small next to u_c (by
default more than 10 % of it), the straight-line propagation is not
reliable and the function refuses, pointing to its Monte Carlo option.

Optionally (n_mc > 0) it also propagates the distributions by Monte
Carlo, drawing every input from a normal distribution with its standard
uncertainty, as described in JCGM 101:2008, "Supplement 1 to the GUM -
Propagation of distributions using a Monte Carlo method".  A component
tolerance of +-t with nothing else known about it is commonly treated
as a rectangular distribution, whose standard uncertainty is t/sqrt(3)
(GUM 4.3.7); convert tolerances that way before passing them.

What the tests check: for the second-order filter the sensitivities to
Icp and Kvco equal closed forms derived from |L| = 1 at the crossover;
an exact scaling of the loop (every C times k, every R divided by k,
Icp times k leaves L unchanged) gives a sum rule that the
sensitivities obey; for a fast loop whose phase crossover is at f_ref/2
the gain margin moves by exactly -20 log10(1 + u) dB per relative
change u of Icp; and the Monte Carlo spread agrees with the linear
propagation for small uncertainties.

Honest limits: the inputs are taken as uncorrelated; f_ref and N are
exact.  The answer is only as good as the model behind it
(`fracpll.stability` or `fracpll.sampled_margins`), and inherits their
refusals.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .filters import loop_filter_impedance
from .loop import stability
from .sampled import _polarity, sampled_margins

__all__ = ["margin_uncertainty"]


def _evaluate(model, icp, kv, n_div, f_ref, c0, br, pol):
    if model == "averaged":
        zf = lambda w: loop_filter_impedance(w, c0, br)  # noqa: E731
        r = stability(icp, kv, n_div, zf, f_ref)
        return {"f_crossover_hz": r["f_crossover_hz"],
                "phase_margin_deg": r["phase_margin_deg"]}
    r = sampled_margins(icp, kv, n_div, f_ref, c0, br, pump_polarity=pol)
    if not np.isfinite(r["gain_margin_db"]):
        raise ValueError(
            "the per-cycle loop gain has no phase crossover, so the gain "
            "margin is infinite and has no uncertainty to propagate")
    return {"f_crossover_hz": r["f_crossover_hz"],
            "phase_margin_deg": r["phase_margin_deg"],
            "gain_margin_db": r["gain_margin_db"]}


def margin_uncertainty(icp: float, kvco_hz_per_v: float, n_div: float,
                       f_ref_hz: float, c_shunt: float,
                       branches: Sequence[tuple[float, float]] = (),
                       u_icp: float = 0.0, u_kvco: float = 0.0,
                       u_c_shunt: float = 0.0,
                       u_branches: Sequence[tuple[float, float]] | None
                       = None,
                       model: str = "averaged", pump_polarity: int = 1,
                       n_mc: int = 0, seed: int | None = None,
                       max_curvature: float = 0.1) -> dict[str, Any]:
    """Standard uncertainty of the loop bandwidth and margins.

    icp, kvco_hz_per_v, n_div, f_ref_hz, c_shunt, branches : the loop,
        as for `fracpll.sampled_margins` (for model="averaged" Kvco must
        be positive, as for `fracpll.stability`).
    u_icp, u_kvco, u_c_shunt : RELATIVE standard uncertainties
        (fractions: 0.02 means 2 %), each in [0, 0.25).  For Kvco from
        `fit_linear_vco` use u_kvco_hz_per_v / |kvco_hz_per_v|.
    u_branches : one (u_R, u_C) pair of relative standard
        uncertainties per branch, or None for exact branches.
    model : "averaged" (`fracpll.stability`: bandwidth and phase
        margin) or "sampled" (`fracpll.sampled_margins`: also the gain
        margin, at any loop bandwidth).
    pump_polarity : passed to the per-cycle model (-1 with a negative
        Kvco); the averaged model needs +1 and a positive Kvco.
    n_mc : number of Monte Carlo draws (0 = none).  seed : for the
        draws, so a run can be repeated.
    max_curvature : refusal threshold, as a fraction of u_c (default
        0.1), for the curvature of an output over +-1 standard
        uncertainty of any one input.

    Returns a dict with
        nominal : the model's outputs at the nominal values;
        u : the combined standard uncertainty of each output (same
            units: Hz, degrees, dB);
        contributions : {input: {output: c_i u_i}}, signed, so that
            u^2 is the sum of their squares;
        sensitivity : {input: {output: c_i}}, the change of the output
            per unit relative change of the input (dy/d ln x);
        curvature : {output: largest |y(+) + y(-) - 2 y(0)|/2 over the
            inputs, divided by u_c};
        mc : (only when n_mc > 0) {output: {"mean", "std", "p2.5",
            "p97.5"}} over the draws.
    Inputs are named "icp", "kvco", "c_shunt", and "branch0_r",
    "branch0_c", "branch1_r", ... for the resistor and capacitor of
    each branch (counted from 0).

    Refuses (ValueError) when the nominal loop is refused by the model;
    when moving any one input by one standard uncertainty either way
    takes the loop out of the model (unstable, beyond f_ref/10 for the
    averaged model, no crossover), since a margin is then not a fair
    summary; when the curvature check fails; and when any Monte Carlo
    draw is refused, giving the fraction of such draws, which itself
    estimates how likely the loop is to fail.
    """
    model = str(model)
    if model not in ("averaged", "sampled"):
        raise ValueError("model must be 'averaged' or 'sampled', got "
                         f"{model!r}")
    pol = _polarity(pump_polarity)
    if model == "averaged" and pol != 1:
        raise ValueError("the averaged model takes |Kvco| with "
                         "pump_polarity=+1; use model='sampled' for a "
                         "signed Kvco with pump_polarity=-1")
    icp, kv = float(icp), float(kvco_hz_per_v)
    n_div, f_ref, c0 = float(n_div), float(f_ref_hz), float(c_shunt)
    br = [(float(r), float(c)) for r, c in branches]
    if u_branches is None:
        ub = [(0.0, 0.0)] * len(br)
    else:
        ub = [(float(a), float(b)) for a, b in u_branches]
        if len(ub) != len(br):
            raise ValueError(f"u_branches has {len(ub)} pairs for "
                             f"{len(br)} branches; give one (u_R, u_C) "
                             "pair per branch")
    # (name, nominal value, relative u, setter)
    names = ["icp", "kvco", "c_shunt"]
    us = [float(u_icp), float(u_kvco), float(u_c_shunt)]
    for k, (ur, uc) in enumerate(ub):
        names += [f"branch{k}_r", f"branch{k}_c"]
        us += [ur, uc]
    for name, u in zip(names, us):
        if not (np.isfinite(u) and 0.0 <= u < 0.25):
            raise ValueError(
                f"relative standard uncertainty of {name} must be in "
                f"[0, 0.25), got {u!r}; a larger spread makes "
                "non-physical (negative) values likely, and a straight-"
                "line propagation meaningless")
    mct = float(max_curvature)
    if not (np.isfinite(mct) and mct > 0.0):
        raise ValueError("max_curvature must be finite and positive")
    n_mc = int(n_mc)
    if n_mc < 0 or n_mc == 1:
        raise ValueError(f"n_mc must be 0 (no Monte Carlo) or >= 2, got "
                         f"{n_mc}")

    def run(scales):
        """The model with every input multiplied by its scale."""
        s = iter(scales)
        a_icp, a_kv, a_c0 = icp * next(s), kv * next(s), c0 * next(s)
        a_br = [(r * next(s), c * next(s)) for r, c in br]
        return _evaluate(model, a_icp, a_kv, n_div, f_ref, a_c0, a_br,
                         pol)

    ones = np.ones(len(names))
    y0 = run(ones)
    outs = list(y0)
    contrib = {}
    sens = {}
    curv = {o: 0.0 for o in outs}
    for i, (name, u) in enumerate(zip(names, us)):
        if u == 0.0:
            contrib[name] = {o: 0.0 for o in outs}
            sens[name] = {o: 0.0 for o in outs}
            continue
        ys = []
        for sign in (1.0, -1.0):
            sc = ones.copy()
            sc[i] = 1.0 + sign * u
            try:
                ys.append(run(sc))
            except ValueError as err:
                raise ValueError(
                    f"moving {name} by {'+' if sign > 0 else '-'}1 "
                    f"standard uncertainty ({u:.3g} relative) takes the "
                    f"loop out of the model: {err}. A margin is not a "
                    "fair summary of a loop that can fail within its "
                    "own uncertainty; reduce the uncertainty, or change "
                    "the design") from None
        yp, ym = ys
        contrib[name] = {o: 0.5 * (yp[o] - ym[o]) for o in outs}
        sens[name] = {o: contrib[name][o] / u for o in outs}
        for o in outs:
            curv[o] = max(curv[o], abs(0.5 * (yp[o] + ym[o]) - y0[o]))
    u_c = {o: float(np.sqrt(sum(contrib[n][o] ** 2 for n in names)))
           for o in outs}
    curv_rel = {}
    for o in outs:
        if u_c[o] == 0.0:
            curv_rel[o] = 0.0 if curv[o] == 0.0 else np.inf
        else:
            curv_rel[o] = curv[o] / u_c[o]
        if curv_rel[o] > mct and n_mc == 0:
            raise ValueError(
                f"{o} is too curved over +-1 standard uncertainty for "
                f"linear propagation (curvature {curv_rel[o]:.3g} of "
                f"u_c, above max_curvature = {mct:g}); pass n_mc (for "
                "example 2000) for a Monte Carlo propagation, or "
                "reduce the uncertainties")
    out = {"nominal": y0, "u": u_c, "contributions": contrib,
           "sensitivity": sens, "curvature": curv_rel}
    if n_mc:
        rng = np.random.default_rng(seed)
        draws = 1.0 + rng.standard_normal((n_mc, len(names))) \
            * np.array(us)[None, :]
        if np.any(draws <= 0.0):
            raise ValueError("a Monte Carlo draw gave a non-positive "
                             "component value; the uncertainties are "
                             "too large for a normal distribution")
        vals = {o: np.empty(n_mc) for o in outs}
        failed = 0
        for k in range(n_mc):
            try:
                y = run(draws[k])
            except ValueError:
                failed += 1
                continue
            for o in outs:
                vals[o][k] = y[o]
        if failed:
            raise ValueError(
                f"in {failed} of {n_mc} Monte Carlo draws "
                f"({100.0 * failed / n_mc:.3g} %) the loop leaves the "
                "model (unstable, no crossover, or beyond its validity "
                "range): quote that fraction as an estimate of the "
                "failure probability, not a margin with an uncertainty")
        out["mc"] = {o: {"mean": float(np.mean(v)),
                         "std": float(np.std(v, ddof=1)),
                         "p2.5": float(np.percentile(v, 2.5)),
                         "p97.5": float(np.percentile(v, 97.5))}
                     for o, v in vals.items()}
    return out
