"""Read measured data from delimited text files (CSV and similar).

Phase-noise analyzers and bench scripts save traces as text: a few
header lines, then one row per point.  `read_noise_csv` turns such a
file into a `NoiseSpec`, and `read_tuning_csv` turns a (control
voltage, frequency) file into a fitted `TuningCurve`.  Both still need
the mandatory `reference` saying where the numbers come from.

Reading rules (stated because a silently misread file is worse than a
refused one):

* blank lines and lines starting with '#' are skipped anywhere;
* lines BEFORE the first numeric row are treated as a header and
  skipped;
* once the data has begun, a row whose chosen columns are not numbers
  is REFUSED with its line number, rather than silently ending the
  data there;
* the delimiter is detected per line (';', then ',', then tab, else
  whitespace) unless you pass one;
* a decimal comma ("1,5", common in files saved with European
  settings) is read with `decimal=","` (since 0.9.0).  The comma then
  cannot also separate the columns: the delimiter is detected among
  ';', tab and whitespace, and `delimiter=","` is refused.  A number
  with a point in such a file ("1.5", or "1.234,5" with a point as a
  thousands separator) is refused with its line number, because the
  point could mean either;
* the x column must be strictly increasing (the error names the first
  line where it is not), and every value must be finite.

No instrument-specific format is assumed or claimed: pick the columns
with `columns` and the frequency unit with `f_scale`.
"""
from __future__ import annotations

import os

import numpy as np

from ._checks import whole
from .noise import NoiseSpec
from .tuning import TuningCurve, fit_tuning

__all__ = ["read_noise_csv", "read_tuning_csv"]


def _split(line, delimiter, decimal="."):
    if delimiter is None:
        seps = (";", ",", "\t") if decimal == "." else (";", "\t")
        for d in seps:
            if d in line:
                return [x.strip() for x in line.split(d)]
        return line.split()
    if delimiter == " ":
        return line.split()
    return [x.strip() for x in line.split(delimiter)]


class _PointInCommaFile(ValueError):
    """A number written with a point in a decimal-comma file."""


def _number(text, decimal):
    """float(text), with a decimal comma when decimal == ","."""
    if decimal == ".":
        return float(text)
    if "." in text:
        try:
            float(text.replace(",", ""))
        except ValueError:
            raise ValueError(text) from None       # not a number at all
        raise _PointInCommaFile(text)
    return float(text.replace(",", "."))


def _check_format(columns, delimiter, decimal):
    try:
        cx, cy = columns
    except (TypeError, ValueError):
        raise ValueError(f"columns must be a pair of column numbers, got "
                         f"{columns!r}") from None
    # whole numbers: before 0.9.0 int() read columns=(0.7, 1.9) as (0, 1)
    cx, cy = whole(cx, "a column number"), whole(cy, "a column number")
    if decimal not in (".", ","):
        raise ValueError(f"decimal must be '.' or ',', got {decimal!r}")
    if decimal == "," and delimiter == ",":
        raise ValueError("with decimal=',' the comma cannot also separate "
                         "the columns: give delimiter=';', '\\t' or ' ', "
                         "or None to detect it")
    return cx, cy


def _read_xy(path, columns, delimiter, decimal="."):
    cx, cy = _check_format(columns, delimiter, decimal)
    xs, ys, lines = [], [], []
    started = False
    with open(os.fspath(path), "r", encoding="utf-8-sig",
              newline="") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = _split(line, delimiter, decimal)
            try:
                x = _number(fields[cx], decimal)
                y = _number(fields[cy], decimal)
            except _PointInCommaFile as err:
                raise ValueError(
                    f"{path}, line {lineno}: the number {str(err)!r} has a "
                    "point in a file read with decimal=',': it could be a "
                    "decimal point or a thousands separator, so it is "
                    "refused rather than guessed") from None
            except (ValueError, IndexError):
                if started:
                    raise ValueError(
                        f"{path}, line {lineno}: columns {cx} and {cy} "
                        f"are not numbers after the data began "
                        f"({line[:60]!r}); a file with text inside its "
                        "data is refused rather than cut short") from None
                continue                       # header line
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError(f"{path}, line {lineno}: non-finite "
                                 f"value ({line[:60]!r})")
            started = True
            xs.append(x)
            ys.append(y)
            lines.append(lineno)
    if not xs:
        hint = ("a decimal comma such as '1,5' needs decimal=','"
                if decimal == "." else "decimal=',' is set")
        raise ValueError(
            f"{path}: no numeric rows found in columns {cx} and {cy} "
            f"(check the delimiter and columns; {hint})")
    x = np.array(xs)
    bad = np.flatnonzero(np.diff(x) <= 0.0)
    if bad.size:
        k = int(bad[0]) + 1
        raise ValueError(
            f"{path}, line {lines[k]}: column {cx} is not strictly "
            f"increasing ({x[k - 1]:.10g} then {x[k]:.10g}); sort the "
            "file or remove the repeated point")
    return x, np.array(ys)


def read_noise_csv(path: str | os.PathLike, reference: str,
                   columns: tuple[int, int] = (0, 1),
                   delimiter: str | None = None, f_scale: float = 1.0,
                   label: str | None = None,
                   decimal: str = ".") -> NoiseSpec:
    """A `NoiseSpec` from a text file of (offset, dBc/Hz) rows.

    path : the file.  reference : where the measurement comes from
        (instrument and date, or a citation; mandatory, >= 8 characters).
    columns : (offset column, dBc/Hz column), counted from 0.
    delimiter : None to detect per line, or the delimiter string.
    f_scale : multiplies the offset column into Hz (1e3 for kHz, 1e6
        for MHz).
    label : defaults to the file name.
    decimal : "." (the default) or "," for a file written with a
        decimal comma ("-92,5"); see the module docstring for the rules.

    The spec interpolates between the rows and refuses offsets outside
    them, as any `NoiseSpec` does; `NoiseSpec.phase_variance` then
    integrates the trace exactly between its points.
    """
    f, dbc = _read_xy(path, columns, delimiter, decimal)
    scale = float(f_scale)
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError("f_scale must be finite and positive")
    return NoiseSpec(tuple(f * scale), tuple(dbc), reference=reference,
                     label=os.path.basename(os.fspath(path))
                     if label is None else label)


def read_tuning_csv(path: str | os.PathLike, reference: str,
                    columns: tuple[int, int] = (0, 1),
                    delimiter: str | None = None, f_scale: float = 1.0,
                    label: str | None = None,
                    decimal: str = ".") -> TuningCurve:
    """A fitted `TuningCurve` from a text file of (V, frequency) rows.

    Same reading rules (and the same `decimal` option) as
    `read_noise_csv`; f_scale multiplies the
    frequency column into Hz (1e9 for GHz).  The rows go to
    `fit_tuning` unchanged, with all its checks (>= 4 points, a
    monotone curve, the fold refusal naming its voltage).
    """
    vc, f = _read_xy(path, columns, delimiter, decimal)
    scale = float(f_scale)
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError("f_scale must be finite and positive")
    return fit_tuning(vc, f * scale, reference,
                      label=os.path.basename(os.fspath(path))
                      if label is None else label)
