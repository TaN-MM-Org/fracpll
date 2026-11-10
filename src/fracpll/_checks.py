"""Input checks shared by several modules (internal)."""
from __future__ import annotations

import numbers

__all__ = []


def whole(value, name):
    """`value` as a Python int, refusing a fraction instead of cutting it.

    Integers of any size (Python or NumPy) pass unchanged, and so does a
    float or a string that holds a whole number (40.0, "64").  Before
    0.8.0 these inputs went through int(), which cut 40.5 to 40 and
    3.7 to 3 without a word: a divider ratio or a fraction num/den
    other than the one asked for."""
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)              # exact, whatever its size
        except ValueError:
            pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number, got "
                         f"{value!r}") from None
    if not v.is_integer():                 # also NaN and +-inf
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return int(v)
