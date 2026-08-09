"""Input validation utilities for realviz.

All public functions share these validation routines to ensure safe,
well-bounded computation and clear error messages.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Callable

# ── module-level constants ──────────────────────────────────────────
_MAX_PARTITIONS = 10_000
_MAX_SAMPLES = 1_000_000
_MAX_DOMAIN = 1e12  # reasonable domain bound for real analysis
_MAX_CANTOR_LEVELS = 10  # each level doubles the interval count: 2**n
_MAX_FIGSIZE = 50  # inches per side; the render buffer scales as size * dpi


def _check_callable(f: Any, name: str = "f") -> None:
    """Validate that *f* is callable."""
    if not callable(f):
        raise TypeError(f"{name} must be callable, got {type(f).__name__!r}")


def _check_interval(a: Any, b: Any) -> None:
    """Validate that *a* and *b* are finite, ordered numbers."""
    if not isinstance(a, Number) or not isinstance(b, Number):
        raise TypeError(
            f"Interval endpoints must be numbers, "
            f"got {type(a).__name__!r} and {type(b).__name__!r}"
        )
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"Endpoints must be finite, got a={a}, b={b}")
    if a >= b:
        raise ValueError(f"a must be strictly less than b, got a={a}, b={b}")
    if abs(a) > _MAX_DOMAIN or abs(b) > _MAX_DOMAIN:
        raise ValueError(
            f"Domain too large: [{a}, {b}]. Maximum absolute value is {_MAX_DOMAIN}."
        )


def _check_partitions(n: Any, name: str = "n_partitions") -> None:
    """Validate that *n* is a positive, bounded integer."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"{name} must be an int, got {type(n).__name__!r}")
    if n <= 0:
        raise ValueError(f"{name} must be positive, got {n}")
    if n > _MAX_PARTITIONS:
        raise ValueError(f"{name} too large: {n}. Maximum is {_MAX_PARTITIONS}.")


def _check_samples(n: Any, name: str = "n_samples") -> None:
    """Validate sample count for dense evaluations."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"{name} must be an int, got {type(n).__name__!r}")
    if n <= 0:
        raise ValueError(f"{name} must be positive, got {n}")
    if n > _MAX_SAMPLES:
        raise ValueError(f"{name} too large: {n}. Maximum is {_MAX_SAMPLES}.")


def _check_cantor_levels(n: Any, name: str = "n_levels") -> None:
    """Validate Cantor-set construction depth.

    Depth is bounded because every level doubles the number of intervals
    (``2**n``), so a generous cap keeps both drawing and validation cheap.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"{name} must be an int, got {type(n).__name__!r}")
    if n < 0:
        raise ValueError(f"{name} must be non-negative, got {n}")
    if n > _MAX_CANTOR_LEVELS:
        raise ValueError(
            f"{name} too large: {n}. Maximum is {_MAX_CANTOR_LEVELS} "
            f"(that is 2**{_MAX_CANTOR_LEVELS} = {2 ** _MAX_CANTOR_LEVELS} intervals)."
        )


def _check_figsize(figsize: Any, name: str = "figsize") -> None:
    """Validate a matplotlib figure size ``(width, height)`` tuple.

    The render buffer is roughly ``figsize_inches * dpi`` pixels per side, so an
    unbounded size can ask matplotlib to allocate gigabytes of memory.  Each
    dimension is therefore capped like every other numeric parameter.
    """
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError(f"{name} must be a (width, height) tuple, got {figsize!r}")
    w, h = figsize
    if not isinstance(w, Number) or not isinstance(h, Number):
        raise TypeError(f"{name} entries must be numbers, got {w!r} and {h!r}")
    if not (math.isfinite(w) and math.isfinite(h)):
        raise ValueError(f"{name} entries must be finite, got {figsize!r}")
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} entries must be positive, got {figsize!r}")
    if w > _MAX_FIGSIZE or h > _MAX_FIGSIZE:
        raise ValueError(
            f"{name} too large: {figsize}. Maximum per dimension is {_MAX_FIGSIZE} inches."
        )


def _validate_function(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    allow_constant: bool = True,
) -> None:
    """Full validation: callable, numeric outputs, finite values on [a, b].

    Parameters
    ----------
    f : callable
        The function to validate.
    a, b : float
        Interval endpoints (assumed already validated by ``_check_interval``).
    allow_constant : bool
        If False, raise when the function appears constant on the domain.
        Useful for Lebesgue plots where constant functions produce degenerate
        preimages.
    """
    _check_callable(f)

    # Evaluate at endpoints and midpoint
    test_points = [a, b, (a + b) / 2]
    for pt in test_points:
        try:
            val = f(pt)
        except Exception as exc:
            raise ValueError(
                f"Function raised {type(exc).__name__!r} at x={pt}: {exc}"
            ) from exc
        if not isinstance(val, Number):
            raise TypeError(
                f"Function must return numbers, got {type(val).__name__!r} at x={pt}"
            )
        if not math.isfinite(val):
            raise ValueError(f"Function returned non-finite value at x={pt}: {val}")

    if not allow_constant:
        # Quick check: if f(a) == f(b), probe a few interior points
        probes = [a + (b - a) * t for t in (0.25, 0.5, 0.75)]
        vals = {f(p) for p in probes}
        if len(vals) == 1 and vals.pop() == f(a):
            raise ValueError(
                f"Function appears constant on [{a}, {b}]. "
                "A non-constant function is required for this plot."
            )
