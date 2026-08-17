"""Simple-function approximation of measurable functions.

``v0.3`` of ``realviz``:
- ``simple_approximation`` — the horizontal (y-axis) refinement that makes
  ``f`` the pointwise limit of simple functions ``s_n <= f``, with the
  Lebesgue-integral convergence that *defines* the integral itself.

The whole figure is the ``lebesgue_plot`` story pushed to the limit:
chop the y-axis finer and finer, and the piecewise-constant staircase
converges to ``f`` while its integral converges to ``∫ f``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._validation import (
    _check_approx_levels,
    _check_callable,
    _check_figsize,
    _check_interval,
    _check_samples,
)
from .integrals import (
    _ALPHA_FILL,
    _CMAP_LEBESGUE,
    _CURVE_COLOUR,
    _CURVE_WIDTH,
    _PARTITION_LINE,
    _make_colormap_norm,
)

# -- style constants --------------------------------------------------
_CURVE_STAIRCASE = "#0f766e"  # the simple function, distinct from f
_COLOR_INTEGRAL = "#2c3e50"
_COLOR_REF = "#c0392b"
_COLOR_ERROR = "#16a085"
_MAX_BAND_WORK = 50_000_000  # bands × samples cap for the band-colouring loop


class SimpleApproximationInfo(NamedTuple):
    """Computed data behind a ``simple_approximation`` figure.

    Attributes
    ----------
    levels : np.ndarray
        The refinement exponents ``k = 0 .. n_levels``.
    s_values : np.ndarray
        The display-level simple function ``s_{n_levels}`` on the sample grid.
    integrals : np.ndarray
        ``∫ s_k`` for every ``k`` — a monotone sequence rising to ``∫ f``.
    sup_errors : np.ndarray
        ``max |f - s_k|`` for every ``k`` — decays to 0.
    ref_integral : float
        High-resolution trapezoidal estimate of ``∫ f`` (the dashed reference).
    n_bands : int
        ``2 ** n_levels`` — how many horizontal bands chop the range.
    band_width : float
        Height of one band, ``(y_max - y_min) / n_bands``.
    """

    levels: np.ndarray
    s_values: np.ndarray
    integrals: np.ndarray
    sup_errors: np.ndarray
    ref_integral: float
    n_bands: int
    band_width: float


def _simple_function_values(
    y: np.ndarray, y_min: float, y_max: float, n_levels: int
) -> np.ndarray:
    """Evaluate the ``n_levels``-th dyadic simple approximation of ``y``.

    Cut the range ``[y_min, y_max]`` into ``2 ** n_levels`` equal bands and
    round every sample down to the *lower* edge of the band that contains it.
    The result is a simple function ``s(x) <= f(x)`` taking at most
    ``2 ** n_levels`` distinct values.  A constant range returns the constant
    itself (a constant function is already a simple function).
    """
    n_bands = 2 ** n_levels
    if y_min == y_max:
        return np.full_like(y, y_min)
    dy = (y_max - y_min) / n_bands
    band = np.clip(np.floor((y - y_min) / dy), 0, n_bands - 1).astype(np.int64)
    return y_min + band * dy


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def simple_approximation(
    f,
    a: float,
    b: float,
    n_levels: int = 5,
    *,
    n_samples: int = 2000,
    figsize: Tuple[float, float] = (15, 6),
) -> Tuple[Figure, Tuple[Axes, Tuple[Axes, Axes]], SimpleApproximationInfo]:
    """Draw the simple-function approximation of a measurable function.

    Every measurable function is the pointwise limit of simple functions.
    The classic construction cuts the *range* of ``f`` into ``2 ** n`` equal
    horizontal bands and lets ``s_n(x)`` be the lower edge of the band that
    contains ``f(x)`` — so ``s_n <= f``, the sequence increases with ``n``,
    and for continuous ``f`` it converges uniformly.  The Lebesgue integral
    is *defined* as the limit of ``∫ s_n``, which the right-hand panels show.

    Three panels:
    - **Left**: ``f`` with ``s_n`` staircased beneath it; each horizontal
      band is a level set ``E_k = {x : c_k <= f(x) < c_{k+1}}`` coloured by
      its y-value (the ``lebesgue_plot`` look).
    - **Top right**: ``∫ s_k`` for every ``k``, rising monotonically toward
      a high-resolution estimate of ``∫ f`` (dashed).
    - **Bottom right**: the uniform error ``max |f - s_k|`` on a log scale,
      decaying to 0.

    Parameters
    ----------
    f : callable
        Real-valued (measurable) function on ``[a, b]``.
    a, b : float
        Domain interval (``a < b``, both finite).
    n_levels : int
        Refinement depth: the range is cut into ``2 ** n_levels`` bands
        (0 <= n_levels <= 10).
    n_samples : int
        Dense evaluation points (2 <= n_samples <= 1e6).
    figsize : (float, float)

    Returns
    -------
    fig : Figure
    (ax_left, (ax_integral, ax_error)) : nested axes
    info : SimpleApproximationInfo
        The computed sequence data (see the class docstring).
    """
    _check_callable(f)
    _check_interval(a, b)
    _check_approx_levels(n_levels)
    _check_samples(n_samples)
    _check_figsize(figsize)
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    # Band-colouring is O(2**n_levels × n_samples) — bound the *product*,
    # not just each factor, so a near-cap combination cannot hang or blow up
    # the artist count.
    if (2 ** n_levels) * n_samples > _MAX_BAND_WORK:
        raise ValueError(
            f"n_levels={n_levels} with n_samples={n_samples} is too expensive: "
            f"{2 ** n_levels} bands × {n_samples} samples exceeds the work bound "
            f"of {_MAX_BAND_WORK}. Lower n_levels or n_samples."
        )

    # -- dense sampling ---------------------------------------------
    x_dense = np.linspace(a, b, n_samples, dtype=np.float64)
    try:
        y_dense = np.array([float(f(xi)) for xi in x_dense], dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            f"Function raised {type(exc).__name__!r} during sampling."
        ) from exc
    if np.any(~np.isfinite(y_dense)):
        raise ValueError("Function returned non-finite values.")

    y_min = float(np.min(y_dense))
    y_max = float(np.max(y_dense))
    if not math.isfinite(y_max - y_min):
        raise ValueError(
            "Function range is too large to subdivide: y_max - y_min overflows. "
            "Scale the function values to a smaller range."
        )
    is_constant = y_min == y_max

    # Trapezoidal weights: the two endpoints carry half weight, every
    # interior sample one full cell.  Summing the weights reproduces the
    # interval length exactly, so a constant function has a *exact* integral
    # (the naive "every sample = dx" overcounts by n/(n-1)).
    dx = (b - a) / (n_samples - 1)
    weights = np.full(n_samples, dx)
    weights[0] = weights[-1] = 0.5 * dx

    # high-resolution trapezoidal reference for ∫ f (reuses the grid)
    ref_integral = float(np.sum(y_dense * weights))

    # -- the whole sequence over every level -------------------------
    ks = np.arange(n_levels + 1, dtype=np.int64)
    s_values = _simple_function_values(y_dense, y_min, y_max, n_levels)
    integrals = np.array(
        [
            float(
                np.sum(
                    _simple_function_values(y_dense, y_min, y_max, int(k)) * weights
                )
            )
            for k in ks
        ],
        dtype=np.float64,
    )
    sup_errors = np.array(
        [
            float(
                np.max(
                    np.abs(y_dense - _simple_function_values(y_dense, y_min, y_max, int(k)))
                )
            )
            for k in ks
        ],
        dtype=np.float64,
    )

    n_bands = 2 ** n_levels
    dy = (y_max - y_min) / n_bands if not is_constant else 0.0

    # -- layout -------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[2, 1], height_ratios=[1, 1],
        hspace=0.38, wspace=0.24,
    )
    ax_l = fig.add_subplot(gs[:, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_rb = fig.add_subplot(gs[1, 1])

    # -- LEFT: f with s_n staircased beneath --------------------------
    if not is_constant:
        _, norm, sm = _make_colormap_norm(y_min, y_max, _CMAP_LEBESGUE)
        for k in range(n_bands):
            y_low = y_min + k * dy
            y_high = y_low + dy
            if k == n_bands - 1:
                mask = (y_dense >= y_low) & (y_dense <= y_max)
            else:
                mask = (y_dense >= y_low) & (y_dense < y_high)
            if not np.any(mask):
                continue
            colour = sm.to_rgba((y_low + y_high) / 2)
            # NaN splits the fill into one rectangle per contiguous run, so a
            # single PolyCollection per band covers even a highly fragmented
            # level set — the artist count stays at O(2**n_levels), not O(n).
            ax_l.fill_between(
                x_dense, y_low, np.where(mask, y_high, np.nan),
                alpha=_ALPHA_FILL, color=colour, edgecolor=colour,
                linewidth=0.4, zorder=3,
            )

        for k in range(n_bands + 1):
            ax_l.axhline(y=y_min + k * dy, color=_PARTITION_LINE,
                         linewidth=0.4, linestyle="--", alpha=0.5, zorder=2)

        cbar = plt.colorbar(sm, ax=ax_l, shrink=0.85, pad=0.02)
        cbar.set_label("$y$-value", fontsize=10)

    ax_l.step(x_dense, s_values, where="post", color=_CURVE_STAIRCASE,
              linewidth=2.0, label=r"$s_n(x)$", zorder=6)
    ax_l.plot(x_dense, y_dense, color=_CURVE_COLOUR, linewidth=_CURVE_WIDTH,
              label=r"$f(x)$", zorder=7)
    ax_l.set_title(
        "Simple function $s_n$ — chop the $y$-axis into "
        + rf"$2^{{{n_levels}}}$ bands"
        + "\n"
        + r"$s_n(x) = \sum_k c_k \mathbf{1}_{E_k}(x)$" + "  ·  "
        + r"$s_n \leq f$",
        fontsize=10.5,
    )
    ax_l.set_xlabel("$x$")
    ax_l.set_ylabel("$y$")
    ax_l.set_xlim(a, b)
    ypad = (y_max - y_min) * 0.05 if not is_constant else max(1.0, abs(y_min)) * 0.05
    ax_l.set_ylim(y_min - ypad, y_max + ypad)
    ax_l.axhline(y=0, color="grey", linewidth=0.5, zorder=1)
    ax_l.legend(loc="upper left", fontsize=9, framealpha=0.92,
                edgecolor="#cccccc")

    # -- TOP RIGHT: ∫s_k rises to ∫f ---------------------------------
    ax_rt.plot(ks, integrals, marker="o", color=_COLOR_INTEGRAL,
               linewidth=1.6, zorder=3)
    ax_rt.axhline(ref_integral, color=_COLOR_REF, linestyle="--",
                  linewidth=1.4, zorder=2, label=r"$\int f$ (high-res)")
    for k, v in zip(ks, integrals):
        ax_rt.text(float(k), float(v), f"{v:.3f}", ha="center", va="bottom",
                   fontsize=6.5)
    lo = min(0.0, float(np.min(integrals)), ref_integral)
    hi = max(float(np.max(integrals)), ref_integral)
    pad = (hi - lo) * 0.16 if hi > lo else 1.0
    ax_rt.set_ylim(lo - pad, hi + pad)  # headroom so labels never clip
    ax_rt.set_title(r"$\int s_k \nearrow \int f$", fontsize=11)
    ax_rt.set_xlabel("level $k$")
    ax_rt.set_ylabel(r"$\int s_k$")
    ax_rt.set_xticks(ks)
    ax_rt.legend(loc="lower right", fontsize=8)

    # -- BOTTOM RIGHT: sup error decays to 0 --------------------------
    if not is_constant:
        ax_rb.semilogy(ks, sup_errors, marker="o", color=_COLOR_ERROR,
                       linewidth=1.6, zorder=3)
        for k, v in zip(ks, sup_errors):
            ax_rb.text(float(k), float(v), f"{v:.1e}", ha="center",
                       va="bottom", fontsize=6.5)
        e_lo = float(np.min(sup_errors))
        e_hi = float(np.max(sup_errors))
        ax_rb.set_ylim(e_lo * 0.25, e_hi * 2.0)  # headroom for labels
    else:
        ax_rb.text(0.5, 0.5, r"$s_n \equiv f$ (exact)", ha="center",
                   va="center", transform=ax_rb.transAxes, fontsize=11)
    ax_rb.set_title(r"$\|f - s_k\|_\infty \to 0$", fontsize=11)
    ax_rb.set_xlabel("level $k$")
    ax_rb.set_ylabel("sup error (log)")
    ax_rb.set_xticks(ks)

    fig.suptitle(
        "Simple-function approximation of $f$"
        + "\n"
        + r"$s_n \leq f$,   $s_n \uparrow f$ pointwise,   "
        + r"$\int s_n \to \int f$",
        fontsize=12,
        y=0.985,
    )
    # Manual margins: the hand-built gridspec (spanning left column +
    # colorbar axes) makes ``tight_layout`` warn, so we place it ourselves.
    # ``top`` must leave room above the axes for BOTH the panel titles and
    # the suptitle — with top too high they collide (the suptitle is pinned
    # near the figure edge, the panel titles rise above their axes).
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.08)

    info = SimpleApproximationInfo(
        levels=ks, s_values=s_values, integrals=integrals,
        sup_errors=sup_errors, ref_integral=ref_integral,
        n_bands=n_bands, band_width=dy,
    )
    return fig, (ax_l, (ax_rt, ax_rb)), info
