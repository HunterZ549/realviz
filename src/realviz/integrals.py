"""Riemann and Lebesgue integral visualizations.

Core module of ``realviz`` - the pedagogical heart of the library.

Design principle
----------------
The key visual distinction:
- Riemann: partitions the **x-axis** -> strips coloured by x-position.
- Lebesgue: partitions the **y-axis** -> strips coloured by y-value.

This colouring makes the conceptual difference immediately obvious
even when both integrals approximate the same area.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from ._validation import (
    _check_callable,
    _check_figsize,
    _check_interval,
    _check_partitions,
    _check_samples,
)

# -- style constants ------------------------------------------------
_CMAP_RIEMANN = "coolwarm"    # colour by x-position
_CMAP_LEBESGUE = "viridis"    # colour by y-value
_ALPHA_FILL = 0.55
_ALPHA_EDGE = 0.9
_LINE_COLOUR = "#1a1a2e"
_CURVE_COLOUR = "#1a1a2e"
_CURVE_WIDTH = 2.2
_PARTITION_LINE = "#555555"
_ANNOTATION_BG = "#fefefe"


def _make_colormap_norm(vmin: float, vmax: float, cmap_name: str):
    """Return a ``(cmap, norm, sm)`` tuple for consistent colour mapping."""
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    return cmap, norm, sm


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def riemann_plot(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_partitions: int = 10,
    *,
    method: str = "midpoint",
    ax: Optional[Axes] = None,
) -> Tuple[Axes, float]:
    """Draw a Riemann integral approximation.

    Vertical strips are **coloured by x-position** (coolwarm: blue=left,
    red=right) to emphasise that we are partitioning the *x-axis*.

    Parameters
    ----------
    f : callable
        Real-valued function on ``[a, b]``.
    a, b : float
        Integration interval (``a < b``, both finite).
    n_partitions : int
        Number of equal-width sub-intervals (<= 10 000).
    method : {"midpoint", "left", "right"}
        Where to evaluate *f* inside each sub-interval.
    ax : Axes or None
        Draw on an existing axes; creates a new one when ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    riemann_sum : float
    """
    # -- validate --------------------------------------------------
    _check_callable(f)
    _check_interval(a, b)
    _check_partitions(n_partitions)
    if method not in {"midpoint", "left", "right"}:
        raise ValueError(f"method must be 'midpoint', 'left', or 'right'; got {method!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    dx = (b - a) / n_partitions
    x_edges = np.linspace(a, b, n_partitions + 1, dtype=np.float64)

    if method == "left":
        sample_x = x_edges[:-1]
    elif method == "right":
        sample_x = x_edges[1:]
    else:
        sample_x = (x_edges[:-1] + x_edges[1:]) / 2

    # evaluate
    try:
        heights = np.array([float(f(xi)) for xi in sample_x], dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            f"Function raised {type(exc).__name__!r} during sampling. "
            f"Check that f is defined on [{a}, {b}]."
        ) from exc
    if np.any(~np.isfinite(heights)):
        raise ValueError(f"Function returned non-finite values on [{a}, {b}].")

    # -- draw function curve ----------------------------------------
    n_dense = max(500, n_partitions * 20)
    x_dense = np.linspace(a, b, n_dense, dtype=np.float64)
    y_dense = np.array([float(f(xi)) for xi in x_dense], dtype=np.float64)
    ax.plot(x_dense, y_dense, color=_CURVE_COLOUR, linewidth=_CURVE_WIDTH,
            label="$f(x)$", zorder=5)

    # -- colour map by x-position ----------------------------------
    _, norm, sm = _make_colormap_norm(a, b, _CMAP_RIEMANN)

    riemann_sum = 0.0
    for i in range(n_partitions):
        x_left, x_right = float(x_edges[i]), float(x_edges[i + 1])
        colour = sm.to_rgba((x_left + x_right) / 2)  # colour by x-position
        ax.fill_between(
            [x_left, x_right], [0, 0], [heights[i], heights[i]],
            alpha=_ALPHA_FILL, color=colour, edgecolor=colour,
            linewidth=0.6, zorder=3,
        )
        riemann_sum += heights[i] * dx

    riemann_sum = float(riemann_sum)

    # -- partition markers ------------------------------------------
    for xv in x_edges:
        ax.axvline(x=xv, color=_PARTITION_LINE, linewidth=0.4, linestyle="--",
                   alpha=0.5, zorder=2)

    # -- colour bar -------------------------------------------------
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("$x$-position  $\\longrightarrow$", fontsize=10)

    # -- annotations -----------------------------------------------
    ax.set_title(
        f"Riemann: partition $x$-axis  $\\longrightarrow$  "
        f"${{\\sum f(x_i)\\Delta x \\approx {riemann_sum:.4f}}}$\n"
        f"({n_partitions} strips, {method} rule)",
        fontsize=12,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y = f(x)$")
    ax.axhline(y=0, color="grey", linewidth=0.5, zorder=1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(a, b)

    return ax, riemann_sum


def lebesgue_plot(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_partitions: int = 10,
    *,
    n_samples: int = 2000,
    ax: Optional[Axes] = None,
) -> Tuple[Axes, float]:
    """Draw a Lebesgue integral approximation.

    Horizontal strips are **coloured by y-value** (viridis: dark=low,
    bright=high) to emphasise that we are partitioning the *y-axis*
    (the range of *f*).

    For each y-slice ``[y_k, y_{k+1})`` we find the set
    ``E_k = {x : y_k <= f(x) < y_{k+1}}``, measure its total length
    ``m(E_k)``, and contribute ``y_k * m(E_k)``.

    Parameters
    ----------
    f : callable
        Real-valued function on ``[a, b]``.  Must not be constant.
    a, b : float
        Domain interval (``a < b``, both finite).
    n_partitions : int
        Number of horizontal strips (<= 10 000).
    n_samples : int
        Dense evaluation points for preimage approximation (<= 106).
    ax : Axes or None

    Returns
    -------
    ax : matplotlib.axes.Axes
    lebesgue_sum : float
    """
    # -- validate --------------------------------------------------
    _check_callable(f)
    _check_interval(a, b)
    _check_partitions(n_partitions)
    _check_samples(n_samples)

    # -- dense sampling --------------------------------------------
    x_dense = np.linspace(a, b, n_samples, dtype=np.float64)
    try:
        y_dense = np.array([float(f(xi)) for xi in x_dense], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Function raised {type(exc).__name__!r} during sampling.") from exc
    if np.any(~np.isfinite(y_dense)):
        raise ValueError("Function returned non-finite values.")

    y_min = float(np.min(y_dense))
    y_max = float(np.max(y_dense))
    if y_min == y_max:
        raise ValueError(
            f"Function is constant (== {y_min}) on [{a}, {b}]. "
            "Constant functions produce degenerate preimages."
        )

    dy = (y_max - y_min) / n_partitions

    # -- setup axes ------------------------------------------------
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_dense, y_dense, color=_CURVE_COLOUR, linewidth=_CURVE_WIDTH,
            label="$f(x)$", zorder=5)

    # -- colour map by y-value -------------------------------------
    _, norm, sm = _make_colormap_norm(y_min, y_max, _CMAP_LEBESGUE)

    dx_sample = (b - a) / (n_samples - 1)
    lebesgue_sum = 0.0

    for k in range(n_partitions):
        y_low = y_min + k * dy
        y_high = y_low + dy

        # preimage of this y-slice
        if k == n_partitions - 1:
            mask = (y_dense >= y_low) & (y_dense <= y_max)
        else:
            mask = (y_dense >= y_low) & (y_dense < y_high)

        if not np.any(mask):
            continue

        # colour by the y-value of this slice (midpoint)
        colour = sm.to_rgba((y_low + y_high) / 2)

        indices = np.flatnonzero(mask)
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        runs: list[np.ndarray] = np.split(indices, breaks)

        for run in runs:
            if len(run) == 0:
                continue
            x_start = float(x_dense[run[0]])
            x_end = float(x_dense[run[-1]])
            measure = (x_end - x_start) + dx_sample

            ax.fill_between(
                [x_start, x_end],
                [y_low, y_low],
                [y_high, y_high],
                alpha=_ALPHA_FILL, color=colour, edgecolor=colour,
                linewidth=0.6, zorder=3,
            )
            lebesgue_sum += y_low * measure

    lebesgue_sum = float(lebesgue_sum)

    # -- partition markers (horizontal) --------------------------
    for k in range(n_partitions + 1):
        y_line = y_min + k * dy
        ax.axhline(y=y_line, color=_PARTITION_LINE, linewidth=0.4,
                   linestyle="--", alpha=0.5, zorder=2)

    # -- colour bar -------------------------------------------------
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("$y$-value  $\\longrightarrow$", fontsize=10)

    # -- annotations -----------------------------------------------
    ax.set_title(
        f"Lebesgue: partition $y$-axis  $\\longrightarrow$  "
        f"${{\\sum y_k \\cdot m(E_k) \\approx {lebesgue_sum:.4f}}}$\n"
        f"({n_partitions} horizontal strips)",
        fontsize=12,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y = f(x)$")
    ax.axhline(y=0, color="grey", linewidth=0.5, zorder=1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(a, b)

    return ax, lebesgue_sum


def compare_integrals(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_partitions: int = 10,
    *,
    figsize: Tuple[float, float] = (16, 6),
) -> Tuple[Figure, Tuple[Axes, Axes], Tuple[float, float]]:
    """Side-by-side comparison: Riemann vs Lebesgue integral.

    Left panel: Riemann (colour by x-position).
    Right panel: Lebesgue (colour by y-value).

    Parameters
    ----------
    f : callable
    a, b : float
    n_partitions : int
    figsize : (float, float)

    Returns
    -------
    fig : Figure
    (ax_left, ax_right) : (Axes, Axes)
    (riemann_sum, lebesgue_sum) : (float, float)
    """
    _check_callable(f)
    _check_interval(a, b)
    _check_partitions(n_partitions)
    _check_figsize(figsize)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    ax_l, r_sum = riemann_plot(f, a, b, n_partitions, ax=ax_l)
    ax_r, l_sum = lebesgue_plot(f, a, b, n_partitions, ax=ax_r)

    # -- global annotation -----------------------------------------
    text = (
        "RIEMANN: chop $x$-axis  $\\rightarrow$  vertical strips, same width\n"
        "LEBESGUE: chop $y$-axis  $\\rightarrow$  horizontal strips, same height\n"
        "Same total area - different philosophy"
    )
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=10,
             style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                       edgecolor="#cccccc", alpha=0.9))

    fig.suptitle(
        f"Riemann vs Lebesgue  -  $f$ on $[{a}, {b}]$  -  "
        f"{n_partitions} partitions",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])

    return fig, (ax_l, ax_r), (r_sum, l_sum)


# -------------------------------------------------------------------
# Bonus: Dirichlet illustration (standalone, no numerical sampling)
# -------------------------------------------------------------------

def dirichlet_illustration(a: float = 0, b: float = 1) -> Figure:
    """Illustrate why the Dirichlet function is Lebesgue- but not
    Riemann-integrable.

    The Dirichlet function ``D(x) = 1`` if x is rational, ``0`` otherwise.

    - Every Riemann upper sum = *b-a*, every lower sum = 0 -> not integrable.
    - Lebesgue: ``m(Q) = 0``, complement set measure ``b-a`` -> int D = 0.
    """
    _check_interval(a, b)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_bars = 8
    dx = (b - a) / n_bars
    x_edges = np.linspace(a, b, n_bars + 1)
    rng = np.random.default_rng(0)

    # ================= LEFT: Riemann =================
    # Upper sum bars: sup D = 1 in every sub-interval (red)
    for i in range(n_bars):
        xl, xr = float(x_edges[i]), float(x_edges[i + 1])
        ax_l.fill_between(
            [xl, xr], [0, 0], [1, 1],
            alpha=0.35, color="#e74c3c", edgecolor="#c0392b", linewidth=0.6,
            label=r"$\sup D = 1$ (contains a rational)" if i == 0 else "",
        )

    # Lower sum: inf D = 0 in every sub-interval.
    # The bars have ZERO height, so we draw a thin blue band just below the
    # axis to make "lower sum = 0" visible.
    for i in range(n_bars):
        xl, xr = float(x_edges[i]), float(x_edges[i + 1])
        ax_l.fill_between(
            [xl, xr], [-0.07, -0.07], [0, 0],
            alpha=0.55, color="#3498db", edgecolor="#2980b9", linewidth=0.6,
            label=r"$\inf D = 0$ (contains an irrational)" if i == 0 else "",
        )

    # partition lines
    for xv in x_edges:
        ax_l.axvline(x=xv, color="#888888", linewidth=0.4, linestyle="--",
                     alpha=0.5, zorder=2)
    ax_l.axhline(y=0, color="#555555", linewidth=0.8, zorder=3)

    # rational points (y=1) and irrational points (y=0) sprinkled in
    ax_l.scatter(rng.uniform(a, b, 25), np.ones(25), s=6, c="#e74c3c",
                 alpha=0.8, zorder=6, edgecolors="white", linewidths=0.3)
    ax_l.scatter(rng.uniform(a, b, 25), np.zeros(25), s=6, c="#3498db",
                 alpha=0.8, zorder=6, edgecolors="white", linewidths=0.3)

    # annotate the gap on one sub-interval
    mid3 = (float(x_edges[3]) + float(x_edges[4])) / 2
    ax_l.annotate(
        "", xy=(mid3, 0.02), xytext=(mid3, 0.98),
        arrowprops=dict(arrowstyle="<->", color="#2c3e50",
                        linewidth=1.5, shrinkA=0, shrinkB=0),
    )
    ax_l.text(mid3 + 0.04, 0.45,
              r"$\sup - \inf = 1$" + "\n" + r"never $\to 0$",
              fontsize=9, color="#2c3e50", va="center",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor="#cccccc", alpha=0.85))

    ax_l.set_title(
        r"Riemann: $\mathbf{NOT}$ integrable"
        + "\n"
        + rf"$U(f,P) = 1\cdot {b - a:.1f} = {b - a:.1f}$   |   "
        + rf"$L(f,P) = 0\cdot {b - a:.1f} = 0$"
        + "\n"
        + rf"gap $(U-L) = {b - a:.1f} \neq 0$ never closes",
        fontsize=11,
    )
    ax_l.set_xlabel(r"$x$")
    ax_l.set_ylabel(r"$D(x)$")
    ax_l.set_ylim(-0.22, 1.35)
    ax_l.set_xlim(a, b)
    ax_l.legend(loc="upper right", fontsize=8, ncol=2)

    # ================= RIGHT: Lebesgue =================
    # rationals: countable -> measure 0
    ax_r.fill_between([a, b], [0.97, 0.97], [1.03, 1.03],
                       alpha=0.2, color="#e74c3c",
                       label=r"$\{x \in \mathbb{Q}\}$  $\mu = 0$")
    # irrationals: measure b-a
    ax_r.fill_between([a, b], [-0.03, -0.03], [0.03, 0.03],
                       alpha=0.55, color="#3498db",
                       label=r"$\{x \notin \mathbb{Q}\}$  $\mu = %.1f$" % (b - a))

    ax_r.axhline(y=0, color="#555555", linewidth=0.8)
    ax_r.axhline(y=1, color="#888888", linewidth=0.3, linestyle="--")

    formula = (
        r"$\int_{[a,b]} D\,d\mu$" + "\n"
        + r"$= 1\cdot\mu(\mathbb{Q}\cap[a,b]) + 0\cdot\mu(\mathbb{R}\setminus\mathbb{Q}\cap[a,b])$"
        + "\n"
        + rf"$= 1\cdot 0 + 0\cdot {b - a:.1f} = \mathbf{{0}}$"
    )
    ax_r.text(0.5, 0.58, formula, transform=ax_r.transAxes, fontsize=13,
              ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="#fafafa",
                        edgecolor="#aaaaaa", alpha=0.92))

    ax_r.set_title(
        r"Lebesgue: integrable, $\int D = 0$" + "\n"
        r"Countable sets $\to$ measure zero",
        fontsize=12,
    )
    ax_r.set_xlabel(r"$x$")
    ax_r.set_ylabel(r"$D(x)$")
    ax_r.set_ylim(-0.22, 1.35)
    ax_r.set_xlim(a, b)
    ax_r.legend(loc="upper right", fontsize=8)

    # shared
    fig.suptitle(
        rf"Dirichlet Function  -  $D(x) = \mathbf{{1}}_{{\mathbb{{Q}}}}(x)$ "
        f"on $[{a},{b}]$",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig

