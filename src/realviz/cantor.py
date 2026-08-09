"""Cantor set and Cantor function visualisations.

``v0.2`` of ``realviz``:
- ``cantor_set``      — the middle-third construction, level by level.
- ``cantor_function`` — the devil's staircase it grows into.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._validation import _check_cantor_levels, _check_figsize, _check_samples
from .integrals import _make_colormap_norm

# -- style constants --------------------------------------------------
_CMAP_CANTOR = "viridis"   # colour by construction level
_BAR_WIDTH = 7              # linewidth of each remaining interval
_MAX_DIGITS = 60            # ternary digits for F (float64 runs out ~2**-34)


def _cantor_function_values(xs: np.ndarray) -> np.ndarray:
    """Cantor function ``F(x)`` sampled on ``xs`` in ``[0, 1]``.

    Read ``x`` in ternary; map each digit 0 -> 0, 2 -> 1 (into binary), and
    at the *first* 1 make every following binary digit 1.  This yields the
    continuous devil's staircase: ``F' = 0`` a.e., yet ``F(1) - F(0) = 1``.
    """
    xs = np.clip(xs, 0.0, 1.0)
    is_one = xs >= 1.0  # x=1 has ternary expansion 0.222... -> F = 1
    frac = np.where(is_one, 0.0, xs).astype(np.float64)
    out = np.where(is_one, 1.0, 0.0)
    done = np.zeros(xs.shape, dtype=bool)
    power = 0.5  # 2 ** -i, i starting at 1
    for _ in range(_MAX_DIGITS):
        frac = frac * 3.0
        digit = np.floor(frac)
        first_one = (digit == 1.0) & ~done
        digit_two = (digit == 2.0) & ~done
        out += np.where(first_one | digit_two, power, 0.0)
        done |= first_one
        frac = frac - digit
        frac = np.where(done, 0.0, frac)
        power *= 0.5
    return out


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def cantor_set(
    n_levels: int = 4,
    *,
    figsize: Tuple[float, float] = (13, 5.5),
) -> Tuple[Figure, Tuple[Axes, Axes], float]:
    """Draw the construction of the Cantor set, level by level.

    Two panels:
    - **Left**: row ``k`` shows ``E_k`` — what is left after ``k`` middle-third
      removals.  Level 0 is the whole interval ``[0, 1]``; each level has
      ``2**k`` intervals of equal length ``(1/3)**k``.  The y-axis labels
      carry the exact measure ``(2/3)**k`` of each row.
    - **Right**: a bar chart of those measures, ``m(E_k) = (2/3)**k``,
      which shrink geometrically to 0 — even though the limiting set is
      uncountable.

    Parameters
    ----------
    n_levels : int
        Number of middle-third removals to draw (0 <= n_levels <= 10).
    figsize : (float, float)

    Returns
    -------
    fig : Figure
    (ax_left, ax_right) : (Axes, Axes)
    total_measure : float
        Total length remaining at the last level, ``(2/3) ** n_levels``.
    """
    _check_cantor_levels(n_levels)
    _check_figsize(figsize)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )

    # -- build the intervals level by level --------------------------
    levels: list[list[Tuple[float, float]]] = [[(0.0, 1.0)]]
    for _ in range(n_levels):
        prev = levels[-1]
        nxt: list[Tuple[float, float]] = []
        for left, right in prev:
            third = (right - left) / 3.0
            nxt.append((left, left + third))
            nxt.append((right - third, right))
        levels.append(nxt)

    # -- colour by construction level --------------------------------
    _, norm, sm = _make_colormap_norm(0, n_levels, _CMAP_CANTOR)

    # -- LEFT: the staircase -----------------------------------------
    for k, intervals in enumerate(levels):
        y = -float(k)
        for left, right in intervals:
            ax_l.hlines(y, left, right, linewidth=_BAR_WIDTH, color=sm.to_rgba(k),
                        zorder=3)

    cbar = plt.colorbar(sm, ax=ax_l, shrink=0.85, pad=0.02)
    cbar.set_ticks(range(n_levels + 1))
    cbar.set_label("construction level $k$", fontsize=10)

    ax_l.set_title("Middle-third construction", fontsize=11)
    ax_l.set_xlabel("$x$")
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(-n_levels - 0.5, 0.5)
    ax_l.set_yticks([-float(k) for k in range(n_levels + 1)])
    ax_l.set_yticklabels(
        [r"$E_0$"]
        + [rf"$E_{k} = {2 ** k}/{3 ** k}$" for k in range(1, n_levels + 1)]
    )
    ax_l.axvline(x=1 / 3, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_l.axvline(x=2 / 3, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)

    total_measure = float((2.0 / 3.0) ** n_levels)

    # -- RIGHT: measure decay ----------------------------------------
    ks = np.arange(n_levels + 1)
    measures = (2.0 / 3.0) ** ks
    ax_r.bar(ks, measures, width=0.62, color=[sm.to_rgba(int(k)) for k in ks],
             zorder=3)
    for k, v in zip(ks, measures):
        ax_r.text(float(k), float(v), f"{v:.3f}", ha="center", va="bottom",
                  fontsize=9)
    ax_r.axhline(y=0, color="grey", linewidth=0.8)
    ax_r.set_title(r"$m(E_k) = (2/3)^k \to 0$", fontsize=11)
    ax_r.set_xlabel("level $k$")
    ax_r.set_ylabel("remaining measure")
    ax_r.set_xticks(ks)
    ax_r.set_ylim(0, 1.08)

    fig.suptitle(
        "Cantor set: remove middle thirds, level by level"
        + "\n"
        + r"$m(E_n) = (2/3)^n \to 0$" + "  as  " + r"$n \to \infty$"
        + "\n"
        + "uncountable, yet measure zero",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, (ax_l, ax_r), total_measure


def cantor_function(
    n_points: int = 1000,
    *,
    ax: Optional[Axes] = None,
) -> Tuple[Axes, np.ndarray]:
    """Draw the Cantor function — the devil's staircase.

    ``F`` is continuous, non-decreasing, ``F(0) = 0``, ``F(1) = 1``, and
    ``F'(x) = 0`` for almost every ``x``.  It is the standard counterexample
    to the naive fundamental theorem of calculus: its derivative is zero a.e.
    yet the function climbs from 0 to 1.  The constant middle thirds are the
    flat steps of the staircase.

    Parameters
    ----------
    n_points : int
        Number of sample points for the curve (<= 1e6).
    ax : Axes or None
        Draw on an existing axes; creates a new one when ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    ys : np.ndarray
        The sampled values ``F(x)`` on a dense grid of ``[0, 1]``.
    """
    _check_samples(n_points, name="n_points")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    xs = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    ys = _cantor_function_values(xs)

    # flat first-level middle third
    ax.axvspan(1 / 3, 2 / 3, color="grey", alpha=0.15, zorder=1)
    ax.plot(xs, ys, color="#1a1a2e", linewidth=2.0, zorder=3)

    ax.annotate(
        r"$F \equiv 1/2$ on $[1/3, 2/3]$",
        xy=(0.5, 0.5), xytext=(0.6, 0.68),
        arrowprops=dict(arrowstyle="->", color="#555555"),
        fontsize=9,
    )

    ax.set_title(
        "Cantor function (devil's staircase)"
        + "\n"
        + r"continuous, non-decreasing, $F' = 0$ a.e., yet $F(1)-F(0)=1$",
        fontsize=11,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$F(x)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color="grey", linewidth=0.5)
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")

    return ax, ys
