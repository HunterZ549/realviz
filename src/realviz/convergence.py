"""Convergence modes of measurable functions.

``v0.4`` of ``realviz``:
- ``convergence_modes`` — one function family, four cameras.  The same
  sequence ``f_n`` is judged under pointwise, uniform, almost-everywhere
  and L1 convergence.  This is exactly how the classic counterexamples
  separate the modes:
  - ``x^n`` on ``[0, 1]`` converges pointwise (and a.e.) but *not* uniformly;
  - the thin spike ``n * 1_[0, 1/n]`` converges a.e. but not in L1;
  - the *typewriter* sequence converges in L1 but nowhere pointwise.
- ``FAMILIES`` — those three counterexamples, ready to draw by name.

The four cameras share one grid and one target ``f``; every metric is a
plain NumPy computation (no magic, no LaTeX).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ._validation import (
    _check_callable,
    _check_epsilon,
    _check_figsize,
    _check_interval,
    _check_n_max,
    _check_samples,
)
from .integrals import _CURVE_COLOUR, _CURVE_WIDTH

# -- style constants ------------------------------------------------
_FAMILY_LO = "#ccfbf1"   # light end of the n-gradient (n = 1)
_FAMILY_HI = "#0f766e"   # dark end of the n-gradient (n = n_max)
_COLOR_BAD = "#dc2626"   # unsettled points (the failure set)
_COLOR_OK = "#16a34a"    # a passing verdict
_COLOR_TUBE = "#64748b"  # the epsilon-tube around f
_ALPHA_TUBE = 0.22
_COLOR_AREA = "#2dd4bf"  # the |f_n - f| area in the L1 panel
_ALPHA_AREA = 0.20

_WINDOW = 3             # settle test: the last W members must agree at each x
_AE_MEASURE_TOL = 0.10  # a.e. verdict: unsettled set < this fraction of (b-a)
_PT_TOL = 0.01          # pointwise verdict: unsettled set < this fraction of (b-a)
_MAX_WORK = 50_000_000  # n_max * n_samples cap for the sampling loop
# Header budget above the 2x2 grid, in inches: one-line suptitle + figure
# legend + the two-line panel titles.  The grid top drops on short figures
# so the legend never collides with the titles or the suptitle.
_HEADER_IN = 1.25


class ConvergenceInfo:
    """Computed data behind a ``convergence_modes`` figure.

    Attributes
    ----------
    levels : np.ndarray
        The member indices ``k = 1 .. n_max``.
    sup_errors : np.ndarray
        ``max_x |f_k - f|`` — the uniform-convergence metric.
    l1_norms : np.ndarray
        ``∫ |f_k - f| dx`` — the L1-convergence metric.
    bad_measures : np.ndarray
        Length of the *unsettled* set at step ``k`` (NaN before the settle
        window fills): the points where the last ``_WINDOW`` members still
        differ by more than the threshold, or where the newest member has
        stopped approaching ``f`` while still ``eps`` away from it.  Shrinking
        to 0 is the honest finite-window test of pointwise / a.e. convergence.
    slice_xs : np.ndarray
        The fixed ``x_0`` values where pointwise convergence is inspected.
    slice_errors : np.ndarray
        ``|f_k(x_0) - f(x_0)|`` for every ``k`` and every slice.
    epsilon_val : float
        The absolute value threshold used by the verdicts.
    pointwise_ok, uniform_ok, ae_ok, l1_ok : bool
        Verdicts.  Pointwise / a.e. hold when the unsettled set is tiny
        (``pt_tol`` / a tenth of the interval); uniform holds when
        ``sup |f_k - f|`` is below ``epsilon_val``; L1 holds when the last
        norm has dropped below a quarter of the first member's norm — the
        honest "→ 0" statement on a finite draw.
    """

    def __init__(
        self,
        levels: np.ndarray,
        sup_errors: np.ndarray,
        l1_norms: np.ndarray,
        bad_measures: np.ndarray,
        slice_xs: np.ndarray,
        slice_errors: np.ndarray,
        epsilon_val: float,
        pointwise_ok: bool,
        uniform_ok: bool,
        ae_ok: bool,
        l1_ok: bool,
    ) -> None:
        self.levels = levels
        self.sup_errors = sup_errors
        self.l1_norms = l1_norms
        self.bad_measures = bad_measures
        self.slice_xs = slice_xs
        self.slice_errors = slice_errors
        self.epsilon_val = float(epsilon_val)
        self.pointwise_ok = bool(pointwise_ok)
        self.uniform_ok = bool(uniform_ok)
        self.ae_ok = bool(ae_ok)
        self.l1_ok = bool(l1_ok)


# -------------------------------------------------------------------
# The three classic counterexamples
# -------------------------------------------------------------------


def _typewriter(n: int, x):
    """The n-th typewriter indicator on [0, 1].

    Block ``k`` splits the unit interval into ``2**k`` equal sub-intervals;
    the members of a block sweep across them left to right.  Every point of
    ``[0, 1]`` is covered infinitely often, so the sequence converges in
    measure / L1 but fails at *every* point.
    """
    x = np.asarray(x, dtype=np.float64)
    k = int(math.floor(math.log2(n + 1)))
    j = n - (2 ** k - 1)
    lo, hi = j / 2 ** k, (j + 1) / 2 ** k
    return np.where((x >= lo) & (x < hi), 1.0, 0.0)


FAMILIES: Dict[str, dict] = {
    "x_pow_n": {
        "label": r"f_n(x) = x^n",
        "a": 0.0, "b": 1.0, "n_max": 16, "epsilon": 0.08,
        "family": lambda n, x: np.power(np.asarray(x, dtype=np.float64), n),
        "f": lambda x: np.where(np.asarray(x, dtype=np.float64) < 1.0, 0.0, 1.0),
        "note": ("converges pointwise (and a.e.) but NOT uniformly: near x = 1 "
                 "the tube of width epsilon never closes, however large n gets"),
    },
    "thin_spike": {
        "label": r"f_n(x) = n \cdot \mathbf{1}_{[0,\ 1/n]}(x)",
        "a": 0.0, "b": 1.0, "n_max": 20, "epsilon": 0.08,
        "family": lambda n, x: np.where(
            np.asarray(x, dtype=np.float64) <= 1.0 / n, float(n), 0.0
        ),
        "f": lambda x: np.zeros_like(np.asarray(x, dtype=np.float64)),
        "note": ("converges to 0 at every x > 0 (the only failure is the point "
                 "x = 0, a null set) yet integral |f_n| = 1 stays put: "
                 "almost everywhere, but not in L1"),
    },
    "typewriter": {
        "label": r"\mathrm{typewriter\ sequence}",
        "a": 0.0, "b": 1.0, "n_max": 16, "epsilon": 0.15,
        "family": _typewriter,
        "f": lambda x: np.zeros_like(np.asarray(x, dtype=np.float64)),
        "note": ("converges to 0 in L1 (the intervals shrink) yet fails at every "
                 "point: the typewriter marches across [0, 1] forever"),
    },
}


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------


def _family_colour(k: int, n_max: int):
    """Interpolate the per-member line colour: light (n=1) to dark (n=n_max)."""
    t = k / (n_max - 1) if n_max > 1 else 0.0
    lo = np.asarray(mcolors.to_rgb(_FAMILY_LO))
    hi = np.asarray(mcolors.to_rgb(_FAMILY_HI))
    return lo + t * (hi - lo)


def _sample_family(family, n_max: int, x: np.ndarray) -> np.ndarray:
    """Evaluate every member ``family(n, x)`` on the grid, defensively.

    Returns an array of shape ``(n_max, len(x))``.  The family is allowed to
    return a scalar (a constant member) or a vector; anything non-finite or
    mis-shaped is reported with the offending ``n``.
    """
    x = np.asarray(x, dtype=np.float64)
    rows = []
    for n in range(1, n_max + 1):
        try:
            arr = np.asarray(family(n, x), dtype=np.float64)
            arr = np.broadcast_to(arr, x.shape)
        except Exception as exc:
            raise ValueError(
                f"Family raised {type(exc).__name__!r} for n={n}. "
                "Check that family(n, x) is defined on the interval."
            ) from exc
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"Family returned non-finite values for n={n}.")
        rows.append(arr)
    return np.stack(rows)


def _sample_target(f, x: np.ndarray) -> np.ndarray:
    """Evaluate the limit function ``f`` on the grid."""
    try:
        arr = np.asarray(f(x), dtype=np.float64)
        return np.broadcast_to(arr, x.shape).copy()
    except Exception as exc:
        raise ValueError(
            f"Target f raised {type(exc).__name__!r} during sampling. "
            "Check that f is defined on the interval."
        ) from exc


def _settle_bad_measures(
    y_all: np.ndarray, y_f: np.ndarray, eps_val: float, dx: float
) -> np.ndarray:
    """Measure the *unsettled* set at every step of the sequence.

    A grid point is unsettled at step ``k`` when, over the last ``_WINDOW``
    members ending at ``k``, either

    - they still disagree among themselves (``oscillation >= eps_val``), or
    - the newest one is ``>= eps_val`` away from the limit ``f`` and no
      closer to it than the window's first member — the sequence has *stuck*
      at the wrong value instead of converging to ``f``.

    This is the honest finite-window test of pointwise/a.e. convergence: it
    separates ``x^n`` (settles everywhere), the thin spike (settles except
    near the tip), the typewriter (a large unsettled region keeps sweeping
    the axis) and a family that converges *to the wrong limit*.  A plain
    ``m({|f_k - f| >= eps})`` cannot — that quantity is exactly *convergence
    in measure* and would swallow the typewriter.
    """
    n_max, _ = y_all.shape
    bad = np.full(n_max, np.nan)
    for k in range(_WINDOW - 1, n_max):
        tail = y_all[k - _WINDOW + 1 : k + 1]
        oscillation = np.max(np.abs(tail - tail[-1]), axis=0)
        err = np.abs(tail - y_f)
        stuck_off_f = (err[-1] >= eps_val) & (err[-1] >= err[0])
        unsettled = (oscillation >= eps_val) | stuck_off_f
        bad[k] = float(np.count_nonzero(unsettled) * dx)
    return bad


def _unsettled_mask(y_all: np.ndarray, y_f: np.ndarray, eps_val: float) -> np.ndarray:
    """Boolean mask of the grid points unsettled by the last drawn members."""
    tail = y_all[-_WINDOW:]
    oscillation = np.max(np.abs(tail - tail[-1]), axis=0)
    err = np.abs(tail - y_f)
    stuck_off_f = (err[-1] >= eps_val) & (err[-1] >= err[0])
    return (oscillation >= eps_val) | stuck_off_f


def _draw_family(ax: Axes, x, y_all: np.ndarray, y_f: np.ndarray) -> None:
    """Draw the gradient family (light -> dark) and the limit f on top."""
    n_max = y_all.shape[0]
    for k in range(n_max):
        ax.plot(x, y_all[k], color=_family_colour(k, n_max), linewidth=1.4,
                zorder=3 + k)
    ax.plot(x, y_f, color=_CURVE_COLOUR, linewidth=_CURVE_WIDTH, zorder=100)


def _metric_inset(ins: Axes, ks, values, threshold: float, ylabel: str) -> None:
    """Small log-scale metric panel tucked into a corner of a camera."""
    vals = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    ins.semilogy(ks, vals, marker="o", markersize=3, color=_FAMILY_HI,
                 linewidth=1.2)
    if threshold > 0:  # a non-positive line cannot sit on a log axis
        ins.axhline(threshold, color=_COLOR_BAD, linewidth=0.8, linestyle="--")
    ins.set_xticks([int(ks[0]), int(ks[-1])])
    ins.set_ylabel(ylabel, fontsize=7)
    ins.tick_params(labelsize=6)
    # translucent so the family curves beneath the corner inset stay visible
    ins.set_facecolor((1.0, 1.0, 1.0, 0.85))


def _panel_title(line1: str, line2: str, ok: bool) -> str:
    glyph = "✓" if ok else "✗"
    return line1 + "\n" + glyph + "  " + line2


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def convergence_modes(
    family,
    a: Optional[float] = None,
    b: Optional[float] = None,
    n_max: Optional[int] = None,
    *,
    f=None,
    epsilon: Optional[float] = None,
    n_samples: int = 2000,
    figsize: Tuple[float, float] = (15, 10),
) -> Tuple[Figure, Tuple[Axes, Axes, Axes, Axes], ConvergenceInfo]:
    """Draw one function family under four convergence cameras.

    The same sequence ``f_n`` is judged four ways — pointwise, uniform,
    almost-everywhere and L1 — which is precisely how the classic examples
    separate the modes: ``x^n`` is pointwise- but not uniformly-convergent,
    the thin spike ``n·1_[0, 1/n]`` converges a.e. but not in L1, and the
    typewriter sequence converges in L1 yet fails at every point.

    Four panels (top-left, top-right, bottom-left, bottom-right):
    - **Pointwise**: the family with fixed-``x_0`` slice markers; the verdict
      reports the measure of the still-*unsettled* set (the last ``_WINDOW``
      members still disagreeing) — the honest finite-window test.
    - **Uniform**: the family inside an ``epsilon``-tube around ``f``; the
      inset shows ``sup_x |f_k - f|`` decaying (or not) on a log axis.
    - **Almost everywhere**: the same unsettled points as red ``x`` markers —
      where the sequence keeps refusing to converge — with the measure of
      that set shrinking (or not) in the inset.
    - **L1**: the ``|f_{n_max} - f|`` area shaded, with ``∫ |f_k - f|``
      decaying (or not) in the inset.

    Parameters
    ----------
    family : callable or str
        Either a callable ``family(n, x) -> float`` giving the n-th member,
        or the name of a built-in counterexample (one of ``FAMILIES``).
    a, b : float
        Domain interval.  Required when ``family`` is a callable; presets
        supply their own defaults (all on ``[0, 1]``).
    n_max : int
        Number of members drawn (``3 <= n_max <= 60``).  At least three
        members are needed for the settle test behind the pointwise/a.e.
        cameras.  Presets pick a value that makes the counterexample visible.
    f : callable or None
        The limit function ``f(x)``.  When ``None`` the *last drawn member*
        is used as the reference, so the four metrics show how the early
        members approach the latest one.
    epsilon : float
        Relative value threshold ``0 < epsilon < 1`` used by the verdicts
        (as a fraction of the family's range).
    n_samples : int
        Dense evaluation points (``2 <= n_samples <= 1e6``).
    figsize : (float, float)

    Returns
    -------
    fig : Figure
    (ax_pt, ax_un, ax_ae, ax_l1) : four axes
    info : ConvergenceInfo
        The per-member metrics and verdicts (see the class docstring).
    """
    # -- resolve a preset name, then validate --------------------------
    preset_label = None
    preset_note = None
    if isinstance(family, str):
        if family not in FAMILIES:
            raise ValueError(
                f"Unknown preset {family!r}. Choose from {sorted(FAMILIES)} "
                "or pass a callable family(n, x)."
            )
        spec = FAMILIES[family]
        preset_label = spec["label"]
        preset_note = spec["note"]
        family = spec["family"]
        a = spec["a"] if a is None else a
        b = spec["b"] if b is None else b
        n_max = spec["n_max"] if n_max is None else n_max
        f = spec["f"] if f is None else f
        epsilon = spec["epsilon"] if epsilon is None else epsilon

    _check_callable(family)
    if a is None or b is None:
        raise ValueError("a and b are required when family is a callable.")
    _check_interval(a, b)
    _check_n_max(n_max if n_max is not None else 8)
    if n_max is not None and n_max < _WINDOW:
        raise ValueError(
            f"n_max={n_max} is too small: the pointwise/a.e. settle test needs "
            f"at least {_WINDOW} members to compare. Use n_max >= {_WINDOW}."
        )
    if f is not None:
        _check_callable(f, name="f")
    _check_epsilon(epsilon if epsilon is not None else 0.08)
    _check_samples(n_samples)
    _check_figsize(figsize)

    n_max = 8 if n_max is None else int(n_max)
    epsilon = 0.08 if epsilon is None else float(epsilon)
    if n_max * n_samples > _MAX_WORK:
        raise ValueError(
            f"n_max={n_max} with n_samples={n_samples} is too expensive: "
            f"{n_max} members x {n_samples} samples exceeds the work bound of "
            f"{_MAX_WORK}. Lower n_max or n_samples."
        )

    # -- dense sampling ---------------------------------------------
    x_dense = np.linspace(a, b, n_samples, dtype=np.float64)
    y_all = _sample_family(family, n_max, x_dense)
    if f is None:
        y_f = y_all[-1].copy()
    else:
        y_f = _sample_target(f, x_dense)
    if np.any(~np.isfinite(y_f)):
        raise ValueError("Target f returned non-finite values.")

    lo = float(np.minimum(y_all.min(), y_f.min()))
    hi = float(np.maximum(y_all.max(), y_f.max()))
    if not math.isfinite(hi - lo):
        raise ValueError(
            "Function family range is too large to measure: y_max - y_min "
            "overflows. Scale the family values to a smaller range."
        )
    eps_val = epsilon * (hi - lo) if hi > lo else 1.0

    # -- metrics ----------------------------------------------------
    dx = (b - a) / (n_samples - 1)
    weights = np.full(n_samples, dx)
    weights[0] = weights[-1] = 0.5 * dx
    sup_errors = np.max(np.abs(y_all - y_f), axis=1)
    l1_norms = np.sum(np.abs(y_all - y_f) * weights, axis=1)
    bad_measures = _settle_bad_measures(y_all, y_f, eps_val, dx)

    ks = np.arange(1, n_max + 1, dtype=np.int64)
    slice_xs = a + (b - a) * np.array([0.25, 0.5, 0.75])
    slice_idx = np.searchsorted(x_dense, slice_xs)
    slice_errors = np.abs(y_all[:, slice_idx] - y_f[slice_idx])

    # -- verdicts ---------------------------------------------------
    # The pointwise and a.e. thresholds are *absolute* fractions of the
    # interval (not grid-cell counts), so the verdicts do not depend on how
    # densely the family was sampled.
    bad_last = bad_measures[n_max - 1]
    pt_tol = _PT_TOL * (b - a)
    ae_tol = _AE_MEASURE_TOL * (b - a)
    pointwise_ok = bool(np.isfinite(bad_last) and bad_last <= pt_tol)
    # a.e. additionally requires the unsettled set to still be shrinking in
    # the final step — a residual that merely plateaus is not converging to 0.
    prev_bad = bad_measures[n_max - 2] if n_max > _WINDOW else np.nan
    ae_shrinking = bool(not np.isfinite(prev_bad) or bad_last <= prev_bad)
    ae_ok = bool(np.isfinite(bad_last) and bad_last <= ae_tol and ae_shrinking)
    uniform_ok = bool(sup_errors[-1] < eps_val)
    # The L1 norm is an *area*; a range-relative tolerance would be inflated
    # by a tall thin tip (the spike's range is ~20 while its integral is 1),
    # so the verdict asks whether ∫|f_k - f| has dropped below a quarter of
    # the first member's value — the honest "→ 0" statement on a finite draw.
    # When the first member already equals f (l1_head == 0), the comparison
    # ``0 <= 0`` still works: it passes exactly when the tail also vanishes.
    l1_head = l1_norms[0]
    l1_ok = bool(l1_norms[-1] <= 0.25 * l1_head)

    info = ConvergenceInfo(
        levels=ks, sup_errors=sup_errors, l1_norms=l1_norms,
        bad_measures=bad_measures, slice_xs=slice_xs,
        slice_errors=slice_errors, epsilon_val=eps_val,
        pointwise_ok=pointwise_ok, uniform_ok=uniform_ok,
        ae_ok=ae_ok, l1_ok=l1_ok,
    )

    # -- layout -------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.20)
    ax_pt = fig.add_subplot(gs[0, 0])
    ax_un = fig.add_subplot(gs[0, 1])
    ax_ae = fig.add_subplot(gs[1, 0])
    ax_l1 = fig.add_subplot(gs[1, 1])

    ylims = (lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.06) if hi > lo else (lo - 1.0, lo + 1.0)

    # -- POINTWISE (top-left) ----------------------------------------
    _draw_family(ax_pt, x_dense, y_all, y_f)
    for x0 in slice_xs:
        ax_pt.axvline(x0, color="#94a3b8", linewidth=0.6, linestyle="--", zorder=2)
    # dots at each slice for the last few members, marching to f's value
    dot_start = max(0, n_max - 6)
    for idx in range(len(slice_xs)):
        x0 = float(slice_xs[idx])
        for k in range(dot_start, n_max):
            ax_pt.plot(x0, y_all[k, slice_idx[idx]], marker="o", markersize=4,
                       color=_family_colour(k, n_max), zorder=150)
        ax_pt.plot(x0, y_f[slice_idx[idx]], marker="*", markersize=11,
                   color=_COLOR_BAD, zorder=160)
    unsettled = _unsettled_mask(y_all, y_f, eps_val)
    x_bad = x_dense[unsettled]
    if x_bad.size:
        ax_pt.plot(x_bad, np.full_like(x_bad, ylims[0]), marker="x",
                   linestyle="none", color=_COLOR_BAD, markersize=5,
                   markeredgewidth=1.2, zorder=120)
    ax_pt.set_title(
        _panel_title(r"Pointwise — $f_n(x_0) \to f(x_0)$",
                     f"unsettled set m = {bad_last:.4f}  (tol {pt_tol:.4f})",
                     pointwise_ok),
        fontsize=10.5, color=_COLOR_OK if pointwise_ok else _COLOR_BAD,
    )
    ax_pt.set_xlim(a, b)
    ax_pt.set_ylim(*ylims)
    ax_pt.set_xlabel("$x$")
    ax_pt.set_ylabel("$y$")

    # -- UNIFORM (top-right) ------------------------------------------
    ax_un.fill_between(x_dense, y_f - eps_val, y_f + eps_val,
                       color=_COLOR_TUBE, alpha=_ALPHA_TUBE, zorder=1)
    _draw_family(ax_un, x_dense, y_all, y_f)
    ins_un = ax_un.inset_axes([0.03, 0.07, 0.32, 0.26])
    _metric_inset(ins_un, ks, sup_errors, eps_val, r"$\sup_x |f_k - f|$")
    ax_un.set_title(
        _panel_title(r"Uniform — $\sup_x |f_n - f| \to 0$",
                     f"sup error = {sup_errors[-1]:.4f}  (tol {eps_val:.3f})",
                     uniform_ok),
        fontsize=10.5, color=_COLOR_OK if uniform_ok else _COLOR_BAD,
    )
    ax_un.set_xlim(a, b)
    ax_un.set_ylim(*ylims)
    ax_un.set_xlabel("$x$")
    ax_un.set_ylabel("$y$")

    # -- A.E. (bottom-left) --------------------------------------------
    _draw_family(ax_ae, x_dense, y_all, y_f)
    if x_bad.size:
        ax_ae.plot(x_bad, np.full_like(x_bad, ylims[0]), marker="x",
                   linestyle="none", color=_COLOR_BAD, markersize=5,
                   markeredgewidth=1.2, zorder=120)
    valid = np.isfinite(bad_measures)
    ins_ae = ax_ae.inset_axes([0.03, 0.07, 0.32, 0.26])
    _metric_inset(ins_ae, ks[valid], bad_measures[valid], ae_tol,
                  r"$m(\text{unsettled})$")
    ax_ae.set_title(
        _panel_title(r"A.e. — failures confined to a null set",
                     f"unsettled m = {bad_last:.4f}  (tol {ae_tol:.3f})",
                     ae_ok),
        fontsize=10.5, color=_COLOR_OK if ae_ok else _COLOR_BAD,
    )
    ax_ae.set_xlim(a, b)
    ax_ae.set_ylim(*ylims)
    ax_ae.set_xlabel("$x$")
    ax_ae.set_ylabel("$y$")

    # -- L1 (bottom-right) ---------------------------------------------
    ax_l1.fill_between(x_dense, y_f, y_all[-1], color=_COLOR_AREA,
                       alpha=_ALPHA_AREA, zorder=1)
    _draw_family(ax_l1, x_dense, y_all, y_f)
    ins_l1 = ax_l1.inset_axes([0.03, 0.07, 0.32, 0.26])
    _metric_inset(ins_l1, ks, l1_norms, 0.25 * l1_head, r"$\int |f_k - f|$")
    ax_l1.set_title(
        _panel_title(r"L1 — $\int |f_n - f| \to 0$",
                     f"L1 norm = {l1_norms[-1]:.4f}  (needs <= {0.25 * l1_head:.4f})",
                     l1_ok),
        fontsize=10.5, color=_COLOR_OK if l1_ok else _COLOR_BAD,
    )
    ax_l1.set_xlim(a, b)
    ax_l1.set_ylim(*ylims)
    ax_l1.set_xlabel("$x$")
    ax_l1.set_ylabel("$y$")

    # -- figure-level ---------------------------------------------------
    # Header budget above the grid: one-line suptitle + the legend + the
    # two-line panel titles.  Drop the grid top on short figures so the
    # legend never collides with the titles below it or the suptitle above.
    _h = float(figsize[1])
    top = min(0.80, 1.0 - _HEADER_IN / _h)
    legend_y = top + 0.52 / _h
    if preset_label:
        head = rf"Convergence modes of ${preset_label}$  ·  one family, four cameras"
    else:
        head = "Convergence modes of a function family  ·  four cameras"
    fig.suptitle(head, fontsize=12.5, y=0.965)
    fig.legend(
        handles=[
            Line2D([0], [0], color=_family_colour(0, n_max), linewidth=1.5,
                   label="n = 1"),
            Line2D([0], [0], color=_family_colour(n_max - 1, n_max),
                   linewidth=2.2, label=f"n = {n_max}"),
            Line2D([0], [0], color=_CURVE_COLOUR, linewidth=_CURVE_WIDTH,
                   label="f (limit)"),
            Line2D([0], [0], marker="x", color=_COLOR_BAD, linestyle="none",
                   label="unsettled point"),
        ],
        loc="lower center", ncol=4, fontsize=9, framealpha=0.9,
        edgecolor="#cccccc", bbox_to_anchor=(0.5, legend_y),
    )
    if preset_note:
        fig.text(0.5, 0.012, preset_note, ha="center", va="bottom", fontsize=10,
                 style="italic",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                           edgecolor="#cccccc", alpha=0.9))
    # Manual margins: the suptitle sits at the very top, the legend just
    # below it, then the 2x2 grid; tight_layout cannot handle the insets.
    fig.subplots_adjust(left=0.07, right=0.97, top=top, bottom=0.08)

    return fig, (ax_pt, ax_un, ax_ae, ax_l1), info
