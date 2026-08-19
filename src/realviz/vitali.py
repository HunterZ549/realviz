"""The Vitali set and non-measurable sets.

``v0.5`` of ``realviz``:
- ``vitali_illustration`` — a conceptual, two-by-two figure that teaches
  *why* a transversal of the rationals cannot be Lebesgue measurable.
  The true Vitali set is not constructible, so nothing here is sampled
  from data: every element is a finite, labeled schematic.

  Four panels, read as a chain of reasoning:

  1. **Equivalence classes** — ``x ~ y`` iff ``x - y in Q``.  Each class
     is a countable dense cloud: every window meets every class.  Three
     genuine cosets are drawn, with anchors ``0``, ``sqrt(2) - 1`` and
     ``sqrt(3) - 1`` whose pairwise differences are provably irrational,
     so the three clouds are truly disjoint.
  2. **Choice** — the Axiom of Choice picks one representative per class;
     the chosen points are ``V = {((k+1) sqrt(2)) mod 1}``.  Amber is
     ``V`` everywhere in the figure.
  3. **Tiling** — the rational translates ``V + r`` (``r in {j/n}``) are
     pairwise disjoint: ``(k - l) sqrt(2) - (i - j)/n`` is irrational for
     ``k != l``, so no two of the drawn points collide mod 1.  That is
     verified numerically and exposed in ``VitaliInfo``.
  4. **The trap** — translation invariance forces ``m(V + r) = m(V)``;
     countable additivity demands ``1 = sum over all translates``.  Both
     ``m(V) = 0`` and ``m(V) > 0`` contradict that sum, so ``V`` is not
     measurable.  The inner/outer measures ``m_*(V) = 0`` and
     ``m^*(V) > 0`` are stated, not derived here: ``m^*(V)`` is positive
     because its countably many translates cover ``[0, 1)``, but the
     transversal can be chosen inside any interval, so the outer measure
     can be arbitrarily small -- it is not ``1`` in general.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._validation import _check_figsize, _check_vitali_counts

# -- style constants ------------------------------------------------
_COL_CLASS = ("#2a78d6", "#eb6834", "#1baf7a")  # one colour per equivalence class
_COL_REP = "#b45309"        # amber: V and its chosen representatives
_COL_TILE = "#7c3aed"       # the non-V translates in the tiling panel; violet
                            # deliberately avoids the panel-1 class blues
_COL_CLOUD = "#cbd5e1"      # the faint grey cloud of one class's many points
_COL_BAR = "#a8a29e"        # the unrolled circle bar / strip baseline
_COL_OK = "#16a34a"
_COL_BAD = "#dc2626"
_ALPHA_CLOUD = 0.75

# Panel-1 cosets: the class of 0 is Q itself; sqrt(2)-1 and sqrt(3)-1 are
# irrational, and sqrt(3) - sqrt(2) is irrational too, so the three classes
# are pairwise distinct.  (sqrt(2)-1 doubles as the first rep of panel 2,
# which is why it is "the chosen one".)
_CLASS_ANCHORS: Tuple[float, ...] = (0.0, math.sqrt(2.0) - 1.0, math.sqrt(3.0) - 1.0)
_QS_MAIN = 48     # reduced-fraction grid on the main classes panel
_QS_FINE = 300    # finer grid for the density zoom (the honesty check)
_DENSITY_WINDOW = (0.40, 0.48)  # the zoom window, provably met by every class


class VitaliInfo(NamedTuple):
    """Computed data behind a ``vitali_illustration`` figure.

    Attributes
    ----------
    class_anchors : Tuple[float, ...]
        The three class representatives drawn in panel 1: ``0``, ``sqrt(2)-1``
        and ``sqrt(3)-1``, with provably irrational pairwise differences.
    reps : np.ndarray
        The chosen representatives ``((k+1) sqrt(2)) mod 1``, ``k = 0..n_reps-1``.
        Amber in the figure; ``V = {reps}``.
    shifts : np.ndarray
        The rational translates ``{j / n_shifts}`` tiling the circle in panel 3.
    translate_points : np.ndarray
        ``(reps[:, None] + shifts[None, :]) mod 1``, shape ``(n_reps, n_shifts)``.
    disjoint_count : int
        The number of pairwise distinct points among the translates
        (``n_reps * n_shifts`` — the tiling copies really are disjoint).
    min_gap : float
        The smallest circular gap between any two translate points.
    n_rows_choice : int
        How many classes (rows) the choice panel draws.
    seed : int
        The jitter seed, so the cloud scatter is reproducible.
    """

    class_anchors: Tuple[float, ...]
    reps: np.ndarray
    shifts: np.ndarray
    translate_points: np.ndarray
    disjoint_count: int
    min_gap: float
    n_rows_choice: int
    seed: int


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------


def _reduced_fractions(q_max: int) -> np.ndarray:
    """All reduced fractions ``p/q`` in ``(0, 1)`` with ``2 <= q <= q_max``."""
    out = set()
    for d in range(2, q_max + 1):
        for k in range(1, d):
            if math.gcd(k, d) == 1:
                out.add(k / d)
    return np.array(sorted(out), dtype=np.float64)


def _reps(n: int) -> np.ndarray:
    """The chosen representatives ``{((k+1) sqrt(2)) mod 1 : k = 0..n-1}``."""
    return ((np.arange(n, dtype=np.float64) + 1.0) * math.sqrt(2.0)) % 1.0


def _min_circular_gap(points: np.ndarray) -> float:
    """Smallest gap between two points on the circle ``[0, 1)``."""
    x = np.sort(np.unique(np.asarray(points, dtype=np.float64) % 1.0))
    gaps = np.diff(x)
    wrap = (x[0] + 1.0) - x[-1]
    return float(np.min(np.concatenate([gaps, [wrap]])))


# -------------------------------------------------------------------
# Panel 1: equivalence classes as dense clouds
# -------------------------------------------------------------------


def _draw_classes(ax: Axes, seed: int) -> None:
    rng = np.random.default_rng(seed)
    qs = _reduced_fractions(_QS_MAIN)
    lo, hi = _DENSITY_WINDOW

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.45, 1.40)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([])

    # the unrolled circle
    ax.plot([0.0, 1.0], [0.0, 0.0], color=_COL_BAR, linewidth=9,
            solid_capstyle="round", zorder=1)
    ax.text(0.0, -0.24, "0", ha="left", fontsize=8, color="#475569")
    ax.text(1.0, -0.24, "1  ≡  0  (mod 1)", ha="right", fontsize=8, color="#475569")
    ax.text(0.42, -0.40, "the circle  $[0,1)$  — rolled up", ha="center",
            fontsize=8, color="#475569")

    # the zoom window and its density inset
    ax.axvspan(lo, hi, color="#94a3b8", alpha=0.10, zorder=0)
    ax.add_patch(Rectangle((lo, -0.16), hi - lo, 0.46, fill=False,
                           edgecolor="#64748b", linestyle="--", linewidth=1.2))
    ins = ax.inset_axes([0.50, 0.44, 0.44, 0.40])
    qs_fine = _reduced_fractions(_QS_FINE)
    for a, col in zip(_CLASS_ANCHORS, _COL_CLASS):
        xs = (a + qs_fine) % 1.0
        mask = (xs >= lo) & (xs < hi)
        n_in = int(mask.sum())
        if n_in < 3:
            raise AssertionError(
                f"Density check failed: class {a!r} has only {n_in} points "
                f"in the window {_DENSITY_WINDOW} at q <= {_QS_FINE}."
            )
        ys = 0.5 + rng.uniform(-0.04, 0.04, n_in)
        ins.scatter(xs[mask], ys, s=4, color=col, alpha=0.9, zorder=3)
    ins.set_xlim(*_DENSITY_WINDOW)
    ins.set_ylim(0.2, 0.8)
    ins.set_yticks([])
    ins.set_xticks([0.40, 0.44, 0.48])
    ins.tick_params(labelsize=6)
    ins.set_title(f"zoom: finer grid $q\\leq{_QS_FINE}$ — every window\n"
                  "meets every class", fontsize=7.5)
    for spine in ins.spines.values():
        spine.set_color("#64748b")

    # the three dense clouds, one per class
    for a, col in zip(_CLASS_ANCHORS, _COL_CLASS):
        xs = (a + qs) % 1.0
        ys = 0.22 + rng.uniform(-0.05, 0.05, xs.size)
        ax.scatter(xs, ys, s=6, color=col, alpha=_ALPHA_CLOUD, zorder=3)
        if a == _CLASS_ANCHORS[1]:
            # sqrt(2)-1 doubles as panel 2's first rep: the anchor itself is
            # a FILLED amber dot, so "amber = chosen into V" is unambiguous
            # (the cloud around it stays the class's orange).
            ax.scatter([a], [0.22], s=46, facecolor=_COL_REP, edgecolor="white",
                       linewidth=0.9, zorder=6)
            ax.text(a + 0.015, 0.22, r"$u_0 \in V$", fontsize=7.5, va="center",
                    color=_COL_REP, zorder=7)
        else:
            ax.scatter([a], [0.22], s=46, facecolor="none",
                       edgecolor=col, linewidth=1.6, zorder=5)

    # legend of the three classes
    ax.text(0.0, 1.28, "three genuine cosets (drawn, of $\\aleph_0$):",
            fontsize=8, color="#475569")
    for i, (a, col, tag) in enumerate(zip(
            _CLASS_ANCHORS, _COL_CLASS,
            (r"$0 + \mathbb{Q}$",
             r"$\sqrt{2}-1 + \mathbb{Q}$  (chosen in panel 2)",
             r"$\sqrt{3}-1 + \mathbb{Q}$"))):
        y = 1.06 - 0.14 * i
        ax.plot([0.0], [y], marker="o", mfc=col, mec="none", ms=7,
                clip_on=False)
        ax.text(0.06, y, tag, fontsize=8.5, color="#1e293b", va="center")

    ax.text(0.02, 0.48, "each vertical strip is a finite sample of ONE class —\n"
            "countable and dense, disjoint from the others",
            ha="left", va="top", fontsize=8, color="#334155")
    ax.text(0.5, -0.13, "schematic: ~%d points of each countable class shown"
            % len(qs), ha="center", fontsize=7.5, style="italic",
            color="#64748b")


# -------------------------------------------------------------------
# Panel 2: choice — one representative per class
# -------------------------------------------------------------------


def _draw_choice(ax: Axes, n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reps = _reps(n_rows)
    ax.set_xlim(-0.06, 1.30)
    ax.set_ylim(-3.8, 8.9)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([])

    for k in range(n_rows):
        y = 7.9 - k
        grid = (reps[k] + np.arange(60) / 60.0) % 1.0
        ax.scatter(grid, np.full(60, y), s=5, color=_COL_CLOUD,
                   alpha=0.7, zorder=3)
        ax.scatter([reps[k]], [y], s=46, color=_COL_REP, edgecolor="white",
                   linewidth=0.8, zorder=6)
        ax.plot([reps[k], reps[k]], [y, -1.15], color=_COL_REP, linewidth=0.6,
                linestyle=":", alpha=0.6, zorder=2)
        ax.text(1.06, y, rf"$C_{k}$  (rep $u_{k}$)", ha="left", va="center",
                fontsize=8, color="#475569")

    # the assembled set V at the bottom of the panel
    ax.plot([0.0, 1.0], [-1.15, -1.15], color=_COL_BAR, linewidth=5,
            solid_capstyle="round", zorder=1)
    ax.scatter(reps, np.full(n_rows, -1.15), s=34, color=_COL_REP,
               edgecolor="white", linewidth=0.7, zorder=4)
    ax.text(0.5, -1.25, "$V = \\{u_0, \\dots, u_{%d}\\}$ — one amber point per\n"
            "class drawn, collected by the Axiom of Choice — a finite sample of $V$"
            % (n_rows - 1),
            ha="center", va="top", fontsize=8.5, color="#1e293b")
    ax.text(0.5, -2.85, "FINITE SCHEMATIC — the true class has countably\n"
            "many points; only 60 shown per row",
            ha="center", va="top", fontsize=7.5, style="italic", color="#64748b")
    ax.text(0.5, 8.32, "row $C_0$ is the $\\sqrt{2}-1$ class from panel 1",
            ha="center", fontsize=7.5, color="#64748b")
    return reps


# -------------------------------------------------------------------
# Panel 3: rational translates tile the circle
# -------------------------------------------------------------------


def _draw_tiling(ax: Axes, reps: np.ndarray, shifts: np.ndarray) -> None:
    n_reps, n_shifts = len(reps), len(shifts)
    ax.set_xlim(-0.06, 1.30)
    ax.set_ylim(-4.8, 9.3)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([])

    for j in range(n_shifts):
        y = 7.4 - j
        pts = (reps + shifts[j]) % 1.0
        col = _COL_REP if j == 0 else _COL_TILE
        ax.scatter(pts, np.full(n_reps, y), s=18, color=col, zorder=4)
        label = "$V$" if j == 0 else rf"$V + {j}/{n_shifts}$"
        ax.text(1.06, y, label, ha="left", va="center", fontsize=8.5,
                color="#475569")

    # the same points on the true circle, so the wrap is natural
    pts_all = (reps[:, None] + shifts[None, :]) % 1.0
    flat = pts_all.reshape(-1)
    ins = ax.inset_axes([0.03, 0.20, 0.22, 0.20])
    theta = 2.0 * np.pi * flat
    # flat is row-major: index i*n_shifts + j is rep u_i + shift j/n, so the
    # V column (j == 0) sits at every n_shifts-th element.
    col_each = np.where(np.tile(np.arange(n_shifts) == 0, n_reps),
                        _COL_REP, _COL_TILE)
    ins.scatter(np.cos(theta), np.sin(theta), s=5, color=col_each, alpha=0.9,
                zorder=4)
    ins.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor=_COL_BAR,
                             linewidth=1.2))
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set_color("#64748b")
    ins.set_title("all drawn copies on the circle:\nno two points collide",
                  fontsize=7.5)

    count = n_reps * n_shifts
    gap = _min_circular_gap(flat)
    ax.text(0.5, 7.85, "each row is one translate $V + r_j$ (drawn, of countably many)\n"
            f"disjointness verified: ${n_reps}\\times{n_shifts} = {count}$ points, "
            f"pairwise distinct (mod 1), min gap {gap:.4f}",
            ha="center", fontsize=8, color="#334155")
    ax.text(0.55, -1.65, "row 0 (amber) is $V$ itself; violet rows are its shifts\n"
            r"$\{j/" + str(n_shifts) + r"\}$ — copies cannot overlap: "
            r"$(k-l)\sqrt{2} - (i-j)/" + str(n_shifts) + r" \notin \mathbb{Z}$",
            ha="center", va="top", fontsize=8, color="#334155")
    ax.text(0.55, -3.35, "taking ALL of $\\mathbb{Q}\\cap[0,1)$ covers the circle —\n"
            "every $x$ lies in exactly one copy (stated, not drawn here)",
            ha="center", va="top", fontsize=8, style="italic", color="#64748b")


# -------------------------------------------------------------------
# Panel 4: the measure trap
# -------------------------------------------------------------------


def _draw_trap(ax: Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.9, 10.8)
    ax.axis("off")

    ax.text(5, 10.25, "Assume $V$ is Lebesgue measurable:   $\\delta = m(V) \\geq 0$",
            ha="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor="#64748b", alpha=0.9))
    ax.text(5, 9.30, "measure is translation-invariant:   "
            "$m(V+r) = m(V) = \\delta$  for every $r \\in \\mathbb{Q}\\cap[0,1)$",
            ha="center", fontsize=10)
    ax.text(5, 8.20, r"$1 \;=\; m([0,1)) \;=\; \sum_{r\in\mathbb{Q}\cap[0,1)} "
            r"m(V+r) \;=\; \sum_{\aleph_0} \delta$",
            ha="center", fontsize=12)

    # the unit interval the copies must fill
    ax.add_patch(Rectangle((2.2, 6.75), 5.6, 0.5, facecolor="#1a1a2e", zorder=5))
    ax.text(5, 7.0, "$m([0,1)) = 1$", color="white", ha="center", va="center",
            fontsize=10)
    ax.text(5, 6.55, "countably many disjoint copies $V+r$ must fill it:  "
            "$\\aleph_0 \\cdot \\delta = 1$",
            ha="center", va="top", fontsize=8.5, color="#334155")

    # fork into the two horns
    ax.annotate("", xy=(2.9, 4.9), xytext=(3.6, 5.5),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.4))
    ax.annotate("", xy=(7.1, 4.9), xytext=(6.4, 5.5),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.4))

    # horn 1: delta = 0
    ax.text(2.9, 4.85, "if $\\delta = 0$", ha="center", fontsize=10.5,
            color=_COL_BAD)
    ax.plot([1.1, 4.7], [3.0, 3.0], color="#94a3b8", linewidth=1.2)
    for x in np.linspace(1.6, 4.2, 6):  # six empty copies
        ax.plot(x, 3.0, marker="o", mfc="none", mec=_COL_BAD, ms=8, mew=1.5,
                zorder=6)
    ax.plot([1.1, 4.7], [4.55, 4.55], color="#0f172a", linewidth=1.8,
            linestyle="--")
    ax.text(4.75, 4.55, "target = 1", fontsize=8, va="center", color="#0f172a")
    ax.annotate("", xy=(1.3, 3.0), xytext=(1.3, 4.55),
                arrowprops=dict(arrowstyle="->", color=_COL_BAD, lw=1.8))
    ax.text(2.9, 3.55, "$\\sum \\delta = 0 \\neq 1$", fontsize=8.5,
            color=_COL_BAD, ha="center")
    ax.text(2.9, 2.30, "✗   $m(V) \\neq 0$", ha="center", fontsize=11,
            color=_COL_BAD)

    # horn 2: delta > 0
    ax.text(7.1, 4.85, "if $\\delta > 0$", ha="center", fontsize=10.5,
            color=_COL_BAD)
    ax.plot([5.3, 8.9], [3.0, 3.0], color="#94a3b8", linewidth=1.2)
    h = 0.35  # bar height: six copies already overshoot the target of 1
    xs = np.linspace(5.8, 8.4, 6)
    for x in xs:
        ax.add_patch(Rectangle((x - 0.24, 3.0), 0.48, h, facecolor=_COL_BAD,
                               alpha=0.6))
    ax.plot(xs, 3.0 + np.cumsum(np.full(6, h)), color=_COL_BAD, linewidth=1.6,
            zorder=6)
    ax.plot([5.3, 8.9], [4.55, 4.55], color="#0f172a", linewidth=1.8,
            linestyle="--")
    ax.text(8.95, 4.55, "1", fontsize=9, va="center", color="#0f172a")
    ax.annotate("", xy=(8.5, 5.45), xytext=(8.4, 5.12),
                arrowprops=dict(arrowstyle="->", color=_COL_BAD, lw=1.8))
    ax.text(8.6, 5.45, "→ $\\infty$", fontsize=9, color=_COL_BAD)
    ax.text(7.1, 2.30, "✗   $m(V) \\neq \\delta > 0$", ha="center", fontsize=11,
            color=_COL_BAD)

    # conclusion
    ax.text(5, 1.35, "neither $\\delta = 0$ nor $\\delta > 0$ can sum to 1 — "
            "the assumption was wrong",
            ha="center", fontsize=9.5, color="#334155")
    ax.text(5, 0.45, "✓   $V$ is NOT Lebesgue measurable", ha="center",
            fontsize=12, color=_COL_OK)
    ax.text(5, -0.30, "$m_*(V) = 0$,   $m^*(V) > 0$  (inner / outer measure) — "
            "stated, not derived here", ha="center", fontsize=8,
            style="italic", color="#475569")


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def vitali_illustration(
    *,
    n_rows_choice: int = 8,
    n_reps: int = 12,
    n_shifts: int = 8,
    seed: int = 7,
    figsize: Tuple[float, float] = (15, 10),
) -> Tuple[Figure, Tuple[Axes, Axes, Axes, Axes], VitaliInfo]:
    """Draw the Vitali construction and the contradiction that kills it.

    The figure reads left-to-right, top-to-bottom as a four-step proof:
    classes are dense clouds (top-left); the Axiom of Choice assembles one
    representative per class into ``V`` (top-right); the rational translates
    tile the circle disjointly (bottom-left); and translation invariance
    makes the measure of ``V`` self-contradictory (bottom-right).

    The true Vitali set is not constructible, so the figure is honest about
    being a schematic: amber marks ``V`` and its chosen points, every dense
    cloud is a finite subset of a countable class, and the load-bearing
    numbers (the translates, their pairwise distinctness, the class anchors)
    are returned in ``VitaliInfo`` so the picture can be checked, not believed.

    Parameters
    ----------
    n_rows_choice : int
        How many classes (rows) the choice panel draws (``1 <= n <= 12``).
    n_reps : int
        How many representatives of ``V`` are sampled in the tiling panel
        (``1 <= n <= 30``).  Disjointness is verified for whatever is drawn.
    n_shifts : int
        How many rational translates ``{j/n}`` tile the circle in the tiling
        panel (``1 <= n <= 30``).
    seed : int
        Jitter seed for the cloud scatter; the geometry (reps, shifts, points)
        does not depend on it, only the vertical jitter.
    figsize : (float, float)

    Returns
    -------
    fig : Figure
    (ax_classes, ax_choice, ax_tiling, ax_trap) : four axes
    info : VitaliInfo
        The computed geometry (see the class docstring).
    """
    _check_vitali_counts(n_rows_choice, n_reps, n_shifts)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__!r}")
    _check_figsize(figsize)

    reps = _reps(n_reps)
    shifts = np.arange(n_shifts, dtype=np.float64) / n_shifts
    translate_points = (reps[:, None] + shifts[None, :]) % 1.0
    flat = translate_points.reshape(-1)
    disjoint_count = int(np.unique(np.round(flat, 12)).size)
    if disjoint_count != n_reps * n_shifts:
        raise AssertionError(
            "Internal check failed: rational translates should be pairwise "
            f"distinct, but found {disjoint_count} distinct points of "
            f"{n_reps * n_shifts}."
        )
    min_gap = _min_circular_gap(flat)
    info = VitaliInfo(
        class_anchors=_CLASS_ANCHORS,
        reps=reps,
        shifts=shifts,
        translate_points=translate_points,
        disjoint_count=disjoint_count,
        min_gap=min_gap,
        n_rows_choice=n_rows_choice,
        seed=seed,
    )

    # -- layout -----------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24,
                          width_ratios=[1.15, 1])
    ax_classes = fig.add_subplot(gs[0, 0])
    ax_choice = fig.add_subplot(gs[0, 1])
    ax_tiling = fig.add_subplot(gs[1, 0])
    ax_trap = fig.add_subplot(gs[1, 1])

    # -- panel titles ------------------------------------------------
    ax_classes.set_title(
        "1 · Equivalence classes — "
        r"$x \sim y \;\Longleftrightarrow\; x - y \in \mathbb{Q}$"
        "\nevery window meets every class (dense), yet the classes are disjoint",
        fontsize=10.5)
    ax_choice.set_title(
        "2 · Choice — one representative per class\n"
        "the Axiom of Choice assembles $V$: one amber point out of each class",
        fontsize=10.5)
    ax_tiling.set_title(
        "3 · Rational translates tile the circle\n"
        "$V + \\{0, \\frac{1}{8}, \\dots, \\frac{7}{8}\\}$ — drawn disjoint "
        r"(verified), covering the circle in the limit",
        fontsize=10.5)
    ax_trap.set_title(
        "4 · The trap — measure is translation-invariant\n"
        "countable additivity forces $1 = \\sum m(V+r) = \\sum \\delta$",
        fontsize=10.5)

    _draw_classes(ax_classes, seed)
    _draw_choice(ax_choice, n_rows_choice, seed)
    _draw_tiling(ax_tiling, reps, shifts)
    _draw_trap(ax_trap)

    # -- figure-level ------------------------------------------------
    fig.suptitle("The Vitali set — a transversal of $\\mathbb{Q}$ that cannot "
                 "be measured", fontsize=13.5, y=0.975)
    fig.text(
        0.5, 0.012,
        "FINITE SCHEMATIC throughout — the true Vitali set is not constructible, "
        "so every element drawn here is a labeled stand-in (amber = $V$). "
        "Anchors $0, \\sqrt{2}-1, \\sqrt{3}-1$ have provably irrational "
        "pairwise differences.",
        ha="center", va="bottom", fontsize=8.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#cccccc", alpha=0.9),
    )
    # Manual margins: the suptitle sits at the very top, the honesty banner
    # at the very bottom; tight_layout cannot handle the inset axes.
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14)

    return fig, (ax_classes, ax_choice, ax_tiling, ax_trap), info
