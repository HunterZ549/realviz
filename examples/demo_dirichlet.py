"""Demo: Dirichlet function — Riemann fails, Lebesgue succeeds.

The Dirichlet function D(x) = 1 if x ∈ ℚ, 0 otherwise.

- Every Riemann upper sum = b - a, every lower sum = 0 → not integrable.
- Lebesgue: m(ℚ) = 0, m(ℝ\ℚ) = b - a → ∫D = 0·1 + (b-a)·0 = 0.

This demo illustrates the conceptual difference. Because we cannot
enumerate rationals numerically, we draw the Lebesgue perspective as a
stylised diagram.

Run with:
    python examples/demo_dirichlet.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def main():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    a, b = 0, 1

    # ── Left: Riemann attempt ──────────────────────────────────
    n = 8
    dx = (b - a) / n
    x_edges = np.linspace(a, b, n + 1)

    # Upper sum: every sub-interval contains a rational → sup = 1
    for i in range(n):
        ax_l.fill_between(
            [x_edges[i], x_edges[i + 1]],
            [0, 0],
            [1, 1],
            alpha=0.3,
            color="steelblue",
            edgecolor="navy",
            linewidth=0.5,
            label="Upper sum" if i == 0 else "",
        )
    # Lower sum: every sub-interval contains an irrational → inf = 0
    ax_l.axhline(y=0, color="gray", linewidth=0.5)

    # Sprinkle "rationals" (stylised)
    np.random.seed(0)
    rational_x = np.sort(np.random.uniform(a, b, 40))
    rational_y = np.ones_like(rational_x)
    ax_l.scatter(rational_x, rational_y, s=4, c="red", alpha=0.6, zorder=5, label="$x\\in\\mathbb{Q}$")
    ax_l.scatter([], [], s=4, c="red", alpha=0.6, label="$D(x)=1$ (rational)")

    ax_l.set_title("Riemann: Not Integrable\n$U(f,P)=1$, $L(f,P)=0$ for every partition")
    ax_l.set_xlabel("$x$")
    ax_l.set_ylabel("$D(x)$")
    ax_l.set_ylim(-0.1, 1.3)
    ax_l.legend(loc="upper right", fontsize=8)

    # ── Right: Lebesgue perspective ────────────────────────────
    # Measure of rationals = 0 → contributes nothing
    # Measure of irrationals = 1 → contributes 1 × 0 = 0
    # Total integral = 0

    ax_r.fill_between([a, b], [0, 0], [1, 1], alpha=0.15, color="coral", label="$\\{x:D(x)=1\\}$ (rationals, $m=0$)")
    ax_r.fill_between([a, b], [0, 0], [0, 0], alpha=0.0)  # invisible, just for legend
    # The whole interval at y=0 — measure 1
    ax_r.fill_between(
        [a, b],
        [-0.05, -0.05],
        [0, 0],
        alpha=0.15,
        color="steelblue",
        label="$\\{x:D(x)=0\\}$ (irrationals, $m=1$)",
    )

    # Annotation
    ax_r.annotate(
        "$\\int D\\,d\\mu = 1\\cdot m(\\mathbb{Q}) + 0\\cdot m(\\mathbb{R}\\setminus\\mathbb{Q})$",
        xy=(0.5, 0.6),
        fontsize=13,
        ha="center",
    )
    ax_r.annotate(
        "$= 1\\cdot 0 + 0\\cdot 1 = 0$",
        xy=(0.5, 0.4),
        fontsize=14,
        ha="center",
        fontweight="bold",
    )

    ax_r.axhline(y=0, color="gray", linewidth=0.5)
    ax_r.set_title("Lebesgue: Integrable = 0\nCountable sets have measure zero")
    ax_r.set_xlabel("$x$")
    ax_r.set_ylabel("$D(x)$")
    ax_r.set_ylim(-0.1, 1.3)
    ax_r.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Dirichlet Function  —  $D(x)=\\mathbf{1}_{\\mathbb{Q}}(x)$ on $[0,1]$",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
