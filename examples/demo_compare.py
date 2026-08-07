"""Demo: Riemann vs Lebesgue integral comparison.

Run with:
    python examples/demo_compare.py
"""

import matplotlib.pyplot as plt
import numpy as np

from realviz import compare_integrals


def main():
    # ── Example 1: x² on [0, 1] ────────────────────────────────
    print("Plotting f(x) = x² on [0, 1] ...")
    compare_integrals(lambda x: x**2, 0, 1, n_partitions=15)

    # ── Example 2: sin(x) on [0, π] ───────────────────────────
    print("Plotting f(x) = sin(x) on [0, π] ...")
    compare_integrals(np.sin, 0, np.pi, n_partitions=12)

    plt.show()


if __name__ == "__main__":
    main()
