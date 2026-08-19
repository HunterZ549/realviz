"""Demo: the same function family under four convergence cameras.

Each of the three classic counterexamples separates two modes of
convergence — pointwise, uniform, almost-everywhere and L1 — by behaving
differently under each.  ``convergence_modes`` draws one family once and
judges it four ways, so the separation is visible side by side:

- ``x^n`` on [0, 1]: converges pointwise (and a.e.) but NOT uniformly;
- the thin spike ``n·1_[0, 1/n]``: converges a.e. but NOT in L1;
- the typewriter sequence: converges in L1 but fails at EVERY point.

Run with:
    python examples/demo_convergence.py
"""

import matplotlib.pyplot as plt

from realviz import FAMILIES, convergence_modes


def main():
    # each preset prints its four verdicts and opens one 2x2 figure
    for name in sorted(FAMILIES):
        _, _, info = convergence_modes(name)
        verdict = (
            f"pointwise={'OK' if info.pointwise_ok else 'FAIL'}  "
            f"uniform={'OK' if info.uniform_ok else 'FAIL'}  "
            f"a.e.={'OK' if info.ae_ok else 'FAIL'}  "
            f"L1={'OK' if info.l1_ok else 'FAIL'}"
        )
        print(f"{name:>10}: {verdict}")

    plt.show()


if __name__ == "__main__":
    main()
