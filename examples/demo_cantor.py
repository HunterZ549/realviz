"""Demo: the Cantor set and the Cantor function.

Top row: the middle-third construction, level by level, with each row's
measure ``m(E_k) = (2/3)^k`` on the y-axis and a bar chart of the decay on
the right.  Bottom: the devil's staircase it grows into — continuous,
``F' = 0`` almost everywhere, yet ``F(1) - F(0) = 1``.

Run with:
    python examples/demo_cantor.py
"""

import matplotlib.pyplot as plt

from realviz import cantor_function, cantor_set


def main():
    fig, (ax_l, ax_r), measure = cantor_set(n_levels=4)
    print(f"Total measure after 4 removals: {measure:.6f}  (= (2/3)^4)")
    cantor_function()
    plt.show()


if __name__ == "__main__":
    main()
