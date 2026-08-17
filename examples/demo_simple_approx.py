"""Demo: simple-function approximation of measurable functions.

Top-left: f with the simple function s_n staircased beneath it — the
y-axis cut into 2^n horizontal bands, each band a level set E_k, colored
by y-value.  Top-right: ∫s_k rising to ∫f.  Bottom-right: the uniform
error ‖f - s_k‖∞ decaying to 0 on a log scale.

Run with:
    python examples/demo_simple_approx.py
"""

import matplotlib.pyplot as plt
import numpy as np

from realviz import simple_approximation


def main():
    # smooth and monotone — each band's preimage is a single interval
    fig, _, info = simple_approximation(lambda x: x**2, 0, 1, n_levels=5)
    print(f"x²: ∫s ≈ {info.integrals[-1]:.4f},  ∫f ≈ {info.ref_integral:.4f}")

    # oscillating — each band's preimage has several runs
    simple_approximation(lambda x: np.sin(3 * x), 0, 2 * np.pi, n_levels=5)

    # a jump discontinuity — simple functions handle it naturally
    def jump(x):
        return x + (0.5 if x > 0.5 else 0.0)

    simple_approximation(jump, 0, 1, n_levels=4)

    plt.show()


if __name__ == "__main__":
    main()
