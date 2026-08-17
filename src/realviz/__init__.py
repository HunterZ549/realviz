"""``realviz`` — Real Analysis Visualizations.

A Python library for interactive mathematical visualizations focused on
measure theory and real analysis.  Starting with Riemann vs Lebesgue
integral comparisons.

Quick start
-----------
>>> from realviz import compare_integrals
>>> compare_integrals(lambda x: x**2, 0, 1, n_partitions=10)

Full documentation: https://github.com/HunterZ549/realviz
"""

from .cantor import cantor_function, cantor_set
from .integrals import compare_integrals, dirichlet_illustration, lebesgue_plot, riemann_plot
from .simple_functions import simple_approximation

__version__ = "0.3.0"
__all__ = [
    "cantor_function",
    "cantor_set",
    "compare_integrals",
    "dirichlet_illustration",
    "lebesgue_plot",
    "riemann_plot",
    "simple_approximation",
]
