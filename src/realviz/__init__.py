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

from .integrals import compare_integrals, dirichlet_illustration, lebesgue_plot, riemann_plot

__version__ = "0.1.0"
__all__ = ["compare_integrals", "dirichlet_illustration", "lebesgue_plot", "riemann_plot"]
