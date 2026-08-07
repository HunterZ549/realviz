"""Generate README preview images (temp script, deleted after use)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from realviz import compare_integrals, lebesgue_plot, riemann_plot, dirichlet_illustration

DPI = 160

# 1. Hero: Riemann vs Lebesgue on x^2
fig, _, _ = compare_integrals(lambda x: x**2, 0, 1, n_partitions=15)
fig.savefig("examples/output_compare_x2.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# 2. Individual panels for the side-by-side table
fig, ax = plt.subplots(figsize=(8, 5))
riemann_plot(lambda x: x**2, 0, 1, n_partitions=15, method="midpoint", ax=ax)
fig.savefig("examples/output_riemann_x2.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
lebesgue_plot(lambda x: x**2, 0, 1, n_partitions=15, ax=ax)
fig.savefig("examples/output_lebesgue_x2.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# 3. Dirichlet illustration
fig = dirichlet_illustration()
fig.savefig("examples/output_dirichlet.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("previews regenerated at dpi=%d" % DPI)
