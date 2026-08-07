# realviz — Real Analysis Visualizations

[![CI](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml/badge.svg)](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Interactive mathematical visualizations for real analysis and measure theory.
**Start here: Riemann vs Lebesgue integrals, side by side.**

## Why?

Most students learn Riemann integration first. Lebesgue integration feels
abstract — "partition the range, not the domain" only clicks when you **see**
it. This library draws both, on the same function, so you can compare them
directly.

## Install

```bash
pip install realviz
```

> *(Coming to PyPI soon. For now:)*
> ```bash
> pip install -e .
> ```

## Quick Start

```python
from realviz import compare_integrals, riemann_plot, lebesgue_plot

# Side-by-side comparison — the main event
compare_integrals(lambda x: x**2, 0, 1, n_partitions=15)

# Individual plots
riemann_plot(lambda x: x**2, 0, 1, n_partitions=10, method="midpoint")
lebesgue_plot(lambda x: x**2, 0, 1, n_partitions=10)
```

### What you'll see

| Riemann | Lebesgue |
|---------|----------|
| Vertical strips partition the **x-axis** | Horizontal strips partition the **y-axis** |
| Σ f(xᵢ) · Δx | Σ yₖ · m(Eₖ) |
| Fails for Dirichlet function | Succeeds — countable sets have measure zero |

## Examples

```bash
# Compare Riemann vs Lebesgue on x² and sin(x)
python examples/demo_compare.py

# Dirichlet function: where Riemann fails and Lebesgue succeeds
python examples/demo_dirichlet.py
```

## Roadmap

- [x] `v0.1.0` — Riemann vs Lebesgue integral comparison
- [ ] `v0.2.0` — Cantor set construction (step-by-step)
- [ ] `v0.3.0` — Simple function approximation of measurable functions
- [ ] `v0.4.0` — Convergence modes (pointwise, uniform, almost everywhere, L¹)
- [ ] `v0.5.0` — Vitali set and non-measurable sets (conceptual)

## Development

```bash
git clone https://github.com/HunterZ549/realviz.git
cd realviz
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
