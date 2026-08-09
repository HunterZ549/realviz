<p align="center">
  <strong>English</strong> · <a href="README_zh.md">简体中文</a>
</p>

<div align="center">

# realviz — Real Analysis Visualizations

**See Riemann and Lebesgue integration side by side. Measure theory for the eyes.**

[![CI](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml/badge.svg)](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-86%20passing-brightgreen.svg)](tests)

> Real analysis is abstract until you **see** it. `realviz` draws the two great
> theories of integration on the same function — so the difference finally clicks.

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_compare_x2.png" width="820" alt="Riemann vs Lebesgue on f(x)=x^2">

</div>

---

## What's the big idea?

Every calculus student learns Riemann integration first: chop the **x-axis** into
vertical strips, add up `f(xᵢ)·Δx`. Lebesgue integration flips it around — chop the
**y-axis** instead, group points by their function value, and weigh each slice by
the *measure* of its preimage: `Σ yₖ·m(Eₖ)`.

| Riemann partitions the x-axis | Lebesgue partitions the y-axis |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_riemann_x2.png" width="380" alt="Riemann sum: vertical strips"> | <img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_lebesgue_x2.png" width="380" alt="Lebesgue sum: horizontal strips"> |
| Same-width **vertical** strips | Same-height **horizontal** slices |
| Sum: `Σ f(xᵢ) · Δx` | Sum: `Σ yₖ · m(Eₖ)` |
| Strip color = **x-position** | Slice color = **y-value** |

Same total area. Completely different philosophy.

## Where Riemann breaks: the Dirichlet function

`D(x) = 1` if `x` is rational, `0` otherwise. Every sub-interval of `[0,1]` contains
both a rational and an irrational — so the upper sum is always `1`, the lower sum
always `0`, and the gap *never closes*. Riemann gives up.

Lebesgue measures the irrationals (`1`) and the rationals (`0`), and gets a clean
answer: **`∫D = 0`**. This single function motivated a whole field.

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_dirichlet.png" width="740" alt="Dirichlet function: Riemann fails, Lebesgue succeeds">

## Measure zero, yet uncountable: the Cantor set

Remove the middle third of `[0, 1]`, then the middle third of every piece that
remains, forever. Row `k` shows what is left after `k` removals; the right
panel tracks the surviving measure. `m(E_k) = (2/3)^k` collapses to 0
exponentially — yet the limiting set has as many points as the real line.

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_cantor.png" width="820" alt="Cantor set construction and measure decay">

The Cantor *function* is the staircase this process grows into: continuous,
non-decreasing, flat almost everywhere (`F' = 0` a.e.), yet it climbs from 0
to 1 — the counterexample that kills the naive fundamental theorem of calculus.

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_cantor_function.png" width="740" alt="Cantor function — the devil's staircase">

## Install

```bash
pip install realviz
```

## Quick start

```python
from realviz import compare_integrals, riemann_plot, lebesgue_plot

# Side-by-side comparison — the main event
compare_integrals(lambda x: x**2, 0, 1, n_partitions=15)

# Or inspect each method alone
riemann_plot(lambda x: x**2, 0, 1, n_partitions=10, method="midpoint")
lebesgue_plot(lambda x: x**2, 0, 1, n_partitions=10)

# The Dirichlet function — where Riemann fails, Lebesgue succeeds
from realviz import dirichlet_illustration
dirichlet_illustration()

# The Cantor set and the devil's staircase
from realviz import cantor_set, cantor_function
cantor_set(n_levels=4)     # construction + measure decay
cantor_function()          # F' = 0 a.e., yet F(1) - F(0) = 1
```

### Try it yourself

```bash
python examples/demo_compare.py     # x² and sin(x), side by side
python examples/demo_dirichlet.py   # the Dirichlet illustration
python examples/demo_cantor.py      # Cantor set + devil's staircase
```

## Design principles

- **Safe by default** — strict input validation: functions are sampled defensively,
  partition counts are bounded, no `eval`/`exec`/`pickle`, no network, no file I/O.
- **Pedagogical color** — every strip is colored by *what is being partitioned*,
  so the axis-choice is visible at a glance.
- **Zero magic** — pure NumPy + Matplotlib, ~800 lines of core code, MIT licensed.

## Roadmap

- [x] `v0.1.0` — Riemann vs Lebesgue integral comparison + Dirichlet illustration
- [x] `v0.2.0` — Cantor set construction + Cantor function (devil's staircase)
- [ ] `v0.3.0` — Simple-function approximation of measurable functions
- [ ] `v0.4.0` — Convergence modes (pointwise, uniform, a.e., in L¹)
- [ ] `v0.5.0` — Vitali set and non-measurable sets (conceptual)

## Development

```bash
git clone https://github.com/HunterZ549/realviz.git
cd realviz
pip install -e ".[dev]"
pytest               # 86 tests, CI runs Python 3.9–3.12
```

## License

MIT — see [LICENSE](LICENSE).
