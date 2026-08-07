<p align="center">
  <a href="README.md"><strong>English</strong></a> · <strong>简体中文</strong>
</p>

<div align="center">

# realviz — 实变函数可视化

**把黎曼积分和勒贝格积分并排画出来,让测度论"看得见"。**

[![CI](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml/badge.svg)](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen.svg)](tests)

> 实变函数很抽象,直到你**亲眼看见**它。`realviz` 把两套积分理论画在同一个
> 函数上——黎曼怎么切、勒贝格怎么切,一眼就懂。

<img src="examples/output_compare_x2.png" width="820" alt="f(x)=x^2 上的黎曼与勒贝格积分对比">

</div>

---

## 这个项目是做什么的?

每个学过微积分的人都先认识黎曼积分:把 **x 轴**切成竖条,加起来 `f(xᵢ)·Δx`。
勒贝格积分反过来——切 **y 轴**,按函数值把点分组,再按每个取值集合的"测度"
加权: `Σ yₖ·m(Eₖ)`。

| 黎曼:切 x 轴 | 勒贝格:切 y 轴 |
|:---:|:---:|
| <img src="examples/output_riemann_x2.png" width="380" alt="黎曼和:竖直分割"> | <img src="examples/output_lebesgue_x2.png" width="380" alt="勒贝格和:水平分割"> |
| 等宽的**竖直**细条 | 等高的**水平**切片 |
| 求和:`Σ f(xᵢ) · Δx` | 求和:`Σ yₖ · m(Eₖ)` |
| 颜色 = **x 位置** | 颜色 = **y 取值** |

面积一样,但背后的思想完全不同。

## 黎曼在哪一步"崩溃":Dirichlet 函数

`D(x) = 1`(x 为有理数),否则为 `0`。`[0,1]` 的每个小区间里既有有理数又有无理数,
所以上和永远是 `1`,下和永远是 `0`,间隙**永远合不拢**——黎曼积分认输。

勒贝格测出无理数的测度是 `1`、有理数的测度是 `0`,干净利落地得到答案:
**`∫D = 0`**。这一个函数,催生了一整个领域。

<img src="examples/output_dirichlet.png" width="740" alt="Dirichlet 函数:黎曼失败,勒贝格成功">

## 安装

```bash
pip install realviz            # 发布到 PyPI 之后
# 目前先这样:
pip install -e .
```

## 快速上手

```python
from realviz import compare_integrals, riemann_plot, lebesgue_plot

# 主菜:并排对比
compare_integrals(lambda x: x**2, 0, 1, n_partitions=15)

# 单独看每一种
riemann_plot(lambda x: x**2, 0, 1, n_partitions=10, method="midpoint")
lebesgue_plot(lambda x: x**2, 0, 1, n_partitions=10)

# Dirichlet 函数——黎曼失败的地方,勒贝格成功
from realviz import dirichlet_illustration
dirichlet_illustration()
```

### 亲自跑一跑

```bash
python examples/demo_compare.py     # x² 和 sin(x),并排对比
python examples/demo_dirichlet.py   # Dirichlet 函数图解
```

## 设计原则

- **安全优先** — 严格输入校验:采样防异常、分割数有上限、不用 `eval`/`exec`/`pickle`、
  无网络、无文件读写。
- **教学用配色** — 每条细条的颜色都反映"切的是哪根轴",分割方式一眼可见。
- **零魔法** — 纯 NumPy + Matplotlib,核心代码约 300 行,MIT 协议。

## 路线图

- [x] `v0.1.0` — 黎曼 vs 勒贝格积分对比 + Dirichlet 图解
- [ ] `v0.2.0` — 康托集(Cantor set)逐步构造
- [ ] `v0.3.0` — 可测函数的简单函数逼近
- [ ] `v0.4.0` — 收敛模式(逐点、一致、几乎处处、L¹)
- [ ] `v0.5.0` — Vitali 集与不可测集(概念图解)

## 开发

```bash
git clone https://github.com/HunterZ549/realviz.git
cd realviz
pip install -e ".[dev]"
pytest               # 53 个测试,CI 跑 Python 3.9–3.12
```

## 协议

MIT — 见 [LICENSE](LICENSE)。
