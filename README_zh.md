<p align="center">
  <a href="README.md"><strong>English</strong></a> · <strong>简体中文</strong>
</p>

<div align="center">

# realviz — 实变函数可视化

**把黎曼积分和勒贝格积分并排画出来,让测度论"看得见"。**

[![CI](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml/badge.svg)](https://github.com/HunterZ549/realviz/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-179%20passing-brightgreen.svg)](tests)

> 实变函数很抽象,直到你**亲眼看见**它。`realviz` 把两套积分理论画在同一个
> 函数上——黎曼怎么切、勒贝格怎么切,一眼就懂。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_compare_x2.png" width="820" alt="f(x)=x^2 上的黎曼与勒贝格积分对比">

</div>

---

## 这个项目是做什么的?

每个学过微积分的人都先认识黎曼积分:把 **x 轴**切成竖条,加起来 `f(xᵢ)·Δx`。
勒贝格积分反过来——切 **y 轴**,按函数值把点分组,再按每个取值集合的"测度"
加权: `Σ yₖ·m(Eₖ)`。

| 黎曼:切 x 轴 | 勒贝格:切 y 轴 |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_riemann_x2.png" width="380" alt="黎曼和:竖直分割"> | <img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_lebesgue_x2.png" width="380" alt="勒贝格和:水平分割"> |
| 等宽的**竖直**细条 | 等高的**水平**切片 |
| 求和:`Σ f(xᵢ) · Δx` | 求和:`Σ yₖ · m(Eₖ)` |
| 颜色 = **x 位置** | 颜色 = **y 取值** |

面积一样,但背后的思想完全不同。

## 黎曼在哪一步"崩溃":Dirichlet 函数

`D(x) = 1`(x 为有理数),否则为 `0`。`[0,1]` 的每个小区间里既有有理数又有无理数,
所以上和永远是 `1`,下和永远是 `0`,间隙**永远合不拢**——黎曼积分认输。

勒贝格测出无理数的测度是 `1`、有理数的测度是 `0`,干净利落地得到答案:
**`∫D = 0`**。这一个函数,催生了一整个领域。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_dirichlet.png" width="740" alt="Dirichlet 函数:黎曼失败,勒贝格成功">

## 零测度,却不可数:康托集

删掉 `[0, 1]` 的中间三分之一,再删掉剩下每段的中间三分之一,一直删下去。
第 `k` 行显示删了 `k` 次之后剩下什么;右栏追踪剩下的总长度。
`m(E_k) = (2/3)^k` 指数级塌缩到 0——可极限集合里的点,却和整个实数轴一样多。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_cantor.png" width="820" alt="康托集构造与度量衰减">

由这个过程长出来的,是康托**函数**(魔鬼阶梯):连续、单调不减、几乎处处导数为 0,
却从 0 一路爬到 1——一个干翻"朴素微积分基本定理"的反例。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_cantor_function.png" width="740" alt="康托函数——魔鬼阶梯">

## 每个可测函数都是简单函数的极限

**简单函数**只取有限多个值——在每个可测集合上取常数。经典构造把 `f` 的
**值域**切成 `2^n` 条等高水平带,把每个点按到所在色带的下边:`sₙ ≤ f`,
且随 `n` 增大 `sₙ` 单调逼近 `f`。这是勒贝格图景的收官:积分就是这样
**定义**的——`∫f = lim ∫sₙ`。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_simple_approx_x2.png" width="820" alt="简单函数逼近:s_n 从下方收敛到 f">

左栏是 `f` 与下方的 `sₙ` 阶梯,每条色带就是一个水平集
`E_k = {x : c_k ≤ f(x) < c_{k+1}}`;右上栏 `∫s_k` 单调上升到 `∫f`;
右下栏一致误差 `‖f − s_k‖∞ → 0`(对数轴)。简单函数同样不在乎间断——
振荡和跳跃都能自然处理:

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_simple_approx_sin.png" width="820" alt="振荡情形:sin(3x) 与其简单函数逼近">

跳跃间断也一样——水平集照样可测,`∫s_n` 依然收敛到真实的 `∫f = 3/4`:

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_simple_approx_jump.png" width="820" alt="间断情形:跳跃函数与其简单函数逼近">

## 收敛的四种含义:一个函数族,四台摄影机

一串函数可以有几种**本质不同**的收敛方式,教科书用三个经典反例把它们区分开。
`convergence_modes` 把同一个函数族**只画一遍**,同时用四台摄影机来判定——
逐点、一致、几乎处处、L¹——层次关系当场瓦解给你看。

`x^n` 在 `[0, 1]` 上逐点收敛(只有 `x = 1` 例外)、几乎处处收敛,也 L¹ 收敛——
但**不一致收敛**:围绕极限函数的那条 `ε` 管道,在 `x = 1` 附近永远合不拢。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_convergence_x_pow_n.png" width="820" alt="x^n:逐点、几乎处处、L¹ 收敛,但非一致收敛">

瘦峰 `n·1_[0, 1/n]` 塌缩到 `x = 0` 这一个点——零测集——所以几乎处处收敛,
可 `∫f_n = 1` 纹丝不动:**几乎处处,但不在 L¹**。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_convergence_spike.png" width="820" alt="瘦峰:几乎处处收敛,但不在 L¹">

打字机序列让一个区间指示函数在 `[0, 1]` 上往返扫过,宽度越扫越窄。
`∫f_n → 0`,所以它 L¹ 收敛——可每一点都被覆盖无穷多次:它**在每一点都不收敛**。

<img src="https://raw.githubusercontent.com/HunterZ549/realviz/main/examples/output_convergence_typewriter.png" width="820" alt="打字机序列:L¹ 收敛,却处处不收敛">

四块面板共用同一根网格、同一个极限 `f`、同一个 `ε`,判定全是纯 NumPy 算的。
逐点与几乎处处用的是同一个诚实的"有限窗口"检验——量"最近几个成员仍在分歧"
的集合——这恰恰是区分瘦峰与打字机的关键。

## 安装

```bash
pip install realviz
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

# 康托集与魔鬼阶梯
from realviz import cantor_set, cantor_function
cantor_set(n_levels=4)     # 构造过程 + 度量衰减
cantor_function()          # F' = 0 a.e.,却 F(1) - F(0) = 1

# 简单函数逼近——勒贝格图景收官
from realviz import simple_approximation
simple_approximation(lambda x: x**2, 0, 1, n_levels=5)   # s_n -> f,∫s_n -> ∫f

# 收敛模式——一个函数族,四台摄影机
from realviz import convergence_modes, FAMILIES
convergence_modes("x_pow_n")     # ✓ 逐点 & 几乎处处,✗ 一致
convergence_modes("thin_spike")  # ✓ 几乎处处,✗ L¹
convergence_modes("typewriter")  # ✓ L¹,每一点都不收敛
```

### 亲自跑一跑

```bash
python examples/demo_compare.py     # x² 和 sin(x),并排对比
python examples/demo_dirichlet.py   # Dirichlet 函数图解
python examples/demo_cantor.py      # 康托集 + 魔鬼阶梯
python examples/demo_simple_approx.py  # 简单函数逼近:s_n -> f
python examples/demo_convergence.py    # 三个经典反例,2x2 网格
```

## 设计原则

- **安全优先** — 严格输入校验:采样防异常、分割数有上限、不用 `eval`/`exec`/`pickle`、
  无网络、无文件读写。
- **教学用配色** — 每条细条的颜色都反映"切的是哪根轴",分割方式一眼可见。
- **零魔法** — 纯 NumPy + Matplotlib,核心代码约 1800 行,MIT 协议。

## 路线图

- [x] `v0.1.0` — 黎曼 vs 勒贝格积分对比 + Dirichlet 图解
- [x] `v0.2.0` — 康托集构造 + 康托函数(魔鬼阶梯)
- [x] `v0.3.0` — 可测函数的简单函数逼近
- [x] `v0.4.0` — 收敛模式(逐点、一致、几乎处处、L¹)
- [ ] `v0.5.0` — Vitali 集与不可测集(概念图解)

## 开发

```bash
git clone https://github.com/HunterZ549/realviz.git
cd realviz
pip install -e ".[dev]"
pytest               # 179 个测试,CI 跑 Python 3.9–3.12
```

## 协议

MIT — 见 [LICENSE](LICENSE)。
