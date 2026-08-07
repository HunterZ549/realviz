"""Tests for ``realviz.integrals``."""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import pytest

from realviz.integrals import compare_integrals, lebesgue_plot, riemann_plot


# ── helpers ─────────────────────────────────────────────────────────
def _close_plots():
    """Prevent figure leaks between tests."""
    plt.close("all")


# ── riemann_plot ────────────────────────────────────────────────────
class TestRiemannPlot:
    def teardown_method(self):
        _close_plots()

    def test_basic_x_squared(self):
        ax, s = riemann_plot(lambda x: x**2, 0, 1, n_partitions=10)
        assert ax is not None
        assert 0.2 < s < 0.45  # true value = 1/3 ≈ 0.333

    def test_midpoint_accurate(self):
        _, s = riemann_plot(lambda x: x**2, 0, 1, n_partitions=100, method="midpoint")
        assert math.isclose(s, 1 / 3, rel_tol=1e-4)

    def test_left_method(self):
        _, s = riemann_plot(lambda x: x, 0, 1, n_partitions=10, method="left")
        assert s < 0.5  # left underestimates increasing f

    def test_right_method(self):
        _, s = riemann_plot(lambda x: x, 0, 1, n_partitions=10, method="right")
        assert s > 0.5  # right overestimates increasing f

    def test_custom_ax(self):
        fig, ax = plt.subplots()
        ret_ax, _ = riemann_plot(lambda x: x, 0, 1, ax=ax)
        assert ret_ax is ax

    def test_rejects_bad_method(self):
        with pytest.raises(ValueError, match="method"):
            riemann_plot(lambda x: x, 0, 1, method="trapezoid")

    def test_rejects_constant_function_for_lebsegue(self):
        """riemann_plot should be fine with constant functions."""
        ax, s = riemann_plot(lambda x: 3.0, 0, 2, n_partitions=5)
        # True integral = 6
        assert math.isclose(s, 6.0, rel_tol=1e-10)

    def test_rejects_non_finite_output(self):
        # Use math.nan so every sample point is non-finite — no chance of
        # accidentally missing a singularity.
        with pytest.raises(ValueError, match="non-finite"):
            riemann_plot(lambda x: math.nan, 0, 1, n_partitions=10)

    def test_rejects_function_that_raises(self):
        def kaboom(x):
            if x > 0.5:
                raise RuntimeError("boom")
            return x

        with pytest.raises(ValueError, match="RuntimeError"):
            riemann_plot(kaboom, 0, 1, n_partitions=10)


# ── lebesgue_plot ───────────────────────────────────────────────────
class TestLebesguePlot:
    def teardown_method(self):
        _close_plots()

    def test_basic_x_squared(self):
        ax, s = lebesgue_plot(lambda x: x**2, 0, 1, n_partitions=10)
        assert ax is not None
        # Should be in the ballpark of 1/3
        assert 0.15 < s < 0.5

    def test_linear_increasing(self):
        _, s = lebesgue_plot(lambda x: 2 * x, 0, 1, n_partitions=50)
        assert math.isclose(s, 1.0, rel_tol=0.1)

    def test_rejects_constant_function(self):
        with pytest.raises(ValueError, match="constant"):
            lebesgue_plot(lambda x: 5.0, 0, 1)

    def test_custom_ax(self):
        fig, ax = plt.subplots()
        ret_ax, _ = lebesgue_plot(lambda x: x, 0, 1, ax=ax)
        assert ret_ax is ax

    def test_rejects_nan_in_range(self):
        def bad(x):
            return math.nan

        with pytest.raises(ValueError, match="non-finite"):
            lebesgue_plot(bad, 0, 1)


# ── compare_integrals ───────────────────────────────────────────────
class TestCompareIntegrals:
    def teardown_method(self):
        _close_plots()

    def test_returns_figure_and_sums(self):
        fig, axes, (r_sum, l_sum) = compare_integrals(lambda x: x**2, 0, 1, n_partitions=10)
        assert fig is not None
        assert len(axes) == 2
        assert isinstance(r_sum, float)
        assert isinstance(l_sum, float)
        assert 0 < r_sum < 1
        assert 0 < l_sum < 1

    def test_custom_n_partitions(self):
        fig, _, _ = compare_integrals(lambda x: x, 0, 2, n_partitions=50)
        assert fig is not None

    def test_input_validation_propagates(self):
        with pytest.raises(TypeError):
            compare_integrals("not callable", 0, 1)  # type: ignore[arg-type]

    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="strictly less"):
            compare_integrals(lambda x: x, 5, 1)

    def test_too_many_partitions(self):
        with pytest.raises(ValueError, match="too large"):
            compare_integrals(lambda x: x, 0, 1, n_partitions=999_999)


# ── edge cases ──────────────────────────────────────────────────────
class TestEdgeCases:
    def teardown_method(self):
        _close_plots()

    def test_negative_domain(self):
        ax, s = riemann_plot(lambda x: x, -3, -1, n_partitions=10)
        # ∫_{-3}^{-1} x dx = [x²/2]_{-3}^{-1} = 1/2 - 9/2 = -4
        assert math.isclose(s, -4.0, rel_tol=0.01)

    def test_small_interval(self):
        _, s = riemann_plot(lambda x: x**3, 0, 0.1, n_partitions=100, method="midpoint")
        # ∫_0^0.1 x³ dx = 0.1⁴/4 = 2.5e-5
        assert math.isclose(s, 2.5e-5, rel_tol=0.01)

    def test_single_partition(self):
        ax, s = riemann_plot(lambda x: x, 0, 1, n_partitions=1, method="midpoint")
        assert math.isclose(s, 0.5)


# ── dirichlet_illustration ───────────────────────────────────────────
class TestDirichletIllustration:
    def teardown_method(self):
        _close_plots()

    def test_renders_without_control_characters(self):
        """Regression: a non-raw string turned ``\\to`` into a TAB, so the
        label read "never <TAB> o 0" instead of "never -> 0".  Every Text
        object must be free of control characters (newlines are OK)."""
        from realviz import dirichlet_illustration

        fig = dirichlet_illustration()

        texts = list(fig.texts)
        if fig._suptitle is not None:
            texts.append(fig._suptitle)
        for ax in fig.axes:
            texts.extend(ax.texts)
            texts.extend(ax.get_xticklabels() + ax.get_yticklabels())

        bad = [
            t.get_text()
            for t in texts
            if any(ord(c) < 32 and c != "\n" for c in t.get_text())
        ]
        assert bad == [], f"control characters in rendered text: {bad!r}"

    def test_gap_annotation_says_never_goes_to_zero(self):
        """The middle-panel annotation should read "never -> 0", not "o 0"."""
        from realviz import dirichlet_illustration

        fig = dirichlet_illustration()
        for ax in fig.axes:
            for t in ax.texts:
                if "never" in t.get_text():
                    assert "\\to" in t.get_text()  # raw backslash survives
                    return
        raise AssertionError("'never' annotation not found")
