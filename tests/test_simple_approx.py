"""Tests for ``realviz.simple_functions``."""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from realviz.simple_functions import (
    SimpleApproximationInfo,
    _simple_function_values,
    simple_approximation,
)


# ── helpers ─────────────────────────────────────────────────────────
def _close_plots():
    """Prevent figure leaks between tests."""
    plt.close("all")


# ── _simple_function_values ─────────────────────────────────────────
class TestSimpleFunctionValues:
    def test_single_band_is_min(self):
        ys = _simple_function_values(np.array([0.1, 0.9]), 0.0, 1.0, 0)
        assert np.allclose(ys, 0.0)

    def test_lower_edge_of_band(self):
        ys = _simple_function_values(np.array([0.25, 0.75]), 0.0, 1.0, 1)
        assert np.allclose(ys, [0.0, 0.5])

    def test_never_above_input(self):
        y = np.linspace(-0.5, 2.5, 300)
        s = _simple_function_values(y, -0.5, 2.5, 5)
        assert np.all(s <= y + 1e-12)
        assert np.all(s >= -0.5 - 1e-12)

    def test_handles_negative_range(self):
        ys = _simple_function_values(np.array([-0.5, 0.5]), -1.0, 1.0, 1)
        assert np.allclose(ys, [-1.0, 0.0])

    def test_top_of_range_in_last_band(self):
        # n_levels=2 -> dy = 0.25; y_max must land in the last band (0.75)
        ys = _simple_function_values(np.array([1.0]), 0.0, 1.0, 2)
        assert math.isclose(float(ys[0]), 0.75, rel_tol=1e-12)

    def test_constant_range_no_crash(self):
        ys = _simple_function_values(np.array([0.2, 0.8]), 3.0, 3.0, 4)
        assert np.allclose(ys, 3.0)

    def test_returns_float_array(self):
        ys = _simple_function_values(np.array([0.5]), 0.0, 1.0, 1)
        assert ys.dtype == np.float64


# ── simple_approximation ────────────────────────────────────────────
class TestSimpleApproximation:
    def teardown_method(self):
        _close_plots()

    def test_returns_figure_axes_and_info(self):
        fig, (ax_l, (ax_rt, ax_rb)), info = simple_approximation(lambda x: x**2, 0, 1)
        assert fig is not None
        assert isinstance(info, SimpleApproximationInfo)
        assert len(info.levels) == 6  # 0 .. 5
        assert info.n_bands == 2**5
        assert len(info.integrals) == 6
        assert len(info.sup_errors) == 6

    def test_default_levels(self):
        _, _, info = simple_approximation(lambda x: x**2, 0, 1)
        assert info.n_bands == 32

    def test_zero_levels_is_single_band(self):
        _, _, info = simple_approximation(lambda x: x**2, 0, 1, n_levels=0)
        assert info.n_bands == 1
        # s_0 = y_min = 0 everywhere -> integral 0, error = max f = 1
        assert math.isclose(info.integrals[0], 0.0, abs_tol=1e-9)
        assert math.isclose(info.sup_errors[0], 1.0, abs_tol=1e-9)

    def test_known_integral_at_level_one(self):
        """x² on [0,1], 2 bands: s=0 on [0,1/√2), s=1/2 on [1/√2,1]."""
        _, _, info = simple_approximation(lambda x: x**2, 0, 1, n_levels=1)
        expected = 0.5 * (1 - 1 / math.sqrt(2))
        assert math.isclose(info.integrals[1], expected, abs_tol=1e-3)

    def test_integrals_monotone_increasing(self):
        _, _, info = simple_approximation(lambda x: x**2, 0, 1, n_levels=5)
        assert np.all(np.diff(info.integrals) >= -1e-12)
        assert np.all(np.diff(info.sup_errors) <= 1e-12)

    def test_sup_error_bounded_by_band_width(self):
        _, _, info = simple_approximation(lambda x: x**2, 0, 1, n_levels=4)
        assert info.sup_errors[-1] <= info.band_width + 1e-12

    def test_integrals_converge_to_true(self):
        _, _, info = simple_approximation(lambda x: x**2, 0, 1, n_levels=8)
        assert math.isclose(info.integrals[-1], 1 / 3, abs_tol=1e-2)
        assert math.isclose(info.ref_integral, 1 / 3, rel_tol=1e-3)

    def test_signed_function(self):
        """f(x)=x on [-1,1]: ∫s_0 = -2 (constant -1), ∫s_1 = -1, ∫f = 0."""
        _, _, info = simple_approximation(lambda x: x, -1, 1, n_levels=1)
        assert math.isclose(info.integrals[0], -2.0, abs_tol=1e-9)
        assert math.isclose(info.integrals[1], -1.0, abs_tol=1e-3)
        assert math.isclose(info.ref_integral, 0.0, abs_tol=1e-9)

    def test_constant_function_is_exact(self):
        _, _, info = simple_approximation(lambda x: 5.0, 0, 2, n_levels=3)
        assert np.allclose(info.integrals, 10.0)
        assert np.all(info.sup_errors == 0.0)
        assert math.isclose(info.ref_integral, 10.0, rel_tol=1e-12)

    def test_jump_function_handled(self):
        def jump(x):
            return x + (0.5 if x > 0.5 else 0.0)

        fig, (ax_l, _), info = simple_approximation(jump, 0, 1, n_levels=4)
        assert ax_l is not None
        assert info.n_bands == 16

    # ── validation ─────────────────────────────────────────────────
    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            simple_approximation("nope", 0, 1)  # type: ignore[arg-type]

    def test_rejects_bad_interval(self):
        with pytest.raises(ValueError, match="strictly less"):
            simple_approximation(lambda x: x, 5, 1)

    def test_rejects_non_int_levels(self):
        with pytest.raises(TypeError, match="n_levels"):
            simple_approximation(lambda x: x, 0, 1, n_levels="4")

    def test_rejects_bool_levels(self):
        with pytest.raises(TypeError, match="n_levels"):
            simple_approximation(lambda x: x, 0, 1, n_levels=True)

    def test_rejects_negative_levels(self):
        with pytest.raises(ValueError, match="non-negative"):
            simple_approximation(lambda x: x, 0, 1, n_levels=-1)

    def test_rejects_too_many_levels(self):
        with pytest.raises(ValueError, match="too large"):
            simple_approximation(lambda x: x, 0, 1, n_levels=11)

    def test_rejects_huge_figsize(self):
        with pytest.raises(ValueError, match="too large"):
            simple_approximation(lambda x: x, 0, 1, figsize=(500, 6))

    def test_rejects_too_many_samples(self):
        with pytest.raises(ValueError, match="too large"):
            simple_approximation(lambda x: x, 0, 1, n_samples=2_000_000)

    def test_rejects_single_sample(self):
        with pytest.raises(ValueError, match="at least 2"):
            simple_approximation(lambda x: x, 0, 1, n_samples=1)

    def test_rejects_function_that_raises(self):
        def kaboom(x):
            if x > 0.5:
                raise RuntimeError("boom")
            return x

        with pytest.raises(ValueError, match="RuntimeError"):
            simple_approximation(kaboom, 0, 1)

    def test_rejects_non_finite_output(self):
        with pytest.raises(ValueError, match="non-finite"):
            simple_approximation(lambda x: math.nan, 0, 1)

    def test_rejects_range_overflow(self):
        """Finite samples whose *range* overflows float64 must be rejected
        cleanly instead of producing NaN geometry (and a matplotlib crash)."""
        with pytest.raises(ValueError, match="too large to subdivide"):
            simple_approximation(
                lambda x: 1e308 * math.sin(2 * math.pi * x), 0, 1
            )

    def test_rejects_expensive_level_sample_product(self):
        """Each of n_levels and n_samples is legal alone, but their product
        must not be able to drive a hang or an artist explosion."""
        with pytest.raises(ValueError, match="too expensive"):
            simple_approximation(lambda x: x, 0, 1,
                                 n_levels=10, n_samples=1_000_000)

    def test_no_control_characters(self):
        """Regression discipline: no raw-string escape can leak a TAB into
        rendered text (see the ``\\to`` -> TAB bug in ``dirichlet_illustration``)."""
        fig, _, _ = simple_approximation(lambda x: x**2, 0, 1, n_levels=4)

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
