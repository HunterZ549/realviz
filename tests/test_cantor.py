"""Tests for ``realviz.cantor``."""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from realviz.cantor import _cantor_function_values, cantor_function, cantor_set


# ── helpers ─────────────────────────────────────────────────────────
def _close_plots():
    """Prevent figure leaks between tests."""
    plt.close("all")


def _row_segments(ax):
    """Collect hlines segments grouped by row y-coordinate."""
    rows = {}
    for coll in ax.collections:
        get_segments = getattr(coll, "get_segments", None)
        if get_segments is None:
            continue
        for seg in coll.get_segments():
            (x0, y0), (x1, y1) = seg
            rows.setdefault(y0, []).append((x0, x1))
    return rows


# ── cantor_set ──────────────────────────────────────────────────────
class TestCantorSet:
    def teardown_method(self):
        _close_plots()

    def test_returns_figure_axes_and_measure(self):
        fig, (ax_l, ax_r), measure = cantor_set(n_levels=4)
        assert fig is not None
        assert len((ax_l, ax_r)) == 2
        assert math.isclose(measure, (2.0 / 3.0) ** 4, rel_tol=1e-12)

    def test_default_levels(self):
        _, _, measure = cantor_set()
        assert math.isclose(measure, (2.0 / 3.0) ** 4, rel_tol=1e-12)

    def test_level_zero_is_full_interval(self):
        fig, (ax_l, ax_r), measure = cantor_set(n_levels=0)
        assert math.isclose(measure, 1.0, rel_tol=1e-12)
        rows = _row_segments(ax_l)
        assert set(rows) == {0.0}  # only E_0
        (x0, x1), = rows[0.0]
        assert math.isclose(x0, 0.0) and math.isclose(x1, 1.0)

    def test_known_intervals_at_level_two(self):
        """After 2 removals: 4 intervals, endpoints at powers of 1/3."""
        _, (ax_l, _), _ = cantor_set(n_levels=2)
        rows = _row_segments(ax_l)

        assert set(rows) == {0.0, -1.0, -2.0}
        assert math.isclose(rows[0.0][0][0], 0.0)
        assert math.isclose(rows[0.0][0][1], 1.0)

        expected_level_1 = [(0, 1 / 3), (2 / 3, 1)]
        got_level_1 = sorted(rows[-1.0])
        for (a0, b0), (a1, b1) in zip(expected_level_1, got_level_1):
            assert math.isclose(a0, a1, rel_tol=1e-12)
            assert math.isclose(b0, b1, rel_tol=1e-12)

        expected_level_2 = [(0, 1 / 9), (2 / 9, 1 / 3), (2 / 3, 7 / 9), (8 / 9, 1)]
        got_level_2 = sorted(rows[-2.0])
        for (a0, b0), (a1, b1) in zip(expected_level_2, got_level_2):
            assert math.isclose(a0, a1, rel_tol=1e-12)
            assert math.isclose(b0, b1, rel_tol=1e-12)

    def test_measure_decreases_with_depth(self):
        _, _, m1 = cantor_set(n_levels=1)
        _close_plots()
        _, _, m2 = cantor_set(n_levels=2)
        _close_plots()
        assert m1 > m2 > 0

    def test_each_level_doubles_interval_count(self):
        _, (ax_l, _), _ = cantor_set(n_levels=4)
        rows = _row_segments(ax_l)
        for k in range(5):
            assert len(rows[-float(k)]) == 2 ** k

    def test_measure_panel_bars_match_formula(self):
        _, (_, ax_r), _ = cantor_set(n_levels=3)
        # the measure bars are the only patch collection on ax_r
        heights = [rect.get_height() for rect in ax_r.patches]
        expected = [(2.0 / 3.0) ** k for k in range(4)]
        assert len(heights) == len(expected)
        for h, e in zip(heights, expected):
            assert math.isclose(h, e, rel_tol=1e-12)

    def test_ytick_labels_carry_measure(self):
        _, (ax_l, _), _ = cantor_set(n_levels=3)
        labels = [t.get_text() for t in ax_l.get_yticklabels()]
        assert labels == [
            "$E_0$",
            "$E_1 = 2/3$",
            "$E_2 = 4/9$",
            "$E_3 = 8/27$",
        ]

    def test_rejects_non_int(self):
        with pytest.raises(TypeError, match="n_levels"):
            cantor_set(n_levels="4")

    def test_rejects_bool(self):
        with pytest.raises(TypeError, match="n_levels"):
            cantor_set(n_levels=True)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            cantor_set(n_levels=-1)

    def test_rejects_too_many_levels(self):
        with pytest.raises(ValueError, match="too large"):
            cantor_set(n_levels=11)

    def test_rejects_huge_figsize(self):
        with pytest.raises(ValueError, match="too large"):
            cantor_set(n_levels=2, figsize=(500, 6))

    def test_no_control_characters(self):
        """Regression discipline: no raw-string escape can leak a TAB into
        rendered text (see the ``\\to`` -> TAB bug in ``dirichlet_illustration``)."""
        fig, (ax_l, ax_r), _ = cantor_set(n_levels=4)

        texts = list(fig.texts)
        if fig._suptitle is not None:
            texts.append(fig._suptitle)
        for ax in (ax_l, ax_r):
            texts.extend(ax.texts)
            texts.extend(ax.get_xticklabels() + ax.get_yticklabels())

        bad = [
            t.get_text()
            for t in texts
            if any(ord(c) < 32 and c != "\n" for c in t.get_text())
        ]
        assert bad == [], f"control characters in rendered text: {bad!r}"


# ── cantor_function ─────────────────────────────────────────────────
class TestCantorFunction:
    def teardown_method(self):
        _close_plots()

    def test_endpoints(self):
        ax, ys = cantor_function()
        assert ax is not None
        assert math.isclose(ys[0], 0.0, abs_tol=1e-8)
        assert math.isclose(ys[-1], 1.0, abs_tol=1e-8)

    def test_middle_third_flat_at_half(self):
        xs = np.array([1 / 3, 0.5, 2 / 3])
        ys = _cantor_function_values(xs)
        for y in ys:
            assert math.isclose(y, 0.5, abs_tol=1e-8)

    def test_exact_values_at_quarter_points(self):
        xs = np.array([1 / 4, 3 / 4, 1 / 9, 2 / 9, 7 / 9, 8 / 9])
        expected = np.array([1 / 3, 2 / 3, 1 / 4, 1 / 4, 3 / 4, 3 / 4])
        got = _cantor_function_values(xs)
        for g, e in zip(got, expected):
            assert math.isclose(g, e, abs_tol=1e-8)

    def test_non_decreasing(self):
        _, ys = cantor_function(n_points=2000)
        # allow tiny float noise on the flat plateaus
        assert np.all(np.diff(ys) >= -1e-12)

    def test_values_inside_unit_interval(self):
        _, ys = cantor_function(n_points=100)
        assert np.all((ys >= 0) & (ys <= 1))

    def test_custom_ax(self):
        fig, ax = plt.subplots()
        ret_ax, _ = cantor_function(ax=ax)
        assert ret_ax is ax

    def test_rejects_non_int_points(self):
        with pytest.raises(TypeError, match="n_points"):
            cantor_function(n_points="1000")

    def test_rejects_too_many_points(self):
        with pytest.raises(ValueError, match="too large"):
            cantor_function(n_points=2_000_000)

    def test_no_control_characters(self):
        fig, ax = plt.subplots()
        cantor_function(ax=ax)

        texts = list(fig.texts)
        if fig._suptitle is not None:
            texts.append(fig._suptitle)
        texts.extend(ax.texts)
        texts.extend(ax.get_xticklabels() + ax.get_yticklabels())

        bad = [
            t.get_text()
            for t in texts
            if any(ord(c) < 32 and c != "\n" for c in t.get_text())
        ]
        assert bad == [], f"control characters in rendered text: {bad!r}"
