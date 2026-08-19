"""Tests for ``realviz.vitali``."""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

import realviz.vitali as vitali_mod
from realviz.vitali import (
    _CLASS_ANCHORS,
    _COL_REP,
    _COL_TILE,
    _DENSITY_WINDOW,
    _QS_FINE,
    _min_circular_gap,
    _reduced_fractions,
    _reps,
    vitali_illustration,
)


# ── helpers ─────────────────────────────────────────────────────────
def _close_plots():
    """Prevent figure leaks between tests."""
    plt.close("all")


def _hex_rgb(h: str):
    """'#b45309' -> [r, g, b] in 0..1, for comparing scatter facecolors."""
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


# ── reduced-fraction grid ───────────────────────────────────────────
class TestReducedFractions:
    def test_q_max_4(self):
        assert np.allclose(_reduced_fractions(4),
                           [0.25, 1 / 3, 0.5, 2 / 3, 0.75])

    def test_always_reduced_and_in_unit_interval(self):
        fracs = _reduced_fractions(20)
        assert np.all(fracs > 0.0) and np.all(fracs < 1.0)
        assert np.all(np.diff(fracs) > 0)  # sorted, no duplicates

    def test_q_max_6_count(self):
        # q=2..6 reduced fractions in (0,1): 1/2, 1/3, 2/3, 1/4, 3/4,
        # 1/5, 2/5, 3/5, 4/5, 1/6, 5/6  -- 11 distinct values.
        assert len(_reduced_fractions(6)) == 11

    def test_larger_grid_is_superset(self):
        small, large = _reduced_fractions(8), _reduced_fractions(16)
        assert set(large) >= set(small)

    def test_q_max_1_yields_nothing(self):
        # range(2, 2) is empty: no denominator available.
        assert _reduced_fractions(1).size == 0

    def test_q_max_2_is_just_one_half(self):
        assert np.allclose(_reduced_fractions(2), [0.5])

    def test_q_max_3_known_values(self):
        assert np.allclose(_reduced_fractions(3), [1 / 3, 0.5, 2 / 3])


# ── the chosen representatives ──────────────────────────────────────
class TestReps:
    def test_values_lie_in_unit_interval(self):
        reps = _reps(12)
        assert reps.shape == (12,)
        assert np.all(reps >= 0.0) and np.all(reps < 1.0)

    def test_first_rep_is_sqrt2_mod_1(self):
        assert math.isclose(_reps(1)[0], math.sqrt(2) - 1.0, rel_tol=1e-12)

    def test_sequence_is_distinct(self):
        reps = _reps(30)
        assert np.unique(reps).size == 30


# ── circular gap ────────────────────────────────────────────────────
class TestMinCircularGap:
    def test_single_point_wraps_whole_circle(self):
        assert math.isclose(_min_circular_gap([0.5]), 1.0)

    def test_antipodal_points(self):
        assert math.isclose(_min_circular_gap([0.0, 0.5]), 0.5)

    def test_three_points_including_wrap(self):
        # 0, 0.4, 0.8 -> linear gaps 0.4, 0.4 and wrap gap 0.2
        assert math.isclose(_min_circular_gap([0.0, 0.4, 0.8]), 0.2)

    def test_returns_float(self):
        assert isinstance(_min_circular_gap([0.1, 0.7]), float)

    def test_duplicates_dedup_to_a_single_point(self):
        # np.unique collapses the pair, so the circle has one point and the
        # "gap" is the whole circumference.
        assert math.isclose(_min_circular_gap([0.5, 0.5]), 1.0)


# ── the class anchors ───────────────────────────────────────────────
class TestClassAnchors:
    def test_three_anchors_with_documented_values(self):
        assert _CLASS_ANCHORS[0] == 0.0
        assert math.isclose(_CLASS_ANCHORS[1], math.sqrt(2) - 1.0)
        assert math.isclose(_CLASS_ANCHORS[2], math.sqrt(3) - 1.0)

    def test_anchors_are_pairwise_distinct(self):
        a, b, c = _CLASS_ANCHORS
        assert len({a, b, c}) == 3

    def test_pairwise_differences_are_the_documented_irrationals(self):
        # 0 + Q, (sqrt2-1)+Q and (sqrt3-1)+Q are distinct cosets:
        # the differences sqrt2-1, sqrt3-1 (irrational) and
        # sqrt3-sqrt2 (irrational: (sqrt3-sqrt2)(sqrt3+sqrt2)=1) are
        # all irrational, so no difference lies in Q.
        a, b, c = _CLASS_ANCHORS
        assert math.isclose(b - a, math.sqrt(2) - 1.0)
        assert math.isclose(c - b, math.sqrt(3) - math.sqrt(2))
        assert math.isclose(c - a, math.sqrt(3) - 1.0)


# ── the density honesty check ───────────────────────────────────────
class TestDensityHonesty:
    def test_every_class_meets_the_zoom_window_at_fine_grid(self):
        """The inset claims ``every window meets every class``; that must be
        literally true of the drawn (finite) schematic at q <= 300."""
        lo, hi = _DENSITY_WINDOW
        fracs = _reduced_fractions(_QS_FINE)
        for a in _CLASS_ANCHORS:
            xs = (a + fracs) % 1.0
            assert int(((xs >= lo) & (xs < hi)).sum()) >= 3, (
                f"class {a!r} misses the window {_DENSITY_WINDOW}")

    def test_illustration_does_not_trip_the_assertion(self):
        fig, _, _ = vitali_illustration()
        plt.close(fig)


# ── vitali_illustration ─────────────────────────────────────────────
class TestVitaliIllustration:
    def teardown_method(self):
        _close_plots()

    def test_returns_figure_four_axes_and_info(self):
        fig, (a1, a2, a3, a4), info = vitali_illustration()
        assert fig is not None
        # not vacuous: the four returned axes are distinct and are exactly
        # the figure's four subplot panels
        panels = (a1, a2, a3, a4)
        assert len({id(a) for a in panels}) == 4
        assert {id(a) for a in panels} <= {id(a) for a in fig.axes}
        assert len(fig.axes) >= 4
        assert tuple(info._fields) == (
            "class_anchors", "reps", "shifts", "translate_points",
            "disjoint_count", "min_gap", "n_rows_choice", "seed")

    def test_default_geometry(self):
        _, _, info = vitali_illustration()
        assert info.reps.shape == (12,)
        assert info.shifts.shape == (8,)
        assert info.translate_points.shape == (12, 8)
        # disjointness recomputed from the raw points, independent of the
        # value the figure itself reports in info.disjoint_count
        pts = np.round(info.translate_points, 12).reshape(-1)
        assert np.unique(pts).size == 12 * 8
        assert info.disjoint_count == 12 * 8 == 96
        assert math.isclose(info.min_gap, 0.007359, rel_tol=1e-4)

    def test_anchors_are_exposed_in_info(self):
        _, _, info = vitali_illustration()
        assert info.class_anchors == _CLASS_ANCHORS

    def test_translate_points_stay_on_the_circle(self):
        _, _, info = vitali_illustration()
        assert np.all(info.translate_points >= 0.0)
        assert np.all(info.translate_points < 1.0)

    def test_custom_counts_resize_the_schematic(self):
        _, _, info = vitali_illustration(n_reps=4, n_shifts=5)
        assert info.translate_points.shape == (4, 5)
        pts = np.round(info.translate_points, 12).reshape(-1)
        assert np.unique(pts).size == 20  # independent of disjoint_count
        assert info.disjoint_count == 20

    def test_circle_inset_marks_exactly_the_v_points_amber(self):
        """Regression: the tiling panel's circle inset colors V (column
        j==0, every n_shifts-th element of the row-major flat array) amber.
        ``np.repeat`` used to color the first n_reps elements instead, which
        silently contradicted the figure's own "amber = V" legend."""
        fig, (_, _, a3, _), _ = vitali_illustration(n_reps=6, n_shifts=4)
        ins = a3.child_axes[0]
        scatter = ins.collections[0]
        face = np.array(scatter.get_facecolors())
        assert face.shape == (6 * 4, 4)
        is_amber = np.all(
            np.isclose(face[:, :3], _hex_rgb(_COL_REP), atol=1e-3), axis=1)
        is_violet = np.all(
            np.isclose(face[:, :3], _hex_rgb(_COL_TILE), atol=1e-3), axis=1)
        assert int(is_amber.sum()) == 6
        assert int(is_violet.sum()) == 6 * 3
        # exactly one representative per row is V; nothing else is amber
        assert np.all(is_amber[::4]) and not np.any(is_amber[1::4])

    def test_n_rows_choice_only_scopes_the_choice_panel(self):
        _, _, info = vitali_illustration(n_rows_choice=3)
        assert info.reps.shape == (12,)  # tiling reps come from n_reps
        assert info.n_rows_choice == 3

    def test_seed_does_not_change_geometry(self):
        _, _, i1 = vitali_illustration(seed=1)
        _, _, i2 = vitali_illustration(seed=999)
        assert np.array_equal(i1.reps, i2.reps)
        assert np.array_equal(i1.translate_points, i2.translate_points)
        assert i1.min_gap == i2.min_gap


# ── validation ──────────────────────────────────────────────────────
class TestValidation:
    def teardown_method(self):
        _close_plots()

    def test_rejects_zero_rows(self):
        with pytest.raises(ValueError, match="positive"):
            vitali_illustration(n_rows_choice=0)

    def test_rejects_too_many_rows(self):
        with pytest.raises(ValueError, match="too large"):
            vitali_illustration(n_rows_choice=13)

    def test_rejects_too_many_reps(self):
        with pytest.raises(ValueError, match="too large"):
            vitali_illustration(n_reps=31)

    def test_rejects_zero_reps(self):
        with pytest.raises(ValueError, match="positive"):
            vitali_illustration(n_reps=0)

    def test_rejects_negative_reps(self):
        with pytest.raises(ValueError, match="positive"):
            vitali_illustration(n_reps=-1)

    def test_rejects_float_reps(self):
        with pytest.raises(TypeError, match="int"):
            vitali_illustration(n_reps=4.0)

    def test_rejects_zero_shifts(self):
        with pytest.raises(ValueError, match="positive"):
            vitali_illustration(n_shifts=0)

    def test_rejects_negative_shifts(self):
        with pytest.raises(ValueError, match="positive"):
            vitali_illustration(n_shifts=-2)

    def test_rejects_too_many_shifts(self):
        with pytest.raises(ValueError, match="too large"):
            vitali_illustration(n_shifts=31)

    def test_rejects_float_shifts(self):
        with pytest.raises(TypeError, match="int"):
            vitali_illustration(n_shifts=3.0)

    def test_rejects_bool_shifts(self):
        with pytest.raises(TypeError, match="int"):
            vitali_illustration(n_shifts=True)  # type: ignore[arg-type]

    def test_rejects_float_rows(self):
        with pytest.raises(TypeError, match="int"):
            vitali_illustration(n_rows_choice=4.0)

    def test_rejects_negative_rows(self):
        with pytest.raises(ValueError, match="positive"):
            vitali_illustration(n_rows_choice=-1)

    def test_rejects_bool_rows(self):
        with pytest.raises(TypeError, match="int"):
            vitali_illustration(n_rows_choice=True)  # type: ignore[arg-type]

    def test_rejects_bool_reps(self):
        with pytest.raises(TypeError, match="int"):
            vitali_illustration(n_reps=True)  # type: ignore[arg-type]

    def test_density_guard_trips_on_a_too_coarse_grid(self, monkeypatch):
        """The zoom-window honesty check is live: with no fraction in the
        window at q <= 2, the illustration refuses to draw the claim."""
        monkeypatch.setattr(vitali_mod, "_QS_FINE", 2)
        with pytest.raises(AssertionError, match="Density check failed"):
            vitali_illustration()

    def test_internal_disjointness_guard_trips_on_duplicate_reps(self, monkeypatch):
        """The tiling cannot be drawn if the rational translates collide."""
        monkeypatch.setattr(vitali_mod, "_reps", lambda n: np.full(n, 0.5))
        with pytest.raises(AssertionError, match="Internal check failed"):
            vitali_illustration(n_reps=4, n_shifts=5)

    def test_rejects_string_seed(self):
        with pytest.raises(TypeError, match="seed"):
            vitali_illustration(seed="7")  # type: ignore[arg-type]

    def test_rejects_float_seed(self):
        with pytest.raises(TypeError, match="seed"):
            vitali_illustration(seed=7.0)  # type: ignore[arg-type]

    def test_rejects_bool_seed(self):
        with pytest.raises(TypeError, match="seed"):
            vitali_illustration(seed=True)  # type: ignore[arg-type]

    def test_rejects_huge_figsize(self):
        with pytest.raises(ValueError, match="too large"):
            vitali_illustration(figsize=(500, 6))

    def test_rejects_malformed_figsize(self):
        with pytest.raises(TypeError, match="tuple"):
            vitali_illustration(figsize=(8,))


# ── rendered text hygiene ───────────────────────────────────────────
class TestTextHygiene:
    def teardown_method(self):
        _close_plots()

    def test_no_control_characters(self):
        """Regression discipline: no raw-string escape can leak a control
        character into rendered text."""
        fig, _, _ = vitali_illustration()

        texts = list(fig.texts)
        if fig._suptitle is not None:
            texts.append(fig._suptitle)
        for ax in fig.axes:
            texts.extend(ax.texts)
            texts.extend(ax.get_xticklabels() + ax.get_yticklabels())
            for ins in ax.child_axes:
                texts.append(ins.title)
                texts.extend(ins.get_xticklabels() + ins.get_yticklabels())

        bad = [
            t.get_text()
            for t in texts
            if any(ord(c) < 32 and c != "\n" for c in t.get_text())
        ]
        assert bad == [], f"control characters in rendered text: {bad!r}"
