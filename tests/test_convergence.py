"""Tests for ``realviz.convergence``."""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from realviz.convergence import (
    FAMILIES,
    ConvergenceInfo,
    _typewriter,
    convergence_modes,
)


# ── helpers ─────────────────────────────────────────────────────────
def _close_plots():
    """Prevent figure leaks between tests."""
    plt.close("all")


# ── the typewriter mapping ──────────────────────────────────────────
class TestTypewriter:
    def test_first_block_sweeps_left_half_then_right_half(self):
        x = np.array([0.25, 0.75])
        assert np.allclose(_typewriter(1, x), [1.0, 0.0])  # [0, 1/2)
        assert np.allclose(_typewriter(2, x), [0.0, 1.0])  # [1/2, 1)

    def test_second_block_quarters(self):
        x = np.array([0.1, 0.3, 0.6, 0.9])
        assert np.allclose(_typewriter(3, x), [1.0, 0.0, 0.0, 0.0])  # [0, 1/4)
        assert np.allclose(_typewriter(6, x), [0.0, 0.0, 0.0, 1.0])  # [3/4, 1)

    def test_third_block_eighths(self):
        x = np.array([0.01, 0.2, 0.3])
        assert np.allclose(_typewriter(7, x), [1.0, 0.0, 0.0])  # [0, 1/8)
        assert np.allclose(_typewriter(8, x), [0.0, 1.0, 0.0])  # [1/8, 1/4)
        assert np.allclose(_typewriter(9, x), [0.0, 0.0, 1.0])  # [1/4, 3/8)

    def test_fourth_block_sixteenths(self):
        x = np.array([0.05, 0.1])
        assert np.allclose(_typewriter(15, x), [1.0, 0.0])  # [0, 1/16)
        assert np.allclose(_typewriter(16, x), [0.0, 1.0])  # [1/16, 1/8)

    def test_vectorized_over_x(self):
        x = np.linspace(0, 1, 200)
        assert np.allclose(_typewriter(1, x),
                           np.where((x >= 0) & (x < 0.5), 1.0, 0.0))

    def test_every_point_covered_somewhere(self):
        # 0.37 has a dyadic expansion, so some member of each late block
        # contains it; in particular block 4 contains 0.37 in [5/16, 6/16).
        assert _typewriter(20, 0.37) == 1.0
        assert _typewriter(23, 0.37) == 0.0


# ── FAMILIES registry ───────────────────────────────────────────────
class TestFamilies:
    def test_has_three_counterexamples(self):
        assert set(FAMILIES) == {"x_pow_n", "thin_spike", "typewriter"}

    def test_each_spec_has_required_fields(self):
        for spec in FAMILIES.values():
            for key in ("label", "a", "b", "n_max", "epsilon", "family", "f", "note"):
                assert key in spec, f"{key} missing from {spec.get('label')}"

    def test_spec_families_are_callable(self):
        for name, spec in FAMILIES.items():
            x = np.linspace(spec["a"], spec["b"], 50)
            y = np.asarray(spec["family"](1, x), dtype=float)
            assert y.shape == x.shape and np.all(np.isfinite(y))

    def test_spec_targets_are_callable(self):
        for name, spec in FAMILIES.items():
            x = np.linspace(spec["a"], spec["b"], 50)
            y = np.asarray(spec["f"](x), dtype=float)
            assert y.shape == x.shape and np.all(np.isfinite(y))


# ── convergence_modes ───────────────────────────────────────────────
class TestConvergenceModes:
    def teardown_method(self):
        _close_plots()

    def test_returns_figure_axes_and_info(self):
        fig, (ax_pt, ax_un, ax_ae, ax_l1), info = convergence_modes(
            "x_pow_n", figsize=(8, 5)
        )
        assert fig is not None
        assert isinstance(info, ConvergenceInfo)
        assert info.levels[-1] == 16
        for name in ("sup_errors", "l1_norms", "bad_measures", "slice_errors"):
            assert len(getattr(info, name)) == 16

    def test_slice_xs_at_quarter_points(self):
        _, _, info = convergence_modes("x_pow_n", figsize=(8, 5))
        assert np.allclose(info.slice_xs, [0.25, 0.5, 0.75])

    def test_bad_measures_nan_before_window_fills(self):
        _, _, info = convergence_modes("x_pow_n", figsize=(8, 5))
        assert np.all(np.isnan(info.bad_measures[:2]))
        assert np.all(np.isfinite(info.bad_measures[2:]))

    def test_default_n_max_is_8_for_callable(self):
        _, _, info = convergence_modes(lambda n, x: x / n, 0, 1,
                                       figsize=(8, 5))
        assert info.levels[-1] == 8

    def test_preset_override_wins(self):
        _, _, info = convergence_modes("x_pow_n", n_max=4, figsize=(8, 5))
        assert info.levels[-1] == 4

    def test_absent_f_uses_last_member(self):
        _, _, info = convergence_modes(lambda n, x: x / n, 0, 1,
                                       figsize=(8, 5))
        assert info.sup_errors[-1] == 0.0  # last member is its own target

    def test_constant_family_all_modes_hold(self):
        _, _, info = convergence_modes(lambda n, x: 3.0, 0, 1, n_max=6,
                                       f=lambda x: 3.0, n_samples=500,
                                       figsize=(8, 5))
        assert info.pointwise_ok and info.uniform_ok
        assert info.ae_ok and info.l1_ok

    # ── preset verdicts (the classic counterexamples) ─────────────
    def test_x_pow_n_pointwise_not_uniform(self):
        """x^n: converges pointwise and a.e., NOT uniformly, and in L1."""
        _, _, info = convergence_modes("x_pow_n", figsize=(8, 5))
        assert info.pointwise_ok is True
        assert info.uniform_ok is False
        assert info.ae_ok is True
        assert info.l1_ok is True

    def test_x_pow_n_metric_values(self):
        _, _, info = convergence_modes("x_pow_n", figsize=(8, 5))
        # sup|f_16 - f| is still ~1 near x=1; ∫x^16 = 1/17; settle set ~ 0
        assert info.sup_errors[-1] > 0.9
        assert math.isclose(info.l1_norms[-1], 1 / 17, rel_tol=0.02)
        assert info.bad_measures[-1] < 1e-6

    def test_x_pow_n_slices_converge(self):
        _, _, info = convergence_modes("x_pow_n", figsize=(8, 5))
        assert np.all(info.slice_errors[-1] < info.epsilon_val)

    def test_thin_spike_ae_not_l1(self):
        """n·1_[0,1/n]: converges a.e. (fails only at {0}), NOT in L1."""
        _, _, info = convergence_modes("thin_spike", figsize=(8, 5))
        assert info.pointwise_ok is False
        assert info.uniform_ok is False
        assert info.ae_ok is True
        assert info.l1_ok is False

    def test_thin_spike_metric_values(self):
        _, _, info = convergence_modes("thin_spike", figsize=(8, 5))
        # the spike is 20 tall, its integral is exactly 1, and the unsettled
        # set is ~[0, 1/18] because the last three members still differ there
        assert math.isclose(info.sup_errors[-1], 20.0, rel_tol=0.01)
        assert math.isclose(info.l1_norms[-1], 1.0, rel_tol=0.02)
        assert 0.03 < info.bad_measures[-1] < 0.08

    def test_thin_spike_f_targets_zero(self):
        _, _, info = convergence_modes("thin_spike", figsize=(8, 5))
        assert np.all(info.slice_errors[-1] == 0.0)  # slices dodge the spike

    def test_typewriter_l1_but_nowhere(self):
        """Typewriter: converges in L1 but fails at every point."""
        _, _, info = convergence_modes("typewriter", figsize=(8, 5))
        assert info.pointwise_ok is False
        assert info.uniform_ok is False
        assert info.ae_ok is False
        assert info.l1_ok is True

    def test_typewriter_metric_values(self):
        _, _, info = convergence_modes("typewriter", figsize=(8, 5))
        # last block is [1/16, 1/8): ∫ = 1/16; a large unsettled region ~0.25
        assert math.isclose(info.l1_norms[-1], 1 / 16, abs_tol=0.003)
        assert math.isclose(info.bad_measures[-1], 0.25, abs_tol=0.02)

    def test_metrics_are_nonnegative(self):
        for name in ("x_pow_n", "thin_spike", "typewriter"):
            _, _, info = convergence_modes(name, figsize=(8, 5))
            assert np.all(info.sup_errors >= 0)
            assert np.all(info.l1_norms >= 0)
            # the settle measure never exceeds the interval length
            assert np.all(info.bad_measures[np.isfinite(info.bad_measures)]
                          <= 1.0 + 1e-12)

    # ── regression: verdicts must not be fooled by the fixes ───────
    def test_family_converging_to_wrong_limit_is_rejected(self):
        """A family that settles at the wrong value must NOT pass the
        pointwise/a.e. cameras — the settle test must notice it is stuck
        off ``f`` (regression for the 'stuck off-f' criterion)."""
        _, _, info = convergence_modes(
            lambda n, x: 3.0, 0, 1, n_max=6, f=lambda x: 5.0,
            n_samples=500, figsize=(8, 5))
        assert info.pointwise_ok is False
        assert info.ae_ok is False

    def test_l1_requires_tail_norm_to_decay(self):
        """When the first member already equals ``f`` but later members
        drift away, the L1 verdict must be False: the norm has to shrink,
        not merely the first member equal ``f`` (regression for the removed
        ``l1_head == 0`` escape)."""
        _, _, info = convergence_modes(
            lambda n, x: 0.0 if n == 1 else 1.0, 0, 1, n_max=6,
            f=lambda x: 0.0, n_samples=500, figsize=(8, 5))
        assert info.l1_norms[0] == 0.0   # first member == f
        assert info.l1_norms[-1] > 0     # ... but the tail diverges
        assert info.l1_ok is False

    def test_thin_spike_pointwise_false_on_coarse_grid(self):
        """The pointwise verdict must not depend on sampling density: at
        n_samples=100 the spike's unsettled set is still ~[0, 1/18], far
        above the absolute tolerance (regression for the removed
        grid-cell-count threshold)."""
        _, _, info = convergence_modes("thin_spike", n_samples=100,
                                       figsize=(8, 5))
        assert info.pointwise_ok is False
        assert info.ae_ok is True
        assert info.bad_measures[-1] > 0.01   # the absolute pt tolerance

    # ── validation ─────────────────────────────────────────────────
    def test_rejects_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            convergence_modes("nope", figsize=(8, 5))

    def test_rejects_non_callable_family(self):
        with pytest.raises(TypeError, match="callable"):
            convergence_modes(42, 0, 1, figsize=(8, 5))  # type: ignore[arg-type]

    def test_rejects_callable_without_interval(self):
        with pytest.raises(ValueError, match="a and b are required"):
            convergence_modes(lambda n, x: x, figsize=(8, 5))

    def test_rejects_bad_interval(self):
        with pytest.raises(ValueError, match="strictly less"):
            convergence_modes(lambda n, x: x, 5, 1, figsize=(8, 5))

    def test_rejects_non_callable_target(self):
        with pytest.raises(TypeError, match="callable"):
            convergence_modes(lambda n, x: x, 0, 1, f="nope",
                              figsize=(8, 5))  # type: ignore[arg-type]

    def test_rejects_zero_n_max(self):
        with pytest.raises(ValueError, match="positive"):
            convergence_modes(lambda n, x: x, 0, 1, n_max=0, figsize=(8, 5))

    def test_rejects_n_max_below_settle_window(self):
        """n_max < the settle window cannot run the pointwise/a.e. test,
        so it must fail loudly instead of crashing on an empty slice."""
        with pytest.raises(ValueError, match="too small"):
            convergence_modes(lambda n, x: x, 0, 1, n_max=2, figsize=(8, 5))

    def test_rejects_float_n_max(self):
        with pytest.raises(TypeError, match="int"):
            convergence_modes(lambda n, x: x, 0, 1, n_max=8.0,
                              figsize=(8, 5))  # type: ignore[arg-type]

    def test_rejects_huge_n_max(self):
        with pytest.raises(ValueError, match="too large"):
            convergence_modes(lambda n, x: x, 0, 1, n_max=61, figsize=(8, 5))

    def test_rejects_bad_epsilon(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            convergence_modes(lambda n, x: x, 0, 1, epsilon=1.5,
                              figsize=(8, 5))

    def test_rejects_string_epsilon(self):
        with pytest.raises(TypeError, match="number"):
            convergence_modes(lambda n, x: x, 0, 1, epsilon="0.08",
                              figsize=(8, 5))  # type: ignore[arg-type]

    def test_rejects_huge_figsize(self):
        with pytest.raises(ValueError, match="too large"):
            convergence_modes("x_pow_n", figsize=(500, 6))

    def test_rejects_family_that_raises(self):
        def kaboom(n, x):
            if n > 1:
                raise RuntimeError("boom")
            return x

        with pytest.raises(ValueError, match="RuntimeError"):
            convergence_modes(kaboom, 0, 1, figsize=(8, 5))

    def test_rejects_non_finite_family(self):
        with pytest.raises(ValueError, match="non-finite"):
            convergence_modes(lambda n, x: math.nan, 0, 1, figsize=(8, 5))

    def test_rejects_expensive_n_max_samples_product(self):
        with pytest.raises(ValueError, match="too expensive"):
            convergence_modes(lambda n, x: x, 0, 1,
                              n_max=60, n_samples=1_000_000, figsize=(8, 5))

    def test_rejects_range_overflow(self):
        """A huge-but-finite range must be rejected instead of producing NaN
        geometry (and a matplotlib crash)."""
        with pytest.raises(ValueError, match="too large to measure"):
            convergence_modes(
                lambda n, x: 1e308 * np.sin(2 * np.pi * np.asarray(x, float)),
                0, 1, figsize=(8, 5))


# ── rendered text hygiene ───────────────────────────────────────────
class TestTextHygiene:
    def teardown_method(self):
        _close_plots()

    def test_no_control_characters(self):
        """Regression discipline: no raw-string escape can leak a control
        character into rendered text."""
        fig, _, _ = convergence_modes("x_pow_n", figsize=(8, 5))

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
