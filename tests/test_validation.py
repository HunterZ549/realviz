"""Tests for ``realviz._validation``."""

import math

import pytest

from realviz._validation import (
    _check_callable,
    _check_figsize,
    _check_interval,
    _check_partitions,
    _check_samples,
    _validate_function,
)


# ── _check_callable ──────────────────────────────────────────────
class TestCheckCallable:
    def test_accepts_function(self):
        _check_callable(lambda x: x)  # does not raise

    def test_accepts_builtin(self):
        _check_callable(abs)

    def test_rejects_none(self):
        with pytest.raises(TypeError, match="callable"):
            _check_callable(None)

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="callable"):
            _check_callable("not a function")

    def test_custom_name_in_error(self):
        with pytest.raises(TypeError, match="my_func"):
            _check_callable(42, name="my_func")


# ── _check_interval ──────────────────────────────────────────────
class TestCheckInterval:
    def test_accepts_valid_interval(self):
        _check_interval(0.0, 1.0)

    def test_accepts_negative_interval(self):
        _check_interval(-5.0, -1.0)

    def test_rejects_a_equals_b(self):
        with pytest.raises(ValueError, match="strictly less"):
            _check_interval(3.0, 3.0)

    def test_rejects_a_greater_than_b(self):
        with pytest.raises(ValueError, match="strictly less"):
            _check_interval(5.0, 2.0)

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            _check_interval(0.0, math.inf)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            _check_interval(0.0, math.nan)

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="numbers"):
            _check_interval("0", "1")

    def test_rejects_huge_domain(self):
        with pytest.raises(ValueError, match="too large"):
            _check_interval(-1e13, 1e13)


# ── _check_partitions ────────────────────────────────────────────
class TestCheckPartitions:
    def test_accepts_positive_int(self):
        _check_partitions(10)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):
            _check_partitions(0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive"):
            _check_partitions(-5)

    def test_rejects_float(self):
        with pytest.raises(TypeError, match="int"):
            _check_partitions(10.5)  # type: ignore[arg-type]

    def test_rejects_bool(self):
        with pytest.raises(TypeError, match="int"):
            _check_partitions(True)  # type: ignore[arg-type]

    def test_rejects_too_large(self):
        with pytest.raises(ValueError, match="too large"):
            _check_partitions(100_000)


# ── _check_samples ───────────────────────────────────────────────
class TestCheckSamples:
    def test_accepts_reasonable_count(self):
        _check_samples(2000)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):
            _check_samples(0)

    def test_rejects_too_large(self):
        with pytest.raises(ValueError, match="too large"):
            _check_samples(2_000_000)


# ── _check_figsize ───────────────────────────────────────────────
class TestCheckFigsize:
    def test_accepts_reasonable_size(self):
        _check_figsize((13, 5.5))

    def test_accepts_integers(self):
        _check_figsize((8, 6))

    def test_rejects_not_tuple(self):
        with pytest.raises(TypeError, match="tuple"):
            _check_figsize([8, 6])  # type: ignore[arg-type]

    def test_rejects_wrong_length(self):
        with pytest.raises(TypeError, match="tuple"):
            _check_figsize((8,))  # type: ignore[arg-type]

    def test_rejects_non_numeric_entries(self):
        with pytest.raises(TypeError, match="numbers"):
            _check_figsize(("8", 6))  # type: ignore[arg-type]

    def test_rejects_infinite_entry(self):
        with pytest.raises(ValueError, match="finite"):
            _check_figsize((math.inf, 6))

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):
            _check_figsize((0, 6))

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive"):
            _check_figsize((8, -1))

    def test_rejects_too_large(self):
        with pytest.raises(ValueError, match="too large"):
            _check_figsize((500, 6))


# ── _validate_function ───────────────────────────────────────────
class TestValidateFunction:
    def test_valid_linear(self):
        _validate_function(lambda x: 2 * x + 1, 0.0, 1.0)

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            _validate_function("boom", 0.0, 1.0)  # type: ignore[arg-type]

    def test_rejects_function_that_raises(self):
        def bad(x):
            raise ZeroDivisionError("nope")

        with pytest.raises(ValueError, match="ZeroDivisionError"):
            _validate_function(bad, 0.0, 1.0)

    def test_rejects_non_numeric_return(self):
        with pytest.raises(TypeError, match="numbers"):
            _validate_function(lambda x: "string", 0.0, 1.0)  # type: ignore[arg-type]

    def test_rejects_non_finite_return(self):
        with pytest.raises(ValueError, match="non-finite"):
            _validate_function(lambda x: math.nan, 0.0, 1.0)

    def test_detects_constant_when_disallowed(self):
        with pytest.raises(ValueError, match="constant"):
            _validate_function(lambda x: 5.0, 0.0, 10.0, allow_constant=False)

    def test_allows_constant_by_default(self):
        _validate_function(lambda x: 5.0, 0.0, 10.0)  # ok — allow_constant=True
