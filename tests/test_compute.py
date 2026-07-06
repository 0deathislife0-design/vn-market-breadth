"""Unit tests cho các hàm tính toán."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pandas as pd
import pytest

from cache_utils import compute_rsi_wilder, compute_rsi_numpy
from momentum_signals import (
    compute_adx, compute_ma_crossover, compute_breakout,
    compute_roc_momentum, compute_hybrid, check_common_filters,
    compute_bonuses,
)


def _make_df(close_values, volume_values=None):
    n = len(close_values)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "TradingDate": dates,
        "Close": close_values,
        "Volume": volume_values if volume_values else [1_000_000] * n,
    })
    return df


class TestRSI:
    def test_rsi_wilder_constant(self):
        close = pd.Series([50.0] * 20)
        assert compute_rsi_wilder(close, 14) == 50.0

    def test_rsi_wilder_up_trend(self):
        close = pd.Series(range(50, 70))  # always up
        rsi = compute_rsi_wilder(close, 14)
        assert 60 <= rsi <= 100

    def test_rsi_numpy_constant(self):
        arr = np.full(20, 50.0)
        assert compute_rsi_numpy(arr, 14) == 100.0  # avg_loss = 0

    def test_rsi_numpy_up_trend(self):
        arr = np.arange(50, 75, dtype=float)
        rsi = compute_rsi_numpy(arr, 14)
        assert 50 <= rsi <= 100

    def test_rsi_too_short(self):
        arr = np.array([50.0] * 5)
        assert compute_rsi_numpy(arr, 14) == 50.0
        assert compute_rsi_wilder(pd.Series(arr), 14) == 50.0


class TestADX:
    def test_adx_too_short(self):
        arr = np.array([50.0] * 10)
        assert compute_adx(arr, 14) == 0.0

    def test_adx_trending(self):
        arr = np.arange(50, 100, dtype=float)  # strong up trend
        adx = compute_adx(arr, 14)
        assert 0 <= adx <= 100

    def test_adx_range_bound(self):
        np.random.seed(42)
        arr = 50 + np.cumsum(np.random.randn(60))  # random walk
        adx = compute_adx(arr, 14)
        assert 0 <= adx <= 100


class TestStrategySignals:
    def test_ma_crossover_basic(self):
        close = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60] * 10
        df = _make_df([float(c) for c in close])

        # Uptrend: ma10 > ma50, price > ma10, RSI > 50 for sure
        result = compute_ma_crossover(df)
        assert isinstance(result, dict)
        assert "signal" in result
        assert result["ma10"] is not None

    def test_ma_crossover_too_short(self):
        df = _make_df(list(range(30)))
        result = compute_ma_crossover(df)
        assert result["signal"] == 0

    def test_breakout_detection(self):
        # Steady then big spike
        base = [50.0] * 25
        close = base + [55.0]
        df = _make_df(close, [1_000_000] * 26)
        result = compute_breakout(df)
        assert "signal" in result

    def test_roc_momentum_too_short(self):
        df = _make_df(list(range(20)))
        result = compute_roc_momentum(df)
        assert result["signal"] == 0

    def test_hybrid_too_short(self):
        df = _make_df(list(range(30)))
        result = compute_hybrid(df)
        assert result["signal"] == 0

    def test_check_common_filters_too_short(self):
        df = _make_df(list(range(100)))
        assert check_common_filters(df) is False

    def test_bonuses_no_volume_surge(self):
        close = list(range(100, 160))
        volume = [1_000_000] * 60
        df = _make_df(close, volume)
        result = compute_bonuses(df)
        assert "total" in result
        assert result["vol_surge"] == 0


class TestCommonFilters:
    def test_volume_filter_fails(self):
        n = 250
        close = list(range(n))
        volume = [50_000] * n  # below MIN_AVG_VOLUME = 500k
        df = _make_df(close, volume)
        # should fail on volume, not on price/RSI
        result = check_common_filters(df)
        assert result is False

    def test_trend_filter_fails(self):
        """MA50 < MA200 scenario."""
        n = 250
        # Downtrend: recent lower than old
        close = list(range(300, 200, -1)) + list(range(200, 50, -1))
        # Ensure exactly 250+ entries for df
        while len(close) < 250:
            close.append(close[-1] - 1)
        close = close[:250]
        df = _make_df(close, [1_000_000] * len(close))
        result = check_common_filters(df)
        # Likely fails trend check
        assert result is False
