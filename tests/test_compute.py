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
from ensemble_signals import compute_breakout_signal_series, compute_momentum_signal_series
from khung4_tplus_signals import compute_khung4_tplus
from mama_positional_signals import compute_mama_positional_system
from advanced_trailstop_signals import compute_advanced_trailstop


def _make_df(close_values, volume_values=None):
    n = len(close_values)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "TradingDate": dates,
        "Close": close_values,
        "Volume": volume_values if volume_values else [1_000_000] * n,
    })
    return df


def _make_ohlcv_df(open_values, high_values, low_values, close_values):
    n = len(close_values)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "TradingDate": dates,
        "Open": open_values,
        "High": high_values,
        "Low": low_values,
        "Close": close_values,
        "Volume": [1_000_000] * n,
    })


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
        assert compute_rsi_numpy(arr, 14) == 50.0

    def test_rsi_numpy_up_trend(self):
        arr = np.arange(50, 75, dtype=float)
        rsi = compute_rsi_numpy(arr, 14)
        assert 50 <= rsi <= 100

    def test_rsi_too_short(self):
        arr = np.array([50.0] * 5)
        assert compute_rsi_numpy(arr, 14) == 50.0
        assert compute_rsi_wilder(pd.Series(arr), 14) == 50.0

    def test_rsi_uses_wilder_seed_and_implementations_agree(self):
        # First 14 deltas: eight +2 gains and six -1 losses; then one -1 loss.
        deltas = [2.0] * 8 + [-1.0] * 7
        close = np.array([100.0] + list(100.0 + np.cumsum(deltas)))
        expected = 100.0 * 52.0 / 75.0

        assert compute_rsi_numpy(close, 14) == pytest.approx(expected)
        assert compute_rsi_wilder(pd.Series(close), 14) == pytest.approx(expected)

    @pytest.mark.parametrize("deltas, expected", [([1.0] * 14, 100.0), ([-1.0] * 14, 0.0)])
    def test_rsi_handles_one_sided_moves(self, deltas, expected):
        close = np.array([100.0] + list(100.0 + np.cumsum(deltas)))
        assert compute_rsi_numpy(close, 14) == expected
        assert compute_rsi_wilder(pd.Series(close), 14) == expected


class TestEnsembleVolumeConditions:
    def test_breakout_uses_the_prior_twenty_session_volume_average(self):
        close = pd.Series([100.0] * 24 + [101.0])
        volume = pd.Series([100.0] * 24 + [151.0])

        assert bool(compute_breakout_signal_series(close, volume).iloc[-1]) is True

    def test_momentum_uses_live_normalized_linear_volume_slope(self):
        close = pd.Series([120.0] * 5 + [100.0] * 10 + list(np.linspace(105.0, 150.0, 10)))
        volume = pd.Series([100.0] * 15 + [100.0] * 5 + [1000.0, 0.0, 0.0, 0.0, 0.0])

        signal, volume_slope = compute_momentum_signal_series(close, volume)

        assert volume_slope.iloc[-1] < 0
        assert bool(signal.iloc[-1]) is False


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


class TestKhung4Tplus:
    def test_requires_previous_d_before_cross(self):
        df = _make_ohlcv_df(
            [10, 10, 10, 10, 16],
            [11, 12, 13, 14, 16],
            [9, 9, 9, 9, 15],
            [10, 10, 10, 10, 15],
        )
        result = compute_khung4_tplus(df)
        assert result["buy"] is False
        assert result["state"] == 0

    def test_buy_price_is_close_on_state_flip_to_one(self):
        df = _make_ohlcv_df(
            [10, 10, 10, 10, 8, 8, 15],
            [11, 12, 13, 14, 9, 9, 15],
            [9, 9, 9, 9, 7, 7, 13],
            [10, 10, 10, 10, 8, 8, 15],
        )
        result = compute_khung4_tplus(df)
        assert result["buy"] is True
        assert result["state"] == 1
        assert result["buy_price"] == 15.0


class TestMamaPositional:
    def test_ehlers_period_is_dynamic(self):
        n = 140
        x = np.arange(n, dtype=float)
        close = 50 + 4 * np.sin(x / 5) + 0.04 * x
        high = close + 1.0
        low = close - 1.0
        df = _make_ohlcv_df(close, high, low, close)

        result = compute_mama_positional_system(df)
        period = result["period_series"]

        assert period.iloc[-1] > 0
        assert period.iloc[30:].nunique() > 1
        assert result["mama"] is not None
        assert result["fama"] is not None

    def test_outputs_confirmed_signal_series(self):
        n = 120
        x = np.arange(n, dtype=float)
        close = 30 + 0.1 * x + 2.5 * np.sin(x / 4)
        high = close + 0.8
        low = close - 0.8
        df = _make_ohlcv_df(close, high, low, close)

        result = compute_mama_positional_system(df)

        assert len(result["buy_series"]) == n
        assert len(result["sell_series"]) == n
        assert "buy_setup_value_series" in result
        assert "sell_setup_value_series" in result


class TestAdvancedTrailstop:
    def test_bs_updates_only_after_nine_bar_low_condition(self):
        close = [10.0] * 9 + [20.0]
        high = [11.0] * 9 + [21.0]
        low = [9.0] * 9 + [19.0]
        df = _make_ohlcv_df(close, high, low, close)

        result = compute_advanced_trailstop(df)

        assert bool(result["up_condition_series"].iloc[8]) is False
        assert bool(result["up_condition_series"].iloc[9]) is True
        assert result["bs_series"].iloc[8] == 0.0
        assert result["bs_series"].iloc[9] > 0.0
        assert result["bs_series"].iloc[9] < low[-1]

    def test_buy_is_close_cross_above_bs_not_bs_update(self):
        df = _make_ohlcv_df(
            [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 35, 40, 46],
            [51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 36, 41, 47],
            [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 34, 39, 45],
            [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 35, 40, 46],
        )

        result = compute_advanced_trailstop(df, mult=0.1, aper=2)
        buy = result["buy_series"]

        assert len(buy) == len(df)
        assert bool(buy.iloc[-1]) == (df["Close"].iloc[-1] > result["bs_series"].iloc[-1] and df["Close"].iloc[-2] <= result["bs_series"].iloc[-2])
        assert "buy_price_series" in result
        assert "sell_price_series" in result


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
