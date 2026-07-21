"""
Shared utilities: cache loading + RSI calculation.
Dung chung cho fetch_and_compute, strategy_signals, ensemble_signals, market_commentary.
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_trading_date(series: pd.Series) -> pd.Series:
    """Parse cache dates safely across dd/mm/YYYY and ISO YYYY-mm-dd formats."""
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")
    return parsed


def load_cache(symbol: str, cache_dir: Path) -> pd.DataFrame:
    """Load OHLC cache for a symbol. Returns DataFrame with TradingDate, OHLC, Volume when available."""
    path = cache_dir / f"{symbol}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df["TradingDate"] = parse_trading_date(df["TradingDate"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            for col in ("Open", "High", "Low", "Volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = float("nan")
            return df.dropna(subset=["Close"])
        except Exception as exc:
            logger.warning("load_cache(%s): %s", symbol, exc)
    return pd.DataFrame(columns=["TradingDate", "Open", "High", "Low", "Close", "Volume"])


def _rsi_wilder_values(values: np.ndarray, period: int) -> np.ndarray:
    """Return textbook Wilder RSI values, using neutral 50 before the seed."""
    result = np.full(len(values), 50.0)
    if len(values) < period + 1:
        return result

    deltas = np.diff(values)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, len(values)):
        if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
            result[i] = 50.0
        elif avg_loss == 0.0:
            result[i] = 100.0 if avg_gain > 0.0 else 50.0
        elif avg_gain == 0.0:
            result[i] = 0.0
        else:
            result[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

        if i < len(deltas):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return result


def compute_rsi_wilder_series(close_series: pd.Series, period: int = 14) -> pd.Series:
    """Return textbook Wilder RSI seeded from the first ``period`` deltas."""
    values = close_series.to_numpy(dtype=float, copy=False)
    return pd.Series(_rsi_wilder_values(values, period), index=close_series.index, dtype=float)


def compute_rsi_wilder(close_series: pd.Series, period: int = 14) -> float:
    """Return the latest textbook Wilder RSI for a pandas close series."""
    return float(compute_rsi_wilder_series(close_series, period).iloc[-1]) if len(close_series) else 50.0


def compute_rsi_numpy(series: np.ndarray, period: int = 14) -> float:
    """Return the latest textbook Wilder RSI for a NumPy close array."""
    values = np.asarray(series, dtype=float)
    return float(_rsi_wilder_values(values, period)[-1]) if len(values) else 50.0
