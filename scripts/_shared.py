"""
Shared utilities — tqdm fallback, paths, constants, helpers.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "ohlc_cache"
DOCS_DATA_DIR = ROOT / "docs" / "data"
WEIGHTS_PATH = DATA_DIR / "backtest_weights.json"

# --- Thresholds ---
MIN_SYMBOL_HISTORY = 220

# --- Score bases (momentum signals + backtest) ---
SCORE_MA = 30
SCORE_BREAKOUT = 35
SCORE_ROC = 25
SCORE_HYBRID = 40

# --- Bonuses ---
BONUS_VOL_SURGE = 15
BONUS_ADX_STRONG = 10
BONUS_RSI_GOLD = 10

# --- Ensemble weights ---
DEFAULT_WEIGHTS = {"ma_crossover": 0.30, "pullback": 0.14, "breakout": 0.28, "momentum": 0.27}

# --- Tqdm fallback (works on CI without tqdm) ---
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self._it = iterable
            self._desc = kwargs.get('desc', '')
            try:
                self._total = len(iterable)
            except Exception:
                self._total = '?'
            self._n = 0
            print(f"{self._desc}: 0/{self._total}")
        def __iter__(self):
            for item in self._it:
                yield item
                self._n += 1
                if self._n % 50 == 0:
                    print(f"{self._desc}: {self._n}/{self._total}")
        def set_postfix_str(self, s, **kw): pass
        def close(self):
            print(f"{self._desc}: {self._n}/{self._total} - Done")
        @staticmethod
        def write(msg): print(msg)


def list_symbols(cache_dir: Path = CACHE_DIR, min_history: int = 20,
                 min_volume: int | None = None, skip_prefix: tuple[str, ...] = ("FU", "E1")) -> list[str]:
    """List sorted symbols from ohlc_cache, filtered by history length and volume."""
    from cache_utils import load_cache as _load_cache
    import pandas as pd
    symbols = []
    for path in sorted(cache_dir.glob("*.csv")):
        sym = path.stem
        if sym == ".gitkeep":
            continue
        if skip_prefix and sym.startswith(skip_prefix):
            continue
        df = _load_cache(sym, cache_dir)
        if len(df) < min_history:
            continue
        if min_volume and "Volume" in df.columns:
            avg_vol = df["Volume"].dropna().iloc[-20:].mean()
            if pd.isna(avg_vol) or avg_vol < min_volume:
                continue
        symbols.append(sym)
    return symbols


def json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
