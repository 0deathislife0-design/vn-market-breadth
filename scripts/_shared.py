"""
Shared utilities — tqdm fallback, paths, constants, helpers.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "ohlc_cache"
DOCS_DATA_DIR = ROOT / "docs" / "data"
WEIGHTS_PATH = DATA_DIR / "backtest_weights.json"
DATE_FMT = "%d/%m/%Y"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# --- Thresholds ---
MIN_SYMBOL_HISTORY = 220


def vn_now() -> datetime:
    """Return an aware timestamp in Vietnam's civil timezone."""
    return datetime.now(VIETNAM_TZ)


def parse_market_date(value) -> datetime | None:
    """Normalize API/JSON market dates to a timezone-naive midnight datetime."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(VIETNAM_TZ).tz_localize(None)
    return datetime.combine(parsed.date(), time.min)


def format_market_date(value) -> str:
    parsed = parse_market_date(value)
    return parsed.strftime(DATE_FMT) if parsed else ""


def latest_pipeline_market_date() -> datetime | None:
    """Read the latest authoritative aggregate date emitted by the breadth pipeline."""
    path = DATA_DIR / "breadth_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return parse_market_date(payload.get("markets", {}).get("ALL", {}).get("date"))
    except (OSError, ValueError, TypeError):
        return None


def latest_cached_market_date(cache_dir: Path = CACHE_DIR) -> datetime | None:
    """Return the newest cached OHLC date, for standalone signal scripts."""
    from cache_utils import load_cache as _load_cache

    latest = None
    for path in cache_dir.glob("*.csv"):
        df = _load_cache(path.stem, cache_dir)
        if df.empty or "TradingDate" not in df.columns:
            continue
        candidate = pd.to_datetime(df["TradingDate"], errors="coerce").max()
        if not pd.isna(candidate) and (latest is None or candidate > latest):
            latest = candidate.to_pydatetime()
    return parse_market_date(latest)


def signal_market_date() -> datetime | None:
    """Prefer the pipeline's market session date; fall back to the local cache."""
    return latest_pipeline_market_date() or latest_cached_market_date()


def max_signal_staleness_days() -> int:
    return max(0, int(os.environ.get("MAX_SIGNAL_STALENESS_DAYS", "10")))


def is_market_data_fresh(last_date, reference_date=None, max_days: int | None = None) -> bool:
    """Allow normal non-trading gaps, but reject quotes older than the market session."""
    last = parse_market_date(last_date)
    reference = parse_market_date(reference_date) if reference_date is not None else signal_market_date()
    if last is None or reference is None:
        return False
    allowed_gap = max_signal_staleness_days() if max_days is None else max_days
    return last >= reference - timedelta(days=allowed_gap)


def is_market_date_stale(market_date, as_of=None, max_days: int | None = None) -> bool:
    """Check pipeline freshness without treating weekends or short holidays as stale."""
    date = parse_market_date(market_date)
    reference = parse_market_date(as_of) if as_of is not None else vn_now().replace(tzinfo=None)
    if date is None or reference is None:
        return True
    allowed_gap = max_signal_staleness_days() if max_days is None else max_days
    return date < reference - timedelta(days=allowed_gap)

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
