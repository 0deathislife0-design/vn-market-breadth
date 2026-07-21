"""
Backtest ensemble signal weights — vectorized over full OHLC history.
For each symbol, slides day by day, records when each signal fires,
then checks forward return (T+10). Win rate per signal → normalized weight.
Output: data/backtest_weights.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache, compute_rsi_wilder_series
from ensemble_signals import compute_breakout_signal_series, compute_momentum_signal_series
from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, vn_now

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

WEIGHTS_JSON = DATA_DIR / "backtest_weights.json"
DOCS_WEIGHTS_JSON = DOCS_DATA_DIR / "backtest_weights.json"

MIN_SYMBOL_HISTORY = 220  # need 200 for MA200 + 10 fwd + 10 buffer
LOOKFORWARD = 10
SUCCESS_THRESHOLD = 0.02  # 2% return in 10 sessions = win
MIN_AVG_VOLUME = 300_000
MIN_OBSERVATIONS = 50


def get_filtered_symbols() -> list[str]:
    symbols = []
    for path in sorted(CACHE_DIR.glob("*.csv")):
        sym = path.stem
        if sym == ".gitkeep":
            continue
        if sym.startswith("FU") or sym.startswith("E1"):
            continue
        df = _load_cache(sym, CACHE_DIR)
        if len(df) < MIN_SYMBOL_HISTORY:
            continue
        symbols.append(sym)
    return symbols


def backtest_symbol(symbol: str) -> dict | None:
    """Vectorized backtest: compute all signals + forward returns for a symbol."""
    df = _load_cache(symbol, CACHE_DIR)
    if len(df) < MIN_SYMBOL_HISTORY:
        return None

    close: pd.Series = df["Close"]
    volume: pd.Series = df["Volume"]
    n = len(close)

    # --- Indicators (vectorized) ---
    ma10 = close.rolling(10).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    rsi = compute_rsi_wilder_series(close, 14)

    # Breakout
    vol_avg20 = volume.rolling(20).mean()

    # Forward return
    fwd_price = close.shift(-LOOKFORWARD)
    fwd_return = fwd_price / close - 1
    is_win = fwd_return >= SUCCESS_THRESHOLD

    # --- Signal booleans ---
    sig_ma = (ma10 > ma50) & (close > ma10) & (rsi > 50)
    near_ma50 = (close / ma50 >= 0.93) & (close / ma50 <= 1.00) & (ma50 > 0)
    sig_pb = (close > ma200) & near_ma50 & (rsi > 45)
    sig_bo = compute_breakout_signal_series(close, volume)
    sig_mo, _ = compute_momentum_signal_series(close, volume)
    liquidity_ok = vol_avg20 >= MIN_AVG_VOLUME

    # Valid range: we need enough leading history AND forward data available
    valid_start = MIN_SYMBOL_HISTORY - 1
    valid_end = n - LOOKFORWARD

    results = {}
    for name, sig in [
        ("ma_crossover", sig_ma),
        ("pullback", sig_pb),
        ("breakout", sig_bo),
        ("momentum", sig_mo),
    ]:
        triggered = sig.iloc[valid_start:valid_end] & liquidity_ok.iloc[valid_start:valid_end]
        wins = (triggered & is_win.iloc[valid_start:valid_end]).sum()
        total = triggered.sum()
        results[name] = {"wins": int(wins), "total": int(total)}

    return results


def calculate_weights(stats: dict) -> dict:
    """Convert per-signal stats into normalized weights."""
    raw = {}
    for signal, data in stats.items():
        total = data["total"]
        if total >= MIN_OBSERVATIONS:
            win_rate = data["wins"] / total
            raw[signal] = win_rate
        else:
            raw[signal] = 0.0

    total_raw = sum(raw.values())
    if total_raw > 0:
        weights = {k: round(v / total_raw, 4) for k, v in raw.items()}
    else:
        equal = 1.0 / len(stats)
        weights = {k: equal for k in stats}

    return weights


def main():
    print("=" * 60)
    print("Backtest: Ensemble Signal Weights")
    print("=" * 60)

    symbols = get_filtered_symbols()
    print(f"\nTesting {len(symbols)} symbols (history >= {MIN_SYMBOL_HISTORY}; liquidity evaluated per date)...\n")

    # Aggregate per-signal stats across all symbols
    agg_stats = {
        "ma_crossover": {"wins": 0, "total": 0},
        "pullback": {"wins": 0, "total": 0},
        "breakout": {"wins": 0, "total": 0},
        "momentum": {"wins": 0, "total": 0},
    }
    symbols_tested = 0
    for sym in symbols:
        result = backtest_symbol(sym)
        if result is None:
            continue
        symbols_tested += 1
        for sig, data in result.items():
            agg_stats[sig]["wins"] += data["wins"]
            agg_stats[sig]["total"] += data["total"]

    # Win rates
    total_obs = sum(d["total"] for d in agg_stats.values())
    print(f"Symbols tested: {symbols_tested}")
    print(f"Total observations: {total_obs}")
    print()
    for sig, data in agg_stats.items():
        wr = data["wins"] / data["total"] if data["total"] else 0
        print(f"  {sig:15s}: {data['wins']:5d} / {data['total']:5d}  ({wr:.1%})")

    weights = calculate_weights(agg_stats)

    # Per-signal win rate for output
    stats_out = {}
    for sig, data in agg_stats.items():
        wr = round(data["wins"] / data["total"], 4) if data["total"] else 0.0
        stats_out[sig] = {
            "wins": data["wins"],
            "total": data["total"],
            "win_rate": wr,
        }

    now = vn_now()
    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "num_symbols_tested": symbols_tested,
        "total_observations": total_obs,
        "lookforward_days": LOOKFORWARD,
        "success_threshold_pct": SUCCESS_THRESHOLD * 100,
        "weights": weights,
        "stats": stats_out,
    }

    WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_WEIGHTS_JSON.write_bytes(WEIGHTS_JSON.read_bytes())

    print(f"\nWeights: {weights}")
    print(f"\nSaved: {WEIGHTS_JSON}")
    print(f"Synced: {DOCS_WEIGHTS_JSON}")


if __name__ == "__main__":
    main()
