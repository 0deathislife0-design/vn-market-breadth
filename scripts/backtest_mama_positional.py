"""Backtest MAMA Positional Buy signals.

For every historical MAMA Positional Buy signal, measure return from signal
Close to Close after T+10 sessions. A win is forward return >= 7%.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from mama_positional_signals import compute_mama_positional_system
from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, list_symbols, vn_now

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_JSON = DATA_DIR / "backtest_mama_positional.json"
DOCS_OUTPUT_JSON = DOCS_DATA_DIR / "backtest_mama_positional.json"

LOOKFORWARD = 10
SUCCESS_THRESHOLD = 0.07
MIN_HISTORY = 90
LIQUID_AVG_VOLUME = 300_000


def _date(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%d/%m/%Y")


def _empty_stats() -> dict:
    return {
        "wins": 0,
        "total": 0,
        "win_rate": 0.0,
        "avg_return_pct": 0.0,
        "median_return_pct": 0.0,
        "best_return_pct": 0.0,
        "worst_return_pct": 0.0,
    }


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return _empty_stats()
    returns = np.array([r["return_pct"] for r in rows], dtype=float)
    wins = int((returns >= SUCCESS_THRESHOLD * 100).sum())
    total = len(rows)
    return {
        "wins": wins,
        "total": total,
        "win_rate": round(wins / total, 4),
        "avg_return_pct": round(float(np.mean(returns)), 2),
        "median_return_pct": round(float(np.median(returns)), 2),
        "best_return_pct": round(float(np.max(returns)), 2),
        "worst_return_pct": round(float(np.min(returns)), 2),
    }


def backtest_symbol(symbol: str) -> list[dict]:
    df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate").reset_index(drop=True)
    required = ("TradingDate", "Open", "High", "Low", "Close", "Volume")
    if len(df) < MIN_HISTORY + LOOKFORWARD or not all(col in df.columns for col in required):
        return []
    if df[list(required)].isna().any().any():
        return []

    signal = compute_mama_positional_system(df)
    buy = signal["buy_series"]
    close = df["Close"].astype(float).reset_index(drop=True)
    volume = df["Volume"].fillna(0).astype(float).reset_index(drop=True)
    vol_avg20 = volume.rolling(20).mean()
    rows = []

    for i in np.flatnonzero(buy.to_numpy()):
        if i + LOOKFORWARD >= len(df):
            continue
        entry = float(close.iloc[i])
        exit_ = float(close.iloc[i + LOOKFORWARD])
        if entry <= 0:
            continue
        ret_pct = (exit_ / entry - 1) * 100
        rows.append({
            "symbol": symbol,
            "date": _date(df["TradingDate"].iloc[i]),
            "exit_date": _date(df["TradingDate"].iloc[i + LOOKFORWARD]),
            "entry_price": entry,
            "exit_price_t10": exit_,
            "return_pct": round(ret_pct, 2),
            "win_7pct": bool(ret_pct >= SUCCESS_THRESHOLD * 100),
            "volume": float(volume.iloc[i]),
            "avg_volume20": None if pd.isna(vol_avg20.iloc[i]) else float(vol_avg20.iloc[i]),
            "liquid_300k": bool(not pd.isna(vol_avg20.iloc[i]) and vol_avg20.iloc[i] >= LIQUID_AVG_VOLUME),
            "mama": round(float(signal["mama_series"].iloc[i]), 2),
            "fama": round(float(signal["fama_series"].iloc[i]), 2),
            "period": round(float(signal["period_series"].iloc[i]), 2),
            "buy_setup_value": None if pd.isna(signal["buy_setup_value_series"].iloc[i]) else float(signal["buy_setup_value_series"].iloc[i]),
        })
    return rows


def main():
    print("=" * 60)
    print("Backtest: MAMA Positional Buy T+10 >= 7%")
    print("=" * 60)

    symbols = [s for s in list_symbols(CACHE_DIR, min_history=MIN_HISTORY) if len(s) <= 3 and not any(c.isdigit() for c in s)]
    print(f"\nTesting {len(symbols)} symbols (history >= {MIN_HISTORY})...\n")

    trades = []
    for sym in symbols:
        trades.extend(backtest_symbol(sym))

    liquid_trades = [r for r in trades if r["liquid_300k"]]
    stats_all = _summarize(trades)
    stats_liquid = _summarize(liquid_trades)

    by_year = {}
    for row in trades:
        year = row["date"][-4:] if row["date"] else "unknown"
        by_year.setdefault(year, []).append(row)
    yearly = {year: _summarize(rows) for year, rows in sorted(by_year.items())}

    top_winners = sorted(trades, key=lambda r: r["return_pct"], reverse=True)[:20]
    worst_trades = sorted(trades, key=lambda r: r["return_pct"])[:20]

    now = vn_now()
    output = {
        "mode": "mama_positional_buy_t10_7pct",
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "lookforward_days": LOOKFORWARD,
        "success_threshold_pct": SUCCESS_THRESHOLD * 100,
        "min_history": MIN_HISTORY,
        "liquid_avg_volume_threshold": LIQUID_AVG_VOLUME,
        "num_symbols_tested": len(symbols),
        "stats_all": stats_all,
        "stats_liquid_300k": stats_liquid,
        "yearly": yearly,
        "top_winners": top_winners,
        "worst_trades": worst_trades,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_OUTPUT_JSON.write_bytes(OUTPUT_JSON.read_bytes())

    print(f"Symbols tested: {len(symbols)}")
    print(f"All MAMA Buy trades: {stats_all['wins']} / {stats_all['total']} = {stats_all['win_rate']:.1%}")
    print(f"Liquid trades:        {stats_liquid['wins']} / {stats_liquid['total']} = {stats_liquid['win_rate']:.1%}")
    print(f"Avg return all:       {stats_all['avg_return_pct']:.2f}%")
    print(f"Median return all:    {stats_all['median_return_pct']:.2f}%")
    print(f"\nSaved: {OUTPUT_JSON.name}")


if __name__ == "__main__":
    main()
