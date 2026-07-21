"""
Backtest Momentum Signals — Bộ 4 chiến lược tìm điểm mua.
Vectorized over full OHLC history, checks forward return (T+5, T+10, T+20).
Output: data/backtest_momentum.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache, compute_rsi_wilder_series
from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, MIN_SYMBOL_HISTORY, vn_now
from _shared import (SCORE_MA, SCORE_BREAKOUT, SCORE_ROC, SCORE_HYBRID,
                     BONUS_VOL_SURGE, BONUS_ADX_STRONG, BONUS_RSI_GOLD)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_JSON = DATA_DIR / "backtest_momentum.json"
DOCS_OUTPUT_JSON = DOCS_DATA_DIR / "backtest_momentum.json"

LOOKFORWARD_OPTIONS = [5, 10, 20]
SUCCESS_THRESHOLD = 0.02
MIN_AVG_VOLUME = 500_000
MIN_OBSERVATIONS = 20


# --- Vectorized ADX (close-based) ---

def _wilderm_ema(series: pd.Series, period: int = 14) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx_series(close: pd.Series, period: int = 14) -> pd.Series:
    tr = close.diff().abs()
    delta = close.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)

    tr_smooth = _wilderm_ema(tr, period)
    plus_smooth = _wilderm_ema(plus_dm, period)
    minus_smooth = _wilderm_ema(minus_dm, period)

    plus_di = (100.0 * plus_smooth / tr_smooth).where(tr_smooth > 0, 0.0)
    minus_di = (100.0 * minus_smooth / tr_smooth).where(tr_smooth > 0, 0.0)

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).where((plus_di + minus_di) > 0, 0.0)
    return _wilderm_ema(dx, period)


def compute_ma_crossover_signal(close: pd.Series, volume: pd.Series, rsi: pd.Series, adx: pd.Series) -> pd.Series:
    ma10 = close.rolling(10).mean()
    ma50 = close.rolling(50).mean()
    vol_avg20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg20

    return (
        (ma10 > ma50)
        & (close > ma10)
        & (rsi > 50)
        & (adx >= 22)
        & (vol_ratio > 1.3)
    ).astype(int)


def compute_breakout_signal(close: pd.Series, volume: pd.Series, rsi: pd.Series) -> pd.Series:
    high_20 = close.rolling(20).max().shift(1)
    vol_avg20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg20
    ma10 = close.rolling(10).mean()

    return (
        (close > high_20)
        & (vol_ratio > 1.3)
        & (rsi > 45)
        & (close > ma10)
    ).astype(int)


def compute_roc_momentum_signal(close: pd.Series, volume: pd.Series, rsi: pd.Series) -> pd.Series:
    roc10 = close.pct_change(10) * 100
    roc20 = close.pct_change(20) * 100
    ma10 = close.rolling(10).mean()
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()

    return (
        (roc10 > roc20)
        & (roc10 > 5.0)
        & (vol_ma5 > vol_ma20)
        & (rsi > 50)
        & (close > ma10)
    ).astype(int)


def compute_hybrid_signal(close: pd.Series, volume: pd.Series, rsi: pd.Series, adx: pd.Series) -> pd.Series:
    ma10 = close.rolling(10).mean()
    ma50 = close.rolling(50).mean()
    high_20 = close.rolling(20).max().shift(1)
    nh20 = close > high_20
    roc10 = close.pct_change(10) * 100
    roc20 = close.pct_change(20) * 100
    vol_avg20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg20

    return (
        (ma10 > ma50)
        & (close > ma10)
        & (nh20 | (roc10 > roc20))
        & (vol_ratio > 1.5)
        & (rsi >= 48) & (rsi <= 73)
        & (adx >= 23)
    ).astype(int)


def compute_bonuses_vectorized(close: pd.Series, volume: pd.Series, rsi: pd.Series, adx: pd.Series) -> pd.DataFrame:
    vol_avg20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg20

    vol_surge = (vol_ratio > 2.0).astype(int)
    adx_strong = (adx > 28).astype(int)
    rsi_gold = ((rsi >= 50) & (rsi <= 68)).astype(int)

    bonus_total = (vol_surge * BONUS_VOL_SURGE
                   + adx_strong * BONUS_ADX_STRONG
                   + rsi_gold * BONUS_RSI_GOLD)
    return pd.DataFrame({
        "vol_surge": vol_surge, "adx_strong": adx_strong, "rsi_gold": rsi_gold,
        "bonus_total": bonus_total,
    })


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
        vol_avg = df["Volume"].dropna().iloc[-20:].mean()
        if pd.isna(vol_avg) or vol_avg < MIN_AVG_VOLUME:
            continue
        symbols.append(sym)
    return symbols


def backtest_symbol(symbol: str) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR)
    if len(df) < MIN_SYMBOL_HISTORY:
        return None

    close = df["Close"]
    volume = df["Volume"].fillna(0)
    n = len(close)

    # Common filters
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    trend_ok = (close > ma50) & (ma50 > ma200)

    rsi = compute_rsi_wilder_series(close, 14)

    adx = compute_adx_series(close, 14)
    common_ok = trend_ok & (adx >= 20) & (rsi >= 48) & (rsi <= 72)

    vol_avg20 = volume.rolling(20).mean()
    vol_ok = vol_avg20 >= MIN_AVG_VOLUME

    # Individual signals
    sig_ma = compute_ma_crossover_signal(close, volume, rsi, adx)
    sig_bo = compute_breakout_signal(close, volume, rsi)
    sig_roc = compute_roc_momentum_signal(close, volume, rsi)
    sig_hy = compute_hybrid_signal(close, volume, rsi, adx)

    # Bonuses
    bonuses = compute_bonuses_vectorized(close, volume, rsi, adx)

    # Score
    strategy_ok = common_ok & vol_ok
    score = (sig_ma * SCORE_MA + sig_bo * SCORE_BREAKOUT
             + sig_roc * SCORE_ROC + sig_hy * SCORE_HYBRID
             + bonuses["bonus_total"])

    # Categorize
    is_strong = (strategy_ok & (score >= 60)).astype(int)
    is_watch = (strategy_ok & (score >= 30) & (score < 60)).astype(int)
    has_signal = (strategy_ok & (score >= 30)).astype(int)

    # Forward returns
    results = {}
    for lf in LOOKFORWARD_OPTIONS:
        valid_start = MIN_SYMBOL_HISTORY
        valid_end = n - lf

        fwd_price = close.shift(-lf)
        fwd_return = fwd_price / close - 1
        is_win = fwd_return >= SUCCESS_THRESHOLD

        # Stats by bracket
        for bracket_name, bracket_mask in [
            ("strong", is_strong),
            ("watch", is_watch),
            ("any_signal", has_signal),
        ]:
            valid = bracket_mask.iloc[valid_start:valid_end]
            wins = (valid & is_win.iloc[valid_start:valid_end]).sum()
            total = valid.sum()
            results[f"{bracket_name}_T+{lf}"] = {"wins": int(wins), "total": int(total)}

        # Stats by individual strategy
        for sig_name, sig_series in [
            ("ma_crossover", sig_ma),
            ("breakout", sig_bo),
            ("roc_momentum", sig_roc),
            ("hybrid", sig_hy),
        ]:
            sig_valid = sig_series.iloc[valid_start:valid_end] & common_ok.iloc[valid_start:valid_end] & vol_ok.iloc[valid_start:valid_end]
            sig_wins = (sig_valid & is_win.iloc[valid_start:valid_end]).sum()
            sig_total = sig_valid.sum()
            results[f"{sig_name}_T+{lf}"] = {"wins": int(sig_wins), "total": int(sig_total)}

    # Score distribution stats
    valid_scores = score.iloc[MIN_SYMBOL_HISTORY:n - max(LOOKFORWARD_OPTIONS)][strategy_ok.iloc[MIN_SYMBOL_HISTORY:n - max(LOOKFORWARD_OPTIONS)]]
    results["score_distribution"] = {
        "mean": round(float(valid_scores.mean()), 2) if not valid_scores.empty else None,
        "median": round(float(valid_scores.median()), 2) if not valid_scores.empty else None,
        "pct_strong": round(float((valid_scores >= 60).mean() * 100), 2) if not valid_scores.empty else None,
        "pct_watch": round(float(((valid_scores >= 30) & (valid_scores < 60)).mean() * 100), 2) if not valid_scores.empty else None,
    }

    return results


def aggregate_results(all_results: list[dict]) -> dict:
    combined = {}

    # Collect all keys
    keys = set()
    for r in all_results:
        keys.update(r.keys())

    for key in keys:
        if key == "score_distribution":
            dists = [r["score_distribution"] for r in all_results if "score_distribution" in r]
            if not dists:
                continue
            combined["score_distribution"] = {
                "mean_mean": round(np.mean([d["mean"] for d in dists if d["mean"] is not None]), 2) if any(d["mean"] is not None for d in dists) else None,
                "mean_median": round(np.mean([d["median"] for d in dists if d["median"] is not None]), 2) if any(d["median"] is not None for d in dists) else None,
                "mean_pct_strong": round(np.mean([d["pct_strong"] for d in dists if d["pct_strong"] is not None]), 2) if any(d["pct_strong"] is not None for d in dists) else None,
                "mean_pct_watch": round(np.mean([d["pct_watch"] for d in dists if d["pct_watch"] is not None]), 2) if any(d["pct_watch"] is not None for d in dists) else None,
            }
            continue

        wins = sum(r.get(key, {}).get("wins", 0) for r in all_results if key in r)
        total = sum(r.get(key, {}).get("total", 0) for r in all_results if key in r)
        combined[key] = {
            "wins": wins,
            "total": total,
            "win_rate": round(wins / total, 4) if total > 0 else None,
        }

    return combined


def main():
    print("=" * 60)
    print("Backtest: Momentum Signals - Bo 4 chien luoc")
    print("=" * 60)

    symbols = get_filtered_symbols()
    print(f"\nTesting {len(symbols)} symbols (history >= {MIN_SYMBOL_HISTORY}, vol >= {MIN_AVG_VOLUME:,})...\n")

    all_results = []
    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(symbols)} symbols processed...")
        result = backtest_symbol(sym)
        if result:
            all_results.append(result)

    stats = aggregate_results(all_results)

    print(f"\nSymbols processed: {len(all_results)}")
    print()

    # Print summary
    for lf in LOOKFORWARD_OPTIONS:
        print(f"--- T+{lf} (threshold {SUCCESS_THRESHOLD*100:.0f}%) ---")
        for bracket in ["strong", "watch", "any_signal"]:
            key = f"{bracket}_T+{lf}"
            if key in stats:
                d = stats[key]
                wr = d["win_rate"]
                wr_str = f"{wr:.1%}" if wr is not None else "N/A"
                print(f"  {bracket:12s}: {d['wins']:6d} / {d['total']:6d} ({wr_str})")
        print()

        for sig in ["ma_crossover", "breakout", "roc_momentum", "hybrid"]:
            key = f"{sig}_T+{lf}"
            if key in stats:
                d = stats[key]
                wr_str = f"{d['win_rate']:.1%}" if d['win_rate'] is not None else "N/A"
                print(f"  {sig:15s}: {d['wins']:6d} / {d['total']:6d} ({wr_str})")
        print()

    if "score_distribution" in stats:
        sd = stats["score_distribution"]
        print(f"Score distribution (per symbol avg):")
        print(f"  Mean score: {sd['mean_mean']}")
        print(f"  Median score: {sd['mean_median']}")
        print(f"  % Strong (>=60): {sd['mean_pct_strong']}%")
        print(f"  % Watch (35-59): {sd['mean_pct_watch']}%")

    now = vn_now()
    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "num_symbols_tested": len(all_results),
        "lookforward_days": LOOKFORWARD_OPTIONS,
        "success_threshold_pct": SUCCESS_THRESHOLD * 100,
        "min_observations": MIN_OBSERVATIONS,
        "stats": {k: v for k, v in stats.items() if k != "score_distribution"},
        "score_distribution": stats.get("score_distribution"),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_OUTPUT_JSON.write_bytes(OUTPUT_JSON.read_bytes())

    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Synced: {DOCS_OUTPUT_JSON}")


if __name__ == "__main__":
    main()
