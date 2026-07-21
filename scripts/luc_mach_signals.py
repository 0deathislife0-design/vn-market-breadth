"""
Luc Mach Signals - Diep original core.

This module intentionally keeps the output close to the original AmiBroker logic:
  - 5 VUDD checks on periods 13/20/35/55/65
  - 1 Tplus check
  - Buy/Sell by score threshold (roboNo, default 3)
  - Filter by today's volume and at least 2 aligned buy/sell checks for watch

Trend, liquidity, pullback, breakout, Darvas and technical sell warnings are not
used to decide Luc Mach output in this file.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from khung4_tplus_signals import compute_khung4_tplus
from _shared import tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR, format_market_date, json_default as _json_default, list_symbols, signal_market_date, vn_now

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "luc_mach_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "luc_mach_signals.json"

VUDD_PERIODS = (13, 20, 35, 55, 65)
LUC_MACH_THRESHOLD = int(os.environ.get("LUC_MACH_THRESHOLD", "3"))
MIN_VOLUME = int(os.environ.get("LUC_MACH_MIN_VOLUME", "20000"))
MIN_HISTORY = int(os.environ.get("LUC_MACH_MIN_HISTORY", "300"))


def _last_bool(series: pd.Series) -> bool:
    return bool(series.iloc[-1]) if len(series) else False


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def _tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def _zero_lag_tema(series: pd.Series, period: int) -> pd.Series:
    tma1 = _tema(series, period)
    tma2 = _tema(tma1, period)
    return tma1 + (tma1 - tma2)


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def has_ohlcv(df: pd.DataFrame) -> bool:
    required = ("Open", "High", "Low", "Close", "Volume")
    if len(df) < MIN_HISTORY or not all(col in df.columns for col in required):
        return False
    return not df[list(required)].tail(MIN_HISTORY).isna().any().any()


def compute_vudd(df: pd.DataFrame, period: int) -> dict:
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    ha_close_raw = (open_ + high + low + close) / 4
    ha_open = ha_close_raw.shift(1).ewm(alpha=0.5, adjust=False).mean()
    ha_high = pd.concat([high, ha_close_raw, ha_open], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_close_raw, ha_open], axis=1).min(axis=1)
    ha_close = (ha_close_raw + ha_open + ha_high + ha_low) / 4
    avg_price = (high + low + close) / 3

    zl_ha = _zero_lag_tema(ha_close, period)
    zl_typ = _zero_lag_tema(avg_price, period)
    buy = _cross_up(zl_typ, zl_ha).fillna(False)
    sell = _cross_up(zl_ha, zl_typ).fillna(False)

    return {
        "buy_series": buy,
        "sell_series": sell,
        "buy": _last_bool(buy),
        "sell": _last_bool(sell),
        "zl_ha": None if pd.isna(zl_ha.iloc[-1]) else round(float(zl_ha.iloc[-1]), 2),
        "zl_typ": None if pd.isna(zl_typ.iloc[-1]) else round(float(zl_typ.iloc[-1]), 2),
    }


def compute_tplus(df: pd.DataFrame) -> dict:
    return compute_khung4_tplus(df)


def analyze_symbol(symbol: str) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate").reset_index(drop=True)
    if not has_ohlcv(df):
        return None

    volume = df["Volume"]
    last_volume = volume.iloc[-1]
    if pd.isna(last_volume) or float(last_volume) <= MIN_VOLUME:
        return None

    vudds = {p: compute_vudd(df, p) for p in VUDD_PERIODS}
    tplus = compute_tplus(df)

    buy_score = sum(1 for p in VUDD_PERIODS if vudds[p]["buy"]) + int(tplus["buy"])
    sell_score = sum(1 for p in VUDD_PERIODS if vudds[p]["sell"]) + int(tplus["sell"])
    active_score = max(buy_score, sell_score)
    if active_score < 2:
        return None

    luc_mach_buy = buy_score >= LUC_MACH_THRESHOLD
    luc_mach_sell = sell_score >= LUC_MACH_THRESHOLD

    if luc_mach_buy and luc_mach_sell:
        status = "CONFLICT"
        signal_type = "conflict"
    elif luc_mach_buy:
        status = "VALID_BUY"
        signal_type = "valid"
    elif luc_mach_sell:
        status = "SELL_WARNING"
        signal_type = "sell_warning"
    else:
        status = "WATCHLIST"
        signal_type = "watch"

    score = int(round(active_score / 6 * 100))
    strategies = []
    if luc_mach_buy:
        strategies.append("luc_mach_buy")
    if luc_mach_sell:
        strategies.append("luc_mach_sell")
    if tplus["buy"]:
        strategies.append("tplus_buy")
    if tplus["sell"]:
        strategies.append("tplus_sell")
    for p in VUDD_PERIODS:
        if vudds[p]["buy"]:
            strategies.append(f"vudd{p}_buy")
        if vudds[p]["sell"]:
            strategies.append(f"vudd{p}_sell")

    close = df["Close"]
    return {
        "symbol": symbol,
        "status": status,
        "signal_type": signal_type,
        "score": score,
        "buy_score": int(buy_score),
        "sell_score": int(sell_score),
        "luc_mach_buy": bool(luc_mach_buy),
        "luc_mach_sell": bool(luc_mach_sell),
        "diep_filter": True,
        "min_volume": MIN_VOLUME,
        "threshold": LUC_MACH_THRESHOLD,
        "tplus_buy": bool(tplus["buy"]),
        "tplus_sell": bool(tplus["sell"]),
        "tplus_state": int(tplus["state"]),
        "vudd_buy_periods": [p for p in VUDD_PERIODS if vudds[p]["buy"]],
        "vudd_sell_periods": [p for p in VUDD_PERIODS if vudds[p]["sell"]],
        "strategies": strategies,
        "last_price": float(close.iloc[-1]),
        "last_volume": float(last_volume),
    }


def get_filtered_symbols() -> list[str]:
    symbols = list_symbols(CACHE_DIR, min_history=20)
    return [s for s in symbols if len(s) <= 3 and not any(c.isdigit() for c in s)]


def main():
    tqdm.write("=" * 60)
    tqdm.write("Luc Mach Signals - Diep original core")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    skipped_ohlc = 0
    bar = tqdm(symbols, desc="[LM] Luc Mach", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result = analyze_symbol(sym)
        if result:
            signals.append(result)
        else:
            df = _load_cache(sym, CACHE_DIR)
            if len(df) < MIN_HISTORY or not has_ohlcv(df):
                skipped_ohlc += 1

    rank = {"VALID_BUY": 3, "CONFLICT": 2, "WATCHLIST": 1, "SELL_WARNING": 0}
    signals.sort(key=lambda x: (rank.get(x["status"], 0), x["score"]), reverse=True)

    valid = [s for s in signals if s["status"] == "VALID_BUY"]
    conflicts = [s for s in signals if s["status"] == "CONFLICT"]
    watch = [s for s in signals if s["status"] == "WATCHLIST"]
    sell = [s for s in signals if s["status"] == "SELL_WARNING"]
    now = vn_now()
    market_date = format_market_date(signal_market_date()) or now.strftime("%d/%m/%Y")

    output = {
        "mode": "diep_original",
        "generated_at": now.isoformat(),
        "date": market_date,
        "threshold": LUC_MACH_THRESHOLD,
        "min_volume": MIN_VOLUME,
        "min_history": MIN_HISTORY,
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "skipped_missing_ohlc": skipped_ohlc,
        "strong_buy": 0,
        "valid_buy": len(valid),
        "watch_count": len(watch),
        "sell_warning_count": len(sell),
        "conflict_count": len(conflicts),
        "strong": [],
        "valid": valid,
        "watch": watch,
        "sell_warning": sell,
        "conflicts": conflicts,
        "all_signals": signals,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON.name}")
    tqdm.write(f"Tin hieu: {output['total_signals']} (Buy: {len(valid)}, Watch: {len(watch)}, Sell: {len(sell)}, Conflict: {len(conflicts)})")
    if skipped_ohlc:
        tqdm.write(f"Bo qua {skipped_ohlc} ma do cache chua co OHLCV day du.")


if __name__ == "__main__":
    main()
