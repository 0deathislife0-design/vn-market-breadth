"""Khung4/Tplus buy signals based on Diep AFL logic.

This is intentionally separate from Luc Mach. It only tracks the Khung4/Tplus
state line `d` and reports buy points where state flips from 0 to 1.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from _shared import tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR, list_symbols, json_default as _json_default

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "khung4_tplus_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "khung4_tplus_signals.json"

MIN_VOLUME = int(os.environ.get("KHUNG4_TPLUS_MIN_VOLUME", "20000"))
MIN_HISTORY = int(os.environ.get("KHUNG4_TPLUS_MIN_HISTORY", "20"))


def _last_bool(series: pd.Series) -> bool:
    return bool(series.iloc[-1]) if len(series) else False


def _num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _date(value):
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%d/%m/%Y")


def has_ohlcv(df: pd.DataFrame) -> bool:
    required = ("Open", "High", "Low", "Close", "Volume")
    if len(df) < MIN_HISTORY or not all(col in df.columns for col in required):
        return False
    return not df[list(required)].tail(MIN_HISTORY).isna().any().any()


def compute_khung4_tplus(df: pd.DataFrame) -> dict:
    high = df["High"].reset_index(drop=True)
    low = df["Low"].reset_index(drop=True)
    close = df["Close"].reset_index(drop=True)
    n = len(df)

    d = pd.Series(np.nan, index=range(n), dtype=float)
    for i in range(4, n):
        recent_high = high.iloc[i - 4:i].max()
        recent_low = low.iloc[i - 4:i].min()
        if close.iloc[i] > recent_high:
            d.iloc[i] = low.iloc[i - 3:i + 1].min()
        elif close.iloc[i] < recent_low:
            d.iloc[i] = high.iloc[i - 3:i + 1].max()
        else:
            d.iloc[i] = d.iloc[i - 1]

    cross_up = ((close > d) & (close.shift(1) <= d.shift(1))).fillna(False)
    cross_down = ((d > close) & (d.shift(1) <= close.shift(1))).fillna(False)

    state = pd.Series(0, index=range(n), dtype=int)
    for i in range(1, n):
        if cross_up.iloc[i]:
            state.iloc[i] = 1
        elif cross_down.iloc[i]:
            state.iloc[i] = 0
        else:
            state.iloc[i] = state.iloc[i - 1]

    buy = (state > state.shift(1)).fillna(False)
    sell = (state < state.shift(1)).fillna(False)

    return {
        "buy_series": buy,
        "sell_series": sell,
        "buy": _last_bool(buy),
        "sell": _last_bool(sell),
        "state": int(state.iloc[-1]) if n else 0,
        "d": None if not n or pd.isna(d.iloc[-1]) else round(float(d.iloc[-1]), 2),
        "prev_d": None if n < 2 or pd.isna(d.iloc[-2]) else round(float(d.iloc[-2]), 2),
        "prev_state": int(state.iloc[-2]) if n >= 2 else 0,
        "cross_up": _last_bool(cross_up),
        "cross_down": _last_bool(cross_down),
        "buy_price": float(close.iloc[-1]) if _last_bool(buy) else None,
        "sell_price": float(close.iloc[-1]) if _last_bool(sell) else None,
    }


def audit_symbol(symbol: str) -> tuple[dict | None, dict]:
    df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate").reset_index(drop=True)
    audit = {
        "symbol": symbol,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "has_ohlcv": bool(has_ohlcv(df)),
        "reason": None,
    }

    if not has_ohlcv(df):
        required = ("Open", "High", "Low", "Close", "Volume")
        audit["reason"] = "missing_ohlcv_or_history"
        audit["missing_columns"] = [c for c in required if c not in df.columns]
        return None, audit

    last_volume = df["Volume"].iloc[-1]
    last = df.iloc[-1]
    i = len(df) - 1
    recent_high = df["High"].iloc[i - 4:i].max() if i >= 4 else np.nan
    recent_low = df["Low"].iloc[i - 4:i].min() if i >= 4 else np.nan

    audit.update({
        "last_date": _date(last.get("TradingDate")),
        "last_open": _num(last.get("Open")),
        "last_high": _num(last.get("High")),
        "last_low": _num(last.get("Low")),
        "last_close": _num(last.get("Close")),
        "last_volume": _num(last_volume),
        "recent_high_4": _num(recent_high),
        "recent_low_4": _num(recent_low),
        "break_up": bool(last.get("Close") > recent_high) if not pd.isna(recent_high) else False,
        "break_down": bool(last.get("Close") < recent_low) if not pd.isna(recent_low) else False,
    })

    if pd.isna(last_volume) or float(last_volume) <= MIN_VOLUME:
        audit["reason"] = "volume_filter"
        return None, audit

    signal = compute_khung4_tplus(df)
    audit.update({
        "d": signal["d"],
        "prev_d": signal["prev_d"],
        "state": int(signal["state"]),
        "prev_state": int(signal["prev_state"]),
        "cross_up": bool(signal["cross_up"]),
        "cross_down": bool(signal["cross_down"]),
        "buy": bool(signal["buy"]),
        "sell": bool(signal["sell"]),
        "buy_price": signal["buy_price"],
        "sell_price": signal["sell_price"],
    })

    if not signal["buy"]:
        audit["reason"] = "no_buy_signal"
        return None, audit

    close = df["Close"]
    audit["reason"] = "buy_signal"
    return {
        "symbol": symbol,
        "status": "BUY",
        "signal_type": "khung4_tplus_buy",
        "score": 100,
        "khung4_tplus_buy": True,
        "khung4_tplus_sell": bool(signal["sell"]),
        "khung4_tplus_state": int(signal["state"]),
        "khung4_tplus_d": signal["d"],
        "buy_price": signal["buy_price"],
        "last_price": float(close.iloc[-1]),
        "last_volume": float(last_volume),
        "strategies": ["khung4_tplus_buy"],
    }, audit


def analyze_symbol(symbol: str) -> dict | None:
    result, _audit = audit_symbol(symbol)
    return result


def get_filtered_symbols() -> list[str]:
    symbols = list_symbols(CACHE_DIR, min_history=MIN_HISTORY)
    return [s for s in symbols if len(s) <= 3 and not any(c.isdigit() for c in s)]


def main():
    tqdm.write("=" * 60)
    tqdm.write("Khung4/Tplus Buy Signals")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    audit = []
    skipped_ohlc = 0
    bar = tqdm(symbols, desc="[K4] Khung4/Tplus", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result, item_audit = audit_symbol(sym)
        audit.append(item_audit)
        if result:
            signals.append(result)
        else:
            if item_audit["reason"] == "missing_ohlcv_or_history":
                skipped_ohlc += 1

    signals.sort(key=lambda x: (x["score"], x["last_volume"]), reverse=True)
    now = datetime.now(timezone.utc) + timedelta(hours=7)
    output = {
        "mode": "khung4_tplus_original",
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "min_volume": MIN_VOLUME,
        "min_history": MIN_HISTORY,
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "skipped_missing_ohlc": skipped_ohlc,
        "buy": signals,
        "all_signals": signals,
        "audit": audit,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON.name}")
    tqdm.write(f"Tin hieu mua Khung4/Tplus: {len(signals)}")
    if skipped_ohlc:
        tqdm.write(f"Bo qua {skipped_ohlc} ma do cache chua co OHLCV day du.")


if __name__ == "__main__":
    main()
