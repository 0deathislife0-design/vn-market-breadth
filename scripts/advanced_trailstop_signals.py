"""Advanced Trailstop signals based on Diep AFL logic.

This module is intentionally separate from Luc Mach, Khung4/Tplus and MAMA.
It calculates the dynamic `bs` line from ATR and 9-bar High/Low conditions,
then emits Buy/Sell when Close crosses that line.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from _shared import (
    tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR, format_market_date,
    is_market_data_fresh, json_default as _json_default, list_symbols,
    signal_market_date, vn_now,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "advanced_trailstop_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "advanced_trailstop_signals.json"

MULT = float(os.environ.get("ATS_MULT", "2.0"))
APER = int(os.environ.get("ATS_ATR_PERIOD", "7"))
MIN_VOLUME = int(os.environ.get("ATS_MIN_VOLUME", "20000"))
MIN_HISTORY = int(os.environ.get("ATS_MIN_HISTORY", "30"))


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


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def _value_when(condition: pd.Series, value: pd.Series) -> pd.Series:
    out = []
    last_value = np.nan
    for cond, val in zip(condition.fillna(False), value):
        if cond:
            last_value = val
        out.append(last_value)
    return pd.Series(out, index=value.index, dtype=float)


def _flip(set_signal: pd.Series, reset_signal: pd.Series) -> pd.Series:
    out = []
    current = False
    for set_, reset in zip(set_signal.fillna(False), reset_signal.fillna(False)):
        if set_:
            current = True
        if reset:
            current = False
        out.append(current)
    return pd.Series(out, index=set_signal.index, dtype=bool)


def atr_wilder(df: pd.DataFrame, period: int = APER) -> pd.Series:
    high = df["High"].reset_index(drop=True).astype(float)
    low = df["Low"].reset_index(drop=True).astype(float)
    close = df["Close"].reset_index(drop=True).astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def has_ohlcv(df: pd.DataFrame) -> bool:
    required = ("Open", "High", "Low", "Close", "Volume")
    if len(df) < MIN_HISTORY or not all(col in df.columns for col in required):
        return False
    return not df[list(required)].tail(MIN_HISTORY).isna().any().any()


def compute_advanced_trailstop(df: pd.DataFrame, mult: float = MULT, aper: int = APER) -> dict:
    high = df["High"].reset_index(drop=True).astype(float)
    low = df["Low"].reset_index(drop=True).astype(float)
    close = df["Close"].reset_index(drop=True).astype(float)
    n = len(df)

    atr = atr_wilder(df, aper)
    atrvalue = mult * atr
    bs = pd.Series(np.nan, index=range(n), dtype=float)
    up_condition = pd.Series(False, index=range(n), dtype=bool)
    down_condition = pd.Series(False, index=range(n), dtype=bool)

    if n:
        bs.iloc[0] = 0.0

    for i in range(1, n):
        if i < 9:
            bs.iloc[i] = bs.iloc[i - 1]
            continue

        up = low.iloc[i] > low.iloc[i - 9:i].max() and close.iloc[i] > bs.iloc[i - 1]
        down = high.iloc[i] < high.iloc[i - 9:i].min() and close.iloc[i] < bs.iloc[i - 1]
        up_condition.iloc[i] = bool(up)
        down_condition.iloc[i] = bool(down)

        if up:
            bs.iloc[i] = low.iloc[i] - atrvalue.iloc[i]
        elif down:
            bs.iloc[i] = high.iloc[i] + atrvalue.iloc[i]
        else:
            bs.iloc[i] = bs.iloc[i - 1]

    buy = _cross_up(close, bs)
    sell = _cross_up(bs, close)
    buy_price = _value_when(buy, close)
    sell_price = _value_when(sell, close)
    long = _flip(buy, sell)
    shrt = _flip(sell, buy)

    return {
        "bs_series": bs,
        "atr_series": atr,
        "atrvalue_series": atrvalue,
        "up_condition_series": up_condition,
        "down_condition_series": down_condition,
        "buy_series": buy,
        "sell_series": sell,
        "cover_series": buy,
        "short_series": sell,
        "buy_price_series": buy_price,
        "sell_price_series": sell_price,
        "long_series": long,
        "shrt_series": shrt,
        "filter_series": buy | sell,
        "buy": _last_bool(buy),
        "sell": _last_bool(sell),
        "cover": _last_bool(buy),
        "short": _last_bool(sell),
        "long": _last_bool(long),
        "shrt": _last_bool(shrt),
        "up_condition": _last_bool(up_condition),
        "down_condition": _last_bool(down_condition),
        "bs": _num(bs.iloc[-1]) if n else None,
        "atr": _num(atr.iloc[-1]) if n else None,
        "atrvalue": _num(atrvalue.iloc[-1]) if n else None,
        "buy_price": float(close.iloc[-1]) if _last_bool(buy) else None,
        "sell_price": float(close.iloc[-1]) if _last_bool(sell) else None,
        "last_buy_price": _num(buy_price.iloc[-1]) if n else None,
        "last_sell_price": _num(sell_price.iloc[-1]) if n else None,
    }


def audit_symbol(symbol: str, reference_date=None) -> tuple[dict | None, dict]:
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

    last = df.iloc[-1]
    last_volume = df["Volume"].iloc[-1]
    audit.update({
        "last_date": _date(last.get("TradingDate")),
        "last_close": _num(last.get("Close")),
        "last_volume": _num(last_volume),
    })
    if not is_market_data_fresh(last.get("TradingDate"), reference_date):
        audit["reason"] = "stale_quote"
        audit["reference_date"] = format_market_date(reference_date)
        return None, audit
    if pd.isna(last_volume) or float(last_volume) <= MIN_VOLUME:
        audit["reason"] = "volume_filter"
        return None, audit

    signal = compute_advanced_trailstop(df)
    audit.update({
        "bs": _num(signal["bs"]),
        "atr": _num(signal["atr"]),
        "atrvalue": _num(signal["atrvalue"]),
        "up_condition": bool(signal["up_condition"]),
        "down_condition": bool(signal["down_condition"]),
        "buy": bool(signal["buy"]),
        "sell": bool(signal["sell"]),
        "long": bool(signal["long"]),
        "shrt": bool(signal["shrt"]),
        "last_buy_price": _num(signal["last_buy_price"]),
        "last_sell_price": _num(signal["last_sell_price"]),
    })

    if not signal["buy"] and not signal["sell"]:
        audit["reason"] = "no_cross_signal"
        return None, audit

    is_buy = bool(signal["buy"])
    audit["reason"] = "buy_signal" if is_buy else "sell_signal"
    return {
        "symbol": symbol,
        "status": "BUY" if is_buy else "SELL",
        "signal_type": "advanced_trailstop_buy" if is_buy else "advanced_trailstop_sell",
        "score": 100,
        "ats_buy": is_buy,
        "ats_sell": not is_buy,
        "ats_cover": is_buy,
        "ats_short": not is_buy,
        "ats_long": bool(signal["long"]),
        "ats_shrt": bool(signal["shrt"]),
        "bs": round(float(signal["bs"]), 2) if signal["bs"] is not None else None,
        "atr": round(float(signal["atr"]), 2) if signal["atr"] is not None else None,
        "atrvalue": round(float(signal["atrvalue"]), 2) if signal["atrvalue"] is not None else None,
        "buy_price": signal["buy_price"],
        "sell_price": signal["sell_price"],
        "last_buy_price": _num(signal["last_buy_price"]),
        "last_sell_price": _num(signal["last_sell_price"]),
        "last_price": float(df["Close"].iloc[-1]),
        "last_volume": float(last_volume),
        "strategies": ["advanced_trailstop_buy" if is_buy else "advanced_trailstop_sell"],
    }, audit


def analyze_symbol(symbol: str) -> dict | None:
    result, _audit = audit_symbol(symbol)
    return result


def get_filtered_symbols() -> list[str]:
    symbols = list_symbols(CACHE_DIR, min_history=MIN_HISTORY)
    return [s for s in symbols if len(s) <= 3 and not any(c.isdigit() for c in s)]


def main():
    tqdm.write("=" * 60)
    tqdm.write("Advanced Trailstop Signals - Diep original")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    audit = []
    skipped_ohlc = 0
    skipped_stale = 0
    reference_date = signal_market_date()
    bar = tqdm(symbols, desc="[ATS] Trailstop", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result, item_audit = audit_symbol(sym, reference_date)
        audit.append(item_audit)
        if result:
            signals.append(result)
        elif item_audit["reason"] == "missing_ohlcv_or_history":
            skipped_ohlc += 1
        elif item_audit["reason"] == "stale_quote":
            skipped_stale += 1

    signals.sort(key=lambda x: (x["status"] == "BUY", x["last_volume"]), reverse=True)
    buys = [s for s in signals if s["ats_buy"]]
    sells = [s for s in signals if s["ats_sell"]]
    now = vn_now()
    market_date = format_market_date(reference_date) or now.strftime("%d/%m/%Y")
    output = {
        "mode": "diep_advanced_trailstop",
        "generated_at": now.isoformat(),
        "date": market_date,
        "market_date": market_date,
        "freshness_reference_date": market_date,
        "mult": MULT,
        "atr_period": APER,
        "min_volume": MIN_VOLUME,
        "min_history": MIN_HISTORY,
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "skipped_missing_ohlc": skipped_ohlc,
        "skipped_stale_quotes": skipped_stale,
        "buy": buys,
        "sell": sells,
        "all_signals": signals,
        "audit": audit,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON.name}")
    tqdm.write(f"Tin hieu ATS: {len(signals)} (Buy: {len(buys)}, Sell: {len(sells)})")
    if skipped_ohlc:
        tqdm.write(f"Bo qua {skipped_ohlc} ma do cache chua co OHLCV day du.")
    if skipped_stale:
        tqdm.write(f"Bo qua {skipped_stale} ma do quote cu hon nguong freshness.")


if __name__ == "__main__":
    main()
