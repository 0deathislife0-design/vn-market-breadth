"""Ehlers MAMA positional signals.

MAMA/FAMA are calculated with the standard John Ehlers adaptive period logic.
The entry/exit confirmation remains the Diep positional system: MAMA/FAMA cross
only creates a setup, while Close crossing the setup High/Low creates the signal.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from _shared import tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR, format_market_date, json_default as _json_default, list_symbols, signal_market_date, vn_now

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "mama_positional_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "mama_positional_signals.json"

MIN_VOLUME = int(os.environ.get("MAMA_MIN_VOLUME", "20000"))
MIN_HISTORY = int(os.environ.get("MAMA_MIN_HISTORY", "80"))
FAST_LIMIT = float(os.environ.get("MAMA_FAST_LIMIT", "0.5"))
SLOW_LIMIT = float(os.environ.get("MAMA_SLOW_LIMIT", "0.05"))
CYCLE_PART = float(os.environ.get("MAMA_CYCLE_PART", "0.5"))


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


def _exrem(signal: pd.Series, reset_signal: pd.Series) -> pd.Series:
    out = []
    active = False
    for sig, reset in zip(signal.fillna(False), reset_signal.fillna(False)):
        if reset:
            active = False
        if sig and not active:
            out.append(True)
            active = True
        else:
            out.append(False)
    return pd.Series(out, index=signal.index, dtype=bool)


def _value_when(condition: pd.Series, value: pd.Series) -> pd.Series:
    out = []
    last_value = np.nan
    for cond, val in zip(condition.fillna(False), value):
        if cond:
            last_value = val
        out.append(last_value)
    return pd.Series(out, index=value.index, dtype=float)


def has_ohlcv(df: pd.DataFrame) -> bool:
    required = ("Open", "High", "Low", "Close", "Volume")
    if len(df) < MIN_HISTORY or not all(col in df.columns for col in required):
        return False
    return not df[list(required)].tail(MIN_HISTORY).isna().any().any()


def compute_mama_positional_system(
    df: pd.DataFrame,
    fast_limit: float = FAST_LIMIT,
    slow_limit: float = SLOW_LIMIT,
    cycle_part: float = CYCLE_PART,
) -> dict:
    high = df["High"].reset_index(drop=True).astype(float)
    low = df["Low"].reset_index(drop=True).astype(float)
    close = df["Close"].reset_index(drop=True).astype(float)
    prc = (high + low) / 2
    n = len(df)

    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.zeros(n)
    smooth_period = np.zeros(n)
    phase = np.zeros(n)
    delta_phase = np.ones(n)
    alpha = np.zeros(n)
    mama = np.zeros(n)
    fama = np.zeros(n)

    for i in range(6, n):
        smooth[i] = (4 * prc.iloc[i] + 3 * prc.iloc[i - 1] + 2 * prc.iloc[i - 2] + prc.iloc[i - 3]) / 10
        adj = 0.075 * period[i - 1] + 0.54

        detrender[i] = (
            0.0962 * smooth[i]
            + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6]
        ) * adj

        q1[i] = (
            0.0962 * detrender[i]
            + 0.5769 * detrender[i - 2]
            - 0.5769 * detrender[i - 4]
            - 0.0962 * detrender[i - 6]
        ) * adj
        i1[i] = detrender[i - 3]

        ji[i] = (
            0.0962 * i1[i]
            + 0.5769 * i1[i - 2]
            - 0.5769 * i1[i - 4]
            - 0.0962 * i1[i - 6]
        ) * adj
        jq[i] = (
            0.0962 * q1[i]
            + 0.5769 * q1[i - 2]
            - 0.5769 * q1[i - 4]
            - 0.0962 * q1[i - 6]
        ) * adj

        i2_raw = i1[i] - jq[i]
        q2_raw = q1[i] + ji[i]
        i2[i] = 0.2 * i2_raw + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2_raw + 0.8 * q2[i - 1]

        re_raw = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_raw = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re[i] = 0.2 * re_raw + 0.8 * re[i - 1]
        im[i] = 0.2 * im_raw + 0.8 * im[i - 1]

        if im[i] != 0 and re[i] != 0:
            period[i] = 2 * np.pi / np.arctan(im[i] / re[i])
        else:
            period[i] = period[i - 1]
        if period[i - 1] > 0:
            period[i] = min(period[i], 1.5 * period[i - 1])
            period[i] = max(period[i], 0.67 * period[i - 1])
        period[i] = min(max(period[i], 6), 50)
        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]

        if i1[i] != 0:
            phase[i] = np.degrees(np.arctan(q1[i] / i1[i]))
        delta_phase[i] = phase[i - 1] - phase[i]
        if delta_phase[i] < 1:
            delta_phase[i] = 1

        alpha[i] = fast_limit / delta_phase[i]
        if alpha[i] < slow_limit:
            alpha[i] = slow_limit
        if alpha[i] > fast_limit:
            alpha[i] = fast_limit

        mama[i] = alpha[i] * prc.iloc[i] + (1 - alpha[i]) * mama[i - 1]
        fama[i] = cycle_part * alpha[i] * prc.iloc[i] + (1 - cycle_part * alpha[i]) * fama[i - 1]

    mama_s = pd.Series(mama, index=df.index, dtype=float)
    fama_s = pd.Series(fama, index=df.index, dtype=float)
    buysetup = _cross_up(mama_s, fama_s)
    sellsetup = _cross_up(fama_s, mama_s)
    buy_setup_value = _value_when(buysetup, high)
    sell_setup_value = _value_when(sellsetup, low)
    longa = _flip(buysetup, sellsetup)
    shrta = _flip(sellsetup, buysetup)

    raw_buy = longa & _cross_up(close, buy_setup_value)
    raw_sell = shrta & _cross_up(sell_setup_value, close)
    buy = _exrem(raw_buy, raw_sell)
    sell = _exrem(raw_sell, buy)
    t1 = _flip(buy, sell)
    t2 = _flip(sell, buy)
    buy_start = t1 & ~t1.shift(1, fill_value=False)
    sell_start = t2 & ~t2.shift(1, fill_value=False)
    bprice = _value_when(buy_start, close)
    sprice = _value_when(sell_start, close)

    return {
        "mama_series": mama_s,
        "fama_series": fama_s,
        "period_series": pd.Series(period, index=df.index, dtype=float),
        "smooth_period_series": pd.Series(smooth_period, index=df.index, dtype=float),
        "alpha_series": pd.Series(alpha, index=df.index, dtype=float),
        "buysetup_series": buysetup,
        "sellsetup_series": sellsetup,
        "buy_setup_value_series": buy_setup_value,
        "sell_setup_value_series": sell_setup_value,
        "longa_series": longa,
        "shrta_series": shrta,
        "raw_buy_series": raw_buy,
        "raw_sell_series": raw_sell,
        "buy_series": buy,
        "sell_series": sell,
        "t1_series": t1,
        "t2_series": t2,
        "bprice_series": bprice,
        "sprice_series": sprice,
        "buy": _last_bool(buy),
        "sell": _last_bool(sell),
        "buysetup": _last_bool(buysetup),
        "sellsetup": _last_bool(sellsetup),
        "longa": _last_bool(longa),
        "shrta": _last_bool(shrta),
        "mama": _num(mama_s.iloc[-1]) if n else None,
        "fama": _num(fama_s.iloc[-1]) if n else None,
        "period": _num(period[-1]) if n else None,
        "smooth_period": _num(smooth_period[-1]) if n else None,
        "alpha": _num(alpha[-1]) if n else None,
        "buy_setup_value": _num(buy_setup_value.iloc[-1]) if n else None,
        "sell_setup_value": _num(sell_setup_value.iloc[-1]) if n else None,
        "bprice": _num(bprice.iloc[-1]) if n else None,
        "sprice": _num(sprice.iloc[-1]) if n else None,
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
    audit.update({
        "last_date": _date(last.get("TradingDate")),
        "last_close": _num(last.get("Close")),
        "last_volume": _num(last_volume),
    })
    if pd.isna(last_volume) or float(last_volume) <= MIN_VOLUME:
        audit["reason"] = "volume_filter"
        return None, audit

    signal = compute_mama_positional_system(df)
    audit.update({
        "mama": _num(signal["mama"]),
        "fama": _num(signal["fama"]),
        "period": _num(signal["period"]),
        "alpha": _num(signal["alpha"]),
        "buysetup": bool(signal["buysetup"]),
        "sellsetup": bool(signal["sellsetup"]),
        "longa": bool(signal["longa"]),
        "shrta": bool(signal["shrta"]),
        "buy": bool(signal["buy"]),
        "sell": bool(signal["sell"]),
        "buy_setup_value": _num(signal["buy_setup_value"]),
        "sell_setup_value": _num(signal["sell_setup_value"]),
        "bprice": _num(signal["bprice"]),
        "sprice": _num(signal["sprice"]),
    })

    if not signal["buy"] and not signal["sell"]:
        audit["reason"] = "no_confirmed_signal"
        return None, audit

    status = "BUY" if signal["buy"] else "SELL"
    audit["reason"] = "buy_signal" if signal["buy"] else "sell_signal"
    return {
        "symbol": symbol,
        "status": status,
        "signal_type": "mama_positional_buy" if signal["buy"] else "mama_positional_sell",
        "score": 100,
        "mama_buy": bool(signal["buy"]),
        "mama_sell": bool(signal["sell"]),
        "mama": round(float(signal["mama"]), 2) if signal["mama"] is not None else None,
        "fama": round(float(signal["fama"]), 2) if signal["fama"] is not None else None,
        "period": round(float(signal["period"]), 2) if signal["period"] is not None else None,
        "alpha": round(float(signal["alpha"]), 4) if signal["alpha"] is not None else None,
        "buy_setup_value": _num(signal["buy_setup_value"]),
        "sell_setup_value": _num(signal["sell_setup_value"]),
        "bprice": _num(signal["bprice"]),
        "sprice": _num(signal["sprice"]),
        "buy_price": signal["buy_price"],
        "sell_price": signal["sell_price"],
        "last_price": float(df["Close"].iloc[-1]),
        "last_volume": float(last_volume),
        "strategies": ["mama_positional_buy" if signal["buy"] else "mama_positional_sell"],
    }, audit


def analyze_symbol(symbol: str) -> dict | None:
    result, _audit = audit_symbol(symbol)
    return result


def get_filtered_symbols() -> list[str]:
    symbols = list_symbols(CACHE_DIR, min_history=MIN_HISTORY)
    return [s for s in symbols if len(s) <= 3 and not any(c.isdigit() for c in s)]


def main():
    tqdm.write("=" * 60)
    tqdm.write("MAMA Positional Signals - Ehlers adaptive period")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    audit = []
    skipped_ohlc = 0
    bar = tqdm(symbols, desc="[MAMA] Positional", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result, item_audit = audit_symbol(sym)
        audit.append(item_audit)
        if result:
            signals.append(result)
        elif item_audit["reason"] == "missing_ohlcv_or_history":
            skipped_ohlc += 1

    signals.sort(key=lambda x: (x["status"] == "BUY", x["last_volume"]), reverse=True)
    buys = [s for s in signals if s["mama_buy"]]
    sells = [s for s in signals if s["mama_sell"]]
    now = vn_now()
    market_date = format_market_date(signal_market_date()) or now.strftime("%d/%m/%Y")
    output = {
        "mode": "ehlers_mama_positional",
        "generated_at": now.isoformat(),
        "date": market_date,
        "fast_limit": FAST_LIMIT,
        "slow_limit": SLOW_LIMIT,
        "cycle_part": CYCLE_PART,
        "min_volume": MIN_VOLUME,
        "min_history": MIN_HISTORY,
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "skipped_missing_ohlc": skipped_ohlc,
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
    tqdm.write(f"Tin hieu MAMA: {len(signals)} (Buy: {len(buys)}, Sell: {len(sells)})")
    if skipped_ohlc:
        tqdm.write(f"Bo qua {skipped_ohlc} ma do cache chua co OHLCV day du.")


if __name__ == "__main__":
    main()
