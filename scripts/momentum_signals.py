"""
Momentum Signals — Bộ 4 chiến lược tìm điểm mua.
  - MA Crossover Momentum (30đ)
  - Breakout Strength (35đ)
  - ROC Momentum (25đ)
  - Hybrid Momentum Break (40đ)

  Score = sum(strategy_base) + sum(bonuses)
    >= 60: Strong Buy
    30-59: Quan sát
    < 30:  Bỏ qua
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone, timedelta
from cache_utils import load_cache as _load_cache, compute_rsi_numpy

import numpy as np
import pandas as pd
from _shared import tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR
from _shared import (SCORE_MA, SCORE_BREAKOUT, SCORE_ROC, SCORE_HYBRID,
                     BONUS_VOL_SURGE, BONUS_ADX_STRONG, BONUS_RSI_GOLD,
                     json_default as _json_default)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "momentum_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "momentum_signals.json"

MIN_AVG_VOLUME = 500_000


def _wilder_ema(values: np.ndarray, period: int = 14) -> np.ndarray:
    alpha = 1.0 / period
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = out[i-1] * (1 - alpha) + values[i] * alpha
    return out


def compute_adx(close: np.ndarray, period: int = 14) -> float:
    """ADX(14) close-based approximation."""
    if len(close) < period * 2:
        return 0.0
    tr = np.abs(np.diff(close))
    deltas = np.diff(close)
    plus_dm = np.where(deltas > 0, deltas, 0.0)
    minus_dm = np.where(deltas < 0, -deltas, 0.0)

    tr_smooth = _wilder_ema(tr, period)
    plus_smooth = _wilder_ema(plus_dm, period)
    minus_smooth = _wilder_ema(minus_dm, period)

    plus_di = np.where(tr_smooth > 0, 100.0 * plus_smooth / tr_smooth, 0.0)
    minus_di = np.where(tr_smooth > 0, 100.0 * minus_smooth / tr_smooth, 0.0)

    dx = np.where((plus_di + minus_di) > 0,
                  100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx_series = _wilder_ema(dx, period)
    return float(adx_series[-1])


# --- 4 Strategies ------------------------------------------------------------

def compute_ma_crossover(df: pd.DataFrame, rsi14: float | None = None, adx14: float | None = None) -> dict:
    close = df["Close"].values
    volume = df["Volume"].values
    if len(close) < 60:
        return {"signal": 0, "ma10": None, "ma50": None, "rsi14": None, "adx14": None, "vol_ratio": None}
    ma10 = close[-10:].mean()
    ma50 = close[-50:].mean()
    if rsi14 is None: rsi14 = compute_rsi_numpy(close, 14)
    if adx14 is None: adx14 = compute_adx(close, 14)
    vol_avg_20 = volume[-20:].mean()
    vol_ratio = volume[-1] / vol_avg_20 if vol_avg_20 > 0 else 0

    signal = 1 if (
        ma10 > ma50
        and close[-1] > ma10
        and rsi14 > 50
        and adx14 >= 22
        and vol_ratio > 1.3
    ) else 0

    return {
        "signal": signal,
        "ma10": round(ma10, 1),
        "ma50": round(ma50, 1),
        "rsi14": round(rsi14, 1),
        "adx14": round(adx14, 1),
        "vol_ratio": round(vol_ratio, 2),
    }


def compute_breakout(df: pd.DataFrame, rsi14: float | None = None) -> dict:
    close = df["Close"].values
    volume = df["Volume"].values
    if len(close) < 25:
        return {"signal": 0, "high_20": None, "rsi14": None, "vol_ratio": None}
    high_20 = close[-21:-1].max()
    ma10 = close[-10:].mean()
    if rsi14 is None: rsi14 = compute_rsi_numpy(close, 14)
    vol_avg_20 = volume[-20:].mean()
    vol_ratio = volume[-1] / vol_avg_20 if vol_avg_20 > 0 else 0

    signal = 1 if (
        close[-1] > high_20
        and vol_ratio > 1.3
        and rsi14 > 45
        and close[-1] > ma10
    ) else 0

    return {
        "signal": signal,
        "high_20": round(high_20, 1),
        "rsi14": round(rsi14, 1),
        "vol_ratio": round(vol_ratio, 2),
    }


def compute_roc_momentum(df: pd.DataFrame, rsi14: float | None = None) -> dict:
    close = df["Close"].values
    volume = df["Volume"].values
    if len(close) < 25:
        return {"signal": 0, "roc10": None, "roc20": None, "rsi14": None, "vol_slope": None}
    roc10 = (close[-1] - close[-11]) / close[-11] * 100 if close[-11] > 0 else 0
    roc20 = (close[-1] - close[-21]) / close[-21] * 100 if close[-21] > 0 else 0
    ma10 = close[-10:].mean()
    if rsi14 is None: rsi14 = compute_rsi_numpy(close, 14)

    vol_ma5 = volume[-5:].mean()
    vol_ma20 = volume[-20:].mean()
    vol_slope = 1 if (vol_ma5 > vol_ma20 and vol_ma20 > 0) else 0

    signal = 1 if (
        roc10 > roc20
        and roc10 > 5.0
        and vol_slope == 1
        and rsi14 > 50
        and close[-1] > ma10
    ) else 0

    return {
        "signal": signal,
        "roc10": round(roc10, 2),
        "roc20": round(roc20, 2),
        "rsi14": round(rsi14, 1),
        "vol_slope": vol_slope,
    }


def compute_hybrid(df: pd.DataFrame, rsi14: float | None = None, adx14: float | None = None) -> dict:
    close = df["Close"].values
    volume = df["Volume"].values
    if len(close) < 60:
        return {"signal": 0, "ma10": None, "ma50": None, "rsi14": None, "adx14": None, "vol_ratio": None}
    ma10 = close[-10:].mean()
    ma50 = close[-50:].mean()
    if rsi14 is None: rsi14 = compute_rsi_numpy(close, 14)
    if adx14 is None: adx14 = compute_adx(close, 14)

    vol_avg_20 = volume[-20:].mean()
    vol_ratio = volume[-1] / vol_avg_20 if vol_avg_20 > 0 else 0

    high_20 = close[-21:-1].max()
    nh20 = close[-1] > high_20

    roc10 = (close[-1] - close[-11]) / close[-11] * 100 if close[-11] > 0 else 0
    roc20 = (close[-1] - close[-21]) / close[-21] * 100 if close[-21] > 0 else 0

    signal = 1 if (
        ma10 > ma50
        and close[-1] > ma10
        and (nh20 or roc10 > roc20)
        and vol_ratio > 1.5
        and 48 <= rsi14 <= 73
        and adx14 >= 23
    ) else 0

    return {
        "signal": signal,
        "ma10": round(ma10, 1),
        "ma50": round(ma50, 1),
        "rsi14": round(rsi14, 1),
        "adx14": round(adx14, 1),
        "vol_ratio": round(vol_ratio, 2),
        "nh20": int(nh20),
    }


# --- Common filters ----------------------------------------------------------

def check_common_filters(df: pd.DataFrame, rsi14: float | None = None, adx14: float | None = None) -> bool:
    """Check common filters. Returns True if all pass."""
    close = df["Close"].values
    volume = df["Volume"].values
    if len(close) < 210:
        return False

    ma50 = close[-50:].mean()
    ma200 = close[-200:].mean()
    if not (close[-1] > ma50 > ma200):
        return False

    if rsi14 is None: rsi14 = compute_rsi_numpy(close, 14)
    if not (48 <= rsi14 <= 72):
        return False

    if adx14 is None: adx14 = compute_adx(close, 14)
    if adx14 < 20:
        return False

    vol_avg_20 = volume[-20:].mean()
    if pd.isna(vol_avg_20) or vol_avg_20 < MIN_AVG_VOLUME:
        return False

    return True


# --- Bonuses ----------------------------------------------------------------

def compute_bonuses(df: pd.DataFrame, rsi14: float | None = None, adx14: float | None = None) -> dict:
    close = df["Close"].values
    volume = df["Volume"].values

    vol_avg_20 = volume[-20:].mean()
    vol_ratio = volume[-1] / vol_avg_20 if vol_avg_20 > 0 else 0
    vol_surge = 1 if vol_ratio > 2.0 else 0

    if adx14 is None: adx14 = compute_adx(close, 14)
    adx_strong = 1 if adx14 > 28 else 0

    if rsi14 is None: rsi14 = compute_rsi_numpy(close, 14)
    rsi_gold = 1 if 50 <= rsi14 <= 68 else 0

    total = (vol_surge * BONUS_VOL_SURGE
             + adx_strong * BONUS_ADX_STRONG
             + rsi_gold * BONUS_RSI_GOLD)

    return {
        "total": total,
        "vol_surge": vol_surge,
        "adx_strong": adx_strong,
        "rsi_gold": rsi_gold,
        "vol_ratio": round(vol_ratio, 2),
        "adx14": round(adx14, 1),
        "rsi14": round(rsi14, 1),
    }


# --- Per-symbol analysis ----------------------------------------------------

def analyze_symbol(symbol: str) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR)
    if len(df) < 210:
        return None

    close = df["Close"].values
    rsi14 = compute_rsi_numpy(close, 14)
    adx14 = compute_adx(close, 14)

    if not check_common_filters(df, rsi14=rsi14, adx14=adx14):
        return None

    ma = compute_ma_crossover(df, rsi14=rsi14, adx14=adx14)
    bo = compute_breakout(df, rsi14=rsi14)
    roc = compute_roc_momentum(df, rsi14=rsi14)
    hy = compute_hybrid(df, rsi14=rsi14, adx14=adx14)
    bonuses = compute_bonuses(df, rsi14=rsi14, adx14=adx14)

    # Score = sum of activated strategy bases + bonus total
    base = (ma["signal"] * SCORE_MA
            + bo["signal"] * SCORE_BREAKOUT
            + roc["signal"] * SCORE_ROC
            + hy["signal"] * SCORE_HYBRID)
    score = base + bonuses["total"]

    if score < 30:
        return None

    activated = []
    if ma["signal"]: activated.append("ma_crossover")
    if bo["signal"]: activated.append("breakout")
    if roc["signal"]: activated.append("roc_momentum")
    if hy["signal"]: activated.append("hybrid")

    bonus_list = []
    if bonuses["vol_surge"]: bonus_list.append("vol_surge")
    if bonuses["adx_strong"]: bonus_list.append("adx_strong")
    if bonuses["rsi_gold"]: bonus_list.append("rsi_gold")

    sig_type = "strong" if score >= 60 else "watch"

    return {
        "symbol": symbol,
        "score": score,
        "signal_type": sig_type,
        "strategies": activated,
        "bonuses": bonus_list,
        "base_score": base,
        "bonus_score": bonuses["total"],
        "details": {
            "ma_crossover": ma,
            "breakout": bo,
            "roc_momentum": roc,
            "hybrid": hy,
        },
        "bonus_details": {
            "vol_surge": {"active": bonuses["vol_surge"], "ratio": bonuses["vol_ratio"]},
            "adx_strong": {"active": bonuses["adx_strong"], "adx": bonuses["adx14"]},
            "rsi_gold": {"active": bonuses["rsi_gold"], "rsi": bonuses["rsi14"]},
        },
        "rsi14": bonuses["rsi14"],
        "adx14": bonuses["adx14"],
        "vol_ratio": bonuses["vol_ratio"],
        "last_price": float(df["Close"].iloc[-1]),
        "last_volume": float(df["Volume"].iloc[-1]) if not pd.isna(df["Volume"].iloc[-1]) else None,
    }


def get_filtered_symbols() -> list[str]:
    from _shared import list_symbols
    return list_symbols(CACHE_DIR, min_history=20, min_volume=MIN_AVG_VOLUME)


def main():
    tqdm.write("=" * 60)
    tqdm.write("Momentum Signals — Bộ 4 chiến lược tìm điểm mua")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    bar = tqdm(symbols, desc="[MOM] Momentum", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result = analyze_symbol(sym)
        if result:
            signals.append(result)

    signals.sort(key=lambda x: (x["signal_type"] == "strong", x["score"]), reverse=True)

    now = datetime.now(timezone.utc) + timedelta(hours=7)
    strong = [s for s in signals if s["signal_type"] == "strong"]
    watch = [s for s in signals if s["signal_type"] == "watch"]

    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "strong_buy": len(strong),
        "watch_count": len(watch),
        "strong": strong,
        "watch": watch,
        "all_signals": signals,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON}")
    tqdm.write(f"Tong phan tich: {output['total_symbols_analyzed']} ma")
    tqdm.write(f"Tin hieu: {output['total_signals']} (Strong: {output['strong_buy']}, Quan sat: {output['watch_count']})")
    if signals:
        tqdm.write(f"\nTop tin hieu:")
        for s in signals[:5]:
            strat_str = "+".join(s["strategies"]) if s["strategies"] else "-"
            bonus_str = "+".join(s["bonuses"]) if s["bonuses"] else ""
            tqdm.write(f"  {s['symbol']:6s} | Score: {s['score']:3d} | {s['signal_type']:7s} | {strat_str:20s} {bonus_str}")


if __name__ == "__main__":
    main()
