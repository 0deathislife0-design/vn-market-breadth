"""
Ensemble Strategy Voting — 4-signal daily swing.
  - MA Crossover (MA10 > MA50 + price > MA10 + RSI > 50)
  - Pullback to Support (uptrend + near MA50 + RSI rebound)
  - Breakout (new 20d high + vol > 1.5x avg)
  - Momentum ROC (ROC10 > ROC20 + vol slope > 0)
  Score >= 3/4 = Strong Buy, >= 2/4 = Weak Buy.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone, timedelta
from cache_utils import load_cache as _load_cache, compute_rsi_numpy

import numpy as np
import pandas as pd
from _shared import tqdm, CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, DEFAULT_WEIGHTS, WEIGHTS_PATH

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "ensemble_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "ensemble_signals.json"

MIN_AVG_VOLUME = 300_000


def _load_weights() -> dict:
    """Load backtest weights, fallback to defaults."""
    try:
        data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        w = data.get("weights", DEFAULT_WEIGHTS)
        return {k: w.get(k, DEFAULT_WEIGHTS[k]) for k in DEFAULT_WEIGHTS}
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def compute_ma_crossover(df: pd.DataFrame) -> dict:
    """MA Crossover: MA10 > MA50, Close > MA10, RSI14 > 50."""
    close = df["Close"].values
    if len(close) < 60:
        return {"signal": 0, "ma10": None, "ma50": None, "rsi14": None}
    ma10 = close[-10:].mean()
    ma50 = close[-50:].mean()
    rsi14 = compute_rsi_numpy(close, 14)
    signal = 1 if (ma10 > ma50 and close[-1] > ma10 and rsi14 > 50) else 0
    return {"signal": signal, "ma10": round(ma10, 1), "ma50": round(ma50, 1), "rsi14": round(rsi14, 1)}


def compute_pullback(df: pd.DataFrame) -> dict:
    """Pullback to MA50 trong uptrend:
    - Price > MA200 (uptrend)
    - Close o gan MA50 (97-100%)
    - RSI14 > 45 (khong qua yeu)."""
    close = df["Close"].values
    if len(close) < 210:
        return {"signal": 0, "ma50": None, "ma200": None, "rsi14": None}
    ma50 = close[-50:].mean()
    ma200 = close[-200:].mean()
    rsi14 = compute_rsi_numpy(close, 14)
    near_ma50 = 0.93 <= close[-1] / ma50 <= 1.00 if ma50 > 0 else False
    signal = 1 if (close[-1] > ma200 and near_ma50 and rsi14 > 45) else 0
    return {"signal": signal, "ma50": round(ma50, 1), "ma200": round(ma200, 1), "rsi14": round(rsi14, 1)}


def compute_breakout(df: pd.DataFrame) -> dict:
    """Breakout: Close > 20-day high, Volume > 1.5x avg(20)."""
    close = df["Close"].values
    volume = df["Volume"].values
    if len(close) < 25:
        return {"signal": 0, "high_20": None, "vol_ratio": None}
    high_20 = close[-21:-1].max()  # 20 phien truoc (khong tinh hom nay)
    vol_avg_20 = volume[-21:-1].mean() if np.sum(~np.isnan(volume[-21:-1])) >= 10 else 0
    vol_ratio = volume[-1] / vol_avg_20 if vol_avg_20 > 0 else 0
    signal = 1 if (close[-1] > high_20 and vol_ratio > 1.5) else 0
    return {"signal": signal, "high_20": round(high_20, 1), "vol_ratio": round(vol_ratio, 2)}


def compute_momentum(df: pd.DataFrame) -> dict:
    """ROC momentum: ROC10 > ROC20, volume slope 10 phien > 0."""
    close = df["Close"].values
    volume = df["Volume"].values
    if len(close) < 25:
        return {"signal": 0, "roc10": None, "roc20": None}
    roc10 = (close[-1] - close[-11]) / close[-11] * 100 if close[-11] > 0 else 0
    roc20 = (close[-1] - close[-21]) / close[-21] * 100 if close[-21] > 0 else 0
    vol_valid = ~np.isnan(volume[-10:])
    vol_slope = 0
    if np.sum(vol_valid) >= 5:
        x = np.where(vol_valid)[0]
        y = volume[-10:][vol_valid]
        vol_slope = np.polyfit(x, y, 1)[0] / np.mean(y) if np.mean(y) > 0 else 0
    signal = 1 if (roc10 > roc20 and vol_slope > 0) else 0
    return {"signal": signal, "roc10": round(roc10, 2), "roc20": round(roc20, 2), "vol_slope": round(vol_slope, 4)}


def analyze_symbol(symbol: str, weights: dict | None = None) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR)
    if len(df) < 210:
        return None
    if "Volume" in df.columns:
        vol_avg = df["Volume"].dropna().iloc[-20:].mean()
        if pd.isna(vol_avg) or vol_avg < MIN_AVG_VOLUME:
            return None

    ma = compute_ma_crossover(df)
    pb = compute_pullback(df)
    bo = compute_breakout(df)
    mo = compute_momentum(df)

    w = weights if weights is not None else _load_weights()
    total = w["ma_crossover"] * ma["signal"] + w["pullback"] * pb["signal"] + w["breakout"] * bo["signal"] + w["momentum"] * mo["signal"]

    if total < 0.35:
        return None

    return {
        "symbol": symbol,
        "total_score": round(total, 3),
        "ma_crossover": ma["signal"],
        "pullback": pb["signal"],
        "breakout": bo["signal"],
        "momentum": mo["signal"],
        "ma10": ma["ma10"],
        "ma50": ma["ma50"] if ma["ma50"] is not None else pb["ma50"],
        "rsi14": ma["rsi14"] if ma["rsi14"] is not None else pb["rsi14"],
        "vol_ratio": bo["vol_ratio"],
        "roc10": mo["roc10"],
        "roc20": mo["roc20"],
        "last_price": float(df["Close"].iloc[-1]),
        "last_volume": float(df["Volume"].iloc[-1]) if not pd.isna(df["Volume"].iloc[-1]) else None,
    }


def get_filtered_symbols() -> list[str]:
    from _shared import list_symbols
    return list_symbols(CACHE_DIR, min_history=20, min_volume=MIN_AVG_VOLUME)


def main():
    tqdm.write("=" * 60)
    tqdm.write("Ensemble Strategy Signals — 4-Signal Voting")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    weights = _load_weights()
    signals = []
    bar = tqdm(symbols, desc="[ALL] Ensemble", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result = analyze_symbol(sym, weights)
        if result:
            signals.append(result)

    signals.sort(key=lambda x: x["total_score"], reverse=True)

    now = datetime.now(timezone.utc) + timedelta(hours=7)
    strong = [s for s in signals if s["total_score"] >= 0.65]
    weak = [s for s in signals if s["total_score"] >= 0.35 and s["total_score"] < 0.65]

    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "strong_buy": len(strong),
        "weak_buy": len(weak),
        "strong": strong,
        "weak": weak,
        "all_signals": signals,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_bytes(SIGNALS_JSON.read_bytes())

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON}")
    tqdm.write(f"Tong phan tich: {output['total_symbols_analyzed']} ma")
    tqdm.write(f"Tin hieu: {output['total_signals']} (Strong: {output['strong_buy']}, Weak: {output['weak_buy']})")
    if signals:
        tqdm.write(f"\nTop tin hieu:")
        for s in signals[:5]:
            parts = []
            if s["ma_crossover"]: parts.append("MA")
            if s["pullback"]: parts.append("PB")
            if s["breakout"]: parts.append("BO")
            if s["momentum"]: parts.append("ROC")
            tqdm.write(f"  {s['symbol']:6s} | Score: {s['total_score']:.2f} | {'+'.join(parts):12s} | Price: {s['last_price']:.0f}")


if __name__ == "__main__":
    main()
