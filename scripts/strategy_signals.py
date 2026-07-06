"""
Post-session strategy: Pre-breakout detection.
  Criteria cho "cổ phiếu đẹp sắp breakout":
    1. Base/consolidation dài (25+ sessions), range hẹp (<=8%)
    2. Giá sát đỉnh base (top 25% range)
    3. Volume co hẹp trong base (volume trend giảm)
    4. OBV tích lũy dương trong base
    5. Hội tụ đủ: base + money flow + OBV momentum
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone, timedelta
from cache_utils import load_cache as _load_cache, compute_rsi_numpy

import numpy as np
import pandas as pd
from _shared import tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "strategy_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "strategy_signals.json"

LOOKBACK = 90
BASE_WINDOW = 25
MAX_BASE_RANGE_PCT = 8.0
MIN_AVG_VOLUME = 300_000





def detect_base_quality(df: pd.DataFrame) -> dict:
    """Phân tích chất lượng base.
    Tra ve dict: base_score [0-1], price_pos, vol_trend, obv_trend, detail.
    base >= 0.6 moi duoc coi la base tot."""
    if len(df) < LOOKBACK:
        return {"base_score": 0.0, "price_pos": 0, "range_pct": 999, "vol_trend": 0}

    close = df["Close"].values
    volume = df["Volume"].values

    # Base window
    base = close[-BASE_WINDOW:]
    base_vol = volume[-BASE_WINDOW:]
    base_hi = base.max()
    base_lo = base.min()
    base_mid = (base_hi + base_lo) / 2
    range_pct = (base_hi - base_lo) / base_mid * 100 if base_mid > 0 else 999

    if range_pct > MAX_BASE_RANGE_PCT:
        return {"base_score": 0.0, "price_pos": 0, "range_pct": range_pct, "vol_trend": 0}

    # 1. Range tightness (40%)
    range_score = max(0, min(1, (MAX_BASE_RANGE_PCT - range_pct) / MAX_BASE_RANGE_PCT * 1.5))

    # 2. Price position in top of range (30%)
    # Gia hien tai phai o 1/4 tren cung cua base
    price_pos = (base[-1] - base_lo) / (base_hi - base_lo) if base_hi > base_lo else 0
    pos_score = max(0, min(1, (price_pos - 0.5) / 0.5))  # >= 0.75 -> score 0.5, >= 1.0 -> score 1.0

    # 3. Volume contraction trend (20%)
    # Volume trend: linear regression slope normalized
    valid_vol = ~np.isnan(base_vol)
    if np.sum(valid_vol) >= 15:
        x = np.arange(len(base_vol))[valid_vol]
        y = base_vol[valid_vol]
        slope = np.polyfit(x, y, 1)[0] / np.mean(y) if np.mean(y) > 0 else 0
        vol_score = max(0, min(1, -slope * 20))  # slope am = volume co hẹp
    else:
        vol_score = 0.0

    # 4. OBV trend trong base (10%)
    # OBV calculation
    obv_base = np.zeros(len(base))
    for i in range(1, len(base)):
        if not np.isnan(base[i]) and not np.isnan(base[i-1]):
            if base[i] > base[i-1]:
                obv_base[i] = obv_base[i-1] + (base_vol[i] if not np.isnan(base_vol[i]) else 0)
            elif base[i] < base[i-1]:
                obv_base[i] = obv_base[i-1] - (base_vol[i] if not np.isnan(base_vol[i]) else 0)
            else:
                obv_base[i] = obv_base[i-1]

    obv_start = obv_base[0]
    obv_end = obv_base[-1]
    obv_trend = (obv_end - obv_start) / max(abs(obv_start), 1)
    obv_score = max(0, min(1, obv_trend * 50))  # obv tang = accumulation

    base_score = 0.40 * range_score + 0.30 * pos_score + 0.20 * vol_score + 0.10 * obv_score

    return {
        "base_score": round(base_score, 3),
        "price_pos": round(price_pos, 3),
        "range_pct": round(range_pct, 2),
        "vol_trend": round(vol_score, 3),
        "obv_base_trend": round(obv_trend, 4),
        "range_score": round(range_score, 3),
        "pos_score": round(pos_score, 3),
    }


def compute_obv_momentum(df: pd.DataFrame) -> dict:
    """OBV momentum: xu huong OBV 5-10-21 phien gan nhat."""
    if len(df) < 30:
        return {"obv_mom5": 0, "obv_mom10": 0, "obv_mom21": 0, "obv_trend_score": 0}

    close = df["Close"].values
    volume = df["Volume"].values

    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    norm = max(abs(obv[-1]), 1)
    mom5 = (obv[-1] - obv[-5]) / norm
    mom10 = (obv[-1] - obv[-10]) / norm
    mom21 = (obv[-1] - obv[-21]) / norm

    # trend score: OBV phai dong bo (ca 3 khung deu duong hoac it nhat la tang dan)
    score = 0
    if mom5 > 0: score += 0.4
    if mom10 > 0: score += 0.3
    if mom21 > 0: score += 0.3
    # Neu ca 3 deu duong, bonus
    if mom5 > 0 and mom10 > 0 and mom21 > 0:
        score = min(1.0, score + 0.2)

    return {
        "obv_mom5": round(mom5, 4),
        "obv_mom10": round(mom10, 4),
        "obv_mom21": round(mom21, 4),
        "obv_trend_score": round(score, 3),
    }


def compute_vol_regime(df: pd.DataFrame) -> dict:
    """Vol-regime metrics: short/long vol ratio, gradient."""
    if len(df) < 63:
        return {"vol_ratio": None, "vol_gradient": 0}

    log_rets = np.log(df["Close"].values[1:] / df["Close"].values[:-1])
    vol_short = np.std(log_rets[-10:]) * np.sqrt(252)
    vol_long = np.std(log_rets[-63:]) * np.sqrt(252)
    vol_ratio = vol_short / vol_long if vol_long > 0 else None

    rolling_vols = pd.Series(log_rets).rolling(10).std().dropna()
    if len(rolling_vols) >= 10:
        recent_ratio = rolling_vols.iloc[-1] / rolling_vols.iloc[-64:-1].mean() if len(rolling_vols) > 63 else 1.0
        prev_ratio = rolling_vols.iloc[-6] / rolling_vols.iloc[-64:-6].mean() if len(rolling_vols) > 63 else 1.0
        vol_gradient = (recent_ratio - prev_ratio) / prev_ratio if prev_ratio != 0 else 0
    else:
        vol_gradient = 0

    return {"vol_ratio": vol_ratio, "vol_gradient": vol_gradient}


def analyze_symbol(symbol: str) -> dict | None:
    """Phan tich 1 symbol, tra ve signal data neu dat, None neu khong."""
    df = _load_cache(symbol, CACHE_DIR)
    if len(df) < 63:
        return None

    base = detect_base_quality(df)
    base_score = base["base_score"]
    if base_score < 0.60:
        return None

    obv = compute_obv_momentum(df)
    obv_score = obv["obv_trend_score"]

    vol = compute_vol_regime(df)
    vol_grad = vol["vol_gradient"]

    # Hard filter 1: Price position > 0.75 (top 25% base)
    if base["price_pos"] < 0.75:
        return None

    # Hard filter 2: OBV short-term momentum > 0 (5 ngay OBV tang)
    if obv["obv_mom5"] <= 0:
        return None

    # Vol gradient score
    grad_score = max(0, min(1, vol_grad * 3)) if vol_grad > 0 else 0

    # Composite: 50% base + 30% OBV + 20% vol gradient
    composite = 0.50 * base_score + 0.30 * obv_score + 0.20 * grad_score
    composite = round(composite * 100, 1)

    # Price near 20-day high
    close_20 = df["Close"].iloc[-20:].values
    high_20 = close_20.max()
    price_near_high_pct = (high_20 - df["Close"].iloc[-1]) / high_20 * 100 if high_20 > 0 else 999

    return {
        "symbol": symbol,
        "composite_score": composite,
        "base_score": round(base_score * 100, 1),
        "base_range_pct": base["range_pct"],
        "price_pos": base["price_pos"],
        "obv_mom5": obv["obv_mom5"],
        "obv_mom10": obv["obv_mom10"],
        "obv_mom21": obv["obv_mom21"],
        "obv_score": round(obv_score * 100, 1),
        "vol_gradient": round(vol_grad, 4),
        "grad_score": round(grad_score * 100, 1),
        "price_near_high_pct": round(price_near_high_pct, 2),
        "last_price": float(df["Close"].iloc[-1]),
        "last_volume": float(df["Volume"].iloc[-1]) if not pd.isna(df["Volume"].iloc[-1]) else None,
    }


def get_filtered_symbols() -> list[str]:
    """Lay danh sach symbol da loc thanh khoan."""
    symbols = []
    for path in sorted(CACHE_DIR.glob("*.csv")):
        sym = path.stem
        if sym == ".gitkeep":
            continue
        df = _load_cache(sym, CACHE_DIR)
        if len(df) < 20:
            continue
        if "Volume" in df.columns:
            avg_vol = df["Volume"].dropna().iloc[-20:].mean()
            if pd.isna(avg_vol) or avg_vol < MIN_AVG_VOLUME:
                continue
        # Filter out ETFs/funds (prefix FU)
        if sym.startswith('FU') or sym.startswith('E1'):
            continue
        symbols.append(sym)
    return symbols


def main():
    tqdm.write("=" * 60)
    tqdm.write("Pre-Breakout Strategy Signals")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    bar = tqdm(symbols, desc="[ALL] Pre-breakout", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result = analyze_symbol(sym)
        if result and result["composite_score"] >= 60:
            signals.append(result)

    signals.sort(key=lambda x: x["composite_score"], reverse=True)

    now = datetime.now(timezone.utc) + timedelta(hours=7)
    strong = [s for s in signals if s["composite_score"] >= 75]
    moderate = [s for s in signals if 60 <= s["composite_score"] < 75]

    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "strong_signals": len(strong),
        "moderate_signals": len(moderate),
        "strong": strong,
        "moderate": moderate,
        "all_signals": signals,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_bytes(SIGNALS_JSON.read_bytes())

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON}")
    tqdm.write(f"Tong phan tich: {output['total_symbols_analyzed']} ma")
    tqdm.write(f"Tin hieu: {output['total_signals']} (Manh: {output['strong_signals']}, TB: {output['moderate_signals']})")
    if signals:
        tqdm.write(f"\nTop tin hieu:")
        for s in signals[:5]:
            tqdm.write(f"  {s['symbol']:6s} | Score: {s['composite_score']:5.1f} | Base: {s['base_score']:5.1f} | Pos: {s['price_pos']:.2f} | OBV5: {s['obv_mom5']:+.4f}")


if __name__ == "__main__":
    main()

