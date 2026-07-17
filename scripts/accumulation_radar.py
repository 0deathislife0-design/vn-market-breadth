"""
Accumulation Radar: find stocks that stay resilient and quietly accumulate
while the broad market is weak.

The benchmark is an equal-weight proxy built from liquid symbols in the local
OHLC cache. This keeps the hub independent from VNINDEX availability.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, json_default, list_symbols, tqdm


warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_JSON = DATA_DIR / "accumulation_radar.json"
DOCS_OUTPUT_JSON = DOCS_DATA_DIR / "accumulation_radar.json"
LATEST_PRICES_JSON = DATA_DIR / "latest_prices.json"

MIN_AVG_VOLUME = 300_000
MIN_HISTORY = 90


def _pct_change(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    start = close.iloc[-days - 1]
    end = close.iloc[-1]
    if pd.isna(start) or start <= 0 or pd.isna(end):
        return None
    return float((end / start - 1.0) * 100.0)


def _score_between(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return float(max(0.0, min(100.0, (value - low) / (high - low) * 100.0)))


def _drawdown(close: pd.Series, days: int) -> float | None:
    window = close.iloc[-days:]
    high = window.max()
    last = window.iloc[-1]
    if pd.isna(high) or high <= 0 or pd.isna(last):
        return None
    return float((last / high - 1.0) * 100.0)


def _range_pct(df: pd.DataFrame, days: int) -> float | None:
    window = df.iloc[-days:]
    high = window["High"].max()
    low = window["Low"].min()
    last = window["Close"].iloc[-1]
    if pd.isna(high) or pd.isna(low) or pd.isna(last) or last <= 0:
        return None
    return float((high - low) / last * 100.0)


def _position_in_range(df: pd.DataFrame, days: int) -> float | None:
    window = df.iloc[-days:]
    high = window["High"].max()
    low = window["Low"].min()
    last = (window["High"].iloc[-1] + window["Low"].iloc[-1]) / 2.0
    if pd.isna(high) or pd.isna(low) or high <= low:
        return None
    return float((last - low) / (high - low) * 100.0)


def _up_down_volume_ratio(df: pd.DataFrame, days: int = 20) -> float | None:
    window = df.iloc[-days:].copy()
    if len(window) < days or "Volume" not in window:
        return None
    diff = window["Close"].diff()
    up_vol = window.loc[diff > 0, "Volume"].mean()
    down_vol = window.loc[diff < 0, "Volume"].mean()
    if pd.isna(up_vol) or pd.isna(down_vol) or down_vol <= 0:
        return None
    return float(up_vol / down_vol)


def _accumulation_distribution_days(df: pd.DataFrame, days: int = 30) -> tuple[int, int]:
    window = df.iloc[-days:].copy()
    if len(window) < days:
        return 0, 0
    vol_ma20 = df["Volume"].rolling(20).mean().iloc[-days:]
    pct = window["Close"].pct_change() * 100.0
    acc = int(((pct > 1.5) & (window["Volume"] > vol_ma20)).sum())
    dist = int(((pct < -1.5) & (window["Volume"] > vol_ma20)).sum())
    return acc, dist


def _benchmark_returns(symbols: list[str]) -> dict[str, float]:
    rets20 = []
    rets60 = []
    dd60 = []
    for symbol in symbols:
        df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate")
        if len(df) < MIN_HISTORY:
            continue
        close = df["Close"]
        r20 = _pct_change(close, 20)
        r60 = _pct_change(close, 60)
        d60 = _drawdown(close, 60)
        if r20 is not None:
            rets20.append(r20)
        if r60 is not None:
            rets60.append(r60)
        if d60 is not None:
            dd60.append(d60)
    return {
        "return_20d": round(float(np.nanmedian(rets20)), 2) if rets20 else 0.0,
        "return_60d": round(float(np.nanmedian(rets60)), 2) if rets60 else 0.0,
        "drawdown_60d": round(float(np.nanmedian(dd60)), 2) if dd60 else 0.0,
    }


def analyze_symbol(symbol: str, benchmark: dict[str, float], latest_prices: dict | None = None) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate").reset_index(drop=True)
    if len(df) < MIN_HISTORY:
        return None
    if df["Volume"].iloc[-20:].mean() < MIN_AVG_VOLUME:
        return None

    close = df["Close"]
    last = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ret20 = _pct_change(close, 20)
    ret60 = _pct_change(close, 60)
    dd60 = _drawdown(close, 60)
    range20 = _range_pct(df, 20)
    range60 = _range_pct(df, 60)
    pos20 = _position_in_range(df, 20)
    up_down_ratio = _up_down_volume_ratio(df, 20)
    acc_days, dist_days = _accumulation_distribution_days(df, 30)
    range10 = _range_pct(df, 10)

    if None in (ret20, ret60, dd60, range20, range60, pos20):
        return None

    rs20 = ret20 - benchmark["return_20d"]
    rs60 = ret60 - benchmark["return_60d"]

    if len(close) >= 26:
        try:
            ret20_5ago = float((close.iloc[-6] / close.iloc[-26] - 1.0) * 100.0)
            rs_trend = round(ret20 - ret20_5ago, 2) if not pd.isna(ret20_5ago) else None
        except Exception:
            rs_trend = None
    else:
        rs_trend = None

    rs_base_score = 0.55 * _score_between(rs20, -5.0, 12.0) + 0.45 * _score_between(rs60, -8.0, 20.0)
    rs_trend_bonus = _score_between(rs_trend or 0.0, -3.0, 5.0) * 0.15
    rs_score = min(100.0, rs_base_score + rs_trend_bonus)

    dd_advantage = dd60 - benchmark["drawdown_60d"]
    resilience_score = 0.0
    resilience_score += 35.0 if last >= ma20 else 0.0
    resilience_score += 25.0 if last >= ma50 else 0.0
    resilience_score += 25.0 * (_score_between(dd_advantage, -8.0, 8.0) / 100.0)
    resilience_score += 15.0 * (pos20 / 100.0)

    vol_ratio_score = _score_between(up_down_ratio or 0.0, 0.8, 1.8)
    acc_day_score = _score_between(acc_days - dist_days, -2.0, 4.0)

    recent_5 = df.iloc[-5:]
    r5_hi = recent_5["High"].max()
    r5_lo = recent_5["Low"].min()
    if r5_hi > r5_lo and "Volume" in recent_5.columns:
        price_pos_5 = (recent_5["Close"] - r5_lo) / (r5_hi - r5_lo)
        near_top = price_pos_5 >= 0.7
        if near_top.any():
            near_vol = recent_5.loc[near_top, "Volume"].mean()
            far_vol = recent_5.loc[~near_top, "Volume"].mean() if (~near_top).any() else near_vol
            vol_dry_ratio = float(near_vol / far_vol) if far_vol > 0 else 1.0
        else:
            vol_dry_ratio = 1.0
    else:
        vol_dry_ratio = 1.0
    vol_dry_score = max(0.0, 100.0 - _score_between(vol_dry_ratio, 0.5, 1.5))

    volume_score = 0.40 * vol_ratio_score + 0.30 * acc_day_score + 0.30 * vol_dry_score

    contraction = max(0.0, min(1.0, (range60 - range20) / range60)) if range60 > 0 else 0.0
    tightness_score = 100.0 - _score_between(range20, 8.0, 35.0)
    contraction_score = 0.55 * (contraction * 100.0) + 0.45 * tightness_score

    distance_to_high = 100.0 - pos20
    tight10 = 100.0 - _score_between(range10 or 0.0, 5.0, 25.0)
    breakout_score = 0.45 * (100.0 - _score_between(distance_to_high, 0.0, 12.0))
    breakout_score += 0.35 * tight10
    breakout_score += 0.20 * vol_dry_score

    score = (
        0.30 * rs_score
        + 0.20 * resilience_score
        + 0.20 * volume_score
        + 0.15 * contraction_score
        + 0.15 * breakout_score
    )

    if score >= 80:
        status = "LEADER"
    elif score >= 65:
        status = "WATCHLIST"
    elif score >= 50:
        status = "EARLY"
    else:
        status = "IGNORE"

    reasons = []
    if rs20 > 0:
        reasons.append("RS20 vuot benchmark")
    if rs_trend is not None and rs_trend >= 2:
        reasons.append("RS20 dang cai thien")
    if last >= ma50:
        reasons.append("giu tren MA50")
    elif last >= ma20:
        reasons.append("giu tren MA20")
    if up_down_ratio and up_down_ratio >= 1.2:
        reasons.append("volume phien tang tot")
    if contraction_score >= 60:
        reasons.append("nen gia co hep")
    if pos20 >= 80:
        reasons.append("gan dinh nen 20 phien")
    if vol_dry_ratio <= 0.8:
        reasons.append("volume can o gan dinh")

    lp = (latest_prices or {}).get(symbol, {})
    fallback_price = round(last, 2)
    fallback_date = df["TradingDate"].iloc[-1].strftime("%d/%m/%Y")

    return {
        "symbol": symbol,
        "score": round(float(score), 1),
        "status": status,
        "last_price": round(float(lp.get("close", fallback_price)), 2),
        "last_date": str(lp.get("date", fallback_date)),
        "cache_date": fallback_date,
        "rs_trend": round(float(rs_trend), 2) if rs_trend is not None else None,
        "rs20": round(float(rs20), 2),
        "rs60": round(float(rs60), 2),
        "return20": round(float(ret20), 2),
        "return60": round(float(ret60), 2),
        "drawdown60": round(float(dd60), 2),
        "range20": round(float(range20), 2),
        "range60": round(float(range60), 2),
        "position20": round(float(pos20), 1),
        "up_down_volume_ratio": round(float(up_down_ratio), 2) if up_down_ratio else None,
        "vol_dry_ratio": round(float(vol_dry_ratio), 2),
        "accumulation_days": acc_days,
        "distribution_days": dist_days,
        "component_scores": {
            "relative_strength": round(float(rs_score), 1),
            "resilience": round(float(resilience_score), 1),
            "volume": round(float(volume_score), 1),
            "contraction": round(float(contraction_score), 1),
            "breakout_readiness": round(float(breakout_score), 1),
        },
        "reasons": reasons[:4],
    }


def main() -> None:
    tqdm.write("=" * 60)
    tqdm.write("Accumulation Radar")
    tqdm.write("=" * 60)

    latest_prices = {}
    if LATEST_PRICES_JSON.exists():
        try:
            latest_prices = json.loads(LATEST_PRICES_JSON.read_text(encoding="utf-8"))
            tqdm.write(f"Da doc latest_prices.json ({len(latest_prices)} ma)\n")
        except Exception:
            tqdm.write("WARN: Khong doc duoc latest_prices.json, dung cache date\n")

    symbols = list_symbols(min_history=MIN_HISTORY, min_volume=MIN_AVG_VOLUME)
    benchmark = _benchmark_returns(symbols)
    tqdm.write(f"Phan tich {len(symbols)} ma, benchmark proxy: {benchmark}\n")

    rows = []
    bar = tqdm(symbols, desc="[ALL] Accumulation", unit="sym")
    for symbol in bar:
        bar.set_postfix_str(symbol, refresh=True)
        result = analyze_symbol(symbol, benchmark, latest_prices)
        if result:
            rows.append(result)

    rows.sort(key=lambda item: item["score"], reverse=True)
    candidates = [row for row in rows if row["score"] >= 50]
    leaders = [row for row in rows if row["score"] >= 80]
    watchlist = [row for row in rows if 65 <= row["score"] < 80]
    early = [row for row in rows if 50 <= row["score"] < 65]

    now = datetime.now(timezone(timedelta(hours=7)))
    parsed_dates = []
    for v in latest_prices.values():
        d = v.get("date", "") if isinstance(v, dict) else ""
        if "/" in d:
            try:
                parsed_dates.append(datetime.strptime(d, "%d/%m/%Y"))
            except ValueError:
                pass
    latest_data_date = max(parsed_dates).strftime("%d/%m/%Y") if parsed_dates else now.strftime("%d/%m/%Y")

    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "latest_data_date": latest_data_date,
        "method": "equal_weight_liquid_market_proxy",
        "benchmark": benchmark,
        "total_symbols_analyzed": len(symbols),
        "total_candidates": len(candidates),
        "leader_count": len(leaders),
        "watchlist_count": len(watchlist),
        "early_count": len(early),
        "leaders": leaders,
        "watchlist": watchlist,
        "early": early,
        "all_candidates": candidates,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    DOCS_OUTPUT_JSON.write_bytes(OUTPUT_JSON.read_bytes())

    tqdm.write(f"\nDa ghi: {OUTPUT_JSON.name}")
    tqdm.write(f"Du lieu metadata: {latest_data_date}")
    tqdm.write(f"Ung vien: {len(candidates)} (Leader: {len(leaders)}, Watchlist: {len(watchlist)}, Early: {len(early)})")
    for row in candidates[:5]:
        tqdm.write(f"  {row['symbol']:6s} | Score: {row['score']:5.1f} | RS20: {row['rs20']:+5.2f} | Range20: {row['range20']:5.1f}%")


if __name__ == "__main__":
    main()
