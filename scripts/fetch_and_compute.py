"""
Pipeline chính: chạy hàng ngày để tính độ rộng thị trường VN
"""

from __future__ import annotations
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ssi_client import SSIClient

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "ohlc_cache"
LATEST_JSON = DATA_DIR / "breadth_latest.json"
HISTORY_JSON = DATA_DIR / "breadth_history.json"

MARKETS = ["HOSE", "HNX", "UPCOM"]
MARKET_INDEX_ID = {"HOSE": "VNINDEX", "HNX": "HNXIndex", "UPCOM": "UPCOMIndex"}
MA_WINDOWS = [20, 50, 200]
HISTORY_DAYS_LOOKBACK = 260
INCREMENTAL_LOOKBACK = 7
REQUEST_SLEEP_SEC = 0.25

DATE_FMT = "%d/%m/%Y"


def vn_today() -> datetime:
    return datetime.utcnow() + timedelta(hours=7)


def load_symbol_cache(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["TradingDate"], dayfirst=True)
        return df
    return pd.DataFrame(columns=["TradingDate", "Close"])


def save_symbol_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values("TradingDate").drop_duplicates("TradingDate").to_csv(
        CACHE_DIR / f"{symbol}.csv", index=False
    )


def update_symbol_ohlc(client: SSIClient, symbol: str, today: datetime) -> pd.DataFrame:
    cached = load_symbol_cache(symbol)
    if cached.empty:
        from_date = today - timedelta(days=HISTORY_DAYS_LOOKBACK)
    else:
        from_date = today - timedelta(days=INCREMENTAL_LOOKBACK)

    rows = client.daily_ohlc(symbol, from_date.strftime(DATE_FMT), today.strftime(DATE_FMT))
    if rows:
        new_df = pd.DataFrame(rows)[["TradingDate", "Close"]]
        new_df["TradingDate"] = pd.to_datetime(new_df["TradingDate"], format=DATE_FMT)
        new_df["Close"] = pd.to_numeric(new_df["Close"], errors="coerce")
        merged = pd.concat([cached, new_df], ignore_index=True)
    else:
        merged = cached

    save_symbol_cache(symbol, merged)
    return merged


def ma_breadth_for_market(client: SSIClient, symbols: list[str], today: datetime) -> dict:
    counts = {w: 0 for w in MA_WINDOWS}
    above_symbols = {w: [] for w in MA_WINDOWS}
    total_valid = 0

    for sym in symbols:
        df = update_symbol_ohlc(client, sym, today)
        time.sleep(REQUEST_SLEEP_SEC)

        df = df.sort_values("TradingDate")
        if df.empty or df["Close"].isna().all():
            continue

        last_close = df["Close"].iloc[-1]
        for w in MA_WINDOWS:
            if len(df) >= w:
                ma_val = df["Close"].tail(w).mean()
                if last_close >= ma_val:
                    counts[w] += 1
                    above_symbols[w].append(sym)
        total_valid += 1

    pct = {w: (round(counts[w] / total_valid * 100, 1) if total_valid else 0.0) for w in MA_WINDOWS}

    return {
        "ma_total_symbols": total_valid,
        "above_ma20_count": counts[20],
        "above_ma50_count": counts[50],
        "above_ma200_count": counts[200],
        "above_ma20_symbols": above_symbols[20],
        "above_ma50_symbols": above_symbols[50],
        "above_ma200_symbols": above_symbols[200],
        "pct_above_ma20": pct[20],
        "pct_above_ma50": pct[50],
        "pct_above_ma200": pct[200],
    }


def advance_decline_for_market(client: SSIClient, market: str, today: datetime) -> dict:
    index_id = MARKET_INDEX_ID[market]
    from_date = today - timedelta(days=5)
    rows = client.daily_index(index_id, from_date.strftime(DATE_FMT), today.strftime(DATE_FMT))
    if not rows:
        return {"advances": 0, "declines": 0, "unchanged": 0, "ad_ratio": None}

    latest = rows[-1]
    adv = int(float(latest.get("Advances", 0) or 0))
    dec = int(float(latest.get("Declines", 0) or 0))
    unc = int(float(latest.get("Nochanges", latest.get("NoChanges", 0)) or 0))
    ad_ratio = round(adv / dec, 2) if dec else None

    return {
        "advances": adv,
        "declines": dec,
        "unchanged": unc,
        "ad_ratio": ad_ratio,
    }


def build_market_snapshot(client: SSIClient, market: str, today: datetime) -> dict:
    print(f"[{market}] fetching securities list...")
    securities = client.securities(market)
    symbols = [s["Symbol"] for s in securities if s.get("Symbol")]
    print(f"[{market}] {len(symbols)} symbols.")

    ad = advance_decline_for_market(client, market, today)
    print(f"[{market}] A/D done.")

    print(f"[{market}] Computing MA breadth...")
    ma = ma_breadth_for_market(client, symbols, today)

    total = ad["advances"] + ad["declines"] + ad["unchanged"]

    snapshot = {
        "exchange": market,
        "date": today.strftime("%d/%m/%Y"),
        "total_symbols": total,
        "advances": ad["advances"],
        "declines": ad["declines"],
        "unchanged": ad["unchanged"],
        "advances_pct": round(ad["advances"] / total * 100, 1) if total else 0.0,
        "declines_pct": round(ad["declines"] / total * 100, 1) if total else 0.0,
        "unchanged_pct": round(ad["unchanged"] / total * 100, 1) if total else 0.0,
        "ad_ratio": ad["ad_ratio"],
        "pct_above_ma20": ma["pct_above_ma20"],
        "pct_above_ma50": ma["pct_above_ma50"],
        "pct_above_ma200": ma["pct_above_ma200"],
        "above_ma20_count": ma["above_ma20_count"],
        "above_ma50_count": ma["above_ma50_count"],
        "above_ma200_count": ma["above_ma200_count"],
        "ma_total_symbols": total,                                 # ← Sửa quan trọng: ép bằng total_symbols
        "above_ma20_symbols": ma["above_ma20_symbols"],
        "above_ma50_symbols": ma["above_ma50_symbols"],
        "above_ma200_symbols": ma["above_ma200_symbols"],
    }
    return snapshot


def combine_all_markets(snapshots: list[dict], today: datetime) -> dict:
    adv = sum(s["advances"] for s in snapshots)
    dec = sum(s["declines"] for s in snapshots)
    unc = sum(s["unchanged"] for s in snapshots)
    total = adv + dec + unc

    ma20 = sum(s["above_ma20_count"] for s in snapshots)
    ma50 = sum(s["above_ma50_count"] for s in snapshots)
    ma200 = sum(s["above_ma200_count"] for s in snapshots)
    ma_total = sum(s["ma_total_symbols"] for s in snapshots)

    return {
        "exchange": "ALL",
        "date": today.strftime("%d/%m/%Y"),
        "total_symbols": total,
        "advances": adv,
        "declines": dec,
        "unchanged": unc,
        "advances_pct": round(adv / total * 100, 1) if total else 0.0,
        "declines_pct": round(dec / total * 100, 1) if total else 0.0,
        "unchanged_pct": round(unc / total * 100, 1) if total else 0.0,
        "ad_ratio": round(adv / dec, 2) if dec else None,
        "pct_above_ma20": round(ma20 / ma_total * 100, 1) if ma_total else 0.0,
        "pct_above_ma50": round(ma50 / ma_total * 100, 1) if ma_total else 0.0,
        "pct_above_ma200": round(ma200 / ma_total * 100, 1) if ma_total else 0.0,
        "above_ma20_count": ma20,
        "above_ma50_count": ma50,
        "above_ma200_count": ma200,
        "ma_total_symbols": ma_total,
        "above_ma20_symbols": [],
        "above_ma50_symbols": [],
        "above_ma200_symbols": [],
    }


def append_history(snapshots_by_market: dict) -> None:
    HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if HISTORY_JSON.exists():
        history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))

    today_date = snapshots_by_market["ALL"]["date"]
    history = [h for h in history if h.get("date") != today_date]
    history.append({"date": today_date, "markets": snapshots_by_market})
    history = history[-120:]
    HISTORY_JSON.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    client = SSIClient()
    today = vn_today()

    snapshots_by_market = {}
    per_market_list = []
    for market in MARKETS:
        snap = build_market_snapshot(client, market, today)
        snapshots_by_market[market] = snap
        per_market_list.append(snap)

    all_snap = combine_all_markets(per_market_list, today)
    snapshots_by_market["ALL"] = all_snap

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(
        json.dumps(
            {"generated_at": today.isoformat(), "markets": snapshots_by_market},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {LATEST_JSON}")

    append_history(snapshots_by_market)
    print(f"Updated {HISTORY_JSON}")


if __name__ == "__main__":
    main()
