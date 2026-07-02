"""
Pipeline chính: chạy hàng ngày (qua GitHub Actions) để:
  1. Lấy/khớp danh sách mã theo sàn (HOSE/HNX/UPCOM)
  2. Cập nhật cache OHLC cục bộ cho từng mã (chỉ tải phần dữ liệu còn thiếu)
  3. Tính MA20/MA50/MA200 -> % mã trên từng đường MA theo sàn
  4. Lấy Advances/Declines/Nochanges từ DailyIndex (VNINDEX/HNXIndex/UPCOMIndex)
  5. Ghi ra data/breadth_latest.json + append vào data/breadth_history.json

Chạy: python scripts/fetch_and_compute.py
Biến môi trường cần có: SSI_CONSUMER_ID, SSI_CONSUMER_SECRET
"""
from __future__ import annotations

import json
import os
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
HISTORY_DAYS_LOOKBACK = 380  # ~380 ngày lịch mới đủ ~260 phiên giao dịch (trừ T7/CN/lễ), dư cho MA200
INCREMENTAL_LOOKBACK = 7     # mỗi lần chạy chỉ cần lấy vài phiên gần nhất để bù vào cache
REQUEST_SLEEP_SEC = 0.25     # giãn cách giữa các lần gọi để tránh rate limit

DATE_FMT = "%d/%m/%Y"


def vn_today() -> datetime:
    return datetime.utcnow() + timedelta(hours=7)  # UTC+7


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
    """Tính MA20/50/200 + danh sách mã trên MA và tín hiệu chuyển đổi"""
    counts = {w: 0 for w in MA_WINDOWS}
    above_symbols = {w: [] for w in MA_WINDOWS}
    newly_above = {w: [] for w in MA_WINDOWS}
    newly_below = {w: [] for w in MA_WINDOWS}
    total_valid = 0

    print(f"[MA] Bắt đầu tính cho {len(symbols)} mã...")

    for sym in symbols:
        df = update_symbol_ohlc(client, sym, today)
        time.sleep(REQUEST_SLEEP_SEC)

        df = df.sort_values("TradingDate").reset_index(drop=True)
        if df.empty or len(df) < 20 or df["Close"].isna().all():
            continue

        last_close = df["Close"].iloc[-1]
        total_valid += 1

        for w in MA_WINDOWS:
            if len(df) >= w:
                ma_today = df["Close"].tail(w).mean()
                is_above = last_close >= ma_today

                if is_above:
                    counts[w] += 1
                    above_symbols[w].append(sym)

                # Tính tín hiệu chuyển đổi (chỉ cho MA20 và MA50)
                if w in (20, 50) and len(df) >= w + 1:
                    prev_close = df["Close"].iloc[-2]
                    prev_ma = df["Close"].iloc[-(w+1):-1].mean()
                    was_above = prev_close >= prev_ma

                    if is_above and not was_above:
                        newly_above[w].append(sym)
                    elif not is_above and was_above:
                        newly_below[w].append(sym)

    pct = {w: (round(counts[w] / total_valid * 100, 1) if total_valid > 0 else 0.0) for w in MA_WINDOWS}

    print(f"[MA] Hoàn tất. Total valid: {total_valid}, MA20: {counts[20]}, MA50: {counts[50]}")

    return {
        "ma_total_symbols": total_valid,
        "above_ma20_count": counts[20],
        "above_ma50_count": counts[50],
        "above_ma200_count": counts[200],
        "pct_above_ma20": pct[20],
        "pct_above_ma50": pct[50],
        "pct_above_ma200": pct[200],
        "above_ma20_symbols": sorted(above_symbols[20]),
        "above_ma50_symbols": sorted(above_symbols[50]),
        "above_ma200_symbols": sorted(above_symbols[200]),
        "newly_above_ma20": sorted(newly_above[20]),
        "newly_below_ma20": sorted(newly_below[20]),
        "newly_above_ma50": sorted(newly_above[50]),
        "newly_below_ma50": sorted(newly_below[50]),
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
        "trading_date": latest.get("TradingDate"),
    }


def build_market_snapshot(client: SSIClient, market: str, today: datetime) -> dict:
    print(f"[{market}] fetching common stock symbols (loại CW/ETF/Bond)...")
    symbols = client.common_stock_symbols(market)
    print(f"[{market}] {len(symbols)} symbols. Computing A/D...")

    ad = advance_decline_for_market(client, market, today)
    print(f"[{market}] A/D done: {ad}")

    print(f"[{market}] Computing MA breadth for {len(symbols)} symbols (this can take a while)...")
    ma = ma_breadth_for_market(client, symbols, today)

    total = ad["advances"] + ad["declines"] + ad["unchanged"]
    snapshot = {
        "exchange": market,
        "date": today.strftime("%d/%m/%Y"),
        "total_symbols": total or ma["total_symbols"],
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
        "above_ma20_count": ma["above_ma20"],
        "above_ma50_count": ma["above_ma50"],
        "above_ma200_count": ma["above_ma200"],
        "ma_total_symbols": ma["total_symbols"],
        "above_ma20_symbols": ma["above_ma20_symbols"],
        "above_ma50_symbols": ma["above_ma50_symbols"],
        "newly_above_ma20": ma["newly_above_ma20"],
        "newly_below_ma20": ma["newly_below_ma20"],
        "newly_above_ma50": ma["newly_above_ma50"],
        "newly_below_ma50": ma["newly_below_ma50"],
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

    def merge(key: str) -> list[str]:
        out = []
        for s in snapshots:
            out.extend(s.get(key, []))
        return sorted(out)

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
        "above_ma20_symbols": merge("above_ma20_symbols"),
        "above_ma50_symbols": merge("above_ma50_symbols"),
        "newly_above_ma20": merge("newly_above_ma20"),
        "newly_below_ma20": merge("newly_below_ma20"),
        "newly_above_ma50": merge("newly_above_ma50"),
        "newly_below_ma50": merge("newly_below_ma50"),
    }


def append_history(snapshots_by_market: dict) -> None:
    HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if HISTORY_JSON.exists():
        history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))

    today_date = snapshots_by_market["ALL"]["date"]
    history = [h for h in history if h.get("date") != today_date]  # tránh trùng nếu chạy lại cùng ngày
    history.append({"date": today_date, "markets": snapshots_by_market})

    # Giữ tối đa ~120 phiên gần nhất để file không phình to
    history = history[-120:]
    HISTORY_JSON.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    client = SSIClient()
    today = vn_today()

    print(f"Đang cập nhật dữ liệu ngày: {today.strftime('%d/%m/%Y')}")

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
    print(f"Đã ghi file latest: {LATEST_JSON}")

    append_history(snapshots_by_market)
    print(f"Đã cập nhật history. Tổng cộng {len(HISTORY_JSON.read_text(encoding='utf-8')) if HISTORY_JSON.exists() else 0} bytes.")

    print("Hoàn tất cập nhật.")


if __name__ == "__main__":
    main()
