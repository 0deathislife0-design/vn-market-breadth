"""
Pipeline chính: chạy hàng ngày qua GitHub Actions.
  1. Lấy danh sách mã theo sàn (HOSE/HNX/UPCOM)
  2. Cập nhật cache OHLC cục bộ cho từng mã
  3. Tính MA20/MA50/MA200
  4. Lấy Advances/Declines/Unchanged từ DailyIndex
  5. Ghi ra data/breadth_latest.json + data/breadth_history.json
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
MARKET_INDEX_ID = {
    "HOSE": "VNINDEX",
    "HNX": "HNXIndex",
    "UPCOM": "UPCOMIndex",
}
MA_WINDOWS = [20, 50, 200]
HISTORY_DAYS_LOOKBACK = 380   # đủ ~260 phiên để tính MA200
INCREMENTAL_LOOKBACK = 7      # chỉ lấy 7 ngày gần nhất nếu đã có cache
REQUEST_SLEEP_SEC = 0.3       # tránh rate limit
DATE_FMT = "%d/%m/%Y"


def vn_today() -> datetime:
    return datetime.utcnow() + timedelta(hours=7)


# ─── Cache OHLC ───────────────────────────────────────────────

def load_cache(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["TradingDate"], dayfirst=True)
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            return df.dropna(subset=["Close"])
        except Exception:
            pass
    return pd.DataFrame(columns=["TradingDate", "Close"])


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("TradingDate").drop_duplicates("TradingDate")
    df.to_csv(CACHE_DIR / f"{symbol}.csv", index=False)


def update_ohlc(client: SSIClient, symbol: str, today: datetime) -> pd.DataFrame:
    cached = load_cache(symbol)
    if cached.empty:
        from_date = today - timedelta(days=HISTORY_DAYS_LOOKBACK)
    else:
        from_date = today - timedelta(days=INCREMENTAL_LOOKBACK)

    rows = client.daily_ohlc(
        symbol,
        from_date.strftime(DATE_FMT),
        today.strftime(DATE_FMT),
    )

    if rows:
        new_df = pd.DataFrame(rows)
        # Đổi tên cột nếu API trả về tên khác
        col_map = {}
        for c in new_df.columns:
            if c.lower() in ("tradingdate", "trading_date", "date"):
                col_map[c] = "TradingDate"
            if c.lower() in ("close", "closeprice", "close_price"):
                col_map[c] = "Close"
        new_df = new_df.rename(columns=col_map)

        if "TradingDate" in new_df.columns and "Close" in new_df.columns:
            new_df = new_df[["TradingDate", "Close"]]
            new_df["TradingDate"] = pd.to_datetime(new_df["TradingDate"], dayfirst=True, errors="coerce")
            new_df["Close"] = pd.to_numeric(new_df["Close"], errors="coerce")
            new_df = new_df.dropna()
            merged = pd.concat([cached, new_df], ignore_index=True)
        else:
            print(f"  [WARN] {symbol}: không tìm thấy cột TradingDate/Close, columns={list(new_df.columns)}")
            merged = cached
    else:
        merged = cached

    merged = merged.sort_values("TradingDate").drop_duplicates("TradingDate").reset_index(drop=True)
    save_cache(symbol, merged)
    return merged


# ─── Tính MA ──────────────────────────────────────────────────

def compute_ma_breadth(client: SSIClient, symbols: list[str], today: datetime) -> dict:
    counts = {w: 0 for w in MA_WINDOWS}
    above_syms = {w: [] for w in MA_WINDOWS}
    newly_above = {20: [], 50: []}
    newly_below = {20: [], 50: []}
    total_valid = 0

    print(f"[MA] Bắt đầu tính cho {len(symbols)} mã...")

    for i, sym in enumerate(symbols):
        if i % 50 == 0:
            print(f"[MA] Đang xử lý {i}/{len(symbols)}...")

        try:
            df = update_ohlc(client, sym, today)
            time.sleep(REQUEST_SLEEP_SEC)
        except Exception as e:
            print(f"  [WARN] {sym}: lỗi khi tải OHLC: {e}")
            continue

        if df.empty or len(df) < 20:
            continue

        close = df["Close"].values
        last_close = close[-1]
        total_valid += 1

        for w in MA_WINDOWS:
            if len(close) >= w:
                ma_val = close[-w:].mean()
                is_above = last_close >= ma_val

                if is_above:
                    counts[w] += 1
                    above_syms[w].append(sym)

                # Tín hiệu chuyển đổi MA20/MA50
                if w in (20, 50) and len(close) >= w + 1:
                    prev_close = close[-2]
                    prev_ma = close[-(w + 1):-1].mean()
                    was_above = prev_close >= prev_ma
                    if is_above and not was_above:
                        newly_above[w].append(sym)
                    elif not is_above and was_above:
                        newly_below[w].append(sym)

    pct = {
        w: round(counts[w] / total_valid * 100, 1) if total_valid > 0 else 0.0
        for w in MA_WINDOWS
    }

    print(f"[MA] Xong. Valid={total_valid}, MA20={counts[20]}, MA50={counts[50]}, MA200={counts[200]}")

    return {
        "ma_total_symbols":   total_valid,
        "above_ma20_count":   counts[20],
        "above_ma50_count":   counts[50],
        "above_ma200_count":  counts[200],
        "pct_above_ma20":     pct[20],
        "pct_above_ma50":     pct[50],
        "pct_above_ma200":    pct[200],
        "above_ma20_symbols":  sorted(above_syms[20]),
        "above_ma50_symbols":  sorted(above_syms[50]),
        "above_ma200_symbols": sorted(above_syms[200]),
        "newly_above_ma20":   sorted(newly_above[20]),
        "newly_below_ma20":   sorted(newly_below[20]),
        "newly_above_ma50":   sorted(newly_above[50]),
        "newly_below_ma50":   sorted(newly_below[50]),
    }


# ─── A/D Ratio ────────────────────────────────────────────────

def get_advance_decline(client: SSIClient, market: str, today: datetime) -> dict:
    index_id = MARKET_INDEX_ID[market]
    from_date = today - timedelta(days=7)
    rows = client.daily_index(
        index_id,
        from_date.strftime(DATE_FMT),
        today.strftime(DATE_FMT),
    )
    if not rows:
        print(f"[{market}] WARN: daily_index trả về rỗng")
        return {"advances": 0, "declines": 0, "unchanged": 0, "ad_ratio": None}

    latest = rows[-1]
    print(f"[{market}] DailyIndex row: {latest}")

    adv = int(float(latest.get("Advances") or latest.get("advances") or 0))
    dec = int(float(latest.get("Declines") or latest.get("declines") or 0))
    unc = int(float(
        latest.get("Nochanges") or latest.get("NoChanges") or
        latest.get("nochanges") or 0
    ))
    ad_ratio = round(adv / dec, 2) if dec else None

    return {
        "advances": adv,
        "declines": dec,
        "unchanged": unc,
        "ad_ratio": ad_ratio,
        "trading_date": latest.get("TradingDate") or latest.get("tradingDate"),
    }


# ─── Snapshot mỗi sàn ─────────────────────────────────────────

def build_snapshot(client: SSIClient, market: str, today: datetime) -> dict:
    print(f"\n{'='*40}")
    print(f"[{market}] Bắt đầu xử lý...")

    symbols = client.common_stock_symbols(market)
    if not symbols:
        print(f"[{market}] WARN: không lấy được mã nào!")

    ad = get_advance_decline(client, market, today)
    print(f"[{market}] A/D: {ad}")

    ma = compute_ma_breadth(client, symbols, today)

    total_ad = ad["advances"] + ad["declines"] + ad["unchanged"]

    return {
        "exchange":        market,
        "date":            today.strftime(DATE_FMT),
        "total_symbols":   total_ad or ma["ma_total_symbols"],
        "advances":        ad["advances"],
        "declines":        ad["declines"],
        "unchanged":       ad["unchanged"],
        "advances_pct":    round(ad["advances"] / total_ad * 100, 1) if total_ad else 0.0,
        "declines_pct":    round(ad["declines"] / total_ad * 100, 1) if total_ad else 0.0,
        "unchanged_pct":   round(ad["unchanged"] / total_ad * 100, 1) if total_ad else 0.0,
        "ad_ratio":        ad["ad_ratio"],
        "pct_above_ma20":  ma["pct_above_ma20"],
        "pct_above_ma50":  ma["pct_above_ma50"],
        "pct_above_ma200": ma["pct_above_ma200"],
        "above_ma20_count":   ma["above_ma20_count"],
        "above_ma50_count":   ma["above_ma50_count"],
        "above_ma200_count":  ma["above_ma200_count"],
        "ma_total_symbols":   ma["ma_total_symbols"],
        "above_ma20_symbols":  ma["above_ma20_symbols"],
        "above_ma50_symbols":  ma["above_ma50_symbols"],
        "above_ma200_symbols": ma["above_ma200_symbols"],
        "newly_above_ma20":   ma["newly_above_ma20"],
        "newly_below_ma20":   ma["newly_below_ma20"],
        "newly_above_ma50":   ma["newly_above_ma50"],
        "newly_below_ma50":   ma["newly_below_ma50"],
    }


# ─── Gộp ALL ──────────────────────────────────────────────────

def combine_all(snapshots: list[dict], today: datetime) -> dict:
    adv  = sum(s["advances"] for s in snapshots)
    dec  = sum(s["declines"] for s in snapshots)
    unc  = sum(s["unchanged"] for s in snapshots)
    total = adv + dec + unc
    ma20  = sum(s["above_ma20_count"] for s in snapshots)
    ma50  = sum(s["above_ma50_count"] for s in snapshots)
    ma200 = sum(s["above_ma200_count"] for s in snapshots)
    ma_tot = sum(s["ma_total_symbols"] for s in snapshots)

    def merge(key):
        out = []
        for s in snapshots:
            out.extend(s.get(key, []))
        return sorted(out)

    return {
        "exchange":        "ALL",
        "date":            today.strftime(DATE_FMT),
        "total_symbols":   total,
        "advances":        adv,
        "declines":        dec,
        "unchanged":       unc,
        "advances_pct":    round(adv / total * 100, 1) if total else 0.0,
        "declines_pct":    round(dec / total * 100, 1) if total else 0.0,
        "unchanged_pct":   round(unc / total * 100, 1) if total else 0.0,
        "ad_ratio":        round(adv / dec, 2) if dec else None,
        "pct_above_ma20":  round(ma20 / ma_tot * 100, 1) if ma_tot else 0.0,
        "pct_above_ma50":  round(ma50 / ma_tot * 100, 1) if ma_tot else 0.0,
        "pct_above_ma200": round(ma200 / ma_tot * 100, 1) if ma_tot else 0.0,
        "above_ma20_count":   ma20,
        "above_ma50_count":   ma50,
        "above_ma200_count":  ma200,
        "ma_total_symbols":   ma_tot,
        "above_ma20_symbols":  merge("above_ma20_symbols"),
        "above_ma50_symbols":  merge("above_ma50_symbols"),
        "above_ma200_symbols": merge("above_ma200_symbols"),
        "newly_above_ma20":   merge("newly_above_ma20"),
        "newly_below_ma20":   merge("newly_below_ma20"),
        "newly_above_ma50":   merge("newly_above_ma50"),
        "newly_below_ma50":   merge("newly_below_ma50"),
    }


# ─── History ──────────────────────────────────────────────────

def append_history(markets_dict: dict) -> None:
    HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if HISTORY_JSON.exists():
        try:
            history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except Exception:
            history = []

    today_date = markets_dict["ALL"]["date"]
    history = [h for h in history if h.get("date") != today_date]
    history.append({"date": today_date, "markets": markets_dict})
    history = history[-120:]  # giữ 120 phiên gần nhất

    HISTORY_JSON.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─── Main ─────────────────────────────────────────────────────

def main():
    client = SSIClient()
    today = vn_today()
    print(f"Ngày xử lý: {today.strftime(DATE_FMT)}")

    markets_dict = {}
    all_list = []

    for market in MARKETS:
        snap = build_snapshot(client, market, today)
        markets_dict[market] = snap
        all_list.append(snap)

    all_snap = combine_all(all_list, today)
    markets_dict["ALL"] = all_snap

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": today.isoformat(),
        "markets": markets_dict,
    }
    LATEST_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nĐã ghi: {LATEST_JSON}")

    append_history(markets_dict)
    print(f"Đã cập nhật history.")
    print("\nHoàn tất.")


if __name__ == "__main__":
    main()
