"""
Pipeline chinh: chay hang ngay qua GitHub Actions.
  1. Lay danh sach ma theo san (HOSE/HNX/UPCOM)
     - Bo ma co chu so trong ten (CW, phai sinh, ...)
  2. Cap nhat cache OHLC cuc bo cho tung ma (Close + Volume)
  3. Loc thanh khoan: bo ma co KL khop TB 20 phien < 300,000 cp
  4. Tinh MA20/MA50/MA200
  5. Lay Advances/Declines/Unchanged tu DailyIndex
  6. Ghi ra data/breadth_latest.json + data/breadth_history.json
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    # Fallback khi chay tren CI/GitHub Actions khong co tqdm
    _HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self._it = iterable
            self._desc = kwargs.get('desc', '')
            self._total = kwargs.get('total', None)
            try:
                self._total = self._total or len(iterable)
            except Exception:
                self._total = '?'
            self._n = 0
            print(f"{self._desc}: 0/{self._total}")
        def __iter__(self):
            for item in self._it:
                yield item
                self._n += 1
                if self._n % 50 == 0:
                    print(f"{self._desc}: {self._n}/{self._total}")
        def set_postfix_str(self, s, **kw): pass
        def close(self):
            print(f"{self._desc}: {self._n}/{self._total} - Done")
        @staticmethod
        def write(msg): print(msg)

# Suppress pandas FutureWarning va Python DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ssi_client import SSIClient

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "ohlc_cache"
DOCS_DATA_DIR = ROOT / "docs" / "data"
LATEST_JSON = DATA_DIR / "breadth_latest.json"
HISTORY_JSON = DATA_DIR / "breadth_history.json"

MARKETS = ["HOSE", "HNX"]
MARKET_INDEX_ID = {
    "HOSE": "VNINDEX",
    "HNX": "HNXIndex",
}
MA_WINDOWS = [20, 50, 200]
HISTORY_DAYS_LOOKBACK = 380   # du ~260 phien de tinh MA200
INCREMENTAL_LOOKBACK = 21     # lay 21 ngay gan nhat neu da co cache (tránh thiếu sau kỳ nghỉ dài)
REQUEST_SLEEP_SEC = 0.5       # chờ giữa các lần gọi API để tránh rate limit
DATE_FMT = "%d/%m/%Y"
MIN_AVG_VOLUME = 300_000      # loc thanh khoan: TB 20 phien >= 300,000 cp


def vn_today() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=7)


# --- Cache OHLC ---------------------------------------------------------------

def load_cache(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            df["TradingDate"] = pd.to_datetime(df["TradingDate"], format="mixed", dayfirst=True, errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
            else:
                df["Volume"] = float("nan")
            return df.dropna(subset=["Close"])
        except Exception:
            pass
    return pd.DataFrame(columns=["TradingDate", "Close", "Volume"])


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("TradingDate").drop_duplicates("TradingDate")
    df.to_csv(CACHE_DIR / f"{symbol}.csv", index=False)


def cache_max_date(symbol: str) -> datetime | None:
    """Trả về ngày giao dịch gần nhất trong cache của symbol, None nếu chưa có."""
    path = CACHE_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    try:
        dates = pd.read_csv(path, usecols=["TradingDate"])["TradingDate"]
        max_dt = pd.to_datetime(dates, dayfirst=True, errors="coerce").max()
        if pd.isna(max_dt):
            return None
        return max_dt
    except Exception:
        return None


def update_ohlc(client: SSIClient, symbol: str, today: datetime) -> pd.DataFrame:
    cached = load_cache(symbol)
    if cached.empty:
        from_date = today - timedelta(days=HISTORY_DAYS_LOOKBACK)
    else:
        # Nếu cache đã có dữ liệu đến hôm nay thì không cần gọi API
        latest_cached = cached["TradingDate"].max()
        if latest_cached is not None and latest_cached.date() >= today.date():
            return cached
        from_date = today - timedelta(days=INCREMENTAL_LOOKBACK)

    rows = client.daily_ohlc(
        symbol,
        from_date.strftime(DATE_FMT),
        today.strftime(DATE_FMT),
    )

    if rows:
        new_df = pd.DataFrame(rows)
        col_map = {}
        for c in new_df.columns:
            cl = c.lower()
            if cl in ("tradingdate", "trading_date", "date"):
                col_map[c] = "TradingDate"
            elif cl in ("close", "closeprice", "close_price"):
                col_map[c] = "Close"
            elif cl in ("volume", "totalvolume", "total_volume", "matchvolume",
                        "match_volume", "tradingvolume", "trading_volume", "vol"):
                col_map[c] = "Volume"
        new_df = new_df.rename(columns=col_map)

        if "TradingDate" in new_df.columns and "Close" in new_df.columns:
            cols = ["TradingDate", "Close"]
            if "Volume" in new_df.columns:
                cols.append("Volume")
            new_df = new_df[cols]
            new_df["TradingDate"] = pd.to_datetime(new_df["TradingDate"], dayfirst=True, errors="coerce")
            new_df["Close"] = pd.to_numeric(new_df["Close"], errors="coerce")
            if "Volume" in new_df.columns:
                new_df["Volume"] = pd.to_numeric(new_df["Volume"], errors="coerce")
            else:
                new_df["Volume"] = float("nan")
            new_df = new_df.dropna(subset=["Close"])

            # Align columns truoc khi concat
            for col in ["Volume"]:
                if col in cached.columns and col not in new_df.columns:
                    new_df[col] = float("nan")
                if col in new_df.columns and col not in cached.columns:
                    cached[col] = float("nan")

            merged = pd.concat([cached, new_df], ignore_index=True)
        else:
            tqdm.write(f"  [WARN] {symbol}: khong tim thay cot TradingDate/Close")
            merged = cached
    else:
        merged = cached

    merged = merged.sort_values("TradingDate").drop_duplicates("TradingDate").reset_index(drop=True)
    save_cache(symbol, merged)
    return merged


# --- Tinh MA ------------------------------------------------------------------
def compute_ma_breadth(client: SSIClient, symbols: list[str], today: datetime, market: str) -> dict:
    counts = {w: 0 for w in MA_WINDOWS}
    above_syms = {w: [] for w in MA_WINDOWS}
    newly_above = {20: [], 50: []}
    newly_below = {20: [], 50: []}
    total_valid = 0
    skipped_volume = 0
    skipped_data = 0

    bar = tqdm(
        symbols,
        desc=f"[{market}] OHLC+MA",
        unit="ma",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=False,
    )

    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)

        try:
            df = update_ohlc(client, sym, today)
            time.sleep(REQUEST_SLEEP_SEC)
        except Exception as e:
            tqdm.write(f"  [WARN] {sym}: loi khi tai OHLC: {e}")
            continue

        if df.empty or len(df) < 20:
            skipped_data += 1
            continue

        if "Volume" in df.columns:
            recent_vol = df["Volume"].iloc[-20:]
            avg_vol = recent_vol.dropna().mean()
            if pd.isna(avg_vol) or avg_vol < MIN_AVG_VOLUME:
                skipped_volume += 1
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

                if w in (20, 50) and len(close) >= w + 1:
                    prev_close = close[-2]
                    prev_ma = close[-(w + 1):-1].mean()
                    was_above = prev_close >= prev_ma
                    if is_above and not was_above:
                        newly_above[w].append(sym)
                    elif not is_above and was_above:
                        newly_below[w].append(sym)

    bar.close()

    pct = {
        w: round(counts[w] / total_valid * 100, 1) if total_valid > 0 else 0.0
        for w in MA_WINDOWS
    }

    tqdm.write(f"[{market}] Xong: valid={total_valid} | bo_kl={skipped_volume} | it_data={skipped_data}")
    tqdm.write(f"[{market}] MA20={counts[20]} ({pct[20]}%) | MA50={counts[50]} ({pct[50]}%) | MA200={counts[200]} ({pct[200]}%)")

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


# --- A/D Ratio ----------------------------------------------------------------

def get_advance_decline(client: SSIClient, market: str, today: datetime) -> dict:
    index_id = MARKET_INDEX_ID[market]
    from_date = today - timedelta(days=7)
    rows = client.daily_index(
        index_id,
        from_date.strftime(DATE_FMT),
        today.strftime(DATE_FMT),
    )
    if not rows:
        print(f"[{market}] WARN: daily_index tra ve rong")
        return {"advances": 0, "declines": 0, "unchanged": 0, "ad_ratio": None}

    latest = rows[-1]
    print(f"[{market}] DailyIndex: {latest}")

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


# --- Snapshot moi san ---------------------------------------------------------

def build_snapshot(client: SSIClient, market: str, today: datetime) -> dict:
    print(f"\n{'='*50}")
    print(f"[{market}] Bat dau xu ly...")

    symbols = client.common_stock_symbols(market)
    if not symbols:
        print(f"[{market}] WARN: khong lay duoc ma nao!")

    ad = get_advance_decline(client, market, today)
    print(f"[{market}] A/D: adv={ad['advances']} dec={ad['declines']} unc={ad['unchanged']}")

    ma = compute_ma_breadth(client, symbols, today, market)

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


# --- Gop ALL ------------------------------------------------------------------

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


# --- History ------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_docs_data():
    """Đồng bộ dữ liệu sang docs/data/ cho GitHub Pages."""
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("breadth_latest.json", "breadth_history.json"):
        src = DATA_DIR / f
        dst = DOCS_DATA_DIR / f
        if src.exists():
            dst.write_bytes(src.read_bytes())


def append_history(markets_dict: dict) -> None:
    history = []
    if HISTORY_JSON.exists():
        try:
            history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except Exception:
            history = []

    today_date = markets_dict["ALL"]["date"]
    history = [h for h in history if h.get("date") != today_date]
    history.append({"date": today_date, "markets": markets_dict})
    history = history[-120:]  # giu 120 phien gan nhat

    _write_json(HISTORY_JSON, history)


# --- Main ---------------------------------------------------------------------

def main():
    client = SSIClient()
    today = vn_today()
    print(f"Ngay xu ly: {today.strftime(DATE_FMT)}")
    print(f"Nguong thanh khoan: TB 20 phien >= {MIN_AVG_VOLUME:,} cp\n")

    markets_dict = {}
    all_list = []

    for market in MARKETS:
        snap = build_snapshot(client, market, today)
        markets_dict[market] = snap
        all_list.append(snap)

    all_snap = combine_all(all_list, today)
    markets_dict["ALL"] = all_snap

    output = {
        "generated_at": today.isoformat(),
        "markets": markets_dict,
    }
    _write_json(LATEST_JSON, output)
    _sync_docs_data()
    print(f"\nDa ghi: {LATEST_JSON}")

    append_history(markets_dict)
    _sync_docs_data()
    print(f"Da cap nhat history.")
    print("\nHoan tat.")


if __name__ == "__main__":
    main()