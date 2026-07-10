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
import os
import time
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
from _shared import tqdm

# Suppress pandas FutureWarning va Python DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ssi_client import SSIClient
from cache_utils import load_cache as _load_cache
from _shared import DATA_DIR, CACHE_DIR, DOCS_DATA_DIR
from market_commentary import generate_commentary
from strategy_signals import main as run_strategy_signals
from ensemble_signals import main as run_ensemble_signals
from backtest_weights import main as run_backtest_weights
from momentum_signals import main as run_momentum_signals
from backtest_momentum import main as run_backtest_momentum
from luc_mach_signals import main as run_luc_mach_signals
from khung4_tplus_signals import main as run_khung4_tplus_signals
from mama_positional_signals import main as run_mama_positional_signals
from advanced_trailstop_signals import main as run_advanced_trailstop_signals

LATEST_JSON = DATA_DIR / "breadth_latest.json"
HISTORY_JSON = DATA_DIR / "breadth_history.json"
COMMENTARY_JSON = DATA_DIR / "market_commentary.json"
DOCS_COMMENTARY_JSON = DOCS_DATA_DIR / "market_commentary.json"
SIGNALS_HISTORY_JSON = DATA_DIR / "signals_history.json"
DOCS_SIGNALS_HISTORY_JSON = DOCS_DATA_DIR / "signals_history.json"
LATEST_PRICES_JSON = DATA_DIR / "latest_prices.json"

MARKETS = ["HOSE", "HNX"]
MARKET_INDEX_ID = {
    "HOSE": "VNINDEX",
    "HNX": "HNXIndex",
}
MA_WINDOWS = [20, 50, 200]
HISTORY_DAYS_LOOKBACK = 800   # du ~500 phien, giup ZeroLagTEMA(65) on dinh hon cho Luc Mach
INCREMENTAL_LOOKBACK = 21     # lay 21 ngay gan nhat neu da co cache (tranh thieu sau ky nghi dai)
REQUEST_SLEEP_SEC = 0.5       # cho giua cac lan goi API de tranh rate limit                                                         
DATE_FMT = "%d/%m/%Y"
MIN_AVG_VOLUME = 300_000      # loc thanh khoan: TB 20 phien >= 300,000 cp


def vn_today() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=7)


# --- Cache OHLC ---------------------------------------------------------------

def save_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("TradingDate").drop_duplicates("TradingDate")
    ordered = [c for c in ["TradingDate", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]
    df.to_csv(CACHE_DIR / f"{symbol}.csv", index=False, date_format="%d/%m/%Y")


def generate_latest_prices() -> None:
    """Ghi map gia dong cua moi nhat tu OHLC cache cho dashboard."""
    prices = {}
    for path in CACHE_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(path)
            if df.empty or "Close" not in df.columns:
                continue
            row = df.iloc[-1]
            close = pd.to_numeric(row.get("Close"), errors="coerce")
            if pd.isna(close):
                continue
            prices[path.stem.upper()] = {
                "close": float(close),
                "date": str(row.get("TradingDate", "")),
            }
        except Exception:
            continue
    _write_json(LATEST_PRICES_JSON, prices)


def cache_max_date(symbol: str) -> datetime | None:
    """Tra ve ngay giao dich gan nhat trong cache cua symbol, None neu chua co."""
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
    cached = _load_cache(symbol, CACHE_DIR)
    if not cached.empty:
        cached = cached.dropna(subset=["TradingDate"])
        if not {"Open", "High", "Low"}.issubset(cached.columns) or cached[["Open", "High", "Low"]].tail(80).isna().any().any():
            cached = pd.DataFrame()
    cached_historical = pd.DataFrame()
    if cached.empty:
        from_date = today - timedelta(days=HISTORY_DAYS_LOOKBACK)
    else:
        latest_cached = cached["TradingDate"].max()
        if pd.isna(latest_cached):
            from_date = today - timedelta(days=HISTORY_DAYS_LOOKBACK)
        elif latest_cached.date() >= today.date():
            cached.attrs["api_called"] = False
            return cached
        else:
            cached_historical = cached[cached["TradingDate"].dt.date < today.date()]
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
            elif cl in ("open", "openprice", "open_price", "referenceopen"):
                col_map[c] = "Open"
            elif cl in ("high", "highest", "highestprice", "highprice", "high_price"):
                col_map[c] = "High"
            elif cl in ("low", "lowest", "lowestprice", "lowprice", "low_price"):
                col_map[c] = "Low"
            elif cl in ("close", "closeprice", "close_price"):
                col_map[c] = "Close"
            elif cl in ("volume", "totalvolume", "total_volume", "matchvolume",
                        "match_volume", "tradingvolume", "trading_volume", "vol"):
                col_map[c] = "Volume"
        new_df = new_df.rename(columns=col_map)

        if "TradingDate" in new_df.columns and "Close" in new_df.columns:
            cols = ["TradingDate"]
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in new_df.columns:
                    cols.append(col)
            new_df = new_df[cols]
            new_df["TradingDate"] = pd.to_datetime(new_df["TradingDate"], dayfirst=True, errors="coerce")
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in new_df.columns:
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
                else:
                    new_df[col] = float("nan")
            new_df = new_df.dropna(subset=["Close"])

            # Align columns truoc khi concat
            for col in ["Open", "High", "Low", "Volume"]:
                if col in cached_historical.columns and col not in new_df.columns:
                    new_df[col] = float("nan")
                if col in new_df.columns and col not in cached_historical.columns:
                    cached_historical[col] = float("nan")

            merged = pd.concat([cached_historical, new_df], ignore_index=True)
        else:
            tqdm.write(f"  [WARN] {symbol}: khong tim thay cot TradingDate/Close")
            merged = cached
    else:
        merged = cached

    merged = merged.sort_values("TradingDate").drop_duplicates("TradingDate").reset_index(drop=True)
    merged.attrs["api_called"] = True
    save_cache(symbol, merged)
    return merged


# --- Tinh MA ------------------------------------------------------------------
def compute_ma_breadth(client: SSIClient, symbols: list[str], today: datetime, market: str) -> dict:
    counts = {w: 0 for w in MA_WINDOWS}
    above_syms = {w: [] for w in MA_WINDOWS}
    newly_above = {20: [], 50: []}
    newly_below = {20: [], 50: []}
    volume_breakout = []
    total_valid = 0
    skipped_volume = 0
    skipped_data = 0
    volume_threshold = float(os.environ.get("VOLUME_BREAKOUT_CLOSE", "1.3"))

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
            if df.attrs.get("api_called"):
                time.sleep(REQUEST_SLEEP_SEC)
        except Exception as e:
            tqdm.write(f"  [WARN] {sym}: loi khi tai OHLC: {e}")
            continue

        if df.empty or len(df) < 20:
            skipped_data += 1
            continue

        avg_vol = None
        volume_today = None
        if "Volume" in df.columns:
            recent_vol = df["Volume"].iloc[-20:]
            avg_vol = recent_vol.dropna().mean()
            volume_today = df["Volume"].iloc[-1]
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
                                        # Volume breakout: >= MA20 + volume > nguong * TB 20 phien
                    if w == 20 and avg_vol is not None and volume_today is not None and avg_vol > 0 and volume_today > avg_vol * volume_threshold:
                        volume_breakout.append(sym)

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
    tqdm.write(f"[{market}] Volume breakout={len(volume_breakout)} (nguong={volume_threshold}x TB20)")

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
        "volume_breakout_symbols": sorted(volume_breakout),
        "volume_breakout_count": len(volume_breakout),
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
        print(f"[{market}] WARN: API Securities tra ve 0 ma, fallback sang cache...")
        from _shared import list_symbols as _list_symbols
        cached = _list_symbols(CACHE_DIR, min_history=20)
        symbols = [s for s in cached if not any(c.isdigit() for c in s) and len(s) <= 3]
        if symbols:
            print(f"[{market}] Fallback: lay {len(symbols)} ma tu cache")
        else:
            print(f"[{market}] WARN: cache cung khong co ma hop le!")

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
        "volume_breakout_symbols": ma["volume_breakout_symbols"],
        "volume_breakout_count": ma["volume_breakout_count"],
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

    volume_breakout = merge("volume_breakout_symbols")

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
        "volume_breakout_symbols": volume_breakout,
        "volume_breakout_count": len(volume_breakout),
    }


# --- History ------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_docs_data():
    """Dong bo du lieu sang docs/data/ cho GitHub Pages."""
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("breadth_latest.json", "breadth_history.json", "market_commentary.json", "strategy_signals.json", "ensemble_signals.json", "backtest_weights.json", "momentum_signals.json", "backtest_momentum.json", "backtest_mama_positional.json", "backtest_advanced_trailstop.json", "luc_mach_signals.json", "khung4_tplus_signals.json", "mama_positional_signals.json", "advanced_trailstop_signals.json", "latest_prices.json"):
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


def append_signals_history() -> None:
    strategy_path = DATA_DIR / "strategy_signals.json"
    ensemble_path = DATA_DIR / "ensemble_signals.json"
    momentum_path = DATA_DIR / "momentum_signals.json"
    luc_mach_path = DATA_DIR / "luc_mach_signals.json"
    khung4_path = DATA_DIR / "khung4_tplus_signals.json"
    mama_path = DATA_DIR / "mama_positional_signals.json"
    ats_path = DATA_DIR / "advanced_trailstop_signals.json"
    if not strategy_path.exists() and not ensemble_path.exists() and not momentum_path.exists() and not luc_mach_path.exists() and not khung4_path.exists() and not mama_path.exists() and not ats_path.exists():
        return

    history = []
    if SIGNALS_HISTORY_JSON.exists():
        try:
            history = json.loads(SIGNALS_HISTORY_JSON.read_text(encoding="utf-8"))
        except Exception:
            history = []

    entry = {"date": "", "strategy": None, "ensemble": None, "momentum": None, "luc_mach": None, "khung4_tplus": None, "mama_positional": None, "advanced_trailstop": None}
    if strategy_path.exists():
        data = json.loads(strategy_path.read_text(encoding="utf-8"))
        entry["date"] = data.get("date", "")
        entry["strategy"] = data
    if ensemble_path.exists():
        data = json.loads(ensemble_path.read_text(encoding="utf-8"))
        entry["date"] = entry["date"] or data.get("date", "")
        entry["ensemble"] = data

    if momentum_path.exists():
        data = json.loads(momentum_path.read_text(encoding="utf-8"))
        entry["date"] = entry["date"] or data.get("date", "")
        entry["momentum"] = data

    if luc_mach_path.exists():
        data = json.loads(luc_mach_path.read_text(encoding="utf-8"))
        entry["date"] = entry["date"] or data.get("date", "")
        entry["luc_mach"] = data

    if khung4_path.exists():
        data = json.loads(khung4_path.read_text(encoding="utf-8"))
        entry["date"] = entry["date"] or data.get("date", "")
        entry["khung4_tplus"] = data

    if mama_path.exists():
        data = json.loads(mama_path.read_text(encoding="utf-8"))
        entry["date"] = entry["date"] or data.get("date", "")
        entry["mama_positional"] = data

    if ats_path.exists():
        data = json.loads(ats_path.read_text(encoding="utf-8"))
        entry["date"] = entry["date"] or data.get("date", "")
        entry["advanced_trailstop"] = data

    if not entry["date"]:
        return

    history = [h for h in history if h.get("date") != entry["date"]]
    history.append(entry)
    history = history[-365:]

    _write_json(SIGNALS_HISTORY_JSON, history)
    DOCS_SIGNALS_HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_SIGNALS_HISTORY_JSON.write_bytes(SIGNALS_HISTORY_JSON.read_bytes())
    print(f"Da cap nhat signals_history.json ({len(history)} ngay).")


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
        "session": "close",
        "markets": markets_dict,
    }
    _write_json(LATEST_JSON, output)
    generate_latest_prices()
    _sync_docs_data()
    print(f"\nDa ghi: {LATEST_JSON}")

    append_history(markets_dict)
    _sync_docs_data()
    print(f"Da cap nhat history.")

    # Generate market commentary
    try:
        commentary_text = generate_commentary(output)
        commentary_output = {
            "generated_at": datetime.now().isoformat(),
            "session": "close",
            "date": output["markets"]["ALL"]["date"],
            "content": commentary_text,
        }
        _write_json(DATA_DIR / "market_commentary.json", commentary_output)
        _write_json(DOCS_DATA_DIR / "market_commentary.json", commentary_output)
        print(f"Da ghi nhan xet thi truong.")
    except Exception as e:
        print(f"Loi sinh nhan xet: {e}")

    # Generate strategy signals
    try:
        run_strategy_signals()
        print(f"Da ghi tin hieu pre-breakout.\n")
    except Exception as e:
        print(f"Loi sinh tin hieu pre-breakout: {e}")

    try:
        run_backtest_weights()
        print(f"Da cap nhat backtest weights.\n")
    except Exception as e:
        print(f"Loi backtest weights: {e}")

    try:
        run_ensemble_signals()
        print(f"Da ghi tin hieu ensemble.\n")
    except Exception as e:
        print(f"Loi sinh tin hieu ensemble: {e}")

    try:
        run_momentum_signals()
        print(f"Da ghi tin hieu momentum.\n")
    except Exception as e:
        print(f"Loi sinh tin hieu momentum: {e}")

    try:
        run_luc_mach_signals()
        print(f"Da ghi tin hieu Luc Mach.\n")
    except Exception as e:
        print(f"Loi sinh tin hieu Luc Mach: {e}")

    try:
        run_khung4_tplus_signals()
        print(f"Da ghi tin hieu Khung4/Tplus.\n")
    except Exception as e:
        print(f"Loi sinh tin hieu Khung4/Tplus: {e}")

    try:
        run_mama_positional_signals()
        print(f"Da ghi tin hieu MAMA Positional.\n")
    except Exception as e:
        print(f"Loi sinh tin hieu MAMA Positional: {e}")

    try:
        run_advanced_trailstop_signals()
        print(f"Da ghi tin hieu Advanced Trailstop.\n")
    except Exception as e:
        print(f"Loi sinh tin hieu Advanced Trailstop: {e}")

    try:
        run_backtest_momentum()
        print(f"Da ghi backtest momentum.\n")
    except Exception as e:
        print(f"Loi backtest momentum: {e}")

    append_signals_history()

    print("\nHoan tat.")


if __name__ == "__main__":
    main()
