"""
Post-session strategy: Pre-breakout detection.
  1. Accumulation base detection (range + structure)
  2. CMF(21) money flow
  3. Vol-regime: ratio, vol-of-vol, skew, kurtosis -> z-score gradient
  4. Ranked signal list
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self._it = iterable; self._n = 0
            print(f"{kwargs.get('desc','')}: 0/{len(iterable)}")
        def __iter__(self):
            for item in self._it: yield item; self._n += 1
            if self._n % 50 == 0: print(f"  {self._n}/{len(self._it)}")
        def set_postfix_str(self, s, **kw): pass
        def close(self): print(f"  {self._n}/{len(self._it)} - Done")
        @staticmethod
        def write(msg): print(msg)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "ohlc_cache"
DOCS_DATA_DIR = ROOT / "docs" / "data"
SIGNALS_JSON = DATA_DIR / "strategy_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "strategy_signals.json"

BASE_LOOKBACK = 60
MIN_BASE_SESSIONS = 15
MAX_BASE_RANGE_PCT = 15.0
HISTORY_LOOKBACK = 120
MIN_AVG_VOLUME = 300_000


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
            if "High" not in df.columns or df["High"].isna().all():
                df["High"] = float("nan")
            if "Low" not in df.columns or df["Low"].isna().all():
                df["Low"] = float("nan")
            return df.dropna(subset=["Close"])
        except Exception:
            pass
    return pd.DataFrame(columns=["TradingDate", "Close", "Volume", "High", "Low"])


def detect_base(df: pd.DataFrame) -> float:
    """Tra ve base_score [0-1]: 0 = khong co base, 1 = base hoan hao."""
    if len(df) < BASE_LOOKBACK:
        return 0.0
    recent = df.iloc[-MIN_BASE_SESSIONS:]
    lookback = df.iloc[-BASE_LOOKBACK:]

    base_high = recent["Close"].max()
    base_low = recent["Close"].min()
    base_center = (base_high + base_low) / 2
    range_pct = (base_high - base_low) / base_center * 100

    if range_pct > MAX_BASE_RANGE_PCT:
        return 0.0

    # Score: narrower range = better
    range_score = max(0, 1 - range_pct / MAX_BASE_RANGE_PCT)

    # Volume dry-up trong base
    vol_series = recent["Volume"].dropna()
    if len(vol_series) >= 10:
        vol_trend = vol_series.iloc[-1] / vol_series.iloc[:10].mean()
        vol_score = max(0, 1 - vol_trend)  # KL cang thap cang tot
    else:
        vol_score = 0.0

    # Higher lows trong base
    if len(recent) >= 10:
        lows = recent["Close"].values
        n_higher = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        hilo_score = n_higher / len(lows)
    else:
        hilo_score = 0.0

    # Touch count: so lan cham vung support/resistance
    support = recent["Close"].rolling(10).min()
    resistance = recent["Close"].rolling(10).max()
    touch_sup = sum(1 for _, r in recent.iterrows() if abs(r["Close"] - support.loc[r.name]) / r["Close"] < 0.015)
    touch_res = sum(1 for _, r in recent.iterrows() if abs(r["Close"] - resistance.loc[r.name]) / r["Close"] < 0.015)
    touch_score = min(1.0, (touch_sup + touch_res) / 10)

    score = 0.35 * range_score + 0.25 * vol_score + 0.20 * hilo_score + 0.20 * touch_score
    return round(score, 3)


def compute_money_flow(df: pd.DataFrame) -> dict:
    """Money flow indicators chi can Close + Volume (khong can High/Low).
    Tra ve OBV slope + Volume Price Trend + Cumulative Money Flow."""
    if len(df) < 30:
        return {"obv_score": 0, "vpt_score": 0, "cmf_approx": 0}

    close = df["Close"].values
    volume = df["Volume"].values

    # OBV: On Balance Volume
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    # OBV slope 10 phien gan nhat
    obv_slope = (obv[-1] - obv[-10]) / max(obv[-10], 1)
    obv_score = min(1.0, max(-1.0, obv_slope / 0.1))

    # VPT: Volume Price Trend (tich luy)
    vpt = np.zeros(len(close))
    for i in range(1, len(close)):
        pct_change = (close[i] - close[i-1]) / close[i-1]
        vpt[i] = vpt[i-1] + volume[i] * pct_change

    vpt_slope = (vpt[-1] - vpt[-10]) / max(abs(vpt[-10]), 1)
    vpt_score = min(1.0, max(-1.0, vpt_slope / 10000))

    # CMF approx: tuong quan giua gia va volume trong 21 phien
    close_recent = close[-21:]
    vol_recent = volume[-21:]
    if len(close_recent) >= 21 and np.std(close_recent) > 0 and np.std(vol_recent) > 0:
        corr = np.corrcoef(close_recent, vol_recent)[0, 1]
        cmf_approx = round(corr, 4) if not np.isnan(corr) else 0
    else:
        cmf_approx = 0

    return {"obv_score": round(obv_score, 3), "vpt_score": round(vpt_score, 3), "cmf_approx": cmf_approx}


def compute_vol_regime(df: pd.DataFrame) -> dict:
    """Vol-regime metrics: short/long vol ratio, vol-of-vol, skew, kurtosis."""
    if len(df) < 63:
        return {"vol_ratio": None, "vol_of_vol": None, "skew": None, "kurtosis": None}

    log_rets = np.log(df["Close"].values[1:] / df["Close"].values[:-1])
    vol_short = np.std(log_rets[-10:]) * np.sqrt(252)
    vol_long = np.std(log_rets[-63:]) * np.sqrt(252)
    vol_ratio = vol_short / vol_long if vol_long > 0 else None

    # Vol-of-vol: bien dong cua short vol
    if len(log_rets) >= 30:
        rolling_vol = pd.Series(log_rets).rolling(10).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.dropna().iloc[-20:].std()
        vol_of_vol = round(vol_of_vol, 6) if not np.isnan(vol_of_vol) else None
    else:
        vol_of_vol = None

    # Skew va Kurtosis cua 63 phien
    rets_63 = log_rets[-63:]
    skew = round(float(pd.Series(rets_63).skew()), 4) if len(rets_63) >= 10 else None
    kurt = round(float(pd.Series(rets_63).kurtosis()), 4) if len(rets_63) >= 10 else None

    return {"vol_ratio": vol_ratio, "vol_of_vol": vol_of_vol, "skew": skew, "kurtosis": kurt}


def compute_zscore_gradient(symbols: list[str], market: str) -> dict:
    """Compute z-score and gradient cua vol-regime cho tung symbol."""
    results = {}

    bar = tqdm(symbols, desc=f"[{market}] Z-score", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        df = load_cache(sym)
        if len(df) < 63:
            continue

        mf = compute_money_flow(df)
        base_score = detect_base(df)
        vol = compute_vol_regime(df)

        # Vol gradient: toc do thay doi cua vol ratio 5 phien
        log_rets = np.log(df["Close"].values[1:] / df["Close"].values[:-1])
        rolling_vols = pd.Series(log_rets).rolling(10).std().dropna()
        if len(rolling_vols) >= 10:
            recent_ratio = rolling_vols.iloc[-1] / rolling_vols.iloc[-64:-1].mean() if len(rolling_vols) > 63 else 1.0
            prev_ratio = rolling_vols.iloc[-6] / rolling_vols.iloc[-64:-6].mean() if len(rolling_vols) > 63 else 1.0
            vol_gradient = (recent_ratio - prev_ratio) / prev_ratio if prev_ratio != 0 else 0
        else:
            vol_gradient = 0

        # Composite z-score cua vol-regime
        z_values = []
        if vol["vol_ratio"] is not None:
            z_values.append((vol["vol_ratio"] - 1.0) / 0.3)
        if vol["vol_of_vol"] is not None:
            z_values.append((vol["vol_of_vol"] - np.mean([0.0001, 0.001])) / 0.0005 * -1)
        if vol["skew"] is not None:
            z_values.append(vol["skew"] * -1)
        if vol["kurtosis"] is not None:
            z_values.append((vol["kurtosis"] - 3.0) / 2.0 * -1)

        z_composite = np.mean(z_values) if z_values else 0

        # Composite score: 40% base + 30% money flow + 30% vol gradient
        cmf_score = max(0, min(1, (mf["obv_score"] + mf["cmf_approx"]) / 1.5))

        grad_score = 0.0
        if vol_gradient > 0:
            grad_score = min(1.0, vol_gradient * 5)

        composite = 0.40 * base_score + 0.30 * cmf_score + 0.30 * grad_score
        composite = round(composite * 100, 1)

        results[sym] = {
            "base_score": round(base_score * 100, 1),
            "obv": mf["obv_score"],
            "vpt": mf["vpt_score"],
            "cmf_approx": mf["cmf_approx"],
            "mf_score": round(cmf_score * 100, 1),
            "vol_ratio": round(vol["vol_ratio"], 3) if vol["vol_ratio"] else None,
            "vol_gradient": round(vol_gradient, 4),
            "grad_score": round(grad_score * 100, 1),
            "z_composite": round(z_composite, 3),
            "composite_score": composite,
            "last_price": float(df["Close"].iloc[-1]),
            "last_volume": float(df["Volume"].iloc[-1]) if not pd.isna(df["Volume"].iloc[-1]) else None,
        }

    return results


def get_filtered_symbols() -> list[str]:
    """Lay danh sach symbol da loc thanh khoan (gio'ng fetch_and_compute)."""
    symbols = []
    for path in sorted(CACHE_DIR.glob("*.csv")):
        sym = path.stem
        if sym == ".gitkeep":
            continue
        df = load_cache(sym)
        if len(df) < 20:
            continue
        if "Volume" in df.columns:
            avg_vol = df["Volume"].dropna().iloc[-20:].mean()
            if pd.isna(avg_vol) or avg_vol < MIN_AVG_VOLUME:
                continue
        symbols.append(sym)
    return symbols


def build_signal_output(all_signals: dict) -> dict:
    """Generate structured output voi ranking."""
    signals = []
    for sym, data in all_signals.items():
        if data["composite_score"] >= 50:
            signals.append({
                "symbol": sym,
                "composite_score": data["composite_score"],
                "base_score": data["base_score"],
                "obv": data["obv"],
                "mf_score": data["mf_score"],
                "vol_ratio": data["vol_ratio"],
                "vol_gradient": data["vol_gradient"],
                "z_composite": data["z_composite"],
                "last_price": data["last_price"],
                "last_volume": data["last_volume"],
            })

    signals.sort(key=lambda x: x["composite_score"], reverse=True)

    # Metadata
    now = datetime.now(timezone.utc) + timedelta(hours=7)
    strong = [s for s in signals if s["composite_score"] >= 75]
    moderate = [s for s in signals if 60 <= s["composite_score"] < 75]
    weak = [s for s in signals if 50 <= s["composite_score"] < 60]

    return {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "total_symbols_analyzed": len(all_signals),
        "total_signals": len(signals),
        "strong_signals": len(strong),
        "moderate_signals": len(moderate),
        "weak_signals": len(weak),
        "strong": strong,
        "moderate": moderate,
        "weak": weak,
        "all_signals": signals,
    }


def main():
    tqdm.write("=" * 60)
    tqdm.write("Pre-Breakout Strategy Signals")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    all_signals = compute_zscore_gradient(symbols, "ALL")

    output = build_signal_output(all_signals)

    # Write
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = SIGNALS_JSON
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_bytes(path.read_bytes())

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON}")
    tqdm.write(f"Tong phan tich: {output['total_symbols_analyzed']} ma")
    tqdm.write(f"Tin hieu: {output['total_signals']} (Manh: {output['strong_signals']}, TB: {output['moderate_signals']}, Yeu: {output['weak_signals']})")
    if output["all_signals"]:
        tqdm.write(f"\nTop 5 tin hieu:")
        for s in output["all_signals"][:5]:
            tqdm.write(f"  {s['symbol']:6s} | Score: {s['composite_score']:5.1f} | OBV: {s['obv']:+.2f} | Base: {s['base_score']:5.1f}")


if __name__ == "__main__":
    main()
