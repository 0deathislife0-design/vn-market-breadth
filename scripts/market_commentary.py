#!/usr/bin/env python3
"""
Sinh nhận định cuối ngày chỉ từ breadth + kỹ thuật thuần túy.
Không dùng sector, news, cơ bản.
"""
from __future__ import annotations
import json
from pathlib import Path
from _shared import DATA_DIR, CACHE_DIR as OHLC_CACHE_DIR, DOCS_DATA_DIR, vn_now

LATEST_JSON = DATA_DIR / "breadth_latest.json"
COMMENTARY_JSON = DATA_DIR / "market_commentary.json"
DOCS_COMMENTARY_JSON = DOCS_DATA_DIR / "market_commentary.json"


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_ohlc(symbol: str):
    path = OHLC_CACHE_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df["TradingDate"] = pd.to_datetime(df["TradingDate"], dayfirst=True, errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        else:
            df["Volume"] = float("nan")
        return df.dropna(subset=["Close"])
    except Exception:
        return None


def compute_rsi(close_series, period=14):
    """RSI Wilder"""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def compute_macd(close_series, fast=12, slow=26, signal=9):
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return (
        macd_line.iloc[-1] if not macd_line.empty else None,
        signal_line.iloc[-1] if not signal_line.empty else None,
        hist.iloc[-1] if not hist.empty else None,
    )


def ma_trend(close_series, windows=(20, 50, 200)):
    last = close_series.iloc[-1]
    return {w: last >= close_series.rolling(w).mean().iloc[-1] for w in windows}


def vnindex_technical():
    df = load_ohlc("VNINDEX")
    if df is None or len(df) < 200:
        return {}
    close = df["Close"]
    last = close.iloc[-1]
    ma = ma_trend(close)
    rsi = compute_rsi(close)
    macd, signal, hist = compute_macd(close)
    return {
        "close": round(last, 2),
        "ma20": ma.get(20),
        "ma50": ma.get(50),
        "ma200": ma.get(200),
        "rsi": round(rsi, 1) if rsi is not None else None,
        "macd_hist": round(hist, 2) if hist is not None else None,
        "macd_up": (hist > 0) if hist is not None else None,
    }


def generate_commentary(breadth: dict) -> str:
    all_m = breadth["markets"]["ALL"]
    hose = breadth["markets"]["HOSE"]
    hnx = breadth["markets"]["HNX"]
    session = breadth.get("session", "close")
    date = all_m.get("date", vn_now().strftime("%d/%m/%Y"))

    lines = []

    # Header
    session_label = "Phiên sáng (11:30)" if session == "midday" else "Đóng cửa (15:10)"
    lines.append(f"# Nhận định phiên {session_label} — {date}\n")

    # 1. Độ rộng thị trường
    ad = all_m.get("ad_ratio")
    ma20 = all_m.get("pct_above_ma20")
    ma50 = all_m.get("pct_above_ma50")
    ma200 = all_m.get("pct_above_ma200")

    lines.append("## 1. Độ rộng thị trường")
    ad_txt = f"A/D Ratio = **{ad:.2f}**" if ad is not None else "A/D Ratio = N/A"
    lines.append(f"- {ad_txt} | % trên MA20 = **{ma20:.1f}%** | MA50 = **{ma50:.1f}%** | MA200 = **{ma200:.1f}%**")

    if ad is not None:
        if ad >= 1.5:
            lines.append("- Tiền lan tỏa rất rộng, tâm lý lạc quan.")
        elif ad >= 1.2:
            lines.append("- Tiền lan tỏa khá tốt, số mã tăng vượt số mã giảm rõ rệt.")
        elif ad >= 1.0:
            lines.append("- Số mã tăng nhỉnh hơn giảm, cân bằng nghiêng về tăng.")
        elif ad >= 0.8:
            lines.append("- Cân bằng, tiền chưa có chiều hướng rõ.")
        else:
            lines.append("- Tiền rút lại, bán áp đảo.")

    if ma20 is not None:
        if ma20 >= 70:
            lines.append("- % trên MA20 > 70% → thị trường nóng, cảnh báo điều chỉnh ngắn.")
        elif ma20 >= 50:
            lines.append("- % trên MA20 50-70% → xu hướng tăng ổn định.")
        elif ma20 >= 30:
            lines.append("- % trên MA20 30-50% → chao đảo, chờ tín hiệu.")
        else:
            lines.append("- % trên MA20 < 30% → bán mạnh, có thể xuất hiện bounce.")

    # 2. Kỹ thuật VN-Index
    vni = vnindex_technical()
    if vni:
        lines.append("\n## 2. Kỹ thuật VN-Index")
        lines.append(f"- Giá: **{vni['close']:.2f}**")
        ma_lines = []
        if vni.get("ma20") is True: ma_lines.append("MA20 ↑")
        if vni.get("ma50") is True: ma_lines.append("MA50 ↑")
        if vni.get("ma200") is True: ma_lines.append("MA200 ↑")
        lines.append(f"- Đường MA: {' | '.join(ma_lines) if ma_lines else 'Dưới các MA chính'}")
        if vni.get("rsi") is not None:
            rsi_txt = f"RSI = **{vni['rsi']}**"
            if vni["rsi"] >= 70: rsi_txt += " (quá mua)"
            elif vni["rsi"] <= 30: rsi_txt += " (quá bán)"
            lines.append(f"- {rsi_txt}")
        if vni.get("macd_hist") is not None:
            hist = vni["macd_hist"]
            macd_state = "MACD histogram dương ⬆️" if hist > 0 else "MACD histogram âm ⬇️"
            lines.append(f"- {macd_state} ({hist:.2f})")

    # 3. Tín hiệu mới (newly above/below)
    lines.append("\n## 3. Tín hiệu mới trong phiên")
    na20 = hose.get("newly_above_ma20", [])
    nb20 = hose.get("newly_below_ma20", [])
    na50 = hose.get("newly_above_ma50", [])
    nb50 = hose.get("newly_below_ma50", [])
    if na20:
        lines.append(f"- **Mới > MA20 (HOSE)**: {', '.join(na20[:15])}")
    if nb20:
        lines.append(f"- **Mới < MA20 (HOSE)**: {', '.join(nb20[:15])}")
    if na50:
        lines.append(f"- **Mới > MA50 (HOSE)**: {', '.join(na50[:10])}")
    if nb50:
        lines.append(f"- **Mới < MA50 (HOSE)**: {', '.join(nb50[:10])}")

    # 4. Volume breakout
    vb = hose.get("volume_breakout_symbols", [])
    if vb:
        lines.append("\n## 4. Volume breakout (giá ≥ MA20 + KL đột biến)")
        lines.append(f"- **HOSE**: {', '.join(vb[:20])}")

    # 5. Tóm tắt hành động
    lines.append("\n## 5. Gợi ý hành động")
    action = []
    if ad is not None and ad > 1.2 and ma20 is not None and ma20 > 50:
        action.append("✅ Mua dần các mã leader vừa breakout MA20 + volume")
    elif ad is not None and ad < 0.8:
        action.append("⚠️ Giảm vị thế, chờ A/D hồi về > 1.0")
    else:
        action.append("⏳ Chờ tín hiệu rõ ràng hơn (A/D hoặc MA20% breakout)")
    if vb:
        action.append(f"👀 Theo dõi volume breakout: {', '.join(vb[:5])}")
    lines.append("\n".join(action))

    return "\n".join(lines)


def main():
    breadth = load_json(LATEST_JSON)
    if not breadth:
        print("Không tìm thấy breadth_latest.json")
        return

    commentary = generate_commentary(breadth)

    output = {
        "generated_at": vn_now().isoformat(),
        "session": breadth.get("session", "close"),
        "date": breadth["markets"]["ALL"].get("date"),
        "content": commentary,
    }

    COMMENTARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    COMMENTARY_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi: {COMMENTARY_JSON}")

    # Sync docs
    DOCS_COMMENTARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_COMMENTARY_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã sync: {DOCS_COMMENTARY_JSON}")


if __name__ == "__main__":
    main()
