# VN Market Breadth Dashboard

Dashboard độ rộng thị trường chứng khoán Việt Nam (A/D Ratio, % cổ phiếu trên MA20/50/200),
lấy dữ liệu từ SSI FastConnect Data API, tự động cập nhật qua GitHub Actions, hiển thị qua GitHub Pages.

## Tính năng

- **Breadth Overview**: A/D Ratio, % trên MA20/MA50/MA200, breadth bar, newly above/below MA
- **Historical Chart**: Biểu đồ Canvas % trên MA20/MA50 theo thời gian với hover tooltip
- **Volume Breakout**: Phát hiện mã có volume đột biến (>=MA20 + KL > 1.3x TB20)
- **Pre-Breakout Signals**: Phát hiện cổ phiếu tích lũy sắp breakout (base score + OBV momentum + vol gradient)
- **Ensemble Signals**: 4-signal voting strategy (MA Crossover + Pullback + Breakout + Momentum)
- **Momentum Signals**: Bộ lọc momentum có điểm số, ADX/RSI/volume bonus
- **Lục Mạch Signals**: bản core theo Diệp gốc, gồm VUDD 13/20/35/55/65 + Tplus
- **Khung4/Tplus Signals**: hub mua riêng theo state Khung4/Tplus, có giá mua tại phiên phát tín hiệu
- **Market Commentary**: Nhận định thị trường tự động (breadth + kỹ thuật VN-Index)
- **Accumulation Radar**: hub riêng phát hiện cổ phiếu khỏe âm thầm trong thị trường xấu

## Cấu trúc repo

```
scripts/
  ssi_client.py           # client gọi SSI FastConnect Data API
  cache_utils.py          # shared: load_cache + compute_rsi_wilder
  fetch_and_compute.py    # pipeline chính: fetch -> compute -> ghi JSON
  strategy_signals.py     # Pre-breakout detection (base + OBV + vol)
  ensemble_signals.py     # 4-signal voting strategy
  momentum_signals.py     # Momentum score + bonuses
  luc_mach_signals.py     # Lục Mạch core: VUDD + Tplus
  khung4_tplus_signals.py # Khung4/Tplus standalone: buy + buy_price
  mama_positional_signals.py # MAMA positional: Ehlers MAMA/FAMA + xác nhận High/Low setup
  advanced_trailstop_signals.py # Advanced Trailstop: bs ATR + Close cross
  accumulation_radar.py # Accumulation Radar: RS + resilience + volume accumulation + base contraction
  market_commentary.py    # Nhận định thị trường (breadth + technical)
  embed_data.py           # Tạo dashboard.html với embedded data
  requirements.txt
data/
  breadth_latest.json     # snapshot breadth mới nhất
  breadth_history.json    # lịch sử ~120 phiên
  strategy_signals.json   # tín hiệu pre-breakout
  ensemble_signals.json   # tín hiệu ensemble
  momentum_signals.json   # tín hiệu momentum
  luc_mach_signals.json   # tín hiệu Lục Mạch
  khung4_tplus_signals.json # tín hiệu mua Khung4/Tplus
  mama_positional_signals.json # tín hiệu MAMA positional
  advanced_trailstop_signals.json # tín hiệu Advanced Trailstop
  accumulation_radar.json # ứng viên tích lũy khỏe hơn market proxy
  market_commentary.json  # nhận định thị trường
  ohlc_cache/             # cache OHLCV theo mã
docs/
  index.html              # dashboard source (fetch JSON từ data/)
  accumulation-radar.html # hub riêng Accumulation Radar
  dashboard.html          # embedded version, dùng được cả Dashboard và Radar qua file://
  data/                   # JSON copies cho GitHub Pages
.github/workflows/update.yml  # cron: 15:10 VN (T2-T6)
```

## Chạy local

```bash
cd scripts
pip install -r requirements.txt

export SSI_CONSUMER_ID="..."
export SSI_CONSUMER_SECRET="..."
python fetch_and_compute.py
```

Pipeline chỉ xuất snapshot đóng cửa từ 15:10 giờ Việt Nam; dùng `ALLOW_PRE_CLOSE_RUN=1` chỉ khi chủ động cần chạy trước giờ đóng cửa.

Sau khi chạy:
- `data/breadth_latest.json` — breadth snapshot
- `data/strategy_signals.json` — pre-breakout signals
- `data/ensemble_signals.json` — ensemble signals
- `data/momentum_signals.json` — momentum signals
- `data/luc_mach_signals.json` — Lục Mạch signals
- `data/mama_positional_signals.json` — MAMA positional signals
- `data/advanced_trailstop_signals.json` — Advanced Trailstop signals
- `data/accumulation_radar.json` — Accumulation Radar candidates
- `data/market_commentary.json` — nhận định thị trường

Mở `docs/index.html` qua Live Server (VSCode) hoặc `python -m http.server`.
Mở `docs/accumulation-radar.html` qua Live Server để xem hub Accumulation Radar.
Mở `docs/dashboard.html` trực tiếp bằng double-click (`file://`) để dùng Dashboard và Accumulation Radar offline. `docs/index.html` và `docs/accumulation-radar.html` vẫn cần được phục vụ qua HTTP để tải JSON.

## Dashboard tabs

| Tab | Nội dung |
|-----|----------|
| ALL / HOSE / HNX | Breadth overview: A/D, MA stats, breadth bar, newly above/below |
| Lịch sử | Biểu đồ Canvas % trên MA20/MA50 theo thời gian |
| Breakout | Volume breakout symbols |
| Nhận định | Market commentary (breadth + VN-Index technical) |
| Pre-Breakout | Cổ phiếu tích lũy sắp breakout (base + OBV + vol gradient) |
| Ensemble | 4-signal voting: MA Crossover + Pullback + Breakout + Momentum |
| Lục Mạch | Core Diệp gốc: VUDD 13/20/35/55/65 + Tplus |

## Triển khai GitHub

1. Push lên GitHub
2. Settings → Secrets: `SSI_CONSUMER_ID`, `SSI_CONSUMER_SECRET`
3. Settings → Pages: branch `main`, folder `/docs`
4. Actions → Run workflow lần đầu

## Strategy Details

### Pre-Breakout (strategy_signals.py)
- Base detection: 25-session window, max range 8%, price in top 25%
- OBV momentum: 5/10/21-day OBV trend
- Volume regime: short/long volatility ratio + gradient
- Composite: 50% base + 30% OBV + 20% vol gradient
- Threshold: composite >= 60, OBV5 > 0, price near high <= 3%

### Ensemble (ensemble_signals.py)
- **MA Crossover**: MA10 > MA50 + Close > MA10 + RSI14 > 50
- **Pullback to Support**: uptrend (Close > MA200) + near MA50 (93-100%) + RSI > 45
- **Breakout**: new 20-day high + volume > 1.5x avg
- **Momentum ROC**: ROC10 > ROC20 + positive volume slope
- **Voting**: >= 3/4 = Strong Buy, 2/4 = Weak Buy

### Lục Mạch (luc_mach_signals.py)
- **Mode**: `diep_original`, chỉ dùng VUDD + Tplus để ra tín hiệu Lục Mạch
- **VUDD**: Heikin Ashi + ZeroLag TEMA trên các chu kỳ 13/20/35/55/65
- **Tplus**: xác nhận phá vùng 4 phiên, không gán trạng thái khi chưa đủ dữ liệu
- **Score**: `buy_score = 5 VUDD buy + Tplus buy`, `sell_score = 5 VUDD sell + Tplus sell`
- **Buy/Sell**: Buy khi `buy_score >= 3`, Sell khi `sell_score >= 3`
- **Filter**: `Volume > 20000` và có ít nhất một tín hiệu mua hoặc bán
- **History**: yêu cầu tối thiểu 300 phiên OHLCV; pipeline backfill 800 ngày lịch để ZeroLagTEMA(65) ổn định hơn
- **Status**: VALID_BUY, WATCHLIST, SELL_WARNING, CONFLICT

### MAMA Positional (mama_positional_signals.py)
- **Mode**: `ehlers_mama_positional`, tính MAMA/FAMA theo John Ehlers với `Period` động
- **Input price**: `(High + Low) / 2`
- **Setup**: `Buysetup = Cross(MAMA, FAMA)`, `Sellsetup = Cross(FAMA, MAMA)`
- **Confirm**: Buy khi Close vượt High của phiên setup mua; Sell khi Close thủng Low của phiên setup bán
- **Filter**: `ExRem` loại tín hiệu lặp; `BPrice`/`SPrice` là Close tại phiên tín hiệu thật
- **Output**: `data/mama_positional_signals.json` gồm nhóm `buy`, `sell`, `all_signals`, `audit`

### Advanced Trailstop (advanced_trailstop_signals.py)
- **Mode**: `diep_advanced_trailstop`, độc lập với Lục Mạch, Khung4/Tplus và MAMA
- **ATR**: `atrvalue = 2.0 * ATR(7)` mặc định, dùng Wilder-style ATR
- **bs tăng**: nếu Low hiện tại cao hơn toàn bộ 9 Low trước và Close > `bs` trước đó, `bs = Low - atrvalue`
- **bs giảm**: nếu High hiện tại thấp hơn toàn bộ 9 High trước và Close < `bs` trước đó, `bs = High + atrvalue`
- **Signal**: Buy khi Close cắt lên `bs`; Sell khi `bs` cắt lên Close
- **Price**: `BuyPrice`/`SellPrice` là Close tại lần Buy/Sell gần nhất
