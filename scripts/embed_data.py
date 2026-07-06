"""Nhúng dữ liệu breadth + commentary vào dashboard HTML để mở trực tiếp không cần server."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST_JSON = ROOT / "data" / "breadth_latest.json"
HISTORY_JSON = ROOT / "data" / "breadth_history.json"
MIDDAY_JSON = ROOT / "data" / "breadth_midday.json"
COMMENTARY_JSON = ROOT / "data" / "market_commentary.json"
SIGNALS_JSON = ROOT / "data" / "strategy_signals.json"
ENSEMBLE_JSON = ROOT / "data" / "ensemble_signals.json"
WEIGHTS_JSON = ROOT / "data" / "backtest_weights.json"
MOMENTUM_JSON = ROOT / "data" / "momentum_signals.json"
MOMENTUM_BT_JSON = ROOT / "data" / "backtest_momentum.json"
SIGNALS_HISTORY_JSON = ROOT / "data" / "signals_history.json"
SRC_HTML = ROOT / "docs" / "index.html"
OUT_HTML = ROOT / "docs" / "dashboard.html"

latest = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
history = json.loads(HISTORY_JSON.read_text(encoding="utf-8") if HISTORY_JSON.exists() else "[]")
midday = json.loads(MIDDAY_JSON.read_text(encoding="utf-8")) if MIDDAY_JSON.exists() else None
commentary = json.loads(COMMENTARY_JSON.read_text(encoding="utf-8")) if COMMENTARY_JSON.exists() else None
signals = json.loads(SIGNALS_JSON.read_text(encoding="utf-8")) if SIGNALS_JSON.exists() else None
ensemble = json.loads(ENSEMBLE_JSON.read_text(encoding="utf-8")) if ENSEMBLE_JSON.exists() else None
weights = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8")) if WEIGHTS_JSON.exists() else None
momentum = json.loads(MOMENTUM_JSON.read_text(encoding="utf-8")) if MOMENTUM_JSON.exists() else None
momentum_bt = json.loads(MOMENTUM_BT_JSON.read_text(encoding="utf-8")) if MOMENTUM_BT_JSON.exists() else None
signals_history = json.loads(SIGNALS_HISTORY_JSON.read_text(encoding="utf-8")) if SIGNALS_HISTORY_JSON.exists() else None

html = SRC_HTML.read_text(encoding="utf-8")

# Inject dữ liệu JSON inline
inject_script = f"""
<script>
const EMBEDDED_LATEST = {json.dumps(latest, ensure_ascii=False)};
const EMBEDDED_HISTORY = {json.dumps(history, ensure_ascii=False)};
const EMBEDDED_MIDDAY = {json.dumps(midday, ensure_ascii=False)};
const EMBEDDED_COMMENTARY = {json.dumps(commentary, ensure_ascii=False)};
const EMBEDDED_SIGNALS = {json.dumps(signals, ensure_ascii=False)};
const EMBEDDED_ENSEMBLE = {json.dumps(ensemble, ensure_ascii=False)};
const EMBEDDED_WEIGHTS = {json.dumps(weights, ensure_ascii=False)};
const EMBEDDED_MOMENTUM = {json.dumps(momentum, ensure_ascii=False)};
const EMBEDDED_MOMENTUM_BT = {json.dumps(momentum_bt, ensure_ascii=False)};
const EMBEDDED_SIGNALS_HISTORY = {json.dumps(signals_history, ensure_ascii=False)};
</script>
"""

# Inject script vào </head>
html = html.replace("</head>", inject_script + "</head>")

# Thay thế hàm loadData() — dùng markers rõ ràng
old_marker = "async function loadData(){"
new_func = """async function loadData(){
  LATEST = EMBEDDED_LATEST;
  HISTORY = EMBEDDED_HISTORY || [];
  MIDDAY = EMBEDDED_MIDDAY;
  COMMENTARY = EMBEDDED_COMMENTARY;
  SIGNALS = EMBEDDED_SIGNALS;
  ENSEMBLE = EMBEDDED_ENSEMBLE;
  ENSEMBLE_WEIGHTS = EMBEDDED_WEIGHTS;
  MOMENTUM = EMBEDDED_MOMENTUM;
  MOMENTUM_BT = EMBEDDED_MOMENTUM_BT;
  SIGNALS_HISTORY = EMBEDDED_SIGNALS_HISTORY;
  if (SIGNALS_HISTORY && SIGNALS_HISTORY.length > 0) { activeSignalsDate = SIGNALS_HISTORY[SIGNALS_HISTORY.length - 1].date; }
  if (MIDDAY && MIDDAY.markets && MIDDAY.markets.ALL.date !== LATEST.markets.ALL.date) { MIDDAY = null; }
  const sessionLabel = LATEST.session === 'midday'
    ? '<span class="session-badge midday">Phiên sáng 11:30</span>'
    : '<span class="session-badge close">Đóng cửa 15:10</span>';
  document.getElementById('metaLine').innerHTML =
    "Cập nhật: " + new Date(LATEST.generated_at).toLocaleString('vi-VN') + " · Ngày dữ liệu: " + LATEST.markets.ALL.date + " " + sessionLabel;
  MARKETS = ['ALL', ...Object.keys(LATEST.markets).filter(k => k !== 'ALL')];
  renderTabs();
  render();
}"""

if old_marker in html:
    idx = html.index(old_marker)
    # Đếm brace để tìm đúng dấu } đóng hàm loadData
    rest = html[idx:]
    brace_count = 0
    actual_end = idx
    for i, ch in enumerate(rest):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                actual_end = idx + i + 1
                break
    html = html[:idx] + new_func + "\n\n" + html[actual_end:]

OUT_HTML.write_text(html, encoding="utf-8")
print(f"Đã tạo: {OUT_HTML}")
print("Mở file này bằng double-click (file://) để xem dashboard.")
