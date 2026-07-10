"""Nhung du lieu breadth + commentary vao dashboard HTML de mo truc tiep khong can server."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST_JSON = ROOT / "data" / "breadth_latest.json"
HISTORY_JSON = ROOT / "data" / "breadth_history.json"
COMMENTARY_JSON = ROOT / "data" / "market_commentary.json"
SIGNALS_JSON = ROOT / "data" / "strategy_signals.json"
ENSEMBLE_JSON = ROOT / "data" / "ensemble_signals.json"
WEIGHTS_JSON = ROOT / "data" / "backtest_weights.json"
MOMENTUM_JSON = ROOT / "data" / "momentum_signals.json"
MOMENTUM_BT_JSON = ROOT / "data" / "backtest_momentum.json"
MAMA_BT_JSON = ROOT / "data" / "backtest_mama_positional.json"
ATS_BT_JSON = ROOT / "data" / "backtest_advanced_trailstop.json"
LUC_MACH_JSON = ROOT / "data" / "luc_mach_signals.json"
KHUNG4_TPLUS_JSON = ROOT / "data" / "khung4_tplus_signals.json"
MAMA_POSITIONAL_JSON = ROOT / "data" / "mama_positional_signals.json"
ADVANCED_TRAILSTOP_JSON = ROOT / "data" / "advanced_trailstop_signals.json"
SIGNALS_HISTORY_JSON = ROOT / "data" / "signals_history.json"
LATEST_PRICES_JSON = ROOT / "data" / "latest_prices.json"
SRC_HTML = ROOT / "docs" / "index.html"
OUT_HTML = ROOT / "docs" / "dashboard.html"

latest = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
history = json.loads(HISTORY_JSON.read_text(encoding="utf-8") if HISTORY_JSON.exists() else "[]")
commentary = json.loads(COMMENTARY_JSON.read_text(encoding="utf-8")) if COMMENTARY_JSON.exists() else None
signals = json.loads(SIGNALS_JSON.read_text(encoding="utf-8")) if SIGNALS_JSON.exists() else None
ensemble = json.loads(ENSEMBLE_JSON.read_text(encoding="utf-8")) if ENSEMBLE_JSON.exists() else None
weights = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8")) if WEIGHTS_JSON.exists() else None
momentum = json.loads(MOMENTUM_JSON.read_text(encoding="utf-8")) if MOMENTUM_JSON.exists() else None
momentum_bt = json.loads(MOMENTUM_BT_JSON.read_text(encoding="utf-8")) if MOMENTUM_BT_JSON.exists() else None
mama_bt = json.loads(MAMA_BT_JSON.read_text(encoding="utf-8")) if MAMA_BT_JSON.exists() else None
ats_bt = json.loads(ATS_BT_JSON.read_text(encoding="utf-8")) if ATS_BT_JSON.exists() else None
luc_mach = json.loads(LUC_MACH_JSON.read_text(encoding="utf-8")) if LUC_MACH_JSON.exists() else None
khung4_tplus = json.loads(KHUNG4_TPLUS_JSON.read_text(encoding="utf-8")) if KHUNG4_TPLUS_JSON.exists() else None
mama_positional = json.loads(MAMA_POSITIONAL_JSON.read_text(encoding="utf-8")) if MAMA_POSITIONAL_JSON.exists() else None
advanced_trailstop = json.loads(ADVANCED_TRAILSTOP_JSON.read_text(encoding="utf-8")) if ADVANCED_TRAILSTOP_JSON.exists() else None
signals_history = json.loads(SIGNALS_HISTORY_JSON.read_text(encoding="utf-8")) if SIGNALS_HISTORY_JSON.exists() else None
latest_prices = json.loads(LATEST_PRICES_JSON.read_text(encoding="utf-8")) if LATEST_PRICES_JSON.exists() else None

html = SRC_HTML.read_text(encoding="utf-8")

# Inject du lieu JSON inline
inject_script = f"""
<script>
const EMBEDDED_LATEST = {json.dumps(latest, ensure_ascii=False)};
const EMBEDDED_HISTORY = {json.dumps(history, ensure_ascii=False)};
const EMBEDDED_COMMENTARY = {json.dumps(commentary, ensure_ascii=False)};
const EMBEDDED_SIGNALS = {json.dumps(signals, ensure_ascii=False)};
const EMBEDDED_ENSEMBLE = {json.dumps(ensemble, ensure_ascii=False)};
const EMBEDDED_WEIGHTS = {json.dumps(weights, ensure_ascii=False)};
const EMBEDDED_MOMENTUM = {json.dumps(momentum, ensure_ascii=False)};
const EMBEDDED_MOMENTUM_BT = {json.dumps(momentum_bt, ensure_ascii=False)};
const EMBEDDED_MAMA_BT = {json.dumps(mama_bt, ensure_ascii=False)};
const EMBEDDED_ATS_BT = {json.dumps(ats_bt, ensure_ascii=False)};
const EMBEDDED_LUC_MACH = {json.dumps(luc_mach, ensure_ascii=False)};
const EMBEDDED_KHUNG4_TPLUS = {json.dumps(khung4_tplus, ensure_ascii=False)};
const EMBEDDED_MAMA_POSITIONAL = {json.dumps(mama_positional, ensure_ascii=False)};
const EMBEDDED_ADVANCED_TRAILSTOP = {json.dumps(advanced_trailstop, ensure_ascii=False)};
const EMBEDDED_SIGNALS_HISTORY = {json.dumps(signals_history, ensure_ascii=False)};
const EMBEDDED_LATEST_PRICES = {json.dumps(latest_prices, ensure_ascii=False)};
</script>
"""

# Inject script vao </head>
html = html.replace("</head>", inject_script + "</head>")

# Thay the ham loadData() - dung markers ro rang
old_marker = "async function loadData(){"
new_func = """async function loadData(){
  LATEST = EMBEDDED_LATEST;
  HISTORY = EMBEDDED_HISTORY || [];
  COMMENTARY = EMBEDDED_COMMENTARY;
  SIGNALS = EMBEDDED_SIGNALS;
  ENSEMBLE = EMBEDDED_ENSEMBLE;
  ENSEMBLE_WEIGHTS = EMBEDDED_WEIGHTS;
  MOMENTUM = EMBEDDED_MOMENTUM;
  MOMENTUM_BT = EMBEDDED_MOMENTUM_BT;
  MAMA_BT = EMBEDDED_MAMA_BT;
  ATS_BT = EMBEDDED_ATS_BT;
  LUC_MACH = EMBEDDED_LUC_MACH;
  KHUNG4_TPLUS = EMBEDDED_KHUNG4_TPLUS;
  MAMA_POSITIONAL = EMBEDDED_MAMA_POSITIONAL;
  ADVANCED_TRAILSTOP = EMBEDDED_ADVANCED_TRAILSTOP;
  SIGNALS_HISTORY = EMBEDDED_SIGNALS_HISTORY;
  LATEST_PRICES = EMBEDDED_LATEST_PRICES;
  const sessionLabel = '<span class="session-badge close">Dong cua 15:10</span>';
  document.getElementById('metaLine').innerHTML =
    "Cap nhat: " + new Date(LATEST.generated_at).toLocaleString('vi-VN') + " - Ngay du lieu: " + LATEST.markets.ALL.date + " " + sessionLabel;
  MARKETS = ['ALL', ...Object.keys(LATEST.markets).filter(k => k !== 'ALL')];
  renderMarketTabs();
  render();
}"""

if old_marker in html:
    idx = html.index(old_marker)
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
print(f"Da tao: {OUT_HTML.name}")
print("Mo file nay bang double-click (file://) de xem dashboard.")
