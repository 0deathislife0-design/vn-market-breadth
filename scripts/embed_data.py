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
ACCUMULATION_RADAR_JSON = ROOT / "data" / "accumulation_radar.json"
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
accumulation_radar = json.loads(ACCUMULATION_RADAR_JSON.read_text(encoding="utf-8")) if ACCUMULATION_RADAR_JSON.exists() else None

html = SRC_HTML.read_text(encoding="utf-8")

def inline_json(data):
    """Serialize JSON for an inline script without allowing a closing script tag."""
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


# Inject du lieu JSON inline.
inject_script = f"""
<script>
const EMBEDDED_LATEST = {inline_json(latest)};
const EMBEDDED_HISTORY = {inline_json(history)};
const EMBEDDED_COMMENTARY = {inline_json(commentary)};
const EMBEDDED_SIGNALS = {inline_json(signals)};
const EMBEDDED_ENSEMBLE = {inline_json(ensemble)};
const EMBEDDED_WEIGHTS = {inline_json(weights)};
const EMBEDDED_MOMENTUM = {inline_json(momentum)};
const EMBEDDED_MOMENTUM_BT = {inline_json(momentum_bt)};
const EMBEDDED_MAMA_BT = {inline_json(mama_bt)};
const EMBEDDED_ATS_BT = {inline_json(ats_bt)};
const EMBEDDED_LUC_MACH = {inline_json(luc_mach)};
const EMBEDDED_KHUNG4_TPLUS = {inline_json(khung4_tplus)};
const EMBEDDED_MAMA_POSITIONAL = {inline_json(mama_positional)};
const EMBEDDED_ADVANCED_TRAILSTOP = {inline_json(advanced_trailstop)};
const EMBEDDED_SIGNALS_HISTORY = {inline_json(signals_history)};
const EMBEDDED_LATEST_PRICES = {inline_json(latest_prices)};
const EMBEDDED_ACCUMULATION_RADAR = {inline_json(accumulation_radar)};
</script>
"""

# Inject script vao </head>
html = html.replace("</head>", inject_script + "</head>")

OUT_HTML.write_text(html, encoding="utf-8")
print(f"Da tao: {OUT_HTML.name}")
print("Mo file nay bang double-click (file://) de xem dashboard.")
