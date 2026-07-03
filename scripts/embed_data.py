"""Nhúng dữ liệu breadth vào dashboard HTML để mở trực tiếp không cần server."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST_JSON = ROOT / "data" / "breadth_latest.json"
HISTORY_JSON = ROOT / "data" / "breadth_history.json"
SRC_HTML = ROOT / "docs" / "index.html"
OUT_HTML = ROOT / "docs" / "dashboard.html"

latest = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))

html = SRC_HTML.read_text(encoding="utf-8")

# Inject dữ liệu JSON vào inline, thay thế fetch() bằng biến toàn cục
inject_script = f"""
<script>
const EMBEDDED_LATEST = {json.dumps(latest, ensure_ascii=False)};
const EMBEDDED_HISTORY = {json.dumps(history, ensure_ascii=False)};
</script>
"""

# Thay thế hàm loadData()
new_load = """
async function loadData(){
  LATEST = EMBEDDED_LATEST;
  HISTORY = EMBEDDED_HISTORY;
  document.getElementById('metaLine').textContent =
    `Cập nhật: ${new Date(LATEST.generated_at).toLocaleString('vi-VN')} · Ngày dữ liệu: ${LATEST.markets.ALL.date}`;
  renderTabs();
  render();
}
"""

# Thực hiện thay thế trong HTML
html = html.replace("</head>", inject_script + "</head>")
html = html.replace(
    "async function loadData(){",
    "async function loadData(){ try { " + new_load.split("async function loadData(){")[1].strip() + " return; } catch(e){}",
    1
)

OUT_HTML.write_text(html, encoding="utf-8")
print(f"Đã tạo: {OUT_HTML}")
print("Mở file này bằng double-click (file://) để xem dashboard.")
