"""
Client gọi SSI FastConnect Data API (FCData v2).
"""
import os
import requests

BASE_URL = "https://fc-data.ssi.com.vn/api/v2/Market"


class SSIClient:
    def __init__(self):
        self.consumer_id = os.environ["SSI_CONSUMER_ID"]
        self.consumer_secret = os.environ["SSI_CONSUMER_SECRET"]
        self._token = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        resp = requests.post(
            f"{BASE_URL}/AccessToken",
            json={"consumerID": self.consumer_id, "consumerSecret": self.consumer_secret},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["data"]["accessToken"]
        print(f"[AUTH] Token lấy thành công")
        return self._token

    def _headers(self):
        return {"Authorization": f"Bearer {self._get_token()}"}

    def _get(self, path: str, params: dict, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                resp = requests.get(
                    f"{BASE_URL}/{path}",
                    params=params,
                    headers=self._headers(),
                    timeout=20,
                )
                if resp.status_code == 401 and attempt == 0:
                    self._token = None
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[ERROR] _get {path} attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    return {}
        return {}

    def securities_details(self, market: str, page_size: int = 1000) -> list[dict]:
        out, page = [], 1
        while True:
            data = self._get("SecuritiesDetails", {
                "market": market,
                "pageIndex": page,
                "pageSize": page_size,
            })
            rows = data.get("data", []) or []
            out.extend(rows)
            if len(rows) < page_size:
                break
            page += 1
            if page > 10:
                break
        return out

    def common_stock_symbols(self, market: str) -> list[str]:
        """Lấy mã cổ phiếu thường, tự động nhận diện SecType từ API."""
        rows = self.securities_details(market)

        # In ra để biết API đang trả SecType gì
        all_types = set(str(r.get("SecType", "")).strip().upper() for r in rows)
        print(f"[{market}] Tổng mã API trả về: {len(rows)}, SecType có: {all_types}")

        # Lọc: giữ lại các loại là cổ phiếu thường, loại bỏ CW/ETF/BOND/FUND
        exclude_keywords = {"CW", "ETF", "BOND", "BO", "FU", "MF", "OF", "EF", "FUND", "DC"}
        symbols = []
        for r in rows:
            sec_type = str(r.get("SecType", "")).strip().upper()
            symbol = r.get("Symbol", "").strip()
            if not symbol:
                continue
            # Giữ lại nếu SecType không chứa từ khóa loại trừ
            if not any(kw in sec_type for kw in exclude_keywords):
                symbols.append(symbol)

        print(f"[{market}] Sau lọc còn: {len(symbols)} mã cổ phiếu thường")
        return symbols

    def daily_ohlc(self, symbol: str, from_date: str, to_date: str, page_size: int = 500) -> list[dict]:
        """OHLCV theo ngày. Định dạng ngày: dd/mm/yyyy"""
        data = self._get("DailyOhlc", {
            "symbol": symbol,
            "fromDate": from_date,
            "toDate": to_date,
            "pageIndex": 1,
            "pageSize": page_size,
            "ascending": "true",
        })
        return data.get("data", []) or []

    def daily_index(self, index_id: str, from_date: str, to_date: str, page_size: int = 100) -> list[dict]:
        """Advances/Declines/Nochanges theo chỉ số."""
        data = self._get("DailyIndex", {
            "indexId": index_id,
            "fromDate": from_date,
            "toDate": to_date,
            "pageIndex": 1,
            "pageSize": page_size,
            "ascending": "true",
        })
        return data.get("data", []) or []
