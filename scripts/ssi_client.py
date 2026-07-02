import os
import re
import time
import requests

BASE_URL = "https://fc-data.ssi.com.vn/api/v2/Market"


class SSIClient:
    def __init__(self):
        self.consumer_id = os.environ.get("SSI_CONSUMER_ID")
        self.consumer_secret = os.environ.get("SSI_CONSUMER_SECRET")
        self._token = None

    def _get_token(self):
        if self._token:
            return self._token

        resp = requests.post(
            BASE_URL + "/AccessToken",
            json={
                "consumerID": self.consumer_id,
                "consumerSecret": self.consumer_secret
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["data"]["accessToken"]
        print("[AUTH] Token lấy thành công")
        return self._token

    def _headers(self):
        return {"Authorization": "Bearer " + self._get_token()}

    def _get(self, path, params, retries=3):
        for attempt in range(retries):
            try:
                resp = requests.get(
                    BASE_URL + "/" + path,
                    params=params,
                    headers=self._headers(),
                    timeout=30,
                )
                if resp.status_code == 401 and attempt == 0:
                    self._token = None
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print("[ERROR] _get " + path + " attempt " + str(attempt+1) + ": " + str(e))
                time.sleep(2)
        if attempt == retries - 1:
            return {}
        return {}

    def common_stock_symbols(self, market):
        """Lấy danh sách mã cổ phiếu thường theo sàn."""
        out = []
        for page in range(1, 11):
            data = self._get("Securities", {
                "market": market,
                "pageIndex": page,
                "pageSize": 1000,
            })
            rows = data.get("data") or []
            print("[" + market + "] Page " + str(page) + " trả về " + str(len(rows)) + " mã")
            if rows:
                print("[" + market + "] Ví dụ mã đầu: " + str(rows[0]))
                out.extend(rows)
            if len(rows) < 1000:
                break
            time.sleep(0.5)

        # Lọc chỉ lấy cổ phiếu thường
        exclude = {"CW", "ETF", "BOND", "BO", "FU", "MF", "OF", "EF", "FUND"}
        symbols = []
        skipped_digit = 0
        for r in out:
            symbol = str(r.get("Symbol") or r.get("symbol") or "").strip()
            sec_type = str(r.get("SecType") or r.get("secType") or r.get("type") or "").strip().upper()
            if not symbol:
                continue
            if any(kw in sec_type for kw in exclude):
                continue
            # Bỏ mã có chứa chữ số (CW, bond, phái sinh thường có số trong mã)
            if re.search(r"\d", symbol):
                skipped_digit += 1
                continue
            symbols.append(symbol)
        print("[" + market + "] Bỏ " + str(skipped_digit) + " mã có chữ số trong tên")
        print("[" + market + "] Sau lọc còn: " + str(len(symbols)) + " mã")
        return symbols

    def daily_ohlc(self, symbol, from_date, to_date, page_size=500):
        """OHLCV theo ngày. Định dạng ngày: dd/mm/yyyy"""
        data = self._get("DailyOhlc", {
            "symbol": symbol,
            "fromDate": from_date,
            "toDate": to_date,
            "pageIndex": 1,
            "pageSize": page_size,
            "ascending": "true",
        })
        return data.get("data") or []

    def daily_index(self, index_id, from_date, to_date, page_size=100):
        """Advances/Declines/Nochanges theo chỉ số."""
        data = self._get("DailyIndex", {
            "indexId": index_id,
            "fromDate": from_date,
            "toDate": to_date,
            "pageIndex": 1,
            "pageSize": page_size,
            "ascending": "true",
        })
        return data.get("data") or []