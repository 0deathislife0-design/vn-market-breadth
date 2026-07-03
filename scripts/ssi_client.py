import os
import re
import time
import random
import threading
import requests
from collections import deque
from datetime import datetime, timedelta

BASE_URL = "https://fc-data.ssi.com.vn/api/v2/Market"


class RateLimiter:
    """Token bucket rate limiter: đảm bảo không vượt quá N request trong khoảng thời gian.
    Thread-safe: dùng Lock để tránh race condition khi nhiều thread cùng gọi."""

    def __init__(self, max_calls: int = 10, per_seconds: int = 3):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._lock = threading.Lock()
        self.timestamps: deque[float] = deque()

    def wait(self):
        now = time.time()
        with self._lock:
            cutoff = now - self.per_seconds
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()

            if len(self.timestamps) >= self.max_calls:
                wait_until = self.timestamps[0] + self.per_seconds
                sleep_time = wait_until - now
                if sleep_time > 0:
                    sleep_time += random.uniform(0, 0.3)
                else:
                    sleep_time = 0
            else:
                sleep_time = 0

            self.timestamps.append(time.time())

        if sleep_time > 0:
            time.sleep(sleep_time)


class SSIClient:
    def __init__(self):
        self.consumer_id = os.environ.get("SSI_CONSUMER_ID")
        self.consumer_secret = os.environ.get("SSI_CONSUMER_SECRET")
        self._token = None
        self._token_expires_at: datetime | None = None
        self._token_lock = threading.Lock()
        self._rate_limiter = RateLimiter(max_calls=8, per_seconds=5)

    def _get_token(self):
        # Double-checked locking cho thread safety
        if self._token and self._token_expires_at and datetime.now() < self._token_expires_at:
            return self._token
        with self._token_lock:
            if self._token and self._token_expires_at and datetime.now() < self._token_expires_at:
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
            self._token_expires_at = datetime.now() + timedelta(hours=23)
            print("[AUTH] Token lấy thành công")
            return self._token

    def _headers(self):
        return {"Authorization": "Bearer " + self._get_token()}

    def _get(self, path, params, retries=3):
        last_exception = None
        for attempt in range(retries):
            try:
                # Áp dụng rate limit trước mỗi request
                self._rate_limiter.wait()

                resp = requests.get(
                    BASE_URL + "/" + path,
                    params=params,
                    headers=self._headers(),
                    timeout=30,
                )

                if resp.status_code == 401 and attempt == 0:
                    # Token hết hạn, refresh 1 lần
                    self._token = None
                    self._token_expires_at = None
                    continue

                if resp.status_code == 429:
                    # Rate limit: đọc Retry-After nếu có, nếu không thì backoff
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    sleep_time = retry_after + random.uniform(0.5, 2.0)
                    print(f"[RATE_LIMIT] {path}: HTTP 429, chờ {sleep_time:.1f}s (attempt {attempt+1}/{retries})")
                    time.sleep(sleep_time * (attempt + 1))  # tăng dần
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as e:
                last_exception = e
                status = e.response.status_code if e.response is not None else 0
                print(f"[ERROR] _get {path} HTTP {status} attempt {attempt+1}/{retries}: {e}")
                if attempt < retries - 1:
                    # Exponential backoff + jitter
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(backoff)

            except Exception as e:
                last_exception = e
                print(f"[ERROR] _get {path} attempt {attempt+1}/{retries}: {e}")
                if attempt < retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(backoff)

        print(f"[ERROR] _get {path}: all {retries} attempts failed")
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
