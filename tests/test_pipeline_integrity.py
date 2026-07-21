"""Regression tests for pipeline date and cache integrity guards."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_and_compute as pipeline
import khung4_tplus_signals as khung4
from market_commentary import generate_commentary


class _Client:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def daily_ohlc(self, symbol, from_date, to_date):
        self.calls.append((symbol, from_date, to_date))
        return self.rows


def test_update_ohlc_replaces_overlapping_api_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    pipeline.save_cache("AAA", pd.DataFrame({
        "TradingDate": pd.to_datetime(["2026-07-15", "2026-07-16"]),
        "Open": [10, 10], "High": [11, 11], "Low": [9, 9],
        "Close": [10, 10], "Volume": [100, 100],
    }))
    client = _Client([
        {"TradingDate": "16/07/2026", "Open": 11, "High": 13, "Low": 10, "Close": 12, "Volume": 200},
        {"TradingDate": "17/07/2026", "Open": 12, "High": 14, "Low": 11, "Close": 13, "Volume": 300},
    ])

    updated = pipeline.update_ohlc(client, "AAA", datetime(2026, 7, 17))

    assert client.calls  # Even a same-day cache is refreshed.
    assert updated.loc[updated["TradingDate"] == pd.Timestamp("2026-07-16"), "Close"].iloc[0] == 12
    assert len(updated) == 3


def test_signals_history_only_accepts_the_expected_market_date(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "SIGNALS_HISTORY_JSON", tmp_path / "signals_history.json")
    monkeypatch.setattr(pipeline, "DOCS_SIGNALS_HISTORY_JSON", tmp_path / "docs" / "signals_history.json")
    (tmp_path / "strategy_signals.json").write_text(json.dumps({"date": "17/07/2026", "name": "new"}), encoding="utf-8")
    (tmp_path / "ensemble_signals.json").write_text(json.dumps({"date": "16/07/2026", "name": "stale"}), encoding="utf-8")

    pipeline.append_signals_history("17/07/2026")

    history = json.loads((tmp_path / "signals_history.json").read_text(encoding="utf-8"))
    assert history[0]["date"] == "17/07/2026"
    assert history[0]["strategy"]["name"] == "new"
    assert history[0]["ensemble"] is None


def test_khung4_rejects_quote_older_than_market_session(tmp_path, monkeypatch):
    monkeypatch.setattr(khung4, "CACHE_DIR", tmp_path)
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    pd.DataFrame({
        "TradingDate": dates,
        "Open": [10.0] * 20,
        "High": [11.0] * 20,
        "Low": [9.0] * 20,
        "Close": [10.0] * 20,
        "Volume": [100_000.0] * 20,
    }).to_csv(tmp_path / "SPI.csv", index=False)

    result, audit = khung4.audit_symbol("SPI", datetime(2026, 7, 17))

    assert result is None
    assert audit["reason"] == "stale_quote"


def test_ad_distribution_keeps_exact_limit_moves_in_outer_buckets():
    assert pipeline._ad_bucket_index(-7.0) == 0
    assert pipeline._ad_bucket_index(7.0) == 10


def test_close_pipeline_does_not_publish_intraday_data(monkeypatch):
    monkeypatch.delenv("ALLOW_PRE_CLOSE_RUN", raising=False)
    assert not pipeline.should_run_close_pipeline(datetime(2026, 7, 21, 15, 9))
    assert pipeline.should_run_close_pipeline(datetime(2026, 7, 21, 15, 10))
    monkeypatch.setenv("ALLOW_PRE_CLOSE_RUN", "1")
    assert pipeline.should_run_close_pipeline(datetime(2026, 7, 21, 12, 0))


def test_commentary_keeps_zero_ad_ratio_as_valid_data():
    markets = {
        market: {
            "date": "17/07/2026",
            "ad_ratio": 0.0,
            "pct_above_ma20": 0.0,
            "pct_above_ma50": 0.0,
            "pct_above_ma200": 0.0,
            "newly_above_ma20": [], "newly_below_ma20": [],
            "newly_above_ma50": [], "newly_below_ma50": [],
            "volume_breakout_symbols": [],
        }
        for market in ("ALL", "HOSE", "HNX")
    }
    commentary = generate_commentary({"session": "close", "markets": markets})
    assert "A/D Ratio = **0.00**" in commentary
    assert "Giảm vị thế" in commentary
