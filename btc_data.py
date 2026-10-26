"""Hourly BTC/USDT prices from Binance, shared by ML2.ipynb and app.py.

Prices are saved to btc_1h.csv, so after the first download only new hours are fetched.
"""
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://data-api.binance.vision/api/v3/klines"  # Binance public market data, no account needed
DATA_FILE = Path(__file__).parent / "btc_1h.csv"
FIRST_HOUR = pd.Timestamp("2018-01-01")
HOUR = pd.Timedelta("1h")


def fetch_hours(start: pd.Timestamp) -> pd.DataFrame:
    """Download every finished hourly BTC/USDT candle from `start` until now."""
    now_ms = int(pd.Timestamp.now("UTC").timestamp() * 1000)
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    rows = []
    while start_ms < now_ms:
        r = requests.get(
            API_URL,
            params={"symbol": "BTCUSDT", "interval": "1h", "startTime": start_ms, "limit": 1000},
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start_ms = batch[-1][0] + 3_600_000
    rows = [row for row in rows if row[6] < now_ms]  # drop the hour that is still in progress

    df = pd.DataFrame([row[:6] for row in rows], columns=["Open time", "Open", "High", "Low", "Close", "Volume"])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    return df.set_index("Open time").astype(float)


def load_prices() -> pd.DataFrame:
    """All hourly candles since 2018 (UTC), up to the newest finished hour.

    If Binance can't be reached, the saved prices are used as they are.
    """
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE, index_col="Open time", parse_dates=True)
        try:
            new = fetch_hours(df.index[-1] + HOUR)
        except requests.RequestException:
            new = df.iloc[:0]  # offline or blocked, so no new hours this time
        if len(new):
            df = pd.concat([df, new])
            df.to_csv(DATA_FILE)
    else:
        df = fetch_hours(FIRST_HOUR)
        df.to_csv(DATA_FILE)
    return df.asfreq("h")  # empty rows for missing hours, so a window never jumps over a gap
