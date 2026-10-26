"""The BTC price model, shared by ML2.ipynb and app.py.

For each period (an hour, a week or 30 days) it guesses:
  - the average price of the next period
  - how big the move will be (volatility)
  - an 80% range the price should land in
  - whether the guess is one of the confident ones
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from btc_data import HOUR

PERIODS = {"hour": 1, "week": 24 * 7, "month": 24 * 30}  # period length in hours
STARTS = {"hour": "h", "week": "W-MON", "month": "MS"}    # guesses start every hour, every Monday, every 1st of the month
WINDOW = 4             # how many periods to look back
RANGE = (0.1, 0.9)     # an 80% range
CONFIDENT_SHARE = 0.2  # the 20% most confident guesses count as "confident"
MIN_TRAIN_ROWS = 24 * 30
EPS = 1e-6             # keeps log() away from zero


# ---------- Features ----------

def build_dataset(prices: pd.DataFrame, period_hours: int, window: int = WINDOW):
    """One row per hour t, for guessing the next period (t+1h to t+period).

    Returns the data and the features for each of the two models:
      - "change": what the price did lately (price and volume changes), for guessing up or down
      - "move":   how wild the market is lately (size of moves, high-low range, hour/day), for guessing the move size
    """
    H = period_hours
    min_periods = max(1, H // 2)  # a few missing hours inside a week or month is fine
    hourly_change = prices["Close"].pct_change()

    def per_period(series):
        return series.rolling(H, min_periods=min_periods).mean()

    avg_price = per_period(prices["Close"])
    avg_volume = per_period(prices["Volume"].replace(0, np.nan))
    avg_move = per_period(hourly_change.abs())
    avg_range = per_period((prices["High"] - prices["Low"]) / prices["Close"])

    change, move = {}, {}
    for k in range(1, window + 1):
        now, before = (k - 1) * H, k * H
        change[f"price{k}"] = avg_price.shift(now) / avg_price.shift(before) - 1
        change[f"volume{k}"] = np.log(avg_volume.shift(now) / avg_volume.shift(before))
        move[f"move{k}"] = np.log(avg_move.shift(now) + EPS)
    for hours in (24, 24 * 7, 24 * 30):  # how wild the last day, week and month were
        move[f"move_{hours}h"] = np.log(hourly_change.abs().rolling(hours, min_periods=hours // 2).mean() + EPS)
    move["range"] = np.log(avg_range + EPS)
    if H < 24:  # the hour of day and day of week only matter for short periods
        next_hour = prices.index + HOUR
        move |= {f"hour_{h}": (next_hour.hour == h).astype(float) for h in range(24)}
        move |= {f"day_{d}": (next_hour.dayofweek == d).astype(float) for d in range(7)}

    data = pd.DataFrame(change | move, index=prices.index)
    data["last_price"] = prices["Close"]
    data["change"] = avg_price.shift(-H) / prices["Close"] - 1  # what we guess; empty if the period isn't over yet
    return data, {"change": list(change), "move": list(move)}


# ---------- Model ----------

class Model:
    """Two ridge regressions: one guesses the % change, the other the size of the move.

    Pass another scikit-learn `regressor` to try a different model in the same setup (see experiments.ipynb).
    """

    def __init__(self, features: dict, regressor=None):
        self.features = features
        self.all_features = features["change"] + features["move"]
        self.regressor = Ridge() if regressor is None else regressor

    def fit(self, rows: pd.DataFrame):
        rows = rows.dropna(subset=self.all_features + ["change"])
        X_change, X_move, change = rows[self.features["change"]], rows[self.features["move"]], rows["change"]

        self.change_model = make_pipeline(StandardScaler(), clone(self.regressor)).fit(X_change, change)
        self.move_model = make_pipeline(StandardScaler(), clone(self.regressor)).fit(X_move, np.log(change.abs() + EPS))
        self.average_move = change.abs().mean()

        # Calibrate on the training data: scale the move, size the 80% range, find the confident cut-off
        guess = self.change_model.predict(X_change)
        typical_move = np.exp(self.move_model.predict(X_move))
        self.move_scale = self.average_move / typical_move.mean()
        move = typical_move * self.move_scale
        self.range_low, self.range_high = np.quantile((change - guess) / move, RANGE)
        self.confident_from = np.quantile(np.abs(guess) / move, 1 - CONFIDENT_SHARE)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        last = rows["last_price"]
        change = self.change_model.predict(rows[self.features["change"]])
        move = np.exp(self.move_model.predict(rows[self.features["move"]])) * self.move_scale
        return pd.DataFrame({
            "predicted": last * (1 + change),
            "low": last * (1 + change + self.range_low * move),
            "high": last * (1 + change + self.range_high * move),
            "move": move,                       # expected size of the move, e.g. 0.004 = 0.4%
            "average_move": self.average_move,  # the simple guess for the move size, to compare with
            "confident": np.abs(change) / move >= self.confident_from,
        }, index=rows.index)


def walk_forward(data: pd.DataFrame, features: dict, period: str, start: pd.Timestamp, end: pd.Timestamp,
                 regressor=None) -> pd.DataFrame:
    """Guess every period ("hour", "week" or "month") that starts between `start` and `end`,
    re-learning at the start of every month.

    Each model only learns from periods that were over before its guesses are made.
    """
    step = PERIODS[period] * HOUR
    columns = features["change"] + features["move"]
    period_starts = pd.date_range(start, end, freq=STARTS[period])
    guesses = data.reindex(period_starts - HOUR)  # each guess is made the hour before its period starts
    guesses = guesses.dropna(subset=columns + ["last_price", "change"])
    if guesses.empty:
        raise ValueError("There are no finished periods with prices in that time yet")

    refits = sorted({start} | set(pd.date_range(start, end, freq="MS")))
    parts = []
    for refit_at, next_refit in zip(refits, refits[1:] + [pd.Timestamp.max]):
        chunk = guesses[(guesses.index + HOUR >= refit_at) & (guesses.index + HOUR < next_refit)]
        if chunk.empty:
            continue
        train = data[data.index + step <= chunk.index[0]].dropna(subset=columns + ["change"])
        if len(train) < MIN_TRAIN_ROWS:
            raise ValueError("Not enough history before the start to learn from. Pick a later start")
        parts.append(chunk[["last_price", "change"]].join(Model(features, regressor).fit(train).predict(chunk)))

    result = pd.concat(parts)
    result["real"] = result["last_price"] * (1 + result["change"])
    result.index = result.index + HOUR  # label each guess by the start of its period
    return result


def forecast(data: pd.DataFrame, features: dict) -> pd.Series:
    """Learn from everything so far and guess the period after the latest hour."""
    latest = data.loc[[data.index[-1]]]
    return Model(features).fit(data).predict(latest).iloc[0]


# ---------- Scoring ----------

def luck(right: pd.Series) -> float:
    """Chance of getting at least this many right by flipping a coin."""
    if len(right) == 0:
        return float("nan")
    return binomtest(int(right.sum()), len(right), 0.5, alternative="greater").pvalue


def report(result: pd.DataFrame) -> dict:
    guessed_up = result["predicted"] > result["last_price"]
    right = guessed_up == (result["change"] > 0)
    confident = result["confident"]
    real_move = result["change"].abs()
    return {
        "guesses": len(result),
        "accuracy %": (100 - (result["predicted"] - result["real"]).abs() / result["real"] * 100).mean(),
        "same-as-last-price %": (100 - (result["last_price"] - result["real"]).abs() / result["real"] * 100).mean(),
        "right direction %": right.mean() * 100,
        "chance it's luck %": luck(right) * 100,
        "confident guesses": int(confident.sum()),
        "confident right direction %": right[confident].mean() * 100,
        "confident chance it's luck %": luck(right[confident]) * 100,
        "inside 80% range %": result["real"].between(result["low"], result["high"]).mean() * 100,
        "move size better than average guess %": (
            1 - (result["move"] - real_move).abs().mean() / (result["average_move"] - real_move).abs().mean()
        ) * 100,
    }
