"""BTC price predictor website, 1999 style.

Flask fills in templates/index.html (styled by static/style.css) with the guesses from btc_model.py. Run with:

    python3 app.py

then open http://localhost:5000. On a server, run it with gunicorn instead (see Dockerfile).
"""
import base64
import io
import time
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from flask import Flask, render_template, request
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

import btc_model
from btc_data import HOUR, load_prices

COUNTER_FILE = Path(__file__).parent / "counter.txt"
TIME_FORMAT = "%Y-%m-%d %H:%M"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MAX_ROWS = 500  # rows in the table of guesses; the score and the chart still use all of them

# How each period is named on the page. The lengths are in btc_model.PERIODS.
PERIODS = {
    "hour": {
        "unit": "hour", "title": "Hour by hour", "future": "Price in the next hour",
        "column": "Hour (UTC)", "label": TIME_FORMAT, "hint": "Every hour of the days you pick, in UTC time.",
    },
    "week": {
        "unit": "week", "title": "Week by week", "future": "Average price over the next 7 days",
        "column": "Week starting", "label": "%a %Y-%m-%d", "hint": "Weeks start on Monday. Only finished weeks are shown.",
    },
    "month": {
        "unit": "month", "title": "Month by month", "future": "Average price over the next 30 days",
        "column": "Month", "label": "%B %Y",
        "hint": "Each guess is the average price over 30 days from the 1st. Only finished months are shown.",
    },
}
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

app = Flask(__name__)
_prices, _loaded_at = None, 0.0
_forecasts = {}


def get_prices() -> pd.DataFrame:
    """Hourly prices, refreshed from Binance at most every 10 minutes."""
    global _prices, _loaded_at
    if _prices is None or time.time() - _loaded_at > 600:
        _prices, _loaded_at = load_prices(), time.time()
    return _prices


def count_visitor() -> int:
    visitors = int(COUNTER_FILE.read_text()) + 1 if COUNTER_FILE.exists() else 1
    COUNTER_FILE.write_text(str(visitors))
    return visitors


def read_form(period: str, latest: pd.Timestamp) -> dict:
    """The dates picked in the form, or a sensible default for the period."""
    if period == "hour":  # the last 7 days
        form = {"from_day": f"{latest - pd.Timedelta(days=6):%Y-%m-%d}", "to_day": f"{latest:%Y-%m-%d}"}
    else:  # weeks: the last 3 months, months: since January last year
        first = latest - pd.DateOffset(months=2) if period == "week" else pd.Timestamp(latest.year - 1, 1, 1)
        form = {"from_month": first.month, "from_year": first.year, "to_month": latest.month, "to_year": latest.year}
    return {key: request.args.get(key, str(default)) for key, default in form.items()}


def parse_day(text: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(text)
    except ValueError:
        day = pd.NaT
    if pd.isna(day):
        raise ValueError(f'"{text}" is not a date')
    return day.normalize()


def interval(period: str, form: dict) -> tuple:
    """The first and the last hour picked in the form."""
    if period == "hour":
        start, end = parse_day(form["from_day"]), parse_day(form["to_day"]) + 23 * HOUR
    else:
        try:
            start = pd.Timestamp(int(form["from_year"]), int(form["from_month"]), 1)
            end = pd.Timestamp(int(form["to_year"]), int(form["to_month"]), 1) + pd.DateOffset(months=1) - HOUR
        except ValueError:
            raise ValueError("Pick a month and a year") from None
    if end < start:
        raise ValueError('"From" must be before "To"')
    return start, end


def money(x: float, cents: bool = True) -> str:
    return f"${x:,.2f}" if cents else f"${x:,.0f}"


def chance(percent: float) -> str:
    return "under 1%" if percent < 1 else f"{percent:.0f}%"


def forecasts(prices: pd.DataFrame) -> list:
    """Guesses for the next hour, week and month, learned from all prices so far."""
    latest = prices.index[-1]
    if latest not in _forecasts:
        items = []
        for key, page in PERIODS.items():
            hours = btc_model.PERIODS[key]
            data, features = btc_model.build_dataset(prices, hours)
            if data.iloc[-1][features["change"] + features["move"]].isna().any():
                continue  # looks back further than the data goes
            f = btc_model.forecast(data, features)
            change = f["predicted"] / prices["Close"].iloc[-1] - 1
            items.append({
                "what": page["future"],
                "from": f"{latest + HOUR:{TIME_FORMAT}}",
                "to": f"{latest + (hours + 1) * HOUR:{TIME_FORMAT}}",
                "price": money(f["predicted"]),
                "change": f"{'up' if change > 0 else 'down'} {abs(change):.2%}",
                "range": f"{money(f['low'], False)}–{money(f['high'], False)}",
                "move": f"±{f['move']:.2%}",
                "confident": bool(f["confident"]),
            })
        _forecasts[latest] = items
    return _forecasts[latest]


def chart_png(result: pd.DataFrame, unit: str) -> str:
    """Predicted vs real price with the 80% range, as a base64 PNG for an <img src="data:...">."""
    what = "price" if unit == "hour" else "average price"
    marker = "o" if len(result) <= 120 else None

    # Figure instead of pyplot, since pyplot is not safe when the server answers several visitors at once
    fig = Figure(figsize=(7, 3.4), dpi=100)
    ax = fig.subplots()
    ax.fill_between(result.index, result["low"], result["high"], color=ORANGE, alpha=0.2, lw=0, label="80% range")
    ax.plot(result.index, result["real"], color=BLUE, lw=2, marker=marker, ms=5, label=f"Real {what}")
    ax.plot(result.index, result["predicted"], color=ORANGE, lw=2, ls="--", marker=marker, ms=5, label=f"Predicted {what}")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(color="#dddddd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def make_results(prices: pd.DataFrame, period: str, form: dict) -> dict:
    start, end = interval(period, form)
    unit, label = PERIODS[period]["unit"], PERIODS[period]["label"]

    data, features = btc_model.build_dataset(prices, btc_model.PERIODS[period])
    result = btc_model.walk_forward(data, features, period, start, end)
    score = btc_model.report(result)
    accuracy = 100 - (result["predicted"] - result["real"]).abs() / result["real"] * 100
    move_skill = score["move size better than average guess %"]

    shown = result.tail(MAX_ROWS)  # thousands of hourly rows make the page huge and slow
    rows = [
        {
            "time": f"{t:{label}}", "predicted": money(row.predicted), "real": money(row.real),
            "range": f"{money(row.low, False)}–{money(row.high, False)}", "accuracy": f"{a:.2f}%", "confident": row.confident,
        }
        for (t, row), a in zip(shown.iterrows(), accuracy.tail(MAX_ROWS))
    ]
    return {
        "rows": rows,
        "rows_cut": len(result) > len(shown),
        "unit": unit,
        "showing": f"{result.index[0]:{label}} to {result.index[-1]:{label}}",
        "what": "price" if unit == "hour" else "average",
        "chart": chart_png(result, unit),
        "guesses": f"{score['guesses']:,}",
        "avg_accuracy": f"{score['accuracy %']:.2f}%",
        "naive_accuracy": f"{score['same-as-last-price %']:.2f}%",
        "avg_miss": money((result["predicted"] - result["real"]).abs().mean()),
        "direction": f"{score['right direction %']:.1f}%",
        "direction_luck": chance(score["chance it's luck %"]),
        "confident_guesses": f"{score['confident guesses']:,}",
        "confident_direction": f"{score['confident right direction %']:.1f}%" if score["confident guesses"] else "-",
        "confident_luck": chance(score["confident chance it's luck %"]) if score["confident guesses"] else "-",
        "in_range": f"{score['inside 80% range %']:.0f}%",
        "move_skill": f"{abs(move_skill):.0f}% {'better' if move_skill >= 0 else 'worse'}",
    }


@app.route("/")
def index():
    prices = get_prices()
    latest = prices.index[-1]
    period = request.args.get("period", "hour")
    if period not in PERIODS:
        period = "hour"
    form = read_form(period, latest)
    page = form | {
        "periods": PERIODS, "period": period, "months": MONTHS, "years": range(2018, latest.year + 1),
        "today": f"{latest:%Y-%m-%d}", "latest": f"{latest:{TIME_FORMAT}}", "visitors": f"{count_visitor():06d}",
    }

    try:
        page["forecasts"] = forecasts(prices)
        if "period" in request.args:  # a tab was clicked or the form was sent
            page |= make_results(prices, period, form)
    except ValueError as e:
        page["error"] = str(e)
    return render_template("index.html", **page)


if __name__ == "__main__":
    app.run(port=5000)
