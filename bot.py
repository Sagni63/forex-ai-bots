import os
import json
from datetime import datetime, timezone

import requests
import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# SETTINGS
# =========================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")

JOURNAL_FILE = "trade_journal.json"

# GOLD only
SYMBOLS = {
    "XAU/USD": "XAUUSD=X"
}

# =========================
# TELEGRAM
# =========================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("BOT_TOKEN or CHAT_ID is missing")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=30
    )

    response.raise_for_status()

    print("Telegram message sent")
    return True


# =========================
# MARKET DATA
# =========================

def get_data(symbol, period, interval):

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    return df.dropna(subset=required)


def make_45m(df):

    if df.empty:
        return df

    result = df.resample("45min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

    return result


def make_4h(df):

    if df.empty:
        return df

    result = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

    return result


# =========================
# INDICATORS
# =========================

def EMA(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def ATR(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


# =========================
# TREND
# =========================

def get_trend(df):

    if len(df) < 60:
        return "NEUTRAL"

    ema20 = EMA(df["Close"], 20)
    ema50 = EMA(df["Close"], 50)

    current20 = ema20.iloc[-1]
    current50 = ema50.iloc[-1]

    old20 = ema20.iloc[-10]

    if current20 > current50 and current20 > old20:
        return "BULLISH"

    if current20 < current50 and current20 < old20:
        return "BEARISH"

    return "NEUTRAL"


# =========================
# SUPPLY / DEMAND
# =========================

def find_supply_demand(df):

    if len(df) < 60:
        return []

    zones = []

    recent = df.tail(100)

    ranges = recent["High"] - recent["Low"]

    average_range = ranges.median()

    for i in range(3, len(recent) - 3):

        base = recent.iloc[i-1:i+2]

        base_range = (
            base["High"] - base["Low"]
        ).mean()

        previous = recent.iloc[i-2]
        future = recent.iloc[i+2]

        movement = future["Close"] - previous["Close"]

        # Demand:
        # small base + strong bullish departure
        if (
            base_range < average_range * 0.9
            and movement > average_range * 1.5
        ):

            zones.append({
                "type": "DEMAND",
                "low": float(base["Low"].min()),
                "high": float(base["High"].max())
            })

        # Supply:
        # small base + strong bearish departure
        elif (
            base_range < average_range * 0.9
            and movement < -average_range * 1.5
        ):

            zones.append({
                "type": "SUPPLY",
                "low": float(base["Low"].min()),
                "high": float(base["High"].max())
            })

    return zones[-10:]


def near_zone(price, zones, atr_value):

    if not zones:
        return None

    for zone in reversed(zones):

        low = zone["low"]
        high = zone["high"]

        if low <= price <= high:
            return zone["type"]

        distance = min(
            abs(price - low),
            abs(price - high)
        )

        if distance <= atr_value * 0.35:
            return zone["type"]

    return None


# =========================
# CANDLE CONFIRMATION
# =========================

def candle_confirmation(df, direction):

    if len(df) < 2:
        return 0, []

    candle = df.iloc[-1]

    body = abs(
        candle["Close"] - candle["Open"]
    )

    upper_wick = (
        candle["High"]
        - max(candle["Open"], candle["Close"])
    )

    lower_wick = (
        min(candle["Open"], candle["Close"])
        - candle["Low"]
    )

    reasons = []

    if direction == "BUY":

        hammer = (
            lower_wick >= body * 2
            and upper_wick <= body * 0.8
        )

        if hammer:
            reasons.append("Hammer")

        previous = df.iloc[-2]

        engulfing = (
            candle["Close"] > candle["Open"]
            and candle["Open"] <= previous["Close"]
            and candle["Close"] >= previous["Open"]
        )

        if engulfing:
            reasons.append("Bullish engulfing")

    else:

        shooting_star = (
            upper_wick >= body * 2
            and lower_wick <= body * 0.8
        )

        if shooting_star:
            reasons.append("Shooting star")

        previous = df.iloc[-2]

        engulfing = (
            candle["Close"] < candle["Open"]
            and candle["Open"] >= previous["Close"]
            and candle["Close"] <= previous["Open"]
        )

        if engulfing:
            reasons.append("Bearish engulfing")

    return (2 if reasons else 0), reasons


# =========================
# FIBONACCI
# =========================

def fibonacci_confirmation(df):

    if len(df) < 80:
        return 0, None

    recent = df.tail(80)

    low = float(recent["Low"].min())
    high = float(recent["High"].max())

    price = float(df["Close"].iloc[-1])

    distance = high - low

    if distance <= 0:
        return 0, None

    levels = {
        "38.2%": low + distance * 0.382,
        "50.0%": low + distance * 0.500,
        "61.8%": low + distance * 0.618
    }

    for name, level in levels.items():

        if abs(price - level) <= distance * 0.025:
            return 1, f"Near Fibonacci {name}"

    return 0, None


# =========================
# ANALYSIS
# =========================

def analyze(symbol_name, ticker):

    m15 = get_data(
        ticker,
        "7d",
        "15m"
    )

    h1 = get_data(
        ticker,
        "30d",
        "1h"
    )

    d1 = get_data(
        ticker,
        "2y",
        "1d"
    )

    w1 = get_data(
        ticker,
        "5y",
        "1wk"
    )

    if (
        m15.empty
        or h1.empty
        or d1.empty
        or w1.empty
    ):
        return None

    m45 = make_45m(m15)
    h4 = make_4h(h1)

    if len(m15) < 80:
        return None

    score = 0
    reasons = []

    # =========================
    # MULTI TIMEFRAME
    # 1W -> 1D -> 4H -> 1H
    # =========================

    higher_frames = [
        w1,
        d1,
        h4,
        h1
    ]

    bullish_count = 0
    bearish_count = 0

    for frame in higher_frames:

        trend = get_trend(frame)

        if trend == "BULLISH":
            bullish_count += 1

        elif trend == "BEARISH":
            bearish_count += 1

    # M15 final trend
    m15_trend = get_trend(m15)

    if m15_trend == "BULLISH":
        direction = "BUY"

    elif m15_trend == "BEARISH":
        direction = "SELL"

    else:
        return None

    # Higher timeframe agreement
    if direction == "BUY" and bullish_count >= 2:
        score += 2

    elif direction == "SELL" and bearish_count >= 2:
        score += 2

    else:
        return None

    # 1H confirmation
    h1_trend = get_trend(h1)

    if (
        direction == "BUY"
        and h1_trend == "BULLISH"
    ):
        score += 1
        reasons.append("Bullish EMA trend")

    elif (
        direction == "SELL"
        and h1_trend == "BEARISH"
    ):
        score += 1
        reasons.append("Bearish EMA trend")

    else:
        return None

    # =========================
    # SUPPLY / DEMAND
    # =========================

    zones = find_supply_demand(h1)

    price = float(
        m15["Close"].iloc[-1]
    )

    atr_value = float(
        ATR(m15).iloc[-1]
    )

    if not np.isfinite(atr_value):
        return None

    zone = near_zone(
        price,
        zones,
        atr_value
    )

    if (
        direction == "BUY"
        and zone == "DEMAND"
    ):
        score += 2
        reasons.append("Near demand zone")

    elif (
        direction == "SELL"
        and zone == "SUPPLY"
    ):
        score += 2
        reasons.append("Near supply zone")

    # =========================
    # CANDLE
    # =========================

    candle_score, candle_reasons = candle_confirmation(
        m15,
        direction
    )

    score += candle_score
    reasons.extend(candle_reasons)

    # =========================
    # FIBONACCI
    # =========================

    fib_score, fib_reason = fibonacci_confirmation(h1)

    score += fib_score

    if fib_reason:
        reasons.append(fib_reason)

    # =========================
    # MINIMUM CONFLUENCE
    # =========================

    if score < MIN_SCORE:
        return None

    # =========================
    # ENTRY / SL / TP
    # =========================

    if direction == "BUY":

        recent_low = float(
            m15["Low"].tail(12).min()
        )

        risk = max(
            price - recent_low,
            atr_value * 1.2
        )

        sl = price - risk
        tp1 = price + risk * 1.5
        tp2 = price + risk * 2.0

    else:

        recent_high = float(
            m15["High"].tail(12).max()
        )

        risk = max(
            recent_high - price,
            atr_value * 1.2
        )

        sl = price + risk
        tp1 = price - risk * 1.5
        tp2 = price - risk * 2.0

    return {
        "symbol": symbol_name,
        "ticker": ticker,
        "direction": direction,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "score": min(score, 10),
        "reasons": reasons,
        "timeframe": "M15",
        "time": datetime.now(
            timezone.utc
        ).isoformat()
    }


# =========================
# JOURNAL
# =========================

def load_journal():

    if not os.path.exists(JOURNAL_FILE):
        return {
            "open": [],
            "closed": []
        }

    try:

        with open(
            JOURNAL_FILE,
            "r"
        ) as file:

            return json.load(file)

    except Exception:

        return {
            "open": [],
            "closed": []
        }


def save_journal(journal):

    with open(
        JOURNAL_FILE,
        "w"
    ) as file:

        json.dump(
            journal,
            file,
            indent=2
        )


# =========================
# CHECK OLD TRADES
# =========================

def check_open_trades(journal):

    still_open = []

    for trade in journal["open"]:

        df = get_data(
            trade["ticker"],
            "2d",
            "15m"
        )

        if df.empty:
            still_open.append(trade)
            continue

        entry_time = pd.to_datetime(
            trade["time"]
        )

        result = None
        exit_price = None

        for timestamp, candle in df.iterrows():

            try:

                candle_time = pd.to_datetime(
                    timestamp
                )

                if candle_time <= entry_time:
                    continue

            except Exception:
                continue

            if trade["direction"] == "BUY":

                # Conservative rule:
                # if SL and TP are both inside
                # one candle, SL is counted first.

                if candle["Low"] <= trade["sl"]:

                    result = "LOSS"
                    exit_price = trade["sl"]
                    break

                if candle["High"] >= trade["tp2"]:

                    result = "WIN"
                    exit_price = trade["tp2"]
                    break

            else:

                if candle["High"] >= trade["sl"]:

                    result = "LOSS"
                    exit_price = trade["sl"]
                    break

                if candle["Low"] <= trade["tp2"]:

                    result = "WIN"
                    exit_price = trade["tp2"]
                    break

        if result:

            trade["result"] = result
            trade["exit"] = exit_price
            trade["closed_at"] = datetime.now(
                timezone.utc
            ).isoformat()

            journal["closed"].append(trade)

        else:

            still_open.append(trade)

    journal["open"] = still_open


# =========================
# WIN RATE
# =========================

def get_statistics(journal):

    wins = sum(
        1
        for trade in journal["closed"]
        if trade.get("result") == "WIN"
    )

    losses = sum(
        1
        for trade in journal["closed"]
        if trade.get("result") == "LOSS"
    )

    total = wins + losses

    if total > 0:
        win_rate = (
            wins / total
        ) * 100
    else:
        win_rate = 0

    return wins, losses, total, win_rate


# =========================
# TELEGRAM MESSAGE
# =========================

def make_message(signal, journal):

    wins, losses, total, win_rate = get_statistics(
        journal
    )

    confirmation = "\n".join(
        f"• {reason}"
        for reason in signal["reasons"]
    )

    return f"""
🚨 FOREX AI SIGNAL

📌 {signal["direction"]} — {signal["symbol"]}
⏱ Timeframe: M15

📍 Entry: {signal["entry"]:.5f}
🛑 Stop Loss: {signal["sl"]:.5f}
🎯 TP1: {signal["tp1"]:.5f}
🎯 TP2: {signal["tp2"]:.5f}

📊 Confluence Score: {signal["score"]}/10

✅ Confirmation:
{confirmation}

🏆 BOT WIN RATE: {win_rate:.1f}%

📈 Closed Trades: {total}
✅ Wins: {wins}
❌ Losses: {losses}

⚠️ Risk management required.
This is a trading signal, not a guaranteed profit.
""".strip()


# =========================
# MAIN
# =========================

def main():

    journal = load_journal()

    # First check previous signals
    check_open_trades(journal)

    already_open = {
        trade["symbol"]
        for trade in journal["open"]
    }

    # BTC + GOLD
    for symbol_name, ticker in SYMBOLS.items():

        try:

            signal = analyze(
                symbol_name,
                ticker
            )

            if not signal:

                print(
                    symbol_name,
                    "NO TRADE"
                )

                continue

            # Do not spam duplicate signals
            if symbol_name in already_open:

                print(
                    symbol_name,
                    "already has an open trade"
                )

                continue

            message = make_message(
                signal,
                journal
            )

            if send_telegram(message):

                journal["open"].append(
                    signal
                )

                already_open.add(
                    symbol_name
                )

                print(
                    symbol_name,
                    signal["direction"],
                    "SCORE",
                    signal["score"]
                )

        except Exception as error:

            print(
                symbol_name,
                "ERROR:",
                repr(error)
            )

    save_journal(journal)


if __name__ == "__main__":
    main()

