import os
import json
import time
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

# የሚፈለገው አነስተኛ የሲግናል ነጥብ (ከ 10)
MIN_SCORE = 4  

# GOLD only
SYMBOLS = {
    "XAU/USD": "XAUUSD=X"
}

# =========================
# TELEGRAM
# =========================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ BOT_TOKEN or CHAT_ID is missing")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=30
        )
        response.raise_for_status()
        print("✅ Telegram message sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Telegram send error: {repr(e)}")
        return False


# =========================
# MARKET DATA
# =========================

def get_data(symbol, period, interval):
    try:
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
    except Exception as e:
        print(f"Data fetch error ({symbol}): {e}")
        return pd.DataFrame()


def make_45m(df):
    if df.empty:
        return df
    return df.resample("45min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()


def make_4h(df):
    if df.empty:
        return df
    return df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()


# =========================
# INDICATORS
# =========================

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()


def ATR(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


# =========================
# TREND
# =========================

def get_trend(df):
    if len(df) < 50:
        return "NEUTRAL"

    ema20 = EMA(df["Close"], 20)
    ema50 = EMA(df["Close"], 50)

    current20 = ema20.iloc[-1]
    current50 = ema50.iloc[-1]

    if current20 > current50:
        return "BULLISH"
    elif current20 < current50:
        return "BEARISH"

    return "NEUTRAL"


# =========================
# SUPPLY / DEMAND
# =========================

def find_supply_demand(df):
    if len(df) < 50:
        return []

    zones = []
    recent = df.tail(80)
    ranges = recent["High"] - recent["Low"]
    average_range = ranges.median()

    for i in range(3, len(recent) - 3):
        base = recent.iloc[i-1:i+2]
        base_range = (base["High"] - base["Low"]).mean()

        previous = recent.iloc[i-2]
        future = recent.iloc[i+2]
        movement = future["Close"] - previous["Close"]

        # Demand Zone
        if base_range < average_range * 1.1 and movement > average_range * 1.2:
            zones.append({
                "type": "DEMAND",
                "low": float(base["Low"].min()),
                "high": float(base["High"].max())
            })

        # Supply Zone
        elif base_range < average_range * 1.1 and movement < -average_range * 1.2:
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

        distance = min(abs(price - low), abs(price - high))
        if distance <= atr_value * 0.5:
            return zone["type"]

    return None


# =========================
# CANDLE CONFIRMATION
# =========================

def candle_confirmation(df, direction):
    if len(df) < 2:
        return 0, []

    candle = df.iloc[-1]
    body = abs(candle["Close"] - candle["Open"])
    upper_wick = candle["High"] - max(candle["Open"], candle["Close"])
    lower_wick = min(candle["Open"], candle["Close"]) - candle["Low"]

    reasons = []

    if direction == "BUY":
        hammer = (lower_wick >= body * 1.5 and upper_wick <= body * 0.8)
        if hammer:
            reasons.append("Hammer Candle")

        previous = df.iloc[-2]
        engulfing = (
            candle["Close"] > candle["Open"]
            and candle["Open"] <= previous["Close"]
            and candle["Close"] >= previous["Open"]
        )
        if engulfing:
            reasons.append("Bullish Engulfing")

    else:
        shooting_star = (upper_wick >= body * 1.5 and lower_wick <= body * 0.8)
        if shooting_star:
            reasons.append("Shooting Star")

        previous = df.iloc[-2]
        engulfing = (
            candle["Close"] < candle["Open"]
            and candle["Open"] >= previous["Close"]
            and candle["Close"] <= previous["Open"]
        )
        if engulfing:
            reasons.append("Bearish Engulfing")

    return (2 if reasons else 0), reasons


# =========================
# FIBONACCI
# =========================

def fibonacci_confirmation(df):
    if len(df) < 60:
        return 0, None

    recent = df.tail(60)
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
        if abs(price - level) <= distance * 0.03:
            return 1, f"Near Fibonacci {name}"

    return 0, None


# =========================
# ANALYSIS
# =========================

def analyze(symbol_name, ticker):
    m15 = get_data(ticker, "7d", "15m")
    h1 = get_data(ticker, "30d", "1h")
    d1 = get_data(ticker, "2y", "1d")
    w1 = get_data(ticker, "5y", "1wk")

    if m15.empty or h1.empty or d1.empty or w1.empty:
        return None

    if len(m15) < 50:
        return None

    score = 0
    reasons = []

    # Trend Checks
    h1_trend = get_trend(h1)
    m15_trend = get_trend(m15)

    if m15_trend == "BULLISH":
        direction = "BUY"
    elif m15_trend == "BEARISH":
        direction = "SELL"
    else:
        return None

    # Trend alignment score
    if direction == "BUY" and h1_trend == "BULLISH":
        score += 2
        reasons.append("H1 & M15 Bullish EMA Trend")
    elif direction == "SELL" and h1_trend == "BEARISH":
        score += 2
        reasons.append("H1 & M15 Bearish EMA Trend")
    else:
        score += 1
        reasons.append(f"M15 Trend Alignment ({direction})")

    # Supply / Demand
    zones = find_supply_demand(h1)
    price = float(m15["Close"].iloc[-1])
    atr_series = ATR(m15)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 1.0

    if not np.isfinite(atr_value) or atr_value == 0:
        atr_value = 1.0

    zone = near_zone(price, zones, atr_value)

    if direction == "BUY" and zone == "DEMAND":
        score += 2
        reasons.append("Near Demand Zone")
    elif direction == "SELL" and zone == "SUPPLY":
        score += 2
        reasons.append("Near Supply Zone")

    # Candle Patterns
    candle_score, candle_reasons = candle_confirmation(m15, direction)
    score += candle_score
    reasons.extend(candle_reasons)

    # Fibonacci
    fib_score, fib_reason = fibonacci_confirmation(h1)
    score += fib_score
    if fib_reason:
        reasons.append(fib_reason)

    # Check minimum score requirement
    if score < MIN_SCORE:
        return None

    # SL / TP Levels
    if direction == "BUY":
        recent_low = float(m15["Low"].tail(10).min())
        risk = max(price - recent_low, atr_value * 1.2)
        sl = price - risk
        tp1 = price + risk * 1.5
        tp2 = price + risk * 2.0
    else:
        recent_high = float(m15["High"].tail(10).max())
        risk = max(recent_high - price, atr_value * 1.2)
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
        "time": datetime.now(timezone.utc).isoformat()
    }


# =========================
# JOURNAL
# =========================

def load_journal():
    if not os.path.exists(JOURNAL_FILE):
        return {"open": [], "closed": []}
    try:
        with open(JOURNAL_FILE, "r") as file:
            return json.load(file)
    except Exception:
        return {"open": [], "closed": []}


def save_journal(journal):
    with open(JOURNAL_FILE, "w") as file:
        json.dump(journal, file, indent=2)


def check_open_trades(journal):
    still_open = []
    for trade in journal["open"]:
        df = get_data(trade["ticker"], "2d", "15m")
        if df.empty:
            still_open.append(trade)
            continue

        entry_time = pd.to_datetime(trade["time"])
        result = None
        exit_price = None

        for timestamp, candle in df.iterrows():
            try:
                candle_time = pd.to_datetime(timestamp)
                if candle_time <= entry_time:
                    continue
            except Exception:
                continue

            if trade["direction"] == "BUY":
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
            trade["closed_at"] = datetime.now(timezone.utc).isoformat()
            journal["closed"].append(trade)
        else:
            still_open.append(trade)

    journal["open"] = still_open


def get_statistics(journal):
    wins = sum(1 for t in journal["closed"] if t.get("result") == "WIN")
    losses = sum(1 for t in journal["closed"] if t.get("result") == "LOSS")
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    return wins, losses, total, win_rate


# =========================
# TELEGRAM MESSAGE
# =========================

def make_message(signal, journal):
    wins, losses, total, win_rate = get_statistics(journal)
    confirmation = "\n".join(f"• {reason}" for reason in signal["reasons"])

    return f"""
🚨 FOREX AI SIGNAL

📌 {signal["direction"]} — {signal["symbol"]}
⏱ Timeframe: M15

📍 Entry: {signal["entry"]:.2f}
🛑 Stop Loss: {signal["sl"]:.2f}
🎯 TP1: {signal["tp1"]:.2f}
🎯 TP2: {signal["tp2"]:.2f}

📊 Confluence Score: {signal["score"]}/10

✅ Confirmation:
{confirmation}

🏆 BOT WIN RATE: {win_rate:.1f}%
📈 Closed Trades: {total} | Wins: {wins} | Losses: {losses}

⚠️ Risk management required.
""".strip()


# =========================
# MAIN JOB
# =========================

def run_bot():
    print(f"\n--- Running Bot Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    journal = load_journal()

    # Check state of open trades
    check_open_trades(journal)

    for symbol_name, ticker in SYMBOLS.items():
        try:
            signal = analyze(symbol_name, ticker)

            if not signal:
                print(f"[{symbol_name}] NO TRADE (Conditions or MIN_SCORE not met)")
                continue

            message = make_message(signal, journal)

            if send_telegram(message):
                journal["open"].append(signal)
                print(f"[{symbol_name}] {signal['direction']} SIGNAL SENT (Score: {signal['score']})")

        except Exception as error:
            print(f"[{symbol_name}] ERROR: {repr(error)}")

    save_journal(journal)


if __name__ == "__main__":
    # በየ 1 ሰዓቱ (3600 ሰከንድ) ኮዱን የሚደግም Loop
    while True:
        run_bot()
        print("Waiting 1 hour for the next run...")
        time.sleep(3600)
