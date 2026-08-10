


    if c < float(ema20.iloc[-1]) < float(ema50.iloc[-1]):
        score_sell += 2
        reasons_sell.append("Bearish EMA trend")

    # Hammer
    if lower_wick > body * 2 and upper_wick < body:
        score_buy += 2
        reasons_buy.append("Hammer")

    # Shooting star
    if upper_wick > body * 2 and lower_wick < body:
        score_sell += 2
        reasons_sell.append("Shooting Star")

    # Bullish / bearish candle
    if c > o:
        score_buy += 1
        reasons_buy.append("Bullish candle")

    if c < o:
        score_sell += 1
        reasons_sell.append("Bearish candle")

    # Recent support/resistance approximation
    support = float(low.iloc[-20:].min())
    resistance = float(high.iloc[-20:].max())

    if abs(c - support) / c < 0.002:
        score_buy += 2
        reasons_buy.append("Near support")

    if abs(resistance - c) / c < 0.002:
        score_sell += 2
        reasons_sell.append("Near resistance")

    # Only high-confluence setups
    if score_buy >= 5 and score_buy > score_sell:
        entry = c
        sl = min(support, entry * 0.995)
        risk = entry - sl

        if risk <= 0:
            return None

        tp1 = entry + risk * 2
        tp2 = entry + risk * 3

        return {
            "side": "BUY",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "score": score_buy,
            "reasons": reasons_buy
        }

    if score_sell >= 5 and score_sell > score_buy:
        entry = c
        sl = max(resistance, entry * 1.005)
        risk = sl - entry

        if risk <= 0:
            return None

        tp1 = entry - risk * 2
        tp2 = entry - risk * 3

        return {
            "side": "SELL",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "score": score_sell,
            "reasons": reasons_sell
        }

    return None


def scan_market():
    for name, ticker in SYMBOLS.items():

        df = get_data(ticker)

        if df is None:
            continue

        signal = detect_signal(df)

        if signal is None:
            print(name, "NO TRADE")
            continue

        message = f"""
🚨 FOREX AI SIGNAL

📌 {signal['side']} — {name}
⏱ Timeframe: M15

📍 Entry: {signal['entry']:.5f}
🛑 Stop Loss: {signal['sl']:.5f}
🎯 TP1: {signal['tp1']:.5f}
🎯 TP2: {signal['tp2']:.5f}

📊 Confluence Score: {signal['score']}

✅ Confirmation:
"""

        for reason in signal["reasons"]:
            message += f"• {reason}\n"

        message += """

        send_telegram(message)


if __name__ == "__main__":
    scan_market()
