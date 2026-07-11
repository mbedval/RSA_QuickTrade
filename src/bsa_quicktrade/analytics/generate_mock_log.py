"""Mock TradeLog Generator for the Backtest Analytics Engine.

Creates a sample trade log CSV with mock indicators and regimes for testing.
"""

import csv
from datetime import datetime, timedelta
import random
from pathlib import Path

# Target file
output_file = Path("output/sample_tradelog.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

# Columns matching designprompt.md schema
headers = [
    "Trade ID", "Strategy Name", "Module Name", "Signal Name",
    "Entry Date", "Exit Date", "Entry Price", "Exit Price", "Quantity",
    "Capital Used", "Stop Loss", "Target Price", "Holding Period",
    "ATR", "ADX", "RSI", "Market Regime", "Exit Reason", "Triggered Rules",
    "Confidence Score", "Brokerage", "Taxes", "Slippage", "Remarks"
]

start_date = datetime(2026, 1, 1, 9, 30, 0)
trades = []

regimes = ["Bull", "Bear", "Sideways"]
reasons = ["Target", "Stop Loss", "Trailing Stop", "Time Exit"]
rules = [
    ["EMA_Cross_Up", "RSI_Oversold"],
    ["EMA_Cross_Down", "RSI_Overbought"],
    ["RSI_Oversold", "BB_Lower_Touch"],
    ["BB_Upper_Touch"],
    ["MACD_Bullish_Cross"],
    ["Support_Bounce"]
]

for i in range(1, 41):  # 40 mock trades
    entry_price = round(random.uniform(100.0, 1500.0), 2)
    qty = random.randint(10, 200)
    capital = round(entry_price * qty, 2)
    
    # Win or Loss
    regime = random.choice(regimes)
    is_win = random.random() < (0.65 if regime == "Bull" else 0.40)
    
    if is_win:
        exit_price = round(entry_price * (1 + random.uniform(0.01, 0.08)), 2)
        reason = "Target"
    else:
        exit_price = round(entry_price * (1 - random.uniform(0.01, 0.04)), 2)
        reason = "Stop Loss"
        
    holding_days = round(random.uniform(0.1, 15.0), 2)
    entry_dt = start_date + timedelta(days=i * 2 + random.uniform(0, 1))
    exit_dt = entry_dt + timedelta(days=holding_days)
    
    atr = round(entry_price * random.uniform(0.01, 0.04), 2)
    adx = round(random.uniform(10.0, 50.0), 2)
    rsi = round(random.uniform(20.0, 80.0), 2)
    
    sl = round(entry_price * 0.97, 2)
    tp = round(entry_price * 1.06, 2)
    
    trade_rules = random.choice(rules)
    
    trades.append({
        "Trade ID": f"T_{i:03d}",
        "Strategy Name": "QuickTrade_Alpha",
        "Module Name": "TrendFollower",
        "Signal Name": trade_rules[0],
        "Entry Date": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Exit Date": exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Entry Price": entry_price,
        "Exit Price": exit_price,
        "Quantity": qty,
        "Capital Used": capital,
        "Stop Loss": sl,
        "Target Price": tp,
        "Holding Period": holding_days,
        "ATR": atr,
        "ADX": adx,
        "RSI": rsi,
        "Market Regime": regime,
        "Exit Reason": reason,
        "Triggered Rules": ",".join(trade_rules),
        "Confidence Score": round(random.uniform(60.0, 95.0), 1),
        "Brokerage": round(capital * 0.0003, 2),
        "Taxes": round(capital * 0.0002, 2),
        "Slippage": round(capital * 0.0005, 2),
        "Remarks": f"Mock trade during {regime} regime"
    })

# Write CSV with DictWriter to handle commas in values correctly
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for t in trades:
        writer.writerow(t)

print(f"Mock trade log with 40 entries written to: {output_file}")
