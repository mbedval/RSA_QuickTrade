# DesignPrompt.md
# Backtest Analytics & Research Engine – Development Prompt

## Objective

You are acting as a Senior Quantitative Software Engineer and Python Architect.

Your goal is to extend an existing modular Python Trading Framework by designing and implementing a **production-grade Backtest Analytics & Research Engine**.

This is **NOT** a one-time script.

The objective is to build a reusable analytics framework that can evaluate any completed backtest and produce professional quantitative reports similar to institutional trading systems.

The implementation must be incremental, modular, testable, and maintainable.

---

# Project Philosophy

The system should support three major use cases:

1. Strategy Research
2. Strategy Validation
3. Strategy Comparison

The system should allow future strategies to plug into the analytics engine without requiring any code changes.

Every analytical report must be reproducible from stored trade data.

The framework should eventually support

- Swing Trading
- Intraday Trading
- Positional Trading
- Multi-timeframe Strategies
- Multi-Asset Support

without architectural changes.

---

# Primary Goal

Transform raw trade execution data into actionable quantitative insights.

The framework should answer questions such as

- Is this strategy profitable?
- Why is it profitable?
- Under which market conditions does it work?
- Which indicators contribute the most?
- Which combinations of indicators perform best?
- Which market regime produces maximum return?
- What are the largest risks?
- Can this strategy be trusted in live trading?

---

# Core Architecture

```
Historical Data
        │
        ▼
Signal Engine
        │
        ▼
Trade Simulator
        │
        ▼
Trade Log
        │
        ▼
Analytics Engine
        │
 ┌──────────────┬──────────────┬──────────────┬──────────────┐
 │              │              │              │
 ▼              ▼              ▼              ▼
Performance   Risk      Attribution     Market Regime
        │
        ▼
Visualization Engine
        │
        ▼
HTML Dashboard
PDF Report
CSV
JSON
```

---

# Golden Rule

TradeLog is the single source of truth.

Every metric, chart, visualization, report, and comparison must be generated from TradeLog.

Never calculate analytics directly from OHLC candles.

Never duplicate calculations.

Everything must be derived from completed trades.

---

# Suggested Folder Structure

```
framework/

    analytics/

        models.py

        trade_log.py

        metrics.py

        performance.py

        distribution.py

        attribution.py

        regime.py

        risk.py

        visualization.py

        report_generator.py

        exporter.py

        utils.py

        __init__.py
```

Each module must have only one responsibility.

---

# Standard TradeLog Schema

Each completed trade must contain at minimum

```
Trade ID
Strategy Name
Module Name
Signal Name

Entry Date
Exit Date

Entry Price
Exit Price

Quantity

Capital Used

Stop Loss
Target Price

Holding Period

ATR
ADX
RSI
EMA Values
VWAP
Volume
Market Breadth

Market Regime

Profit
Profit %
Risk
Reward
Risk Reward Ratio
R-Multiple

Exit Reason

Entry Indicator Snapshot
Exit Indicator Snapshot

Triggered Rules

Confidence Score

Brokerage
Taxes
Slippage

Remarks
```

TradeLog should be serializable to

- CSV
- JSON
- SQLite (future)
- PostgreSQL (future)

---

# Functional Requirements

The analytics engine should support the following categories.

---

# 1. Performance Analytics

Implement

- Total Trades
- Winning Trades
- Losing Trades
- Win Rate
- Average Return
- Gross Profit
- Gross Loss
- Net Profit
- CAGR
- Profit Factor
- Payoff Ratio
- Average Win
- Average Loss
- Largest Win
- Largest Loss
- Expectancy
- Average R-Multiple

---

# 2. Trade Distribution

Generate

- Profit Histogram
- Loss Histogram
- Return Distribution
- Trade Duration Histogram
- Holding Period Distribution
- Win/Loss Distribution
- Profit by Weekday
- Profit by Month
- Trade Frequency
- Consecutive Wins
- Consecutive Losses
- Win/Loss Streak Analysis

---

# 3. Market Regime Analysis

Automatically classify every trade into

- Bull Market
- Bear Market
- Sideways Market

Classification should be configurable.

Possible indicators

- EMA200
- ADX
- ATR
- Market Breadth
- Index Trend

Generate

- Performance per Regime
- Win Rate per Regime
- Sharpe Ratio
- Profit Factor
- Drawdown
- Average Holding Time
- Best Strategy per Regime
- Worst Strategy per Regime

---

# 4. Volatility Analysis

Classify every trade into

- Low ATR
- Medium ATR
- High ATR

Generate

- Return Distribution
- Win Rate
- Average Holding Time
- Drawdown
- Expectancy
- Profit Factor

---

# 5. Signal Attribution

Every trade must record

- Triggered Indicators
- Triggered Rules
- Indicator Values
- Confidence Score

Generate

Contribution of

- EMA
- SMA
- RSI
- MACD
- VWAP
- ATR
- Ichimoku
- Candlestick
- Volume
- Chart Pattern
- Market Breadth

Also generate

- Best Indicator Combination
- Worst Indicator Combination
- False Signal Frequency
- Indicator Success Rate
- Indicator Failure Rate

---

# 6. Risk Analytics

Implement

- Maximum Drawdown
- Average Drawdown
- Recovery Factor
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Ulcer Index
- MAR Ratio
- Profit Factor
- Expectancy
- Payoff Ratio
- Consecutive Losses
- Consecutive Wins
- Longest Drawdown
- Time to Recovery

---

# 7. Time Analysis

Generate

- Monthly Returns
- Quarterly Returns
- Yearly Returns
- Day-wise Performance
- Week-wise Performance
- Month-wise Performance
- Session Performance
- Intraday Time Analysis (future)

---

# 8. Visualization Engine

Generate professional charts.

Required

- Equity Curve
- Drawdown Curve
- Monthly Return Heatmap
- Yearly Return Heatmap
- Profit Histogram
- Holding Time Histogram
- Rolling 30-day Sharpe Ratio
- Rolling Return
- Rolling Drawdown
- Rolling Win Rate
- Risk vs Return Scatter Plot
- ATR vs Return
- Volume vs Return
- Indicator Contribution Chart
- Regime Performance Chart

Preferred library

- Plotly

Static fallback

- Matplotlib

---

# 9. Trade Replay

This is a future premium feature.

Every trade should be replayable.

Display

- Candlestick Chart
- Indicators
- Entry Candle
- Exit Candle
- Stop Loss
- Target
- Trailing Stop
- Triggered Rules
- Indicator Values
- Profit/Loss Evolution

The replay should explain

WHY

the trade was entered

and

WHY

the trade exited.

---

# 10. Trade Overlay

Overlay trades on historical charts.

Display

BUY Marker

SELL Marker

Entry Price

Exit Price

Stop Loss

Target

Indicator Values

Profit Label

Trade Number

Exit Reason

Use interactive Plotly charts.

---

# 11. Strategy Comparison

Support comparison between multiple strategies.

Generate comparison tables for

- CAGR
- Win Rate
- Drawdown
- Sharpe
- Sortino
- Calmar
- Recovery Factor
- Profit Factor
- Expectancy
- Average Holding
- Number of Trades

---

# 12. Professional Report

Generate a professional HTML report.

Sections

Executive Summary

Strategy Overview

Performance Summary

Risk Metrics

Trade Distribution

Market Regime Analysis

Indicator Contribution

Trade Statistics

Monthly Performance

Yearly Performance

Worst Trades

Best Trades

Recommendations

Charts

Appendix

Export formats

- HTML
- PDF
- CSV
- JSON

---

# Engineering Standards

Use

- Python 3.12+
- SOLID Principles
- Clean Architecture
- Dependency Injection where appropriate
- Type Hints
- Dataclasses or Pydantic Models
- Comprehensive Logging
- Configuration Driven
- Unit-Test Friendly Design
- Modular Components

Avoid

- Global Variables
- Duplicate Code
- Hardcoded Constants
- Circular Dependencies
- Monolithic Files
- Placeholder Implementations

---

# Development Process

Implement only ONE milestone at a time.

Each milestone must produce

- Working Code
- Unit Tests
- Sample Input
- Sample Output
- Updated README
- Example CLI Command
- HTML Preview (if applicable)

Do not continue to the next milestone until the current milestone is fully functional and reviewed.

---

# Suggested Development Milestones

### Milestone 1
TradeLog Standardization

### Milestone 2
Performance Analytics

### Milestone 3
Trade Distribution

### Milestone 4
Risk Metrics

### Milestone 5
Market Regime Analysis

### Milestone 6
Volatility Analysis

### Milestone 7
Signal Attribution

### Milestone 8
Visualization Engine

### Milestone 9
Trade Overlay

### Milestone 10
Professional HTML Report

### Milestone 11
Strategy Comparison

### Milestone 12
Trade Replay

---

# Definition of Done

A milestone is considered complete only if

- Code compiles successfully
- All unit tests pass
- Documentation is updated
- Example execution works
- HTML report is generated correctly
- Code follows project architecture
- Public APIs are documented
- No placeholder code remains
- Feature is reusable by future strategies without modification

The end goal is to create a reusable quantitative research platform capable of analyzing, validating, comparing, and visualizing trading strategies at a professional level while remaining modular, maintainable, and extensible.