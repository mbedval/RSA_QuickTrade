# SwingTrader.md

# Swing Trading Research Engine
## Design Prompt

---

# Objective

You are a Senior Quantitative Research Engineer and Python Architect.

Your responsibility is to redesign the existing Backtesting Framework into a professional Swing Trading Research Engine.

The framework is NOT intended for intraday trading.

The framework is NOT intended for long-term investing.

The objective is to identify high-probability swing trading opportunities that typically last between **5 and 26 trading days (approximately 1–5 weeks).**

The system should focus on

- High probability entries
- Low drawdown
- Trend following
- Position trading
- Risk controlled exits
- Capital preservation

The goal is NOT to maximize trade count.

The goal is to maximize expected return per selected trade.

---

# Primary Design Philosophy

The engine should answer

> Which stocks have the highest probability of producing profitable returns within the next 5–26 trading days?

instead of

> Which indicator generated a Buy signal?

Signal quality is more important than signal quantity.

---

# Trading Philosophy

The strategy should behave like an experienced swing trader.

The engine should

✓ Wait patiently

✓ Ignore mediocre setups

✓ Enter only during favorable market conditions

✓ Ride established trends

✓ Exit when trend weakens

✓ Avoid unnecessary trading

---

# Target Holding Period

Minimum

5 Trading Days

Preferred

10–20 Trading Days

Maximum

26 Trading Days

Trades should naturally exit earlier if

- Trend weakens
- Stop Loss hits
- Target achieved
- Trailing stop triggered

---

# Trading Workflow

Market Analysis

↓

Sector Analysis

↓

Stock Strength Analysis

↓

Weekly Trend Check

↓

Daily Setup Detection

↓

Entry Score Calculation

↓

Risk Validation

↓

Position Sizing

↓

Trade Execution

↓

Dynamic Exit Management

↓

Trade Analytics

---

# Stage 1
Market Filter

Never trade against the overall market.

Evaluate

NIFTY Trend

BANKNIFTY Trend

VIX

Market Breadth

Advance Decline Ratio

New High / New Low Ratio

If overall market is Bearish

Reduce exposure

or

Avoid new long trades.

---

# Stage 2
Sector Strength Filter

Calculate sector momentum.

Rank all sectors.

Only trade stocks belonging to the strongest sectors.

Example

Technology

★★★★★

Banking

★★★★☆

FMCG

★★☆☆☆

Metals

★☆☆☆☆

---

# Stage 3
Stock Relative Strength

Compare stock performance against

NIFTY

Sector Index

Relative Strength should be increasing.

Prefer

Stocks outperforming both

Market

and

Sector

---

# Stage 4
Weekly Trend Filter

Primary trend should always come from Weekly timeframe.

Examples

Weekly EMA20 > EMA50

Weekly Close > EMA20

Weekly Higher Highs

Weekly ADX > 25

Weekly Trend Score

Only Long trades when Weekly trend is bullish.

---

# Stage 5
Daily Entry Setup

Daily chart should provide entry timing.

Preferred entries

Pullback to EMA20

Bullish Engulfing

Inside Bar Breakout

Higher Low Formation

Volume Expansion

ATR Contraction before breakout

Avoid

Extended moves

Late breakouts

Exhaustion candles

---

# Stage 6
Trade Quality Score

Every trade should receive a score.

Example

Weekly Trend

20

Daily Trend

15

Relative Strength

15

Sector Strength

10

Market Strength

10

Volume Confirmation

10

ATR Condition

5

Trend Quality

10

Risk Reward

5

Price Structure

10

Total

100

Only execute trades above configurable threshold.

Example

Minimum

75/100

---

# Stage 7
Entry Validation

Before entering

Verify

Risk Reward

Minimum

1 : 2

Distance from EMA20

Not excessive

ATR

Within acceptable range

Volume

Above average

Trend

Healthy

If validation fails

Reject trade

---

# Stage 8
Dynamic Position Sizing

Never use fixed quantity.

Position size should depend on

Account Size

Risk Per Trade

ATR

Stop Distance

Maximum Capital Allocation

Example

Risk

1%

Position Size

Automatically calculated

---

# Stage 9
Exit Strategy

Do NOT use fixed exits.

Implement layered exits.

Initial Stop

2 ATR

Initial Target

3 ATR

When profit reaches

2 ATR

Move stop to Break Even.

After

3 ATR

Enable Trailing Stop.

Trailing methods

EMA20

SuperTrend

Chandelier Exit

ATR Trail

Maximum holding

26 trading days.

---

# Stage 10
Trade Health Monitoring

During open trade monitor

ATR Expansion

Volume Dry Up

Weekly Trend

RSI Failure

EMA Cross

Gap Risk

Reduce exposure if

Trend quality deteriorates.

---

# Stage 11
Forward Return Analysis

For every Buy Signal

Calculate average return after

5 Days

10 Days

15 Days

20 Days

26 Days

Generate

Average Return

Median Return

Win Rate

Maximum Return

Maximum Loss

This determines optimal holding period.

---

# Stage 12
Benchmark Comparison

Compare strategy against

Buy & Hold

Metrics

Return

Drawdown

Sharpe

Sortino

Calmar

Profit Factor

Holding Time

Capital Utilization

The strategy must outperform Buy & Hold on a risk-adjusted basis.

---

# Stage 13
Trade Explainability

Every trade must explain

Why entered

Why exited

Indicators involved

Confidence

Risk

Expected Reward

Market Regime

Sector Strength

Relative Strength

---

# Stage 14
Strategy Robustness

Validate strategy using

Walk Forward Testing

Rolling Window Testing

Multi-Year Testing

Different Market Regimes

Different Sectors

Different Market Caps

Large Cap

Mid Cap

Small Cap

The strategy should not rely on one stock.

---

# Stage 15
Research Dashboard

Generate

Trade Quality Distribution

Forward Return Analysis

Equity Curve

Drawdown Curve

Sector Performance

Market Regime Performance

Trade Duration Histogram

Win/Loss Streak

Indicator Contribution

Trade Replay

Monthly Heatmap

Yearly Heatmap

Capital Curve

---

# Engineering Standards

Python 3.12+

SOLID

Dependency Injection

Configuration Driven

Type Hints

Comprehensive Logging

Modular Architecture

Reusable Components

Unit Tested

No Placeholder Code

---

# Development Rules

Implement only one milestone at a time.

Each milestone must provide

Working code

Unit tests

README update

CLI example

Configuration changes

Example report

HTML preview

No milestone is complete until all tests pass.

---

# Success Criteria

The strategy should aim for

Average Holding

10–20 Trading Days

Maximum Holding

26 Trading Days

Trade Frequency

20–50 trades per stock per year

Win Rate

55–65%

Profit Factor

>1.50

Sharpe Ratio

>1.50

Sortino Ratio

>2.00

Maximum Drawdown

<15%

Recovery Factor

>2

Positive Expectancy

Across Bull, Bear, and Sideways markets

Consistent profitability across multiple stocks and sectors

The framework should prioritize high-quality trades with controlled risk over high trade frequency.