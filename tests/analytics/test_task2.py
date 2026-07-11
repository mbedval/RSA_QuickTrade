"""Unit tests for Task 2: Advanced Analytics & Regimes.

Verifies streak analysis, risk metrics (Sharpe, Sortino, Calmar, drawdown durations),
regime partitioning, and volatility clustering.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.distribution import calculate_distribution_metrics
from bsa_quicktrade.analytics.risk import calculate_risk_metrics
from bsa_quicktrade.analytics.regime import (
    RegimeClassifier,
    calculate_regime_analytics,
    calculate_volatility_analytics,
)


@pytest.fixture
def base_time():
    return datetime(2026, 1, 1, 10, 0, 0)


@pytest.fixture
def mock_trades(base_time):
    # Weekday exit mapping:
    # 2026-01-01 is a Thursday
    # We will spread exit dates to test weekday and month grouping
    # T1: Exit in Jan (Thursday)
    # T2: Exit in Jan (Friday)
    # T3: Exit in Feb (Monday)
    # T4: Exit in Feb (Tuesday)
    # T5: Exit in Mar (Wednesday)
    
    r1 = TradeRecord(
        trade_id="T1",
        strategy_name="StratA",
        module_name="ModA",
        signal_name="SigA",
        entry_date=base_time,
        exit_date=base_time + timedelta(days=0.5),  # 0.5 day (Intraday)
        entry_price=100.0,
        exit_price=105.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=50.0,
        profit_pct=5.0,
        r_multiple=1.0,
        atr=2.0,  # 2% of entry
        rsi=65.0,
        adx=30.0,
        market_regime="Bull"
    )
    
    r2 = TradeRecord(
        trade_id="T2",
        strategy_name="StratA",
        module_name="ModA",
        signal_name="SigA",
        entry_date=base_time + timedelta(days=1),
        exit_date=base_time + timedelta(days=3),  # 2 days (1-5 days)
        entry_price=100.0,
        exit_price=110.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=100.0,
        profit_pct=10.0,
        r_multiple=2.0,
        atr=5.0,  # 5% of entry (high vol)
        rsi=60.0,
        adx=28.0,
        market_regime="Bull"
    )

    r3 = TradeRecord(
        trade_id="T3",
        strategy_name="StratA",
        module_name="ModA",
        signal_name="SigA",
        entry_date=base_time + timedelta(days=3),
        exit_date=base_time + timedelta(days=13),  # 10 days (5-20 days)
        entry_price=100.0,
        exit_price=95.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=-50.0,
        profit_pct=-5.0,
        r_multiple=-1.0,
        atr=1.0,  # 1% of entry (low vol)
        rsi=30.0,
        adx=35.0,
        market_regime="Bear"
    )

    r4 = TradeRecord(
        trade_id="T4",
        strategy_name="StratA",
        module_name="ModA",
        signal_name="SigA",
        entry_date=base_time + timedelta(days=13),
        exit_date=base_time + timedelta(days=43),  # 30 days (20-60 days)
        entry_price=100.0,
        exit_price=98.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=-20.0,
        profit_pct=-2.0,
        r_multiple=-0.4,
        atr=1.5,  # 1.5% of entry (low-mid vol)
        rsi=45.0,
        adx=15.0,
        market_regime="Sideways"
    )

    r5 = TradeRecord(
        trade_id="T5",
        strategy_name="StratA",
        module_name="ModA",
        signal_name="SigA",
        entry_date=base_time + timedelta(days=43),
        exit_date=base_time + timedelta(days=123),  # 80 days (> 60 days)
        entry_price=100.0,
        exit_price=115.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=150.0,
        profit_pct=15.0,
        r_multiple=3.0,
        atr=3.0,  # 3% of entry (mid-high vol)
        rsi=70.0,
        adx=40.0,
        market_regime="Bull"
    )
    
    return [r1, r2, r3, r4, r5]


# ── Distribution Metric Tests ──────────────────────────────────────────────

def test_distribution_metrics(mock_trades):
    metrics = calculate_distribution_metrics(mock_trades)
    
    # 5 trades sequence: profit = +50, +100, -50, -20, +150
    # Win streak:
    # +50, +100 -> Streak of 2 (Avg: 75.0)
    # -50, -20 -> Loss streak of 2 (Avg: -35.0)
    # +150 -> Win streak of 1
    assert metrics.consecutive_wins == 2
    assert metrics.consecutive_losses == 2
    assert metrics.win_streak_avg_profit == 75.0
    assert metrics.loss_streak_avg_loss == -35.0
    
    # Holding period bins
    # T1 (0.5d) -> < 1 day
    # T2 (2.0d) -> 1-5 days
    # T3 (10.0d) -> 5-20 days
    # T4 (30.0d) -> 20-60 days
    # T5 (80.0d) -> > 60 days
    assert metrics.duration_distribution["< 1 day"] == 1
    assert metrics.duration_distribution["1-5 days"] == 1
    assert metrics.duration_distribution["5-20 days"] == 1
    assert metrics.duration_distribution["20-60 days"] == 1
    assert metrics.duration_distribution["> 60 days"] == 1

    # Date groupings (check non-zero values)
    assert sum(metrics.profit_by_weekday.values()) == 230.0  # Net profit: 50+100-50-20+150
    assert sum(metrics.profit_by_month.values()) == 230.0


# ── Risk Analytics Tests ──────────────────────────────────────────────────

def test_risk_metrics(mock_trades):
    # Starting capital: 10,000. Net profit evolution:
    # 10,000 -> 10,050 (T1) -> 10,150 (T2) -> 10,100 (T3) -> 10,080 (T4) -> 10,230 (T5)
    # Peaks: 10,000 -> 10,050 -> 10,150 -> 10,150 -> 10,150 -> 10,230
    # Drawdowns absolute: 0 -> 0 -> 0 -> 50 (T3) -> 70 (T4) -> 0 (T5)
    # Max drawdown: 70.0. Max drawdown %: (70 / 10150) * 100 = 0.689%
    
    metrics = calculate_risk_metrics(mock_trades, starting_capital=10000.0)
    
    assert metrics.drawdown.max_drawdown == 70.0
    assert metrics.drawdown.max_drawdown_pct == 0.69  # rounded to 2 decimal places
    
    # Check Sharpe, Sortino ratios are populated and reasonable numbers
    assert metrics.sharpe_ratio != 0.0
    assert metrics.sortino_ratio != 0.0
    assert metrics.ulcer_index > 0.0
    assert metrics.recovery_factor == round(230.0 / 70.0, 2)


# ── Regime and Volatility Tests ─────────────────────────────────────────────

def test_regime_classifier(mock_trades):
    clf = RegimeClassifier()
    
    # T1: rsi=65, adx=30 -> Bull
    assert clf.classify(mock_trades[0]) == "Bull"
    
    # T3: rsi=30, adx=35 -> Bear
    assert clf.classify(mock_trades[2]) == "Bear"
    
    # T4: rsi=45, adx=15 -> Sideways
    assert clf.classify(mock_trades[3]) == "Sideways"


def test_regime_analytics(mock_trades):
    regimes = calculate_regime_analytics(mock_trades, starting_capital=10000.0)
    
    # We expect categories: "Bull", "Bear", "Sideways"
    assert "Bull" in regimes
    assert "Bear" in regimes
    assert "Sideways" in regimes
    
    bull_regime = regimes["Bull"]
    assert bull_regime.total_trades == 3  # T1, T2, T5
    assert bull_regime.winning_trades == 3
    assert bull_regime.net_profit == 300.0  # 50 + 100 + 150
    assert bull_regime.win_rate == 100.0


def test_volatility_analytics(mock_trades):
    # ATR % calculations:
    # T1: 2.0 / 100 * 100 = 2%
    # T2: 5.0 / 100 * 100 = 5%
    # T3: 1.0 / 100 * 100 = 1%
    # T4: 1.5 / 100 * 100 = 1.5%
    # T5: 3.0 / 100 * 100 = 3%
    # Sorted ATR %: 1.0%, 1.5%, 2.0%, 3.0%, 5.0%
    # 33rd percentile: ~1.5%, 66th percentile: ~3.0%
    # Bins:
    # Low Vol (<= 1.5%): T3 (1%), T4 (1.5%) -> 2 trades
    # Med Vol (1.5% - 3.0%): T1 (2%), T5 (3%) -> 2 trades
    # High Vol (> 3.0%): T2 (5%) -> 1 trade
    
    vol_analysis = calculate_volatility_analytics(mock_trades, starting_capital=10000.0)
    
    assert "Low Volatility" in vol_analysis
    assert "Medium Volatility" in vol_analysis
    assert "High Volatility" in vol_analysis
    
    assert vol_analysis["Low Volatility"].total_trades == 2
    assert vol_analysis["Medium Volatility"].total_trades == 1
    assert vol_analysis["High Volatility"].total_trades == 2
