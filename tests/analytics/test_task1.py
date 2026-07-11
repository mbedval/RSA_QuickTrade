"""Unit tests for Task 1: Core Foundation & Data Modeling.

Verifies utility parsing, trade log loading/saving, and performance calculations.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path

import pytest

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.performance import calculate_performance_metrics
from bsa_quicktrade.analytics.trade_log import load_trade_log, save_trade_log
from bsa_quicktrade.analytics.utils import parse_date, parse_dict, parse_float, parse_list


# ── Utility Parsing Tests ──────────────────────────────────────────────────

def test_parse_date():
    expected = datetime(2026, 7, 10, 15, 30, 0)
    
    # ISO strings
    assert parse_date("2026-07-10T15:30:00") == expected
    assert parse_date("2026-07-10 15:30:00") == expected
    
    # Common formats
    assert parse_date("10-07-2026 15:30:00") == expected
    assert parse_date("10-07-2026") == datetime(2026, 7, 10, 0, 0)
    assert parse_date("10/07/2026") == datetime(2026, 7, 10, 0, 0)
    
    # Date/datetime objects
    dt = datetime(2026, 7, 10, 12, 0)
    assert parse_date(dt) == dt
    
    # Epoch timestamps
    ts = datetime(1970, 1, 1, 5, 30, 0) # 0 UTC in IST (UTC+5:30) or UTC depending on local timezone
    assert parse_date(0).year == 1970

    with pytest.raises(ValueError):
        parse_date("invalid-date-string")


def test_parse_float():
    assert parse_float(12.34) == 12.34
    assert parse_float("1,234.56") == 1234.56
    assert parse_float("  $100.50 ") == 100.50
    assert parse_float("5.5%") == 5.5
    assert parse_float(None, default=9.9) == 9.9
    assert parse_float("NaN", default=0.0) == 0.0
    assert parse_float("invalid", default=-1.0) == -1.0


def test_parse_dict():
    # Valid JSON
    assert parse_dict('{"ema_20": 100.5, "ema_50": 98.2}') == {"ema_20": 100.5, "ema_50": 98.2}
    
    # Comma-separated pairs
    assert parse_dict("ema_20: 100.5, ema_50: 98.2") == {"ema_20": 100.5, "ema_50": 98.2}
    
    # Empty / None
    assert parse_dict("") == {}
    assert parse_dict(None) == {}
    
    # Pass-through dict
    d = {"test": 1}
    assert parse_dict(d) is not d  # should create a copy
    assert parse_dict(d) == d


def test_parse_list():
    # JSON list
    assert parse_list('["rule1", "rule2"]') == ["rule1", "rule2"]
    
    # Comma-separated
    assert parse_list("rule1, rule2") == ["rule1", "rule2"]
    
    # Empty
    assert parse_list("") == []
    assert parse_list(None) == []


# ── Trade Log Parsing & Loading Tests ──────────────────────────────────────

@pytest.fixture
def sample_csv_data():
    return (
        "Trade ID,Strategy Name,Entry Date,Exit Date,Entry Price,Exit Price,Quantity,Capital Used,Market Regime,Signal Name\n"
        "T1,TrendFollowing,2026-07-01 09:30:00,2026-07-05 15:30:00,100,110,10,1000,Bull,EMA_Cross_Up\n"
        "T2,MeanReversion,2026-07-02 10:00:00,2026-07-03 12:00:00,200,190,5,1000,Bear,RSI_Oversold\n"
    )


@pytest.fixture
def sample_json_data():
    return [
        {
            "trade_id": "T1",
            "strategy_name": "TrendFollowing",
            "entry_date": "2026-07-01 09:30:00",
            "exit_date": "2026-07-05 15:30:00",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "quantity": 10.0,
            "capital_used": 1000.0,
            "market_regime": "Bull",
            "signal_name": "EMA_Cross_Up"
        },
        {
            "trade_id": "T2",
            "strategy_name": "MeanReversion",
            "entry_date": "2026-07-02 10:00:00",
            "exit_date": "2026-07-03 12:00:00",
            "entry_price": 200.0,
            "exit_price": 190.0,
            "quantity": 5.0,
            "capital_used": 1000.0,
            "market_regime": "Bear",
            "signal_name": "RSI_Oversold"
        }
    ]


def test_load_save_csv(sample_csv_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "tradelog.csv"
        csv_path.write_text(sample_csv_data, encoding="utf-8")
        
        # Load
        records = load_trade_log(csv_path)
        assert len(records) == 2
        
        # Verify first record fields
        r1 = records[0]
        assert r1.trade_id == "T1"
        assert r1.strategy_name == "TrendFollowing"
        assert r1.entry_price == 100.0
        assert r1.exit_price == 110.0
        assert r1.quantity == 10.0
        assert r1.profit == 100.0  # (110 - 100) * 10
        assert r1.profit_pct == 10.0  # 100 / 1000 * 100
        assert r1.holding_period > 4.0  # 4.25 days
        assert r1.market_regime == "Bull"
        assert r1.signal_name == "EMA_Cross_Up"

        # Verify second record fields (loss)
        r2 = records[1]
        assert r2.trade_id == "T2"
        assert r2.profit == -50.0  # (190 - 200) * 5
        assert r2.profit_pct == -5.0
        
        # Save and Reload
        save_path = Path(tmpdir) / "saved_tradelog.csv"
        save_trade_log(records, save_path)
        
        reloaded = load_trade_log(save_path)
        assert len(reloaded) == 2
        assert reloaded[0].trade_id == "T1"
        assert reloaded[0].profit == 100.0


def test_load_save_json(sample_json_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "tradelog.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_json_data, f)
            
        records = load_trade_log(json_path)
        assert len(records) == 2
        assert records[0].trade_id == "T1"
        assert records[1].trade_id == "T2"
        
        # Save and Reload
        save_path = Path(tmpdir) / "saved_tradelog.json"
        save_trade_log(records, save_path)
        
        reloaded = load_trade_log(save_path)
        assert len(reloaded) == 2
        assert reloaded[0].trade_id == "T1"


# ── Performance Metric Calculator Tests ───────────────────────────────────

def test_performance_calculation():
    # Setup standard records
    base_time = datetime(2026, 1, 1, 10, 0, 0)
    
    r1 = TradeRecord(
        trade_id="T1",
        strategy_name="S1",
        module_name="M1",
        signal_name="Sig1",
        entry_date=base_time,
        exit_date=base_time + timedelta(days=5),
        entry_price=100.0,
        exit_price=120.0,  # +20 profit
        quantity=10.0,
        capital_used=1000.0,
        profit=200.0,
        profit_pct=20.0,
        r_multiple=2.0
    )
    
    r2 = TradeRecord(
        trade_id="T2",
        strategy_name="S1",
        module_name="M1",
        signal_name="Sig1",
        entry_date=base_time + timedelta(days=2),
        exit_date=base_time + timedelta(days=4),
        entry_price=100.0,
        exit_price=90.0,  # -10 profit
        quantity=10.0,
        capital_used=1000.0,
        profit=-100.0,
        profit_pct=-10.0,
        r_multiple=-1.0
    )

    r3 = TradeRecord(
        trade_id="T3",
        strategy_name="S1",
        module_name="M1",
        signal_name="Sig1",
        entry_date=base_time + timedelta(days=5),
        exit_date=base_time + timedelta(days=10),
        entry_price=100.0,
        exit_price=110.0,  # +10 profit
        quantity=10.0,
        capital_used=1000.0,
        profit=100.0,
        profit_pct=10.0,
        r_multiple=1.0
    )

    # 3 trades: 2 wins, 1 loss.
    # Total wins = 200 + 100 = 300
    # Total loss = -100
    # Net profit = 200
    # Win rate = 66.67%
    # Avg Return = 200 / 3 = 66.67
    # Avg Return % = (20 - 10 + 10) / 3 = 6.67%
    # Avg Win = 150
    # Avg Loss = -100
    # Profit Factor = 300 / 100 = 3.0
    # Payoff Ratio = 150 / 100 = 1.5
    # Expectancy = (2/3 * 150) + (1/3 * -100) = 100 - 33.33 = 66.67
    # Avg R-Multiple = (2 - 1 + 1) / 3 = 0.667
    
    metrics = calculate_performance_metrics([r1, r2, r3], starting_capital=1000.0)
    
    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 66.67
    assert metrics.gross_profit == 300.0
    assert metrics.gross_loss == 100.0
    assert metrics.net_profit == 200.0
    assert metrics.net_profit_pct == 20.0
    assert metrics.average_return == 66.67
    assert metrics.average_return_pct == 6.667
    assert metrics.average_win == 150.0
    assert metrics.average_loss == -100.0
    assert metrics.largest_win == 200.0
    assert metrics.largest_loss == -100.0
    assert metrics.profit_factor == 3.0
    assert metrics.payoff_ratio == 1.5
    assert metrics.expectancy == 66.67
    assert metrics.average_r_multiple == 0.667
    
    # Test empty input
    empty_metrics = calculate_performance_metrics([])
    assert empty_metrics.total_trades == 0
    assert empty_metrics.win_rate == 0.0
    assert empty_metrics.net_profit == 0.0
