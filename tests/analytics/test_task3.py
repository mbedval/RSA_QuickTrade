"""Unit tests for Task 3: Signal Attribution & Visualization Engine.

Verifies signal contribution analysis, rule combination ranking, and Plotly/Matplotlib
chart generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import pytest

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.attribution import calculate_signal_attribution
from bsa_quicktrade.analytics.visualization import VisualizationEngine
from bsa_quicktrade.analytics.regime import calculate_regime_analytics


@pytest.fixture
def base_time():
    return datetime(2026, 1, 1, 10, 0, 0)


@pytest.fixture
def mock_trades_for_attr(base_time):
    # We want to test signal attribution and rule combinations.
    # T1: Rules = [EMA_Cross, RSI_Oversold] -> Win (+100)
    # T2: Rules = [EMA_Cross, RSI_Oversold] -> Win (+200)
    # T3: Rules = [EMA_Cross, MACD_Bullish] -> Loss (-100)
    # T4: Rules = [RSI_Oversold] -> Loss (-50)
    
    r1 = TradeRecord(
        trade_id="T1",
        strategy_name="S1",
        module_name="M1",
        signal_name="EMA_Cross",
        entry_date=base_time,
        exit_date=base_time + timedelta(days=2),
        entry_price=100.0,
        exit_price=110.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=100.0,
        profit_pct=10.0,
        triggered_rules=["EMA_Cross", "RSI_Oversold"],
        entry_indicator_snapshot={"RSI": 30.0, "ADX": 25.0}
    )
    
    r2 = TradeRecord(
        trade_id="T2",
        strategy_name="S1",
        module_name="M1",
        signal_name="EMA_Cross",
        entry_date=base_time + timedelta(days=2),
        exit_date=base_time + timedelta(days=4),
        entry_price=100.0,
        exit_price=120.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=200.0,
        profit_pct=20.0,
        triggered_rules=["EMA_Cross", "RSI_Oversold"],
        entry_indicator_snapshot={"RSI": 28.0}
    )

    r3 = TradeRecord(
        trade_id="T3",
        strategy_name="S1",
        module_name="M1",
        signal_name="EMA_Cross",
        entry_date=base_time + timedelta(days=4),
        exit_date=base_time + timedelta(days=6),
        entry_price=100.0,
        exit_price=90.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=-100.0,
        profit_pct=-10.0,
        triggered_rules=["EMA_Cross", "MACD_Bullish"],
        entry_indicator_snapshot={"MACD": 0.5}
    )

    r4 = TradeRecord(
        trade_id="T4",
        strategy_name="S1",
        module_name="M1",
        signal_name="RSI_Oversold",
        entry_date=base_time + timedelta(days=6),
        exit_date=base_time + timedelta(days=8),
        entry_price=100.0,
        exit_price=95.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=-50.0,
        profit_pct=-5.0,
        triggered_rules=["RSI_Oversold"],
        entry_indicator_snapshot={"RSI": 25.0}
    )

    return [r1, r2, r3, r4]


# ── Attribution Tests ───────────────────────────────────────────────────────

def test_signal_attribution(mock_trades_for_attr):
    attr = calculate_signal_attribution(mock_trades_for_attr)
    
    # Check individual indicators
    # EMA_Cross is in T1, T2, T3 (2 wins, 1 loss) -> WR = 66.67%
    # RSI_Oversold is in T1, T2, T4 (2 wins, 1 loss) -> WR = 66.67%
    # MACD_Bullish is in T3 (1 loss) -> WR = 0%
    # RSI (from snapshot) is in T1, T2, T4 (2 wins, 1 loss)
    # ADX (from snapshot) is in T1 (1 win) -> WR = 100%
    
    assert "EMA_Cross" in attr.indicators
    assert attr.indicators["EMA_Cross"].total_signals == 3
    assert attr.indicators["EMA_Cross"].winning_signals == 2
    assert attr.indicators["EMA_Cross"].win_rate == 66.67
    assert attr.indicators["EMA_Cross"].false_signal_rate == 33.33
    
    assert "ADX" in attr.indicators
    assert attr.indicators["ADX"].win_rate == 100.0
    
    # Check combos
    # Combo 1: EMA_Cross + RSI_Oversold (T1, T2 -> 2 wins) -> WR = 100%
    # Combo 2: EMA_Cross + MACD_Bullish (T3 -> 1 loss) -> WR = 0%
    # Combo 3: RSI_Oversold (T4 -> 1 loss) -> WR = 0%
    assert attr.best_combination == "EMA_Cross + RSI_Oversold"
    assert attr.best_combination_win_rate == 100.0
    assert attr.worst_combination in ("EMA_Cross + MACD_Bullish", "RSI_Oversold")
    assert attr.worst_combination_win_rate == 0.0


# ── Visualization Engine Tests ──────────────────────────────────────────────

def test_visualization_engine(mock_trades_for_attr):
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = VisualizationEngine(output_dir=tmpdir)
        
        # 1. Test Interactive Plotly Generation (returns HTML div strings)
        eq_html = engine.generate_equity_curve(mock_trades_for_attr, use_plotly=True)
        assert isinstance(eq_html, str)
        assert "plotly-html-element" in eq_html or "div" in eq_html
        
        dd_html = engine.generate_drawdown_curve(mock_trades_for_attr, use_plotly=True)
        assert isinstance(dd_html, str)
        
        heatmap_html = engine.generate_monthly_heatmap(mock_trades_for_attr, use_plotly=True)
        assert isinstance(heatmap_html, str)
        
        dist_html = engine.generate_profit_distribution(mock_trades_for_attr, use_plotly=True)
        assert isinstance(dist_html, str)
        
        # Mock regime summary for bar chart
        regimes = calculate_regime_analytics(mock_trades_for_attr)
        regime_html = engine.generate_regime_chart(regimes, use_plotly=True)
        assert isinstance(regime_html, str)

        # 2. Test Static Matplotlib Generation (returns file Paths)
        eq_path = engine.generate_equity_curve(mock_trades_for_attr, use_plotly=False)
        assert isinstance(eq_path, Path)
        assert eq_path.exists()
        
        dd_path = engine.generate_drawdown_curve(mock_trades_for_attr, use_plotly=False)
        assert isinstance(dd_path, Path)
        assert dd_path.exists()
        
        heatmap_path = engine.generate_monthly_heatmap(mock_trades_for_attr, use_plotly=False)
        assert isinstance(heatmap_path, Path)
        assert heatmap_path.exists()
        
        dist_path = engine.generate_profit_distribution(mock_trades_for_attr, use_plotly=False)
        assert isinstance(dist_path, Path)
        assert dist_path.exists()
        
        regime_path = engine.generate_regime_chart(regimes, use_plotly=False)
        assert isinstance(regime_path, Path)
        assert regime_path.exists()
