"""Market Regime and Volatility Analytics.

Classifies trades into market regimes (Bull/Bear/Sideways) and volatility levels
(Low/Medium/High ATR) and calculates performance metrics per category.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from bsa_quicktrade.analytics.models import RegimeMetrics, TradeRecord
from bsa_quicktrade.analytics.performance import calculate_performance_metrics
from bsa_quicktrade.analytics.risk import calculate_risk_metrics

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """Configurable classifier for market regimes based on trade parameters."""

    def __init__(self, config: Optional[Dict[str, any]] = None) -> None:
        self.config = config or {}

    def classify(self, trade: TradeRecord) -> str:
        """Classify a single trade into a market regime.
        
        Falls back to pre-recorded regime if indicator rules are not met or missing.
        """
        # 1. Custom rule-based classification if indicators are available
        # Example rules using ADX, RSI, ATR, and Breadth
        adx = trade.adx
        rsi = trade.rsi
        breadth = trade.market_breadth

        # If we have sufficient indicators, apply rules
        if adx is not None and rsi is not None:
            # High ADX (> 25) means trending
            if adx > 25.0:
                if rsi > 55.0:
                    return "Bull"
                elif rsi < 45.0:
                    return "Bear"
            # Low ADX or neutral RSI means sideways
            if adx < 20.0 or (40.0 <= rsi <= 60.0):
                return "Sideways"

        if breadth is not None:
            if breadth > 70.0:
                return "Bull"
            elif breadth < 30.0:
                return "Bear"
            else:
                return "Sideways"

        # 2. Fall back to the pre-recorded regime in the trade log
        if trade.market_regime and trade.market_regime.lower() not in ("unknown", ""):
            # Capitalize first letter
            return trade.market_regime.strip().capitalize()

        return "Unknown"


def calculate_regime_analytics(
    records: List[TradeRecord],
    classifier: Optional[RegimeClassifier] = None,
    starting_capital: float = 100000.0
) -> Dict[str, RegimeMetrics]:
    """Partition trades by market regime and calculate performance summaries for each."""
    if not records:
        return {}

    clf = classifier or RegimeClassifier()
    
    # Group trades by regime
    regime_groups: Dict[str, List[TradeRecord]] = {}
    for r in records:
        regime = clf.classify(r)
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(r)

    # Compute metrics for each group
    results: Dict[str, RegimeMetrics] = {}
    for regime, group in regime_groups.items():
        # Calculate sub-performance and risk
        perf = calculate_performance_metrics(group, starting_capital=starting_capital)
        risk = calculate_risk_metrics(group, starting_capital=starting_capital)
        
        results[regime] = RegimeMetrics(
            regime_name=regime,
            total_trades=perf.total_trades,
            winning_trades=perf.winning_trades,
            losing_trades=perf.losing_trades,
            win_rate=perf.win_rate,
            net_profit=perf.net_profit,
            profit_factor=perf.profit_factor,
            avg_holding_period=round(float(np.mean([t.holding_period for t in group])), 2),
            max_drawdown_pct=risk.drawdown.max_drawdown_pct,
            sharpe_ratio=risk.sharpe_ratio,
            expectancy=perf.expectancy,
        )

    return results


def calculate_volatility_analytics(
    records: List[TradeRecord],
    starting_capital: float = 100000.0
) -> Dict[str, RegimeMetrics]:
    """Partition trades by asset volatility (ATR as % of Entry Price) and calculate performance.
    
    Uses percentiles to dynamically bin trades into Low, Medium, and High ATR.
    """
    if not records:
        return {}

    # Calculate ATR as % of entry price for each trade (if ATR is available)
    atr_pcts: List[float] = []
    valid_records: List[TradeRecord] = []
    
    for r in records:
        if r.atr is not None and r.entry_price > 0:
            atr_pct = (r.atr / r.entry_price) * 100.0
            atr_pcts.append(atr_pct)
            valid_records.append(r)
            
    # If we don't have enough valid ATR records, fall back to classification by exit reason or dummy
    if len(valid_records) < 3:
        # Fall back to uniform classification or return empty
        logger.warning("Fewer than 3 trades with valid ATR data. Skipping dynamic volatility partitioning.")
        return {}

    # Compute percentiles
    q33 = np.percentile(atr_pcts, 33.3)
    q66 = np.percentile(atr_pcts, 66.7)

    # Group trades
    groups: Dict[str, List[TradeRecord]] = {
        "Low Volatility": [],
        "Medium Volatility": [],
        "High Volatility": [],
    }

    for r in valid_records:
        atr_pct = (r.atr / r.entry_price) * 100.0
        if atr_pct <= q33:
            groups["Low Volatility"].append(r)
        elif atr_pct <= q66:
            groups["Medium Volatility"].append(r)
        else:
            groups["High Volatility"].append(r)

    # Compute metrics for each group
    results: Dict[str, RegimeMetrics] = {}
    for vol_label, group in groups.items():
        if not group:
            continue
        perf = calculate_performance_metrics(group, starting_capital=starting_capital)
        risk = calculate_risk_metrics(group, starting_capital=starting_capital)
        
        results[vol_label] = RegimeMetrics(
            regime_name=vol_label,
            total_trades=perf.total_trades,
            winning_trades=perf.winning_trades,
            losing_trades=perf.losing_trades,
            win_rate=perf.win_rate,
            net_profit=perf.net_profit,
            profit_factor=perf.profit_factor,
            avg_holding_period=round(float(np.mean([t.holding_period for t in group])), 2),
            max_drawdown_pct=risk.drawdown.max_drawdown_pct,
            sharpe_ratio=risk.sharpe_ratio,
            expectancy=perf.expectancy,
        )

    return results
