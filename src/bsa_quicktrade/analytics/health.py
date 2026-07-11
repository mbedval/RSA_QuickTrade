"""Strategy Health Scoring module.

Evaluates trade logs against key quantitative criteria to calculate a consolidated
Strategy Health Score (0-100) and provide deployment recommendations.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from bsa_quicktrade.analytics.models import StrategyHealth, TradeRecord
from bsa_quicktrade.analytics.performance import calculate_performance_metrics
from bsa_quicktrade.analytics.risk import calculate_risk_metrics
from bsa_quicktrade.analytics.regime import calculate_regime_analytics
from bsa_quicktrade.analytics.attribution import calculate_signal_attribution

logger = logging.getLogger(__name__)


def calculate_strategy_health(
    records: List[TradeRecord],
    starting_capital: float = 100000.0
) -> StrategyHealth:
    """Calculate the Strategy Health Score (0-100) and generate rules-based advice."""
    if not records:
        return StrategyHealth(
            score=0.0,
            status="REJECT",
            reasons=["No trade records found"],
            recommendation="Do not deploy."
        )

    # 1. Compute all metric dependencies
    perf = calculate_performance_metrics(records, starting_capital)
    risk = calculate_risk_metrics(records, starting_capital)
    regimes = calculate_regime_analytics(records, starting_capital=starting_capital)
    attr = calculate_signal_attribution(records)

    # 2. Calculate sub-scores (0-100)
    
    # A. Profit Factor Score (25%)
    pf = perf.profit_factor
    if pf < 1.0:
        pf_score = 0.0
    elif pf >= 2.5:
        pf_score = 100.0
    else:
        pf_score = ((pf - 1.0) / 1.5) * 100.0

    # B. Sharpe Ratio Score (20%)
    sr = risk.sharpe_ratio
    if sr <= 0.0:
        sr_score = 0.0
    elif sr >= 3.0:
        sr_score = 100.0
    else:
        sr_score = (sr / 3.0) * 100.0

    # C. Drawdown Score (15%)
    dd = risk.drawdown.max_drawdown_pct
    if dd >= 30.0:
        dd_score = 0.0
    elif dd <= 5.0:
        dd_score = 100.0
    else:
        dd_score = ((30.0 - dd) / 25.0) * 100.0

    # D. Expectancy Score (15%)
    exp = perf.expectancy
    avg_cap = float(np.mean([r.capital_used for r in records])) if records else 100000.0
    exp_pct = (exp / avg_cap) * 100.0 if avg_cap > 0 else 0.0
    if exp_pct <= 0.0:
        exp_score = 0.0
    elif exp_pct >= 2.0:
        exp_score = 100.0
    else:
        exp_score = (exp_pct / 2.0) * 100.0

    # E. CAGR Score (10%)
    cagr = perf.cagr
    if cagr <= 0.0:
        cagr_score = 0.0
    elif cagr >= 30.0:
        cagr_score = 100.0
    else:
        cagr_score = (cagr / 30.0) * 100.0

    # F. Market Regime Stability (10%)
    active_regimes = [r for r in regimes.values() if r.total_trades > 0]
    if not active_regimes:
        regime_score = 50.0
    else:
        profitable_regimes = [r for r in active_regimes if r.net_profit > 0]
        regime_score = (len(profitable_regimes) / len(active_regimes)) * 100.0

    # G. Trade Count Score (5%)
    n = len(records)
    if n <= 5:
        n_score = 0.0
    elif n >= 40:
        n_score = 100.0
    else:
        n_score = ((n - 5) / 35.0) * 100.0

    # 3. Weighted score compilation
    weighted_score = (
        (pf_score * 0.25) +
        (sr_score * 0.20) +
        (dd_score * 0.15) +
        (exp_score * 0.15) +
        (cagr_score * 0.10) +
        (regime_score * 0.10) +
        (n_score * 0.05)
    )
    weighted_score = round(max(0.0, min(100.0, weighted_score)), 1)

    # 4. Generate reasons for warn/reject
    reasons = []
    if perf.expectancy < 0:
        reasons.append("Negative expectancy")
    if perf.profit_factor < 1.4:
        reasons.append("Poor profit factor")
    if risk.sharpe_ratio < 1.0:
        reasons.append("Low Sharpe ratio")
    if risk.drawdown.max_drawdown_pct > 20.0:
        reasons.append("High drawdown risk")
    if regime_score < 50.0:
        reasons.append("No profitable market regime")
    
    # Check false signal rate of indicators
    high_false_sig = False
    for ind_m in attr.indicators.values():
        if ind_m.false_signal_rate > 50.0 and ind_m.total_signals >= 5:
            high_false_sig = True
            break
    if high_false_sig:
        reasons.append("High false signal rate")
        
    if n < 15:
        reasons.append("Low statistical significance (insufficient trade count)")

    # 5. Determine Status and Recommendation
    if weighted_score >= 70.0:
        status = "PASS"
        recommendation = "Highly suitable for live deployment. Start with standard sizing."
    elif weighted_score >= 50.0:
        status = "WARNING"
        recommendation = "Deploy with caution. Reduce position sizing (e.g. 50%) and monitor closely."
    else:
        status = "REJECT"
        recommendation = "Do not deploy."

    return StrategyHealth(
        score=weighted_score,
        status=status,
        reasons=reasons if reasons else ["Meets all quantitative thresholds"],
        recommendation=recommendation
    )
