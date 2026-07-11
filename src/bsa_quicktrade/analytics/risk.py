"""Risk Analytics calculator.

Computes Sharpe, Sortino, Calmar, Ulcer Index, and detailed drawdown analysis.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import List, Tuple

import numpy as np

from bsa_quicktrade.analytics.models import DrawdownInfo, RiskMetrics, TradeRecord

logger = logging.getLogger(__name__)


def calculate_risk_metrics(
    records: List[TradeRecord],
    starting_capital: float = 100000.0
) -> RiskMetrics:
    """Calculate risk metrics, including Sharpe, Sortino, Calmar, and Drawdown parameters.
    
    Trades are sorted by exit_date to construct a chronological equity curve.
    """
    if not records:
        return RiskMetrics()

    # Sort trades by exit date
    sorted_records = sorted(records, key=lambda r: r.exit_date)
    
    # 1. Build Equity Curve
    equity = [starting_capital]
    dates = [sorted_records[0].entry_date - (sorted_records[0].exit_date - sorted_records[0].entry_date)]  # dummy start date
    
    current_equity = starting_capital
    for r in sorted_records:
        current_equity += r.net_profit
        equity.append(current_equity)
        dates.append(r.exit_date)

    # 2. Drawdown Analysis
    peaks = []
    current_peak = -1.0
    
    drawdowns_val = []
    drawdowns_pct = []
    
    for eq in equity:
        if eq > current_peak:
            current_peak = eq
        peaks.append(current_peak)
        
        dd_val = current_peak - eq
        dd_pct = (dd_val / current_peak) * 100.0 if current_peak > 0 else 0.0
        
        drawdowns_val.append(dd_val)
        drawdowns_pct.append(dd_pct)

    max_dd_val = max(drawdowns_val)
    max_dd_pct = max(drawdowns_pct)

    # Filter for non-zero drawdowns to compute average drawdown
    active_dd_val = [d for d in drawdowns_val if d > 0]
    active_dd_pct = [d for d in drawdowns_pct if d > 0]
    
    avg_dd_val = float(np.mean(active_dd_val)) if active_dd_val else 0.0
    avg_dd_pct = float(np.mean(active_dd_pct)) if active_dd_pct else 0.0

    # 3. Drawdown Durations & Recovery Times
    # A drawdown starts at index i where equity drops below the previous peak
    # and ends at index j where equity >= the peak at start.
    drawdown_durations: List[float] = []
    
    in_drawdown = False
    dd_start_date: datetime | None = None
    dd_peak_val = 0.0
    
    for i in range(1, len(equity)):
        eq = equity[i]
        dt = dates[i]
        prev_peak = peaks[i-1]
        
        if not in_drawdown and eq < prev_peak:
            # Entered drawdown
            in_drawdown = True
            dd_start_date = dates[i-1]  # start from the peak date
            dd_peak_val = prev_peak
            
        elif in_drawdown and eq >= dd_peak_val:
            # Recovered from drawdown
            if dd_start_date and dt:
                duration = (dt - dd_start_date).total_seconds() / 86400.0
                drawdown_durations.append(duration)
            in_drawdown = False
            dd_start_date = None
            
    # If still in drawdown at the end, measure duration up to the last date
    if in_drawdown and dd_start_date and dates[-1]:
        duration = (dates[-1] - dd_start_date).total_seconds() / 86400.0
        drawdown_durations.append(duration)

    longest_dd_days = max(drawdown_durations) if drawdown_durations else 0.0
    avg_recovery_days = float(np.mean(drawdown_durations)) if drawdown_durations else 0.0

    drawdown_info = DrawdownInfo(
        max_drawdown=round(max_dd_val, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        avg_drawdown=round(avg_dd_val, 2),
        avg_drawdown_pct=round(avg_dd_pct, 2),
        longest_drawdown_days=round(longest_dd_days, 1),
        time_to_recovery_days=round(avg_recovery_days, 1),
    )

    # 4. Ratios: Sharpe, Sortino, Calmar, Ulcer Index
    returns_pct = [r.profit_pct for r in sorted_records]
    
    # Sharpe Ratio (Trade-level returns standardized)
    # Using 252 trading sessions as standard annualization factor
    if len(returns_pct) > 1:
        ret_mean = float(np.mean(returns_pct))
        ret_std = float(np.std(returns_pct))
        
        # Annualized Sharpe: (mean / std) * sqrt(average trades per year)
        # We can estimate trades per year or fall back to standard sqrt(252) if total duration is small
        min_date = sorted_records[0].entry_date
        max_date = sorted_records[-1].exit_date
        total_years = (max_date - min_date).days / 365.25 if max_date and min_date else 0.0
        
        trades_per_year = len(returns_pct) / total_years if total_years > 0 else 252
        if trades_per_year <= 0:
            trades_per_year = 252
            
        sharpe = (ret_mean / ret_std * np.sqrt(trades_per_year)) if ret_std > 0 else 0.0
    else:
        sharpe = 0.0

    # Sortino Ratio
    # Downside deviation takes standard deviation of only negative returns
    if len(returns_pct) > 1:
        neg_returns = [r for r in returns_pct if r < 0]
        # downside deviation = sqrt( sum(neg_return^2) / N )
        if neg_returns:
            downside_dev = np.sqrt(np.sum(np.square(returns_pct)) / len(returns_pct))  # standard formulation uses total trade count
            # Let's use standard downside deviation:
            squared_negative_diffs = [r**2 for r in returns_pct if r < 0]
            downside_deviation = np.sqrt(sum(squared_negative_diffs) / len(returns_pct))
            
            # Annualized Sortino
            sortino = (ret_mean / downside_deviation * np.sqrt(trades_per_year)) if downside_deviation > 0 else 0.0
        else:
            sortino = float("inf") if ret_mean > 0 else 0.0
    else:
        sortino = 0.0

    # Ulcer Index
    # UI = sqrt( mean( drawdown_pct^2 ) )
    # Calculated on the series of drawdown percentages
    if drawdowns_pct:
        ulcer_index = np.sqrt(np.mean(np.square(drawdowns_pct)))
    else:
        ulcer_index = 0.0

    # Calmar Ratio & MAR Ratio
    # Calmar = CAGR / Max Drawdown %
    # First get CAGR
    net_profit = current_equity - starting_capital
    cagr = 0.0
    min_date = sorted_records[0].entry_date
    max_date = sorted_records[-1].exit_date
    if min_date and max_date:
        delta_days = (max_date - min_date).days
        years = delta_days / 365.25
        if years > 0.0027:
            if current_equity > 0 and starting_capital > 0:
                cagr = ((current_equity / starting_capital) ** (1 / years) - 1) * 100.0

    if max_dd_pct > 0:
        calmar = cagr / max_dd_pct
        mar = cagr / max_dd_pct
    else:
        calmar = 999.0 if cagr > 0 else 0.0
        mar = 999.0 if cagr > 0 else 0.0

    # Recovery Factor = Net Profit / Max Drawdown (absolute currency values)
    if max_dd_val > 0:
        recovery_factor = net_profit / max_dd_val
    else:
        recovery_factor = 999.0 if net_profit > 0 else 0.0

    return RiskMetrics(
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2) if sortino != float("inf") else 999.0,
        calmar_ratio=round(calmar, 2),
        ulcer_index=round(ulcer_index, 2),
        mar_ratio=round(mar, 2),
        recovery_factor=round(recovery_factor, 2),
        drawdown=drawdown_info,
    )
