"""Performance Analytics calculator.

Computes basic performance metrics from a list of standardized TradeRecord objects.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from bsa_quicktrade.analytics.models import PerformanceMetrics, TradeRecord

logger = logging.getLogger(__name__)


def calculate_performance_metrics(
    records: List[TradeRecord],
    starting_capital: Optional[float] = None
) -> PerformanceMetrics:
    """Calculate baseline performance metrics for the list of trade records."""
    total_trades = len(records)
    
    if total_trades == 0:
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            net_profit_pct=0.0,
            average_return=0.0,
            average_return_pct=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            payoff_ratio=0.0,
            expectancy=0.0,
            average_r_multiple=0.0,
            cagr=0.0,
        )

    # Separate wins and losses (using net_profit of each trade record)
    wins = [r.net_profit for r in records if r.net_profit > 0]
    losses = [r.net_profit for r in records if r.net_profit <= 0]
    
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = (winning_trades / total_trades) * 100.0

    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    net_profit = gross_profit - gross_loss

    # Capital computations
    total_capital_used = sum(r.capital_used for r in records)
    max_capital_used = max((r.capital_used for r in records), default=0.0)
    
    # If starting capital is not provided, estimate it from max capital used or fallback to 100,000
    if starting_capital is None or starting_capital <= 0:
        starting_capital = max_capital_used if max_capital_used > 0 else 100000.0

    net_profit_pct = (net_profit / starting_capital) * 100.0

    # Returns list
    returns = [r.net_profit for r in records]
    returns_pct = [r.profit_pct for r in records]  # raw profit pct per trade

    average_return = float(np.mean(returns))
    average_return_pct = float(np.mean(returns_pct))

    average_win = float(np.mean(wins)) if wins else 0.0
    average_loss = float(np.mean(losses)) if losses else 0.0

    largest_win = float(max(wins)) if wins else 0.0
    largest_loss = float(min(losses)) if losses else 0.0

    # Profit Factor: Gross Profit / Gross Loss
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 1.0

    # Payoff Ratio: Avg Win / Avg Loss (absolute value)
    abs_avg_loss = abs(average_loss)
    if abs_avg_loss > 0:
        payoff_ratio = average_win / abs_avg_loss
    else:
        payoff_ratio = float("inf") if average_win > 0 else 1.0

    # Expectancy: (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
    # Note: average_loss is already negative or zero, so addition is correct
    win_prob = winning_trades / total_trades
    loss_prob = losing_trades / total_trades
    expectancy = (win_prob * average_win) + (loss_prob * average_loss)

    # R-Multiples
    r_multiples = [r.r_multiple for r in records if r.r_multiple is not None]
    average_r_multiple = float(np.mean(r_multiples)) if r_multiples else 0.0

    # CAGR Calculation
    cagr = 0.0
    entry_dates = [r.entry_date for r in records if r.entry_date is not None]
    exit_dates = [r.exit_date for r in records if r.exit_date is not None]
    
    if entry_dates and exit_dates:
        min_date = min(entry_dates)
        max_date = max(exit_dates)
        delta_days = (max_date - min_date).days
        years = delta_days / 365.25
        
        if years > 0.0027:  # More than 1 day
            ending_capital = starting_capital + net_profit
            if ending_capital > 0 and starting_capital > 0:
                cagr = ((ending_capital / starting_capital) ** (1 / years) - 1) * 100.0

    return PerformanceMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=round(win_rate, 2),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        net_profit=round(net_profit, 2),
        net_profit_pct=round(net_profit_pct, 2),
        average_return=round(average_return, 2),
        average_return_pct=round(average_return_pct, 3),
        average_win=round(average_win, 2),
        average_loss=round(average_loss, 2),
        largest_win=round(largest_win, 2),
        largest_loss=round(largest_loss, 2),
        profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        payoff_ratio=round(payoff_ratio, 2) if payoff_ratio != float("inf") else 999.0,
        expectancy=round(expectancy, 2),
        average_r_multiple=round(average_r_multiple, 3),
        cagr=round(cagr, 2),
    )
