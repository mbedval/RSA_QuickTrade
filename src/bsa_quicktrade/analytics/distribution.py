"""Trade Distribution analytics.

Computes duration distribution, streak analytics, and time-based performance.
"""

from __future__ import annotations

from typing import List

import numpy as np

from bsa_quicktrade.analytics.models import DistributionMetrics, TradeRecord


def calculate_distribution_metrics(records: List[TradeRecord]) -> DistributionMetrics:
    """Calculate distribution metrics for the trade records.
    
    Trades are sorted by exit_date to ensure chronological execution order.
    """
    if not records:
        return DistributionMetrics()

    # Sort trades by exit date
    sorted_records = sorted(records, key=lambda r: r.exit_date)

    # 1. Holding period statistics
    holding_periods = [r.holding_period for r in sorted_records]
    avg_holding = float(np.mean(holding_periods)) if holding_periods else 0.0

    # Categorize durations into standard buckets
    duration_bins = {
        "< 1 day": 0,
        "1-5 days": 0,
        "5-20 days": 0,
        "20-60 days": 0,
        "> 60 days": 0,
    }
    for hp in holding_periods:
        if hp < 1.0:
            duration_bins["< 1 day"] += 1
        elif hp <= 5.0:
            duration_bins["1-5 days"] += 1
        elif hp <= 20.0:
            duration_bins["5-20 days"] += 1
        elif hp <= 60.0:
            duration_bins["20-60 days"] += 1
        else:
            duration_bins["> 60 days"] += 1

    # 2. Streak analysis
    max_win_streak = 0
    max_loss_streak = 0
    
    current_win_streak: List[float] = []
    current_loss_streak: List[float] = []
    
    longest_win_streak_trades: List[float] = []
    longest_loss_streak_trades: List[float] = []

    for r in sorted_records:
        profit = r.net_profit
        if profit > 0:
            # End of loss streak
            if len(current_loss_streak) > max_loss_streak:
                max_loss_streak = len(current_loss_streak)
                longest_loss_streak_trades = list(current_loss_streak)
            current_loss_streak = []
            
            # Continue win streak
            current_win_streak.append(profit)
        else:
            # End of win streak
            if len(current_win_streak) > max_win_streak:
                max_win_streak = len(current_win_streak)
                longest_win_streak_trades = list(current_win_streak)
            current_win_streak = []
            
            # Continue loss streak
            current_loss_streak.append(profit)

    # Final check after loop ends
    if len(current_loss_streak) > max_loss_streak:
        max_loss_streak = len(current_loss_streak)
        longest_loss_streak_trades = list(current_loss_streak)
    if len(current_win_streak) > max_win_streak:
        max_win_streak = len(current_win_streak)
        longest_win_streak_trades = list(current_win_streak)

    win_streak_avg_profit = float(np.mean(longest_win_streak_trades)) if longest_win_streak_trades else 0.0
    loss_streak_avg_loss = float(np.mean(longest_loss_streak_trades)) if longest_loss_streak_trades else 0.0

    # 3. Time-based performance (Weekday & Month)
    weekday_profit = {
        "Monday": 0.0,
        "Tuesday": 0.0,
        "Wednesday": 0.0,
        "Thursday": 0.0,
        "Friday": 0.0,
        "Saturday": 0.0,
        "Sunday": 0.0,
    }
    
    month_profit = {
        "January": 0.0,
        "February": 0.0,
        "March": 0.0,
        "April": 0.0,
        "May": 0.0,
        "June": 0.0,
        "July": 0.0,
        "August": 0.0,
        "September": 0.0,
        "October": 0.0,
        "November": 0.0,
        "December": 0.0,
    }

    for r in sorted_records:
        if r.exit_date:
            weekday_name = r.exit_date.strftime("%A")
            month_name = r.exit_date.strftime("%B")
            
            if weekday_name in weekday_profit:
                weekday_profit[weekday_name] += r.net_profit
            if month_name in month_profit:
                month_profit[month_name] += r.net_profit

    # Round all values for clean reporting
    weekday_profit = {k: round(v, 2) for k, v in weekday_profit.items()}
    month_profit = {k: round(v, 2) for k, v in month_profit.items()}

    return DistributionMetrics(
        consecutive_wins=max_win_streak,
        consecutive_losses=max_loss_streak,
        win_streak_avg_profit=round(win_streak_avg_profit, 2),
        loss_streak_avg_loss=round(loss_streak_avg_loss, 2),
        profit_by_weekday=weekday_profit,
        profit_by_month=month_profit,
        avg_holding_period=round(avg_holding, 2),
        duration_distribution=duration_distribution_clean(duration_bins),
    )


def duration_distribution_clean(bins: dict[str, int]) -> dict[str, int]:
    """Return only buckets that have values (optional) or all for standard layout."""
    return bins
