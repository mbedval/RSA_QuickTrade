"""Signal Attribution analytics.

Analyzes performance contributions and win/loss success rates for indicators
and rule combinations.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from bsa_quicktrade.analytics.models import AttributionSummary, IndicatorAttribution, TradeRecord


def calculate_signal_attribution(records: List[TradeRecord]) -> AttributionSummary:
    """Calculate attribution statistics for indicators and rule combinations."""
    if not records:
        return AttributionSummary()

    # 1. Identify all active indicators / rules for each trade
    # We will gather them from:
    # - triggered_rules (list of strings)
    # - entry_indicator_snapshot (keys of the dictionary)
    # - signal_name (as a fallback/primary indicator)
    
    indicator_trades: Dict[str, List[TradeRecord]] = {}
    combination_trades: Dict[str, List[TradeRecord]] = {}

    for r in records:
        active_indicators: Set[str] = set()
        
        # Add primary signal name
        if r.signal_name and r.signal_name.lower() not in ("unknown", "defaultsignal", ""):
            active_indicators.add(r.signal_name)
            
        # Add triggered rules
        for rule in r.triggered_rules:
            if rule:
                active_indicators.add(rule)
                
        # Add keys from indicator snapshot
        for ind_name in r.entry_indicator_snapshot.keys():
            if ind_name:
                active_indicators.add(ind_name)

        # Update individual indicator stats
        for ind in active_indicators:
            if ind not in indicator_trades:
                indicator_trades[ind] = []
            indicator_trades[ind].append(r)

        # Update combination stats (if length of rules >= 2)
        rules_list = sorted(list(r.triggered_rules))
        if len(rules_list) >= 2:
            combo_key = " + ".join(rules_list)
            if combo_key not in combination_trades:
                combination_trades[combo_key] = []
            combination_trades[combo_key].append(r)
        elif len(rules_list) == 1:
            combo_key = rules_list[0]
            if combo_key not in combination_trades:
                combination_trades[combo_key] = []
            combination_trades[combo_key].append(r)

    # 2. Compute individual indicator attributions
    indicators_attr: Dict[str, IndicatorAttribution] = {}
    for ind, group in indicator_trades.items():
        total_signals = len(group)
        wins = [t for t in group if t.net_profit > 0]
        losses = [t for t in group if t.net_profit <= 0]
        
        winning_signals = len(wins)
        losing_signals = len(losses)
        win_rate = (winning_signals / total_signals) * 100.0 if total_signals > 0 else 0.0
        false_signal_rate = (losing_signals / total_signals) * 100.0 if total_signals > 0 else 0.0
        
        net_profit = sum(t.net_profit for t in group)
        avg_ret = float(np.mean([t.profit_pct for t in group])) if group else 0.0
        
        indicators_attr[ind] = IndicatorAttribution(
            indicator_name=ind,
            total_signals=total_signals,
            winning_signals=winning_signals,
            losing_signals=losing_signals,
            win_rate=round(win_rate, 2),
            net_profit=round(net_profit, 2),
            avg_return_pct=round(avg_ret, 3),
            false_signal_rate=round(false_signal_rate, 2),
        )

    # 3. Find best and worst rule combinations (minimum 1 trade)
    best_combo = "None"
    best_wr = -1.0
    worst_combo = "None"
    worst_wr = 101.0

    for combo, group in combination_trades.items():
        total = len(group)
        wins = len([t for t in group if t.net_profit > 0])
        wr = (wins / total) * 100.0
        
        # Prioritize combinations with more trades to avoid single-trade bias
        # but if everything has 1 trade, it's fine.
        if wr > best_wr:
            best_wr = wr
            best_combo = combo
        if wr < worst_wr:
            worst_wr = wr
            worst_combo = combo

    # Adjust default outputs if no combinations exist
    if best_wr < 0:
        best_wr = 0.0
    if worst_wr > 100:
        worst_wr = 0.0

    return AttributionSummary(
        indicators=indicators_attr,
        best_combination=best_combo,
        best_combination_win_rate=round(best_wr, 2),
        worst_combination=worst_combo,
        worst_combination_win_rate=round(worst_wr, 2),
    )
