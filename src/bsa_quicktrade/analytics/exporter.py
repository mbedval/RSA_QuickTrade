"""CSV and JSON Exporters for Backtest Analytics.

Saves computed performance, risk, regime, and attribution summaries to files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.performance import calculate_performance_metrics
from bsa_quicktrade.analytics.distribution import calculate_distribution_metrics
from bsa_quicktrade.analytics.risk import calculate_risk_metrics
from bsa_quicktrade.analytics.regime import calculate_regime_analytics, calculate_volatility_analytics
from bsa_quicktrade.analytics.attribution import calculate_signal_attribution


def compute_summary_dict(records: List[TradeRecord], starting_capital: float = 100000.0) -> Dict[str, Any]:
    """Calculate and return all analytics metrics structured as a dictionary."""
    perf = calculate_performance_metrics(records, starting_capital)
    risk = calculate_risk_metrics(records, starting_capital)
    dist = calculate_distribution_metrics(records)
    regimes = calculate_regime_analytics(records, starting_capital=starting_capital)
    vols = calculate_volatility_analytics(records, starting_capital=starting_capital)
    attr = calculate_signal_attribution(records)

    # Convert dataclasses to dicts
    return {
        "performance": {
            "total_trades": perf.total_trades,
            "winning_trades": perf.winning_trades,
            "losing_trades": perf.losing_trades,
            "win_rate_pct": perf.win_rate,
            "gross_profit": perf.gross_profit,
            "gross_loss": perf.gross_loss,
            "net_profit": perf.net_profit,
            "net_profit_pct": perf.net_profit_pct,
            "average_return": perf.average_return,
            "average_return_pct": perf.average_return_pct,
            "average_win": perf.average_win,
            "average_loss": perf.average_loss,
            "largest_win": perf.largest_win,
            "largest_loss": perf.largest_loss,
            "profit_factor": perf.profit_factor,
            "payoff_ratio": perf.payoff_ratio,
            "expectancy": perf.expectancy,
            "average_r_multiple": perf.average_r_multiple,
            "cagr_pct": perf.cagr,
        },
        "risk": {
            "sharpe_ratio": risk.sharpe_ratio,
            "sortino_ratio": risk.sortino_ratio,
            "calmar_ratio": risk.calmar_ratio,
            "ulcer_index": risk.ulcer_index,
            "mar_ratio": risk.mar_ratio,
            "recovery_factor": risk.recovery_factor,
            "drawdown": {
                "max_drawdown": risk.drawdown.max_drawdown,
                "max_drawdown_pct": risk.drawdown.max_drawdown_pct,
                "avg_drawdown": risk.drawdown.avg_drawdown,
                "avg_drawdown_pct": risk.drawdown.avg_drawdown_pct,
                "longest_drawdown_days": risk.drawdown.longest_drawdown_days,
                "time_to_recovery_days": risk.drawdown.time_to_recovery_days,
            }
        },
        "distribution": {
            "consecutive_wins": dist.consecutive_wins,
            "consecutive_losses": dist.consecutive_losses,
            "win_streak_avg_profit": dist.win_streak_avg_profit,
            "loss_streak_avg_loss": dist.loss_streak_avg_loss,
            "avg_holding_period_days": dist.avg_holding_period,
            "profit_by_weekday": dist.profit_by_weekday,
            "profit_by_month": dist.profit_by_month,
            "duration_distribution": dist.duration_distribution,
        },
        "regimes": {
            reg: {
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "net_profit": m.net_profit,
                "max_drawdown_pct": m.max_drawdown_pct,
                "sharpe_ratio": m.sharpe_ratio,
            } for reg, m in regimes.items()
        },
        "volatility": {
            v: {
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "net_profit": m.net_profit,
                "max_drawdown_pct": m.max_drawdown_pct,
                "sharpe_ratio": m.sharpe_ratio,
            } for v, m in vols.items()
        },
        "attribution": {
            "best_combination": attr.best_combination,
            "best_combination_win_rate": attr.best_combination_win_rate,
            "worst_combination": attr.worst_combination,
            "worst_combination_win_rate": attr.worst_combination_win_rate,
            "indicators": {
                ind: {
                    "total_signals": m.total_signals,
                    "win_rate": m.win_rate,
                    "net_profit": m.net_profit,
                    "false_signal_rate": m.false_signal_rate,
                } for ind, m in attr.indicators.items()
            }
        }
    }


def export_to_json(records: List[TradeRecord], file_path: Union[str, Path], starting_capital: float = 100000.0) -> None:
    """Export summary metrics to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_summary_dict(records, starting_capital)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def export_to_csv(records: List[TradeRecord], file_path: Union[str, Path], starting_capital: float = 100000.0) -> None:
    """Export key summary metrics to a flat CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_summary_dict(records, starting_capital)

    # Flatten the dict into key-value pairs
    flat_data: Dict[str, Any] = {}
    
    for category in ["performance", "risk"]:
        for k, v in summary[category].items():
            if isinstance(v, dict):  # e.g., risk.drawdown
                for dk, dv in v.items():
                    flat_data[f"{category}_{k}_{dk}"] = dv
            else:
                flat_data[f"{category}_{k}"] = v
                
    flat_data["best_combination"] = summary["attribution"]["best_combination"]
    flat_data["best_combination_win_rate"] = summary["attribution"]["best_combination_win_rate"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric Name", "Value"])
        for k, v in flat_data.items():
            writer.writerow([k, v])
