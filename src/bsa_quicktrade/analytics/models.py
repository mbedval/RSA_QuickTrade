"""Standardized data models for the Backtest Analytics & Research Engine.

Defines the core structures for completed trades, metrics, and analyses
following clean architecture and type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TradeRecord:
    """Represents a single completed trade.
    
    This is the standardized representation of a trade log entry.
    All analytics are derived from this model.
    """

    trade_id: str
    strategy_name: str
    module_name: str
    signal_name: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    
    # Financial metrics
    capital_used: float
    profit: float
    profit_pct: float
    
    # Risk/Reward info
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk: Optional[float] = None
    reward: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    r_multiple: Optional[float] = None
    
    # Market & Indicators
    holding_period: float = 0.0  # in days
    atr: Optional[float] = None
    adx: Optional[float] = None
    rsi: Optional[float] = None
    ema_values: Dict[str, float] = field(default_factory=dict)
    vwap: Optional[float] = None
    volume: Optional[float] = None
    market_breadth: Optional[float] = None
    market_regime: str = "Unknown"  # e.g., "Bull", "Bear", "Sideways"
    
    # Execution metrics
    exit_reason: str = "Unknown"  # e.g., "Target", "Stop Loss", "Time Exit"
    entry_indicator_snapshot: Dict[str, Any] = field(default_factory=dict)
    exit_indicator_snapshot: Dict[str, Any] = field(default_factory=dict)
    triggered_rules: List[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0 to 100
    
    # Friction costs
    brokerage: float = 0.0
    taxes: float = 0.0
    slippage: float = 0.0
    
    remarks: str = ""
    direction: str = "long"  # "long" or "short"

    def __post_init__(self) -> None:
        # Auto-calculate holding period in days if not set
        if not self.holding_period and self.entry_date and self.exit_date:
            delta = self.exit_date - self.entry_date
            # convert total seconds to fractional days
            self.holding_period = max(0.0, delta.total_seconds() / 86400.0)

        # Force direction lower case
        self.direction = self.direction.lower()
        if self.direction not in ("long", "short"):
            self.direction = "long"

        # Calculate transaction costs
        self.total_cost = self.brokerage + self.taxes + self.slippage
        self.net_profit = self.profit - self.total_cost


@dataclass
class PerformanceMetrics:
    """Contains basic performance analytics for a backtest log.
    
    Represents the output of the Performance Analytics module.
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # Percentage (0-100)
    
    gross_profit: float
    gross_loss: float
    net_profit: float
    net_profit_pct: float  # Profit as % of starting/total capital used
    
    average_return: float  # Average return per trade
    average_return_pct: float  # Average return % per trade
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    profit_factor: float
    payoff_ratio: float  # Average Win / Average Loss
    expectancy: float  # Expectancy in currency per trade
    average_r_multiple: float
    cagr: float = 0.0  # Compound Annual Growth Rate


@dataclass
class DrawdownInfo:
    """Detailed drawdown analysis details."""

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    avg_drawdown_pct: float = 0.0
    longest_drawdown_days: float = 0.0
    time_to_recovery_days: float = 0.0


@dataclass
class RiskMetrics:
    """Contains risk analytics for a backtest log."""

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0
    mar_ratio: float = 0.0
    recovery_factor: float = 0.0
    drawdown: DrawdownInfo = field(default_factory=DrawdownInfo)


@dataclass
class DistributionMetrics:
    """Contains trade distribution statistics."""

    consecutive_wins: int = 0
    consecutive_losses: int = 0
    win_streak_avg_profit: float = 0.0
    loss_streak_avg_loss: float = 0.0
    profit_by_weekday: Dict[str, float] = field(default_factory=dict)
    profit_by_month: Dict[str, float] = field(default_factory=dict)
    avg_holding_period: float = 0.0
    duration_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class RegimeMetrics:
    """Performance statistics partitioned by a specific market regime or volatility condition."""

    regime_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    avg_holding_period: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    expectancy: float = 0.0


@dataclass
class IndicatorAttribution:
    """Attribution statistics for a specific indicator or signal rule."""

    indicator_name: str
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    win_rate: float = 0.0
    net_profit: float = 0.0
    avg_return_pct: float = 0.0
    false_signal_rate: float = 0.0


@dataclass
class AttributionSummary:
    """Aggregated signal attribution report."""

    indicators: Dict[str, IndicatorAttribution] = field(default_factory=dict)
    best_combination: str = "None"
    best_combination_win_rate: float = 0.0
    worst_combination: str = "None"
    worst_combination_win_rate: float = 0.0


@dataclass
class StrategyHealth:
    """Evaluation of the strategy's overall health, viability, and risk."""

    score: float = 0.0  # 0 to 100
    status: str = "REJECT"  # "PASS", "WARNING", "REJECT"
    reasons: List[str] = field(default_factory=list)
    recommendation: str = ""



