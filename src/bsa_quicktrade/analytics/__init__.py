"""Backtest Analytics & Research Engine.

A production-grade, reusable analytics framework that parses trade logs
and produces professional quantitative performance reports and dashboards.
"""

from bsa_quicktrade.analytics.models import (
    AttributionSummary,
    DistributionMetrics,
    DrawdownInfo,
    IndicatorAttribution,
    PerformanceMetrics,
    RegimeMetrics,
    RiskMetrics,
    StrategyHealth,
    TradeRecord,
)
from bsa_quicktrade.analytics.performance import calculate_performance_metrics
from bsa_quicktrade.analytics.distribution import calculate_distribution_metrics
from bsa_quicktrade.analytics.risk import calculate_risk_metrics
from bsa_quicktrade.analytics.regime import (
    RegimeClassifier,
    calculate_regime_analytics,
    calculate_volatility_analytics,
)
from bsa_quicktrade.analytics.attribution import calculate_signal_attribution
from bsa_quicktrade.analytics.health import calculate_strategy_health
from bsa_quicktrade.analytics.visualization import VisualizationEngine
from bsa_quicktrade.analytics.trade_log import load_trade_log, save_trade_log

__all__ = [
    "TradeRecord",
    "PerformanceMetrics",
    "DrawdownInfo",
    "RiskMetrics",
    "DistributionMetrics",
    "RegimeMetrics",
    "RegimeClassifier",
    "IndicatorAttribution",
    "AttributionSummary",
    "StrategyHealth",
    "VisualizationEngine",
    "load_trade_log",
    "save_trade_log",
    "calculate_performance_metrics",
    "calculate_distribution_metrics",
    "calculate_risk_metrics",
    "calculate_regime_analytics",
    "calculate_volatility_analytics",
    "calculate_signal_attribution",
    "calculate_strategy_health",
]
