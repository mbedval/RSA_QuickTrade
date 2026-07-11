"""Forward Return Analyzer — Stage 11 of the swing trading pipeline.

For every entry signal, measures actual returns at 5, 10, 15, 20, and 26
trading days into the future. Used to determine the optimal holding period
and to validate that the strategy generates positive expectancy across all
time horizons.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = [5, 10, 15, 20, 26]


@dataclass
class HorizonStats:
    """Statistics for a single forward-return horizon."""
    days: int
    avg_return_pct: float
    median_return_pct: float
    win_rate: float             # 0.0–1.0
    max_return_pct: float
    max_loss_pct: float
    trade_count: int


@dataclass
class ForwardReturnSummary:
    """Complete forward return analysis across all horizons."""
    horizons: dict[int, HorizonStats]       # day → stats
    optimal_horizon: int                    # horizon with best avg return
    recommendation: str                     # human-readable advice
    signal_dates: list                      # entry dates analyzed
    direction: str                          # "long" or "short"


class ForwardReturnAnalyzer:
    """Compute forward returns for each signal date in a daily price series.

    Parameters
    ----------
    horizons:
        List of forward-looking day counts to measure. Default: [5, 10, 15, 20, 26].
    commission_pct:
        Round-trip commission + slippage estimate (default 0.1%).
    """

    def __init__(
        self,
        horizons: list[int] = HORIZONS,
        commission_pct: float = 0.1,
    ) -> None:
        self.horizons = sorted(horizons)
        self.commission = commission_pct / 100.0

    def analyze(
        self,
        daily: pd.DataFrame,
        signal_dates: list,
        direction: str = "long",
    ) -> ForwardReturnSummary:
        """Measure future returns from each signal date.

        Parameters
        ----------
        daily:
            Daily OHLCV DataFrame.
        signal_dates:
            List of entry dates (must exist in daily.index).
        direction:
            "long" or "short".
        """
        close = self._get_close(daily)
        if close is None or len(signal_dates) == 0:
            return ForwardReturnSummary(
                horizons={d: HorizonStats(d, 0, 0, 0, 0, 0, 0) for d in self.horizons},
                optimal_horizon=10,
                recommendation="Insufficient data for forward return analysis.",
                signal_dates=signal_dates,
                direction=direction,
            )

        # Build return matrix: signal × horizon
        returns_by_horizon: dict[int, list[float]] = {h: [] for h in self.horizons}

        idx_map = {ts: i for i, ts in enumerate(close.index)}

        for sig_date in signal_dates:
            # Find closest date in index
            if sig_date not in idx_map:
                continue
            i = idx_map[sig_date]
            entry_price = float(close.iloc[i])
            if entry_price <= 0:
                continue

            for h in self.horizons:
                future_i = i + h
                if future_i >= len(close):
                    continue
                future_price = float(close.iloc[future_i])
                raw = (future_price - entry_price) / entry_price
                net = raw - self.commission if direction == "long" else (-raw - self.commission)
                returns_by_horizon[h].append(net * 100.0)

        # Compute stats per horizon
        stats: dict[int, HorizonStats] = {}
        for h, rets in returns_by_horizon.items():
            if not rets:
                stats[h] = HorizonStats(h, 0, 0, 0, 0, 0, 0)
                continue
            arr = np.array(rets)
            stats[h] = HorizonStats(
                days=h,
                avg_return_pct=round(float(np.mean(arr)), 2),
                median_return_pct=round(float(np.median(arr)), 2),
                win_rate=round(float(np.mean(arr > 0)), 3),
                max_return_pct=round(float(np.max(arr)), 2),
                max_loss_pct=round(float(np.min(arr)), 2),
                trade_count=len(rets),
            )

        # Find optimal horizon (best avg return)
        valid = {h: s for h, s in stats.items() if s.trade_count > 0}
        if valid:
            optimal = max(valid, key=lambda h: valid[h].avg_return_pct)
        else:
            optimal = 10

        # Human-readable recommendation
        opt_stats = stats.get(optimal)
        if opt_stats and opt_stats.avg_return_pct > 0:
            recommendation = (
                f"Optimal holding period: {optimal} days "
                f"(avg return: {opt_stats.avg_return_pct:+.2f}%, "
                f"win rate: {opt_stats.win_rate * 100:.0f}%). "
                f"Strategy shows positive expectancy at this horizon."
            )
        else:
            recommendation = (
                "No horizon shows consistent positive returns. "
                "Review entry quality filters or widen data window."
            )

        return ForwardReturnSummary(
            horizons=stats,
            optimal_horizon=optimal,
            recommendation=recommendation,
            signal_dates=signal_dates,
            direction=direction,
        )

    def _get_close(self, df: pd.DataFrame) -> Optional[pd.Series]:
        for col in df.columns:
            if str(col).lower() in {"close", "adj close"}:
                return df[col].dropna()
        return None
