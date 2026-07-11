"""Relative Strength Analyzer — Stage 3 of swing trading pipeline.

Compares a stock's recent performance against the NIFTY benchmark and its
sector index. Stocks outperforming both market and sector are preferred.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RelativeStrengthResult:
    """Relative strength comparison result."""
    rs_vs_nifty: float          # stock % gain - nifty % gain (positive = outperforming)
    rs_vs_sector: float         # stock % gain - sector % gain
    rs_score: float             # 0–100 composite score
    is_outperforming_market: bool
    is_outperforming_sector: bool
    trend: str                  # "improving", "stable", "weakening"
    reasons: list[str]


class RelativeStrengthAnalyzer:
    """Calculate stock relative strength vs. NIFTY and sector index.

    Uses a configurable lookback window (default 20 days) for momentum comparison.
    Also checks if RS is improving (increasing slope over recent 10 days).
    """

    def __init__(self, window: int = 20, trend_window: int = 10) -> None:
        self.window = window
        self.trend_window = trend_window

    def analyze(
        self,
        stock_daily: pd.DataFrame,
        nifty_daily: Optional[pd.DataFrame] = None,
        sector_daily: Optional[pd.DataFrame] = None,
    ) -> RelativeStrengthResult:
        """Compare stock performance vs. benchmark and sector.

        Parameters
        ----------
        stock_daily:
            Stock OHLCV DataFrame.
        nifty_daily:
            NIFTY 50 OHLCV DataFrame (optional).
        sector_daily:
            Sector index OHLCV DataFrame (optional).
        """
        stock_close = self._get_close(stock_daily)
        nifty_close = self._get_close(nifty_daily) if nifty_daily is not None else None
        sector_close = self._get_close(sector_daily) if sector_daily is not None else None

        reasons: list[str] = []

        if stock_close is None or len(stock_close) < self.window + 5:
            return RelativeStrengthResult(
                rs_vs_nifty=0.0, rs_vs_sector=0.0, rs_score=50.0,
                is_outperforming_market=False, is_outperforming_sector=False,
                trend="unknown", reasons=["Insufficient stock data"],
            )

        # Stock momentum over window
        stock_ret = self._momentum(stock_close, self.window)

        # RS vs NIFTY
        rs_nifty = 0.0
        outperform_market = False
        if nifty_close is not None and len(nifty_close) >= self.window + 5:
            nifty_ret = self._momentum(nifty_close, self.window)
            rs_nifty = stock_ret - nifty_ret
            outperform_market = rs_nifty > 0
            reasons.append(
                f"RS vs NIFTY: {'+' if rs_nifty >= 0 else ''}{rs_nifty:.1f}% "
                f"({'outperforming' if outperform_market else 'underperforming'})"
            )
        else:
            reasons.append("No NIFTY data — RS vs market not calculated")

        # RS vs Sector
        rs_sector = 0.0
        outperform_sector = False
        if sector_close is not None and len(sector_close) >= self.window + 5:
            sector_ret = self._momentum(sector_close, self.window)
            rs_sector = stock_ret - sector_ret
            outperform_sector = rs_sector > 0
            reasons.append(
                f"RS vs Sector: {'+' if rs_sector >= 0 else ''}{rs_sector:.1f}% "
                f"({'outperforming' if outperform_sector else 'underperforming'})"
            )
        else:
            reasons.append("No sector data — RS vs sector not calculated")

        # Trend direction (is RS improving over last trend_window days?)
        rs_trend = self._rs_trend(stock_close, nifty_close)
        reasons.append(f"RS trend: {rs_trend}")

        # Composite score (0–100)
        score = 50.0
        if outperform_market:
            score += min(20.0, abs(rs_nifty) * 2)
        else:
            score -= min(20.0, abs(rs_nifty) * 2)
        if outperform_sector:
            score += min(15.0, abs(rs_sector) * 1.5)
        else:
            score -= min(15.0, abs(rs_sector) * 1.5)
        if rs_trend == "improving":
            score += 10
        elif rs_trend == "weakening":
            score -= 10

        return RelativeStrengthResult(
            rs_vs_nifty=round(rs_nifty, 2),
            rs_vs_sector=round(rs_sector, 2),
            rs_score=round(max(0.0, min(100.0, score)), 1),
            is_outperforming_market=outperform_market,
            is_outperforming_sector=outperform_sector,
            trend=rs_trend,
            reasons=reasons,
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _momentum(self, close: pd.Series, window: int) -> float:
        """Percentage return over last `window` bars."""
        if len(close) < window + 1:
            return 0.0
        return (float(close.iloc[-1]) / float(close.iloc[-window - 1]) - 1.0) * 100.0

    def _rs_trend(
        self,
        stock_close: pd.Series,
        nifty_close: Optional[pd.Series],
    ) -> str:
        """Detect if relative strength is improving or weakening."""
        if nifty_close is None or len(stock_close) < self.trend_window + 5:
            return "unknown"

        # Align on common dates
        stock = stock_close.tail(self.trend_window + 5)
        nifty = nifty_close.reindex(stock.index, method="ffill").dropna()
        stock = stock.reindex(nifty.index).dropna()

        if len(stock) < 5:
            return "unknown"

        rs_series = stock / nifty
        # Simple linear slope
        xs = range(len(rs_series))
        ys = rs_series.values
        if len(xs) < 3:
            return "stable"
        slope = (ys[-1] - ys[0]) / max(1, len(xs) - 1)

        if slope > 0.0005:
            return "improving"
        elif slope < -0.0005:
            return "weakening"
        return "stable"

    def _get_close(self, df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if df is None:
            return None
        for col in df.columns:
            if str(col).lower() in {"close", "adj close"}:
                return df[col].dropna()
        return None
