"""Swing Quality Score — Stage 6 of the swing trading pipeline.

Computes a 10-component Trade Quality Score (0–100) for every potential
swing trade entry. Only trades scoring ≥ the configured threshold (default 75)
are executed.

Score Components:
  Weekly Trend          20 pts  — primary trend alignment
  Daily Trend           15 pts  — daily trend support
  Relative Strength     15 pts  — outperformance vs market/sector
  Sector Strength       10 pts  — top sector membership
  Market Strength       10 pts  — NIFTY bullish environment
  Volume Confirmation   10 pts  — volume expanding on entry
  Trend Quality (ADX)   10 pts  — ADX above 20/25
  ATR Condition          5 pts  — ATR squeeze / not excessive
  Risk/Reward            5 pts  — minimum 1:2 R:R
  Price Structure (bonus) up to 5 pts extra
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta

from bsa_quicktrade.entry.setup_detector import EntrySetup
from bsa_quicktrade.filters.market_filter import MarketCondition
from bsa_quicktrade.filters.relative_strength import RelativeStrengthResult
from bsa_quicktrade.filters.sector_filter import SectorRanking

logger = logging.getLogger(__name__)

MIN_SCORE_DEFAULT = 75.0


@dataclass
class SwingQualityScore:
    """Complete swing trade quality score breakdown."""
    total: float                    # 0–100
    components: dict[str, float]    # component name → points earned
    passes_threshold: bool
    threshold: float
    rr_ratio: float                 # calculated risk/reward
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ PASS" if self.passes_threshold else "❌ SKIP"
        return f"Swing Score: {self.total:.1f}/100  {status}"


class SwingScorer:
    """Calculate trade quality score for a potential swing trade entry.

    Usage::

        scorer = SwingScorer(min_score=75)
        score = scorer.score(
            entry_setup=setup,
            market_condition=market,
            rs_result=rs,
            sector_ranking=sectors,
            stock_daily=df,
            entry_price=1234.0,
            stop_price=1210.0,
            target_price=1282.0,
            sector=stock_sector,
        )
        if score.passes_threshold:
            # proceed with trade
    """

    # Component max points
    _MAX = {
        "weekly_trend":       20,
        "daily_trend":        15,
        "relative_strength":  15,
        "sector_strength":    10,
        "market_strength":    10,
        "volume":             10,
        "adx_quality":        10,
        "atr_condition":       5,
        "risk_reward":         5,
        "price_structure":     5,   # bonus
    }

    def __init__(self, min_score: float = MIN_SCORE_DEFAULT) -> None:
        self.min_score = min_score

    def score(
        self,
        entry_setup: EntrySetup,
        market_condition: Optional[MarketCondition],
        rs_result: Optional[RelativeStrengthResult],
        sector_ranking: Optional[SectorRanking],
        stock_daily: pd.DataFrame,
        entry_price: float,
        stop_price: float,
        target_price: float,
        sector: str = "",
    ) -> SwingQualityScore:
        """Compute the composite swing quality score."""
        components: dict[str, float] = {}
        reasons: list[str] = []

        # 1. Weekly Trend (20 pts)
        wt_score = entry_setup.weekly_trend_score / 100.0 * self._MAX["weekly_trend"]
        components["weekly_trend"] = round(wt_score, 1)
        reasons.extend(entry_setup.details[:2])

        # 2. Daily Trend (15 pts)
        dt_score = entry_setup.daily_pattern_score / 100.0 * self._MAX["daily_trend"]
        components["daily_trend"] = round(dt_score, 1)
        if entry_setup.daily_pattern != "No Clear Setup":
            reasons.append(f"Daily: {entry_setup.daily_pattern} ({entry_setup.daily_pattern_score:.0f}/100)")

        # 3. Relative Strength (15 pts)
        if rs_result is not None:
            rs_pts = rs_result.rs_score / 100.0 * self._MAX["relative_strength"]
        else:
            rs_pts = self._MAX["relative_strength"] * 0.5  # neutral if no data
        components["relative_strength"] = round(rs_pts, 1)

        # 4. Sector Strength (10 pts)
        if sector_ranking is not None and sector_ranking.allow_sectors:
            is_top = sector_ranking.is_allowed(sector)
            sec_pts = self._MAX["sector_strength"] if is_top else self._MAX["sector_strength"] * 0.2
            if is_top:
                reasons.append(f"Sector '{sector}' in top sectors ✓")
            else:
                reasons.append(f"Sector '{sector}' not in top sectors ✗")
        else:
            sec_pts = self._MAX["sector_strength"] * 0.5
        components["sector_strength"] = round(sec_pts, 1)

        # 5. Market Strength (10 pts)
        if market_condition is not None:
            mkt_pts = market_condition.score / 100.0 * self._MAX["market_strength"]
            if not market_condition.allow_long:
                mkt_pts = 0.0
                reasons.append("Market not suitable for longs ✗")
        else:
            mkt_pts = self._MAX["market_strength"] * 0.5
        components["market_strength"] = round(mkt_pts, 1)

        # 6. Volume Confirmation (10 pts)
        vol_pts = self._score_volume(stock_daily)
        components["volume"] = round(vol_pts, 1)

        # 7. ADX / Trend Quality (10 pts)
        adx_pts = self._score_adx(stock_daily)
        components["adx_quality"] = round(adx_pts, 1)

        # 8. ATR Condition (5 pts)
        atr_pts = self._score_atr_condition(stock_daily, entry_price, stop_price)
        components["atr_condition"] = round(atr_pts, 1)

        # 9. Risk/Reward (5 pts)
        risk = abs(entry_price - stop_price) if entry_price != stop_price else 1.0
        reward = abs(target_price - entry_price)
        rr = reward / risk if risk > 0 else 0.0
        if rr >= 2.5:
            rr_pts = self._MAX["risk_reward"]
        elif rr >= 2.0:
            rr_pts = self._MAX["risk_reward"] * 0.8
        elif rr >= 1.5:
            rr_pts = self._MAX["risk_reward"] * 0.5
        else:
            rr_pts = 0.0
        components["risk_reward"] = round(rr_pts, 1)
        reasons.append(f"R:R = 1:{rr:.1f} ({'+' if rr >= 2 else '✗'})")

        # 10. Price Structure bonus (5 pts)
        ps_pts = self._score_price_structure(stock_daily, entry_price)
        components["price_structure"] = round(ps_pts, 1)

        total = sum(components.values())
        passes = total >= self.min_score

        return SwingQualityScore(
            total=round(min(total, 100.0), 1),
            components=components,
            passes_threshold=passes,
            threshold=self.min_score,
            rr_ratio=round(rr, 2),
            reasons=reasons,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def _score_volume(self, daily: pd.DataFrame, window: int = 20) -> float:
        if "Volume" not in daily.columns or len(daily) < window + 1:
            return self._MAX["volume"] * 0.5
        avg = float(daily["Volume"].iloc[-window - 1:-1].mean())
        cur = float(daily["Volume"].iloc[-1])
        if avg <= 0:
            return self._MAX["volume"] * 0.5
        ratio = cur / avg
        if ratio >= 2.0:
            return float(self._MAX["volume"])
        elif ratio >= 1.5:
            return self._MAX["volume"] * 0.8
        elif ratio >= 1.0:
            return self._MAX["volume"] * 0.5
        return 0.0

    def _score_adx(self, daily: pd.DataFrame) -> float:
        close = self._get_close(daily)
        if close is None or len(close) < 20:
            return self._MAX["adx_quality"] * 0.4
        adx_df = ta.adx(daily["High"], daily["Low"], close, length=14)
        if adx_df is None or pd.isna(adx_df.iloc[-1, 0]):
            return self._MAX["adx_quality"] * 0.4
        adx = float(adx_df.iloc[-1, 0])
        if adx >= 30:
            return float(self._MAX["adx_quality"])
        elif adx >= 25:
            return self._MAX["adx_quality"] * 0.8
        elif adx >= 20:
            return self._MAX["adx_quality"] * 0.5
        return self._MAX["adx_quality"] * 0.2

    def _score_atr_condition(
        self, daily: pd.DataFrame, entry_price: float, stop_price: float
    ) -> float:
        close = self._get_close(daily)
        if close is None or len(close) < 20:
            return self._MAX["atr_condition"] * 0.5
        atr_s = ta.atr(daily["High"], daily["Low"], close, length=14)
        if atr_s is None or pd.isna(atr_s.iloc[-1]):
            return self._MAX["atr_condition"] * 0.5
        atr = float(atr_s.iloc[-1])
        stop_dist = abs(entry_price - stop_price)
        # Check stop distance is approximately 2×ATR (healthy)
        ratio = stop_dist / atr if atr > 0 else 2.0
        if 1.5 <= ratio <= 2.5:
            return float(self._MAX["atr_condition"])
        elif 1.0 <= ratio <= 3.0:
            return self._MAX["atr_condition"] * 0.6
        return self._MAX["atr_condition"] * 0.2

    def _score_price_structure(
        self, daily: pd.DataFrame, entry_price: float
    ) -> float:
        """Bonus points if price is at a structural support (Fibonacci, prior S/R)."""
        if len(daily) < 30:
            return 0.0
        # Simple check: is price near a prior swing low (support)?
        recent_lows = [float(daily["Low"].iloc[j]) for j in range(-20, -1)]
        swing_low = min(recent_lows)
        if abs(entry_price - swing_low) / entry_price < 0.02:
            return float(self._MAX["price_structure"])
        return 0.0

    def _get_close(self, df: pd.DataFrame):
        for col in df.columns:
            if str(col).lower() in {"close", "adj close"}:
                return df[col].dropna()
        return None
