"""Entry Setup Detector — Stages 4 & 5 of swing trading pipeline.

Validates that:
  - The weekly timeframe shows a healthy bullish trend (Stage 4)
  - The daily chart shows a high-quality entry setup (Stage 5)

Avoids extended moves, late breakouts, and exhaustion candles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


@dataclass
class EntrySetup:
    """Result of entry setup detection."""
    weekly_trend_bullish: bool
    weekly_trend_score: float           # 0–100
    daily_pattern: str                  # e.g. "Pullback to EMA20", "Inside Bar Breakout"
    daily_pattern_score: float          # 0–100
    avoid_reason: Optional[str]         # reason to skip entry, if any
    is_valid: bool                      # True if entry is safe to proceed
    details: list[str] = field(default_factory=list)


class SetupDetector:
    """Detect swing entry setups from weekly and daily price data."""

    def __init__(
        self,
        weekly_ema_short: int = 20,
        weekly_ema_long: int = 50,
        weekly_adx_min: float = 20.0,
        daily_ema: int = 20,
        volume_avg_window: int = 20,
    ) -> None:
        self.weekly_ema_short = weekly_ema_short
        self.weekly_ema_long = weekly_ema_long
        self.weekly_adx_min = weekly_adx_min
        self.daily_ema = daily_ema
        self.volume_avg_window = volume_avg_window

    def detect(
        self,
        daily: pd.DataFrame,
        weekly: Optional[pd.DataFrame] = None,
    ) -> EntrySetup:
        """Run weekly trend filter and daily pattern detection.

        Parameters
        ----------
        daily:
            Daily OHLCV DataFrame (at least 60 bars).
        weekly:
            Weekly OHLCV DataFrame. If None, derived by resampling daily.
        """
        if weekly is None or len(weekly) < 30:
            weekly = self._resample_weekly(daily)

        weekly_result, weekly_score, weekly_details = self._check_weekly_trend(weekly)
        daily_pattern, daily_score, daily_details, avoid = self._check_daily_setup(daily)

        all_details = weekly_details + daily_details
        is_valid = weekly_result and avoid is None

        return EntrySetup(
            weekly_trend_bullish=weekly_result,
            weekly_trend_score=weekly_score,
            daily_pattern=daily_pattern,
            daily_pattern_score=daily_score,
            avoid_reason=avoid,
            is_valid=is_valid,
            details=all_details,
        )

    # ── Weekly trend check ───────────────────────────────────────────────

    def _check_weekly_trend(
        self, weekly: pd.DataFrame
    ) -> tuple[bool, float, list[str]]:
        """Validate weekly trend. Returns (bullish, score 0-100, reasons)."""
        close = self._get_close(weekly)
        if close is None or len(close) < 30:
            return False, 0.0, ["Insufficient weekly data"]

        ema20 = ta.ema(close, length=self.weekly_ema_short)
        ema50 = ta.ema(close, length=self.weekly_ema_long)
        adx_df = ta.adx(weekly["High"], weekly["Low"], close, length=14)

        cur = float(close.iloc[-1])
        e20 = float(ema20.iloc[-1]) if ema20 is not None and not pd.isna(ema20.iloc[-1]) else 0.0
        e50 = float(ema50.iloc[-1]) if ema50 is not None and not pd.isna(ema50.iloc[-1]) else 0.0
        adx = float(adx_df.iloc[-1, 0]) if adx_df is not None and not pd.isna(adx_df.iloc[-1, 0]) else 0.0

        # Higher Highs check
        higher_highs = self._check_higher_highs(weekly, lookback=8)

        score = 0.0
        details: list[str] = []
        bullish = True

        if cur > e20 > 0:
            score += 30
            details.append(f"Weekly close {cur:.0f} above EMA20 {e20:.0f} ✓")
        else:
            bullish = False
            details.append(f"Weekly close {cur:.0f} below EMA20 {e20:.0f} ✗")

        if e20 > e50 > 0:
            score += 30
            details.append("Weekly EMA20 > EMA50 ✓")
        else:
            bullish = False
            details.append("Weekly EMA20 ≤ EMA50 ✗")

        if adx >= self.weekly_adx_min:
            score += 20
            details.append(f"Weekly ADX {adx:.0f} ≥ {self.weekly_adx_min:.0f} ✓")
        else:
            details.append(f"Weekly ADX {adx:.0f} weak trend ✗")

        if higher_highs:
            score += 20
            details.append("Weekly Higher Highs confirmed ✓")
        else:
            details.append("Weekly Higher Highs not confirmed")

        return bullish, round(score, 1), details

    # ── Daily entry setup check ──────────────────────────────────────────

    def _check_daily_setup(
        self, daily: pd.DataFrame
    ) -> tuple[str, float, list[str], Optional[str]]:
        """Detect daily entry pattern. Returns (pattern, score, details, avoid_reason)."""
        close = self._get_close(daily)
        if close is None or len(close) < 30:
            return "Unknown", 0.0, ["Insufficient daily data"], "No daily data"

        ema20_series = ta.ema(close, length=self.daily_ema)
        atr_series = ta.atr(daily["High"], daily["Low"], close, length=14)
        vol = daily["Volume"] if "Volume" in daily.columns else None

        cur = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else cur
        high_0 = float(daily["High"].iloc[-1])
        low_0 = float(daily["Low"].iloc[-1])
        high_1 = float(daily["High"].iloc[-2]) if len(daily) > 1 else high_0
        low_1 = float(daily["Low"].iloc[-2]) if len(daily) > 1 else low_0
        e20 = float(ema20_series.iloc[-1]) if ema20_series is not None and not pd.isna(ema20_series.iloc[-1]) else 0.0
        atr = float(atr_series.iloc[-1]) if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else 0.0

        details: list[str] = []
        score = 50.0

        # --- Avoid conditions (Stage 5 filters) ---
        # Extended move: price too far above EMA20
        if e20 > 0 and atr > 0:
            dist_from_ema = (cur - e20) / atr
            if dist_from_ema > 4.0:
                return "Extended Move", 10.0, ["Price too extended above EMA20"], \
                    f"Price {dist_from_ema:.1f}×ATR above EMA20 — extended, avoid entry"

        # Exhaustion candle: huge range candle closing near low
        if atr > 0:
            candle_range = high_0 - low_0
            close_position = (cur - low_0) / candle_range if candle_range > 0 else 0.5
            if candle_range > 2.5 * atr and close_position < 0.3:
                return "Exhaustion", 15.0, ["Large bearish exhaustion candle"], \
                    "Bearish exhaustion candle — avoid entry"

        # --- Preferred setup patterns ---
        pattern = "No Clear Setup"

        # 1. Pullback to EMA20
        if e20 > 0 and atr > 0:
            dist_to_ema = abs(cur - e20) / atr
            if dist_to_ema < 0.5 and cur > e20:
                pattern = "Pullback to EMA20"
                score = 85.0
                details.append(f"Price {cur:.0f} pulling back to EMA20 {e20:.0f} — ideal entry ✓")

        # 2. Inside Bar Breakout
        if high_0 > high_1 and low_0 > low_1 and prev < high_1:
            if pattern == "No Clear Setup":
                pattern = "Inside Bar Breakout"
                score = 78.0
            details.append("Inside Bar Breakout above prior high ✓")

        # 3. Bullish Engulfing
        open_0 = float(daily["Open"].iloc[-1]) if "Open" in daily.columns else cur
        open_1 = float(daily["Open"].iloc[-2]) if "Open" in daily.columns and len(daily) > 1 else prev
        if open_0 < prev and cur > open_1 and (cur - open_0) > (open_1 - prev):
            if pattern == "No Clear Setup":
                pattern = "Bullish Engulfing"
                score = 80.0
            details.append("Bullish Engulfing candle ✓")

        # 4. Higher Low Formation
        if len(close) >= 10:
            recent_lows = [float(daily["Low"].iloc[j]) for j in range(-5, 0)]
            if recent_lows[-1] > min(recent_lows[:-1]):
                if pattern == "No Clear Setup":
                    pattern = "Higher Low"
                    score = 72.0
                details.append("Higher Low Formation ✓")

        # 5. Volume Expansion
        if vol is not None and len(vol) >= self.volume_avg_window + 1:
            avg_vol = float(vol.iloc[-self.volume_avg_window - 1:-1].mean())
            cur_vol = float(vol.iloc[-1])
            if avg_vol > 0 and cur_vol > avg_vol * 1.5:
                score = min(100.0, score + 8)
                details.append(f"Volume {cur_vol / avg_vol:.1f}×avg — expansion ✓")

        # 6. ATR Contraction (squeeze before breakout)
        if atr_series is not None and len(atr_series) >= 10:
            recent_atr = [float(v) for v in atr_series.iloc[-5:] if not pd.isna(v)]
            prior_atr = [float(v) for v in atr_series.iloc[-15:-5] if not pd.isna(v)]
            if recent_atr and prior_atr:
                avg_recent = np.mean(recent_atr)
                avg_prior = np.mean(prior_atr)
                if avg_recent < avg_prior * 0.8:
                    score = min(100.0, score + 7)
                    details.append("ATR contracting — squeeze setup ✓")

        return pattern, round(score, 1), details, None

    # ── Helpers ──────────────────────────────────────────────────────────

    def _check_higher_highs(self, weekly: pd.DataFrame, lookback: int = 8) -> bool:
        """Check if weekly chart shows Higher Highs over the lookback bars."""
        if len(weekly) < lookback:
            return False
        highs = [float(weekly["High"].iloc[j]) for j in range(-lookback, 0)]
        mid = len(highs) // 2
        return max(highs[mid:]) > max(highs[:mid])

    def _resample_weekly(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Resample daily OHLCV to weekly."""
        try:
            return daily.resample("W").agg({
                "Open": "first", "High": "max",
                "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna()
        except Exception:
            return daily

    def _get_close(self, df: pd.DataFrame) -> Optional[pd.Series]:
        for col in df.columns:
            if str(col).lower() in {"close", "adj close"}:
                return df[col].dropna()
        return None
