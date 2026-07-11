"""Market Filter — Stage 1 of swing trading pipeline.

Evaluates overall market conditions (NIFTY trend, VIX, advance-decline ratio)
to determine if the environment is suitable for new long swing trades.
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
class MarketCondition:
    """Summary of current market health for swing trading."""
    trend: str                      # "bull", "bear", "sideways"
    risk_level: str                 # "low", "medium", "high"
    vix: Optional[float]            # current VIX value, if available
    nifty_above_ema20: bool
    nifty_above_ema50: bool
    nifty_adx: float
    allow_long: bool                # True if safe to enter long swing trades
    allow_short: bool               # True if safe to enter short swing trades
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """0–100 score representing market friendliness for longs."""
        s = 50.0
        if self.nifty_above_ema20:
            s += 15
        if self.nifty_above_ema50:
            s += 15
        if self.nifty_adx > 25:
            s += 10
        if self.vix is not None:
            if self.vix < 15:
                s += 10
            elif self.vix > 25:
                s -= 20
            elif self.vix > 20:
                s -= 10
        return max(0.0, min(100.0, s))


class MarketFilter:
    """Determine market conditions from NIFTY daily data."""

    # VIX thresholds
    VIX_HIGH = 25.0
    VIX_ELEVATED = 20.0

    def __init__(self, ema_short: int = 20, ema_long: int = 50) -> None:
        self.ema_short = ema_short
        self.ema_long = ema_long

    def evaluate(
        self,
        nifty_daily: pd.DataFrame,
        vix_value: Optional[float] = None,
    ) -> MarketCondition:
        """Evaluate market condition from NIFTY OHLCV data.

        Parameters
        ----------
        nifty_daily:
            DataFrame with at least a Close column and DatetimeIndex.
        vix_value:
            Current VIX reading, optional.
        """
        close = self._get_close(nifty_daily)
        if close is None or len(close) < 60:
            return MarketCondition(
                trend="unknown", risk_level="high",
                vix=vix_value, nifty_above_ema20=False,
                nifty_above_ema50=False, nifty_adx=0.0,
                allow_long=False, allow_short=False,
                reasons=["Insufficient NIFTY data"],
            )

        ema20 = ta.ema(close, length=self.ema_short)
        ema50 = ta.ema(close, length=self.ema_long)
        adx_df = ta.adx(nifty_daily["High"], nifty_daily["Low"], close, length=14)

        current_price = float(close.iloc[-1])
        e20 = float(ema20.iloc[-1]) if ema20 is not None and not pd.isna(ema20.iloc[-1]) else 0.0
        e50 = float(ema50.iloc[-1]) if ema50 is not None and not pd.isna(ema50.iloc[-1]) else 0.0
        adx_val = float(adx_df.iloc[-1, 0]) if adx_df is not None and not pd.isna(adx_df.iloc[-1, 0]) else 0.0

        above_ema20 = current_price > e20 > 0
        above_ema50 = current_price > e50 > 0

        reasons: list[str] = []
        allow_long = True
        allow_short = False

        # Trend determination
        if above_ema20 and above_ema50:
            trend = "bull"
            reasons.append("NIFTY above EMA20 and EMA50 — bullish trend")
        elif not above_ema50:
            trend = "bear"
            allow_long = False
            allow_short = True
            reasons.append("NIFTY below EMA50 — bearish trend; avoid new longs")
        else:
            trend = "sideways"
            reasons.append("NIFTY between EMA20 and EMA50 — sideways; reduce exposure")

        # ADX trend strength
        if adx_val < 20:
            reasons.append(f"ADX {adx_val:.0f} — weak trend; prefer patience")
            if allow_long:
                allow_long = adx_val >= 15  # allow if not completely flat

        # VIX risk adjustment
        risk_level = "low"
        if vix_value is not None:
            if vix_value > self.VIX_HIGH:
                risk_level = "high"
                allow_long = False
                reasons.append(f"VIX {vix_value:.1f} > {self.VIX_HIGH} — high volatility; avoid longs")
            elif vix_value > self.VIX_ELEVATED:
                risk_level = "medium"
                reasons.append(f"VIX {vix_value:.1f} elevated — reduce position sizing")
            else:
                reasons.append(f"VIX {vix_value:.1f} normal — safe conditions")

        return MarketCondition(
            trend=trend,
            risk_level=risk_level,
            vix=vix_value,
            nifty_above_ema20=above_ema20,
            nifty_above_ema50=above_ema50,
            nifty_adx=adx_val,
            allow_long=allow_long,
            allow_short=allow_short,
            reasons=reasons,
        )

    def _get_close(self, df: pd.DataFrame) -> Optional[pd.Series]:
        for col in df.columns:
            if str(col).lower() in {"close", "adj close"}:
                return df[col].dropna()
        return None
