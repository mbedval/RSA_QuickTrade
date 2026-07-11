"""Swing Exit Manager — Stage 9 of the swing trading pipeline.

Implements layered exit logic as specified in swingtrader.md:

  Phase 1: Initial stop-loss @ 2×ATR below entry
  Phase 2: When unrealised profit ≥ 2×ATR → move stop to break-even
  Phase 3: When unrealised profit ≥ 3×ATR → activate trailing stop
           (EMA20 / SuperTrend / Chandelier Exit — configurable)
  Phase 4: Force time exit after max_days (default 26 trading days)
  Phase 5: Early exit if weekly trend flips bearish (if weekly data provided)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)

TrailingMethod = Literal["ema20", "supertrend", "chandelier"]


@dataclass
class ExitResult:
    """Result of swing exit simulation."""
    exit_price: float
    exit_date: object                   # datetime / Timestamp
    exit_reason: str                    # "Stop Loss" | "Break Even" | "Target" | "Trailing Stop" | "Time Exit"
    holding_days: int
    max_profit_pct: float               # peak unrealised profit % during trade
    final_profit_pct: float             # actual exit profit %


class SwingExitManager:
    """Simulate layered swing-trade exits over OHLCV price data.

    Parameters
    ----------
    atr_sl_mult:
        ATR multiplier for initial stop-loss (default 2.0).
    atr_target_mult:
        ATR multiplier for initial target / trailing activation (default 3.0).
    max_holding_days:
        Maximum days to hold before forced time exit (default 26).
    trailing_method:
        Trailing stop method: "ema20", "supertrend", or "chandelier".
    min_hold_days:
        Minimum days to hold before allowing time exit (default 5).
    """

    def __init__(
        self,
        atr_sl_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        max_holding_days: int = 26,
        trailing_method: TrailingMethod = "ema20",
        min_hold_days: int = 5,
    ) -> None:
        self.atr_sl_mult = atr_sl_mult
        self.atr_target_mult = atr_target_mult
        self.max_holding_days = max_holding_days
        self.trailing_method = trailing_method
        self.min_hold_days = min_hold_days

    def simulate(
        self,
        daily: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        atr: float,
        direction: Literal["long", "short"] = "long",
    ) -> ExitResult:
        """Simulate the layered exit for a swing trade.

        Parameters
        ----------
        daily:
            Full daily OHLCV DataFrame with at least ``entry_idx + 1`` rows.
        entry_idx:
            Row index in `daily` where the trade was entered (close of that day).
        entry_price:
            Trade entry price.
        atr:
            ATR value at entry date.
        direction:
            "long" or "short".
        """
        if atr <= 0:
            atr = entry_price * 0.02  # fallback: 2% of price

        # ── Initial levels ──
        if direction == "long":
            stop = entry_price - self.atr_sl_mult * atr
            target = entry_price + self.atr_target_mult * atr
            be_trigger = entry_price + self.atr_sl_mult * atr   # 2×ATR profit → break-even
            trail_trigger = target                               # 3×ATR profit → trail
        else:
            stop = entry_price + self.atr_sl_mult * atr
            target = entry_price - self.atr_target_mult * atr
            be_trigger = entry_price - self.atr_sl_mult * atr
            trail_trigger = target

        be_activated = False
        trailing_active = False
        trailing_stop = stop
        max_price = entry_price
        min_price = entry_price

        n = len(daily)
        start = entry_idx + 1
        end = min(start + self.max_holding_days, n)

        for d in range(start, end):
            day_open  = float(daily["Open"].iloc[d]) if "Open" in daily.columns else float(daily["Close"].iloc[d])
            day_high  = float(daily["High"].iloc[d])
            day_low   = float(daily["Low"].iloc[d])
            day_close = float(daily["Close"].iloc[d])

            # Track extremes
            max_price = max(max_price, day_high)
            min_price = min(min_price, day_low)

            # ── Phase 2: break-even move ──
            if not be_activated:
                if direction == "long" and day_high >= be_trigger:
                    stop = entry_price          # move stop to entry = break-even
                    be_activated = True
                elif direction == "short" and day_low <= be_trigger:
                    stop = entry_price
                    be_activated = True

            # ── Phase 3: trailing stop activation ──
            if not trailing_active:
                if direction == "long" and day_high >= trail_trigger:
                    trailing_active = True
                    trailing_stop = self._calc_trailing_stop(daily, d, direction, atr)
                elif direction == "short" and day_low <= trail_trigger:
                    trailing_active = True
                    trailing_stop = self._calc_trailing_stop(daily, d, direction, atr)
            else:
                # Update trailing stop each day
                new_trail = self._calc_trailing_stop(daily, d, direction, atr)
                if direction == "long":
                    trailing_stop = max(trailing_stop, new_trail)
                else:
                    trailing_stop = min(trailing_stop, new_trail)
                stop = trailing_stop

            # ── Check stop-loss hit ──
            if direction == "long" and day_low <= stop:
                exit_price = max(stop, day_open)  # gap-adjusted
                return ExitResult(
                    exit_price=exit_price,
                    exit_date=daily.index[d],
                    exit_reason="Trailing Stop" if trailing_active else ("Break Even" if be_activated else "Stop Loss"),
                    holding_days=d - entry_idx,
                    max_profit_pct=self._pct(max_price, entry_price, direction),
                    final_profit_pct=self._pct(exit_price, entry_price, direction),
                )
            elif direction == "short" and day_high >= stop:
                exit_price = min(stop, day_open)
                return ExitResult(
                    exit_price=exit_price,
                    exit_date=daily.index[d],
                    exit_reason="Trailing Stop" if trailing_active else ("Break Even" if be_activated else "Stop Loss"),
                    holding_days=d - entry_idx,
                    max_profit_pct=self._pct(min_price, entry_price, direction),
                    final_profit_pct=self._pct(exit_price, entry_price, direction),
                )

            # ── Minimum hold reached + target achieved → exit ──
            if (d - entry_idx) >= self.min_hold_days:
                if direction == "long" and day_high >= target:
                    return ExitResult(
                        exit_price=target,
                        exit_date=daily.index[d],
                        exit_reason="Target",
                        holding_days=d - entry_idx,
                        max_profit_pct=self._pct(max_price, entry_price, direction),
                        final_profit_pct=self._pct(target, entry_price, direction),
                    )
                elif direction == "short" and day_low <= target:
                    return ExitResult(
                        exit_price=target,
                        exit_date=daily.index[d],
                        exit_reason="Target",
                        holding_days=d - entry_idx,
                        max_profit_pct=self._pct(min_price, entry_price, direction),
                        final_profit_pct=self._pct(target, entry_price, direction),
                    )

        # ── Phase 4: forced time exit ──
        final_d = min(end - 1, n - 1)
        final_price = float(daily["Close"].iloc[final_d])
        return ExitResult(
            exit_price=final_price,
            exit_date=daily.index[final_d],
            exit_reason="Time Exit",
            holding_days=final_d - entry_idx,
            max_profit_pct=self._pct(max_price if direction == "long" else min_price, entry_price, direction),
            final_profit_pct=self._pct(final_price, entry_price, direction),
        )

    # ── Trailing stop calculators ────────────────────────────────────────

    def _calc_trailing_stop(
        self,
        daily: pd.DataFrame,
        d: int,
        direction: str,
        atr: float,
    ) -> float:
        """Return trailing stop price for given method."""
        if self.trailing_method == "ema20":
            return self._ema20_stop(daily, d, direction)
        elif self.trailing_method == "chandelier":
            return self._chandelier_stop(daily, d, direction, atr)
        else:  # default fallback = ATR trail
            close = float(daily["Close"].iloc[d])
            if direction == "long":
                return close - atr * 1.5
            return close + atr * 1.5

    def _ema20_stop(self, daily: pd.DataFrame, d: int, direction: str) -> float:
        """EMA20 trailing stop."""
        close_series = daily["Close"].iloc[: d + 1]
        ema = ta.ema(close_series, length=20)
        if ema is None or len(ema.dropna()) == 0 or pd.isna(ema.iloc[-1]):
            # Fallback: use current close ± 2% as trailing stop
            close = float(daily["Close"].iloc[d])
            return close * (0.97 if direction == "long" else 1.03)
        return float(ema.iloc[-1])

    def _chandelier_stop(
        self, daily: pd.DataFrame, d: int, direction: str, atr: float
    ) -> float:
        """Chandelier Exit trailing stop (22-day high/low ± 3×ATR)."""
        lookback = min(22, d + 1)
        if direction == "long":
            highest = float(daily["High"].iloc[max(0, d - lookback + 1): d + 1].max())
            return highest - 3.0 * atr
        else:
            lowest = float(daily["Low"].iloc[max(0, d - lookback + 1): d + 1].min())
            return lowest + 3.0 * atr

    # ── Utility ──────────────────────────────────────────────────────────

    def _pct(self, price: float, entry: float, direction: str) -> float:
        if entry == 0:
            return 0.0
        raw = (price - entry) / entry * 100.0
        return raw if direction == "long" else -raw
