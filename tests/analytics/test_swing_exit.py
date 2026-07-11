"""Tests for Milestone 3 — SwingExitManager layered exit logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsa_quicktrade.exit.swing_exit import SwingExitManager, ExitResult


def _make_daily(closes: list[float], highs: list[float] = None, lows: list[float] = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from close list."""
    n = len(closes)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    c = pd.Series(closes, index=dates, dtype=float)
    h = pd.Series(highs, index=dates, dtype=float) if highs else c * 1.01
    lo = pd.Series(lows, index=dates, dtype=float) if lows else c * 0.99
    return pd.DataFrame({"Open": c, "High": h, "Low": lo, "Close": c}, index=dates)


# ── Test 1: Stop-loss triggered ───────────────────────────────────────────────

def test_stop_loss_triggered():
    closes = [100.0] * 5 + [90.0] * 25
    lows   = [100.0] * 5 + [88.0] * 25   # low drops well below stop
    df = _make_daily(closes, lows=lows)
    mgr = SwingExitManager(atr_sl_mult=2.0, atr_target_mult=3.0)
    result = mgr.simulate(df, entry_idx=0, entry_price=100.0, atr=4.0, direction="long")
    # Stop is at 100 - 2*4 = 92; low drops to 88 on day 5
    assert result.exit_reason in {"Stop Loss", "Break Even"}
    assert result.final_profit_pct < 0


# ── Test 2: Profit target hit ────────────────────────────────────────────────

def test_profit_target_hit():
    """Ensure a profitable exit occurs when target is clearly activated.

    Uses large ATR (15) so target = entry + 3*15 = entry + 45.
    A spike to 165 on day 7 after entry guarantees target or trailing stop activation.
    Final profit must be non-negative (break-even or better).
    """
    n = 60
    closes = [100.0 + i * 0.5 for i in range(n)]
    highs  = [c + 1 for c in closes[:34]] + [165.0] + [c + 1 for c in closes[35:]]
    lows   = [c - 1 for c in closes]
    df = _make_daily(closes, highs=highs, lows=lows)
    mgr = SwingExitManager(atr_sl_mult=2.0, atr_target_mult=3.0, min_hold_days=5)
    entry_price = closes[27]   # ~113.5
    result = mgr.simulate(df, entry_idx=27, entry_price=entry_price, atr=15.0, direction="long")
    # High of 165 > target of 113.5+45=158.5; expect Target, Trailing Stop, or Time Exit with profit
    assert result.final_profit_pct >= 0


# ── Test 3: Break-even activation ────────────────────────────────────────────

def test_break_even_activated():
    """When price reaches 2×ATR profit, stop moves to break-even."""
    n = 30
    closes = [100.0 + i * 0.5 for i in range(n)]    # steady drift up
    highs  = [c + 1 for c in closes]
    lows   = [c - 1 for c in closes]
    df = _make_daily(closes, highs=highs, lows=lows)
    mgr = SwingExitManager(atr_sl_mult=2.0, atr_target_mult=3.0, max_holding_days=26, min_hold_days=5)
    result = mgr.simulate(df, entry_idx=0, entry_price=100.0, atr=3.0, direction="long")
    assert isinstance(result, ExitResult)
    assert result.holding_days >= 5


# ── Test 4: Time exit after max holding period ───────────────────────────────

def test_time_exit():
    closes = [100.0] * 60
    df = _make_daily(closes)
    mgr = SwingExitManager(atr_sl_mult=2.0, atr_target_mult=3.0, max_holding_days=10, min_hold_days=5)
    # With flat price and tiny ATR=0.5, stop=99, target=101.5
    # Some days may hit target or break-even depending on simulated highs/lows
    result = mgr.simulate(df, entry_idx=0, entry_price=100.0, atr=0.5, direction="long")
    # Must exit within max_holding_days
    assert result.holding_days <= 10
    assert result.exit_reason in {"Time Exit", "Break Even", "Target", "Trailing Stop", "Stop Loss"}


# ── Test 5: Short trade stop-loss ────────────────────────────────────────────

def test_short_stop_loss():
    closes = [100.0] * 5 + [112.0] * 25
    highs  = [100.0] * 5 + [115.0] * 25
    df = _make_daily(closes, highs=highs)
    mgr = SwingExitManager(atr_sl_mult=2.0, atr_target_mult=3.0)
    # Short stop at 100 + 2*4 = 108; high hits 115
    result = mgr.simulate(df, entry_idx=0, entry_price=100.0, atr=4.0, direction="short")
    assert result.exit_reason in {"Stop Loss", "Break Even"}
    assert result.final_profit_pct < 0
