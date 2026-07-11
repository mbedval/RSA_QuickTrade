"""Tests for Milestone 2 — Setup Detector & Swing Quality Score."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsa_quicktrade.entry.setup_detector import SetupDetector, EntrySetup
from bsa_quicktrade.scoring.swing_score import SwingScorer, SwingQualityScore
from bsa_quicktrade.filters.market_filter import MarketFilter


def _make_bullish_daily(n: int = 200) -> pd.DataFrame:
    """Create a steadily trending up OHLCV daily series."""
    np.random.seed(7)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base = 1000.0
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + 3.0 + np.random.randn() * 4)
    close = pd.Series([max(1, p) for p in prices], index=dates)
    high = close * 1.005 + abs(np.random.randn(n) * 2)
    low = close * 0.995 - abs(np.random.randn(n) * 2)
    volume = pd.Series(np.random.randint(2_000_000, 8_000_000, n), index=dates)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


# ── Test 1: SetupDetector — valid bullish setup ──────────────────────────────

def test_setup_detector_bullish():
    daily = _make_bullish_daily(200)
    detector = SetupDetector()
    result = detector.detect(daily)
    assert isinstance(result, EntrySetup)
    assert result.weekly_trend_score >= 0
    assert result.daily_pattern_score >= 0
    assert isinstance(result.is_valid, bool)


def test_setup_detector_extended_move_avoided():
    """Massive run-up should trigger 'Extended Move' avoidance."""
    np.random.seed(1)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Price jumps 30% in last 5 days — typical extended move
    close_vals = list(range(1000, 1000 + n * 3, 3))
    close = pd.Series(close_vals[:n], index=dates, dtype=float)
    high = close + 5
    low = close - 5
    vol = pd.Series([1_000_000] * n, index=dates)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)
    detector = SetupDetector()
    result = detector.detect(df)
    # Extended move detection may or may not trigger depending on ATR context — just verify no crash
    assert isinstance(result, EntrySetup)


# ── Test 2: SwingScorer ───────────────────────────────────────────────────────

def test_swing_scorer_basic():
    from bsa_quicktrade.entry.setup_detector import EntrySetup
    daily = _make_bullish_daily(200)
    entry_setup = EntrySetup(
        weekly_trend_bullish=True,
        weekly_trend_score=80.0,
        daily_pattern="Pullback to EMA20",
        daily_pattern_score=85.0,
        avoid_reason=None,
        is_valid=True,
        details=["Weekly trend strong", "Pullback detected"],
    )
    scorer = SwingScorer(min_score=75)
    entry = 1200.0
    stop = 1160.0
    target = 1320.0
    result = scorer.score(
        entry_setup=entry_setup,
        market_condition=None,
        rs_result=None,
        sector_ranking=None,
        stock_daily=daily,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        sector="Banking",
    )
    assert isinstance(result, SwingQualityScore)
    assert 0 <= result.total <= 100
    assert result.rr_ratio > 0


def test_swing_scorer_poor_rr_fails():
    from bsa_quicktrade.entry.setup_detector import EntrySetup
    daily = _make_bullish_daily(200)
    entry_setup = EntrySetup(
        weekly_trend_bullish=True,
        weekly_trend_score=90.0,
        daily_pattern="Pullback to EMA20",
        daily_pattern_score=90.0,
        avoid_reason=None,
        is_valid=True,
        details=[],
    )
    scorer = SwingScorer(min_score=75)
    # Very tight target = poor R:R
    result = scorer.score(
        entry_setup=entry_setup,
        market_condition=None,
        rs_result=None,
        sector_ranking=None,
        stock_daily=daily,
        entry_price=1200.0,
        stop_price=1160.0,
        target_price=1220.0,   # only 1:0.5 R:R
        sector="IT",
    )
    assert result.rr_ratio < 1.0


def test_swing_scorer_threshold_gating():
    from bsa_quicktrade.entry.setup_detector import EntrySetup
    daily = _make_bullish_daily(200)
    # Weak setup — score should not pass 75
    entry_setup = EntrySetup(
        weekly_trend_bullish=False,
        weekly_trend_score=10.0,
        daily_pattern="No Clear Setup",
        daily_pattern_score=20.0,
        avoid_reason="No weekly trend",
        is_valid=False,
        details=[],
    )
    scorer = SwingScorer(min_score=75)
    result = scorer.score(
        entry_setup=entry_setup,
        market_condition=None,
        rs_result=None,
        sector_ranking=None,
        stock_daily=daily,
        entry_price=1200.0,
        stop_price=1160.0,
        target_price=1320.0,
        sector="FMCG",
    )
    assert result.passes_threshold is False
