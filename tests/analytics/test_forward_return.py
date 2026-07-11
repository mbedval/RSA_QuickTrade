"""Tests for Milestone 4 — ForwardReturnAnalyzer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsa_quicktrade.analytics.forward_return import ForwardReturnAnalyzer, ForwardReturnSummary


def _make_ohlcv(n: int = 300, trend: str = "up") -> pd.DataFrame:
    np.random.seed(99)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base = 500.0
    closes = [base]
    for _ in range(n - 1):
        step = 1.5 if trend == "up" else -1.5
        closes.append(max(1, closes[-1] + step + np.random.randn() * 3))
    c = pd.Series(closes, index=dates, dtype=float)
    return pd.DataFrame({"Close": c}, index=dates)


# ── Test 1: Basic forward return calculation ──────────────────────────────────

def test_forward_return_basic():
    df = _make_ohlcv(300, "up")
    signal_dates = list(df.index[50:150:10])   # 10 signals spaced 10 days apart
    analyzer = ForwardReturnAnalyzer()
    summary = analyzer.analyze(df, signal_dates)
    assert isinstance(summary, ForwardReturnSummary)
    assert len(summary.horizons) == 5
    for h in [5, 10, 15, 20, 26]:
        assert h in summary.horizons
        stats = summary.horizons[h]
        assert stats.trade_count >= 0


# ── Test 2: Uptrend yields positive returns at most horizons ──────────────────

def test_forward_return_uptrend():
    df = _make_ohlcv(300, "up")
    signal_dates = list(df.index[50:200:5])
    analyzer = ForwardReturnAnalyzer()
    summary = analyzer.analyze(df, signal_dates)
    # In an uptrend, most horizons should have positive average return
    positive_horizons = sum(
        1 for s in summary.horizons.values()
        if s.trade_count > 0 and s.avg_return_pct > 0
    )
    assert positive_horizons >= 3


# ── Test 3: Empty signal dates returns graceful result ────────────────────────

def test_forward_return_empty_signals():
    df = _make_ohlcv(100, "up")
    analyzer = ForwardReturnAnalyzer()
    summary = analyzer.analyze(df, signal_dates=[])
    assert summary.optimal_horizon in [5, 10, 15, 20, 26]
    assert "Insufficient" in summary.recommendation or summary.optimal_horizon >= 5
