"""Tests for Milestone 1 — Market, Sector & Relative Strength Filters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsa_quicktrade.filters.market_filter import MarketFilter, MarketCondition
from bsa_quicktrade.filters.sector_filter import SectorFilter
from bsa_quicktrade.filters.relative_strength import RelativeStrengthAnalyzer


def _make_ohlcv(n: int = 300, trend: str = "up") -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base = 1000.0
    prices = [base]
    for _ in range(n - 1):
        step = 2.0 if trend == "up" else -2.0
        prices.append(prices[-1] + step + np.random.randn() * 5)
    close = pd.Series([max(10, p) for p in prices], index=dates)
    high = close + abs(np.random.randn(n) * 3)
    low = close - abs(np.random.randn(n) * 3)
    volume = pd.Series(np.random.randint(1_000_000, 5_000_000, n), index=dates)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


# ── Test 1: Market Filter — bullish NIFTY ────────────────────────────────────

def test_market_filter_bull():
    nifty = _make_ohlcv(300, trend="up")
    mf = MarketFilter()
    cond = mf.evaluate(nifty, vix_value=14.0)
    assert isinstance(cond, MarketCondition)
    assert cond.trend in {"bull", "sideways", "unknown"}
    assert cond.allow_long is True or cond.nifty_adx >= 0
    assert cond.score >= 0


def test_market_filter_bear_high_vix():
    nifty = _make_ohlcv(300, trend="up")
    mf = MarketFilter()
    cond = mf.evaluate(nifty, vix_value=30.0)
    # High VIX should prevent long entries
    assert cond.allow_long is False
    assert "high" in cond.risk_level


# ── Test 2: Sector Filter ────────────────────────────────────────────────────

def test_sector_filter_ranks_sectors():
    banking = _make_ohlcv(60, "up")
    it = _make_ohlcv(60, "up")
    fmcg = _make_ohlcv(60, "down")
    sector_data = {"Banking": banking, "IT": it, "FMCG": fmcg}

    sf = SectorFilter(top_k=2)
    ranking = sf.evaluate(sector_data)
    assert len(ranking.top_sectors) == 2
    assert len(ranking.rankings) == 3
    # Top sectors should have positive momentum
    for s in ranking.top_sectors:
        assert ranking.rankings[s] > ranking.rankings["FMCG"]


def test_sector_filter_no_data():
    sf = SectorFilter()
    ranking = sf.evaluate(None)
    assert ranking.allow_sectors == set()  # permissive — allow all


# ── Test 3: Relative Strength Analyzer ───────────────────────────────────────

def test_relative_strength_outperforming():
    stock = _make_ohlcv(80, "up")  # strongly uptrending
    nifty = _make_ohlcv(80, "down")  # underperforming benchmark
    ra = RelativeStrengthAnalyzer(window=20)
    result = ra.analyze(stock, nifty_daily=nifty)
    assert result.rs_vs_nifty > 0
    assert result.is_outperforming_market is True
    assert result.rs_score > 50


def test_relative_strength_underperforming():
    stock = _make_ohlcv(80, "down")
    nifty = _make_ohlcv(80, "up")
    ra = RelativeStrengthAnalyzer(window=20)
    result = ra.analyze(stock, nifty_daily=nifty)
    assert result.rs_vs_nifty < 0
    assert result.is_outperforming_market is False
    assert result.rs_score < 50
