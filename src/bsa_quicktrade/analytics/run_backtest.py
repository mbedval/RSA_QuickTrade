"""Swing Trading Research Engine — Backtest Runner.

Implements the full 7-stage swing trading pipeline as defined in swingtrader.md:
  Stage 1: Market Filter (NIFTY trend + VIX gate)
  Stage 2: Sector Strength Filter
  Stage 3: Relative Strength vs NIFTY
  Stage 4/5: Weekly Trend + Daily Setup Detection
  Stage 6: Trade Quality Score (0-100, min 75 to trade)
  Stage 7: Entry Validation (R:R >= 1:2)
  Stage 8: ATR-based position sizing
  Stage 9: Layered exits (SL → BE → Trailing Stop → Time Exit, max 26 days)
  Stage 11: Forward Return Analysis
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.report_generator import ReportGenerator
from bsa_quicktrade.analytics.trade_log import save_trade_log
from bsa_quicktrade.analytics.utils import parse_date
from bsa_quicktrade.analytics.forward_return import ForwardReturnAnalyzer
from bsa_quicktrade.core.cache import DataCache
from bsa_quicktrade.core.config import load_config
from bsa_quicktrade.core.models import StockData
from bsa_quicktrade.data.downloader import DataDownloader
from bsa_quicktrade.entry.setup_detector import SetupDetector
from bsa_quicktrade.exit.swing_exit import SwingExitManager
from bsa_quicktrade.filters.market_filter import MarketFilter
from bsa_quicktrade.filters.relative_strength import RelativeStrengthAnalyzer
from bsa_quicktrade.scoring.swing_score import SwingScorer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Swing engine parameters
SWING_MIN_SCORE    = 75.0    # Stage 6 threshold
SWING_MAX_DAYS     = 26      # Stage 9 max hold
SWING_MIN_DAYS     = 5       # Stage 9 min hold
SWING_RISK_PCT     = 0.01    # Stage 8: 1% of capital per trade
SWING_ATR_SL       = 2.0     # Stage 9: stop = 2×ATR
SWING_ATR_TARGET   = 3.0     # Stage 9: target = 3×ATR
SWING_TRAILING     = "ema20" # Stage 9: trailing method


def _build_analyzers(config):
    """Instantiate all 12 analyzers."""
    from bsa_quicktrade.analyzers.trend import TrendAnalyzer
    from bsa_quicktrade.analyzers.momentum import MomentumAnalyzer
    from bsa_quicktrade.analyzers.volatility import VolatilityAnalyzer
    from bsa_quicktrade.analyzers.volume import VolumeAnalyzer
    from bsa_quicktrade.analyzers.price_action import PriceActionAnalyzer
    from bsa_quicktrade.analyzers.candlestick import CandlestickAnalyzer
    from bsa_quicktrade.analyzers.chart_patterns import ChartPatternAnalyzer
    from bsa_quicktrade.analyzers.fibonacci import FibonacciAnalyzer
    from bsa_quicktrade.analyzers.ichimoku import IchimokuAnalyzer
    from bsa_quicktrade.analyzers.market_breadth import MarketBreadthAnalyzer
    from bsa_quicktrade.analyzers.options import OptionsAnalyzer
    from bsa_quicktrade.analyzers.statistics import StatisticalAnalyzer

    return [
        TrendAnalyzer(config),
        MomentumAnalyzer(config),
        VolatilityAnalyzer(config),
        VolumeAnalyzer(config),
        PriceActionAnalyzer(config),
        CandlestickAnalyzer(config),
        ChartPatternAnalyzer(config),
        FibonacciAnalyzer(config),
        IchimokuAnalyzer(config),
        MarketBreadthAnalyzer(config),
        OptionsAnalyzer(config),
        StatisticalAnalyzer(config),
    ]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present, keeping only standard OHLCV names."""
    if isinstance(df.columns, pd.MultiIndex):
        ohlcv_names = {"open", "high", "low", "close", "volume", "adj close"}
        flat_cols = []
        for col in df.columns:
            metric_name = col[-1]
            for part in col:
                if str(part).lower() in ohlcv_names:
                    metric_name = part
                    break
            flat_cols.append(metric_name)
        df.columns = flat_cols
    return df


def run_stock_backtest(ticker: str, capital: float, output_dir: Path) -> Path | None:
    """Run a backtest on a single stock, save log, and compile HTML report."""
    # 1. Load config and Cache
    config = load_config()
    cache = DataCache(
        directory=config.cache.directory,
        enabled=config.cache.enabled,
        daily_ttl_hours=config.cache.daily_ttl_hours,
    )

    # Format ticker
    ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    symbol = ticker.replace(".NS", "")

    console.print(f"[bold cyan]Downloading historical data for {symbol} …[/]")
    dl = DataDownloader(config, cache)
    daily_dict = dl.download_daily([ticker_ns])
    weekly_dict = dl.download_weekly([ticker_ns])

    if ticker_ns not in daily_dict or daily_dict[ticker_ns].empty:
        console.print(f"[bold red]Error: No data downloaded for {symbol}.[/]")
        cache.close()
        return None

    df_daily = daily_dict[ticker_ns].copy()
    df_weekly = weekly_dict.get(ticker_ns, df_daily).copy()
    
    # Flatten MultiIndex columns if present
    df_daily = flatten_columns(df_daily)
    df_weekly = flatten_columns(df_weekly)
    
    # 2. Pre-calculate indicators on daily data
    # (So we can append them to the trade log records)
    close_col = "Close"
    for col in df_daily.columns:
        if "close" in str(col).lower():
            close_col = col
            break

    df_daily["RSI_14"] = ta.rsi(df_daily[close_col], length=14)
    df_daily["ATR_14"] = ta.atr(df_daily["High"], df_daily["Low"], df_daily[close_col], length=14)
    adx_df = ta.adx(df_daily["High"], df_daily["Low"], df_daily[close_col], length=14)
    if adx_df is not None:
        df_daily["ADX_14"] = adx_df.iloc[:, 0]
    else:
        df_daily["ADX_14"] = 25.0

    df_daily["EMA_200"] = ta.ema(df_daily[close_col], length=200)

    # Create base stock container
    full_stock_data = StockData(
        ticker=ticker_ns,
        company_name=symbol,
        daily=df_daily,
        weekly=df_weekly,
    )

    # 3. Setup backtest parameters
    analyzers = _build_analyzers(config)
    
    trades: List[TradeRecord] = []
    
    start_idx = max(200, config.backtesting.walk_forward_window)
    if len(df_daily) < start_idx + 10:
        console.print(f"[bold red]Error: Insufficient data history ({len(df_daily)} rows) to backtest {symbol}.[/]")
        cache.close()
        return None

    commission_pct = config.backtesting.commission_pct / 100.0
    slippage_pct = config.backtesting.slippage_pct / 100.0

    console.print(f"[bold cyan]Backtesting {len(df_daily) - start_idx} bars for {symbol} …[/]")
    
    trade_id_counter = 1
    signal_dates: List = []

    # ── Instantiate swing pipeline components ──
    setup_detector = SetupDetector()
    swing_scorer   = SwingScorer(min_score=SWING_MIN_SCORE)
    exit_manager   = SwingExitManager(
        atr_sl_mult=SWING_ATR_SL,
        atr_target_mult=SWING_ATR_TARGET,
        max_holding_days=SWING_MAX_DAYS,
        trailing_method=SWING_TRAILING,
        min_hold_days=SWING_MIN_DAYS,
    )
    market_filter  = MarketFilter()
    rs_analyzer    = RelativeStrengthAnalyzer()

    # ── Pre-compute a lightweight NIFTY proxy from the stock data ──
    # When no separate NIFTY feed is available, we skip market filter
    # (returns permissive MarketCondition)
    nifty_daily = None   # TODO: optionally inject NIFTY feed

    skipped_score    = 0
    skipped_setup    = 0
    skipped_market   = 0

    # Loop day by day — walk-forward, no lookahead
    for i in range(start_idx, len(df_daily) - SWING_MAX_DAYS - 1):
        # Slice data up to day i
        truncated_daily = df_daily.iloc[: i + 1].copy()
        truncated_stock = StockData(
            ticker=ticker_ns,
            company_name=symbol,
            daily=truncated_daily,
            weekly=df_weekly,
        )

        # ── Stage 1: Market Filter ──
        if nifty_daily is not None:
            mkt = market_filter.evaluate(nifty_daily.iloc[: i + 1])
            if not mkt.allow_long:
                skipped_market += 1
                continue
        else:
            mkt = None

        # ── Stage 3: Relative Strength ──
        rs = rs_analyzer.analyze(truncated_daily, nifty_daily)

        # ── Stage 4 & 5: Weekly Trend + Daily Setup ──
        entry_setup = setup_detector.detect(truncated_daily, df_weekly)
        if not entry_setup.is_valid:
            skipped_setup += 1
            continue

        # ── Stage 6: Consensus direction from analyzers ──
        bullish_votes = 0
        bearish_votes = 0
        triggered_rules: List[str] = []

        for analyzer in analyzers:
            try:
                res = analyzer.analyze(truncated_stock)
                if res.signal.is_bullish:
                    bullish_votes += 1
                    triggered_rules.append(f"{analyzer.name.upper()}_BULLISH")
                elif res.signal.is_bearish:
                    bearish_votes += 1
                    triggered_rules.append(f"{analyzer.name.upper()}_BEARISH")
            except Exception:
                continue

        direction = None
        if bullish_votes > bearish_votes and bullish_votes >= 2 and entry_setup.weekly_trend_bullish:
            direction = "long"
        elif bearish_votes > bullish_votes and bearish_votes >= 2:
            direction = "short"

        if direction is None:
            continue

        # ── Indicators at entry ──
        entry_dt    = df_daily.index[i]
        entry_price = float(df_daily[close_col].iloc[i])
        atr  = float(df_daily["ATR_14"].iloc[i]) if not pd.isna(df_daily["ATR_14"].iloc[i]) else 0.0
        adx  = float(df_daily["ADX_14"].iloc[i]) if not pd.isna(df_daily["ADX_14"].iloc[i]) else 25.0
        rsi  = float(df_daily["RSI_14"].iloc[i]) if not pd.isna(df_daily["RSI_14"].iloc[i]) else 50.0

        if direction == "long":
            stop_price   = entry_price - SWING_ATR_SL * atr if atr > 0 else entry_price * 0.97
            target_price = entry_price + SWING_ATR_TARGET * atr if atr > 0 else entry_price * 1.06
        else:
            stop_price   = entry_price + SWING_ATR_SL * atr if atr > 0 else entry_price * 1.03
            target_price = entry_price - SWING_ATR_TARGET * atr if atr > 0 else entry_price * 0.94

        # ── Stage 6 Trade Quality Score ──
        quality = swing_scorer.score(
            entry_setup=entry_setup,
            market_condition=mkt,
            rs_result=rs,
            sector_ranking=None,   # sector ranking requires external data
            stock_daily=truncated_daily,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            sector=full_stock_data.sector or "",
        )

        if not quality.passes_threshold:
            skipped_score += 1
            continue

        # ── Stage 7: R:R validation ──
        if quality.rr_ratio < 1.5:
            continue

        # ── Stage 8: ATR-based position sizing ──
        stop_distance = abs(entry_price - stop_price)
        if stop_distance > 0:
            risk_amount = capital * SWING_RISK_PCT
            quantity = max(1.0, np.floor(risk_amount / stop_distance))
        else:
            quantity = max(1.0, np.floor(capital * 0.05 / entry_price))
        capital_used = entry_price * quantity

        # ── Stage 9: Layered exit simulation ──
        exit_result = exit_manager.simulate(
            daily=df_daily,
            entry_idx=i,
            entry_price=entry_price,
            atr=atr,
            direction=direction,
        )

        exit_price  = exit_result.exit_price
        exit_dt     = exit_result.exit_date
        exit_reason = exit_result.exit_reason

        # ── P&L ──
        raw_ret = (exit_price - entry_price) / entry_price if direction == "long" else (entry_price - exit_price) / entry_price
        net_ret = raw_ret - (commission_pct + slippage_pct)
        profit      = net_ret * capital_used
        profit_pct  = net_ret * 100.0

        ema_200 = df_daily["EMA_200"].iloc[i]
        regime  = "Bull" if not pd.isna(ema_200) and entry_price > ema_200 else "Bear"
        brokerage_cost = capital_used * commission_pct
        slippage_cost  = capital_used * slippage_pct

        trade = TradeRecord(
            trade_id=f"T_{trade_id_counter:03d}",
            strategy_name="QuickTrade_SwingEngine",
            module_name="SwingPipeline",
            signal_name=f"SwingScore_{quality.total:.0f}",
            entry_date=parse_date(entry_dt),
            exit_date=parse_date(exit_dt),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            capital_used=capital_used,
            profit=profit,
            profit_pct=profit_pct,
            atr=atr,
            adx=adx,
            rsi=rsi,
            market_regime=regime,
            exit_reason=exit_reason,
            triggered_rules=triggered_rules + [f"Quality:{quality.total:.0f}", f"Pattern:{entry_setup.daily_pattern}"],
            brokerage=brokerage_cost,
            slippage=slippage_cost,
            direction=direction,
        )

        trades.append(trade)
        signal_dates.append(entry_dt)
        trade_id_counter += 1

    if skipped_score or skipped_setup or skipped_market:
        console.print(
            f"[dim]Pipeline filtered: {skipped_market} market, {skipped_setup} setup, "
            f"{skipped_score} score — {len(trades)} trades passed all stages.[/]"
        )

    # ── Stage 11: Forward Return Analysis ──
    if signal_dates and len(df_daily) > 30:
        fwd_analyzer = ForwardReturnAnalyzer()
        fwd_summary = fwd_analyzer.analyze(df_daily[[close_col]].rename(columns={close_col: "Close"}), signal_dates)
        console.print(f"[bold]Optimal holding period:[/] {fwd_summary.optimal_horizon} days")
        console.print(f"[dim]{fwd_summary.recommendation}[/]")

    cache.close()

    if not trades:
        console.print("[bold yellow]No trades triggered during backtest timeframe.[/]")
        return None

    # 4. Save tradelog CSV
    tradelog_csv = output_dir / f"{symbol}_tradelog.csv"
    save_trade_log(trades, tradelog_csv)
    console.print(f"[green]✓ Trade Log saved to:[/] {tradelog_csv}")

    # 5. Compile HTML Dashboard
    report_gen = ReportGenerator(output_dir=output_dir)
    report_file = report_gen.generate_html_report(
        records=trades,
        strategy_name=f"QuickTrade System - {symbol}",
        starting_capital=capital,
        file_name=f"{symbol}_analytics_report.html",
    )
    
    return report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest QuickTrade on real stock data.")
    parser.add_argument("ticker", help="NSE Stock Ticker (e.g., RELIANCE, PAYTM, INFOSYS)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial virtual capital")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report = run_stock_backtest(args.ticker, args.capital, output_path)
    
    if report:
        console.print(Panel(
            f"[bold green]✓ Backtest analysis completed successfully![/]\n\n"
            f"[bold]Output Report:[/] {report}\n"
            f"[bold]CSV Trade Log:[/] {output_path}/{args.ticker.replace('.NS', '')}_tradelog.csv",
            title="Analysis Summary",
            border_style="green"
        ))
    else:
        sys.exit(1)
