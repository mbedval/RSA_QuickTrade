"""Backtest Runner and Analytics Dashboard Generator for Real Stocks.

Downloads historical NSE data, runs walk-forward trading system backtest,
creates TradeLog CSV, and compiles the HTML dashboard report.
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
from bsa_quicktrade.core.cache import DataCache
from bsa_quicktrade.core.config import load_config
from bsa_quicktrade.core.models import StockData
from bsa_quicktrade.data.downloader import DataDownloader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()


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
    
    # Loop day by day simulating walk-forward scoring
    for i in range(start_idx, len(df_daily) - 1):
        # Slice data up to day i (preventing lookahead bias)
        truncated_daily = df_daily.iloc[: i + 1]
        truncated_stock = StockData(
            ticker=ticker_ns,
            company_name=symbol,
            daily=truncated_daily,
            weekly=df_weekly,
        )

        bullish_votes = 0
        bearish_votes = 0
        triggered_rules = []

        # Gather signals from all 12 modules
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

        # Vote threshold (min 2 votes to trade - earlier entries)
        direction = None
        if bullish_votes > bearish_votes and bullish_votes >= 2:
            direction = "long"
        elif bearish_votes > bullish_votes and bearish_votes >= 2:
            direction = "short"

        if direction:
            # Entry details
            entry_dt = df_daily.index[i]
            entry_price = float(df_daily[close_col].iloc[i])
            
            # Extract indicators at entry
            atr = float(df_daily["ATR_14"].iloc[i]) if not pd.isna(df_daily["ATR_14"].iloc[i]) else 0.0
            adx = float(df_daily["ADX_14"].iloc[i]) if not pd.isna(df_daily["ADX_14"].iloc[i]) else 25.0
            rsi = float(df_daily["RSI_14"].iloc[i]) if not pd.isna(df_daily["RSI_14"].iloc[i]) else 50.0
            
            # Calculate dynamic targets & stop-losses
            if direction == "long":
                stop_loss = entry_price - 2.0 * atr if atr > 0 else entry_price * 0.97
                target_1 = entry_price + 3.0 * atr if atr > 0 else entry_price * 1.06
            else:
                stop_loss = entry_price + 2.0 * atr if atr > 0 else entry_price * 1.03
                target_1 = entry_price - 3.0 * atr if atr > 0 else entry_price * 0.94
                
            # Simulate holding the position
            exit_price = entry_price
            exit_dt = df_daily.index[i + 1]
            exit_reason = "Consensus Close"
            
            max_holding_days = 10
            exited = False
            
            for d in range(i + 1, min(i + 1 + max_holding_days, len(df_daily))):
                day_low = float(df_daily["Low"].iloc[d])
                day_high = float(df_daily["High"].iloc[d])
                day_close = float(df_daily[close_col].iloc[d])
                
                # Check stop-loss
                if direction == "long" and day_low <= stop_loss:
                    exit_price = stop_loss
                    exit_dt = df_daily.index[d]
                    exit_reason = "Stop Loss"
                    exited = True
                    break
                elif direction == "short" and day_high >= stop_loss:
                    exit_price = stop_loss
                    exit_dt = df_daily.index[d]
                    exit_reason = "Stop Loss"
                    exited = True
                    break
                    
                # Check profit target
                if direction == "long" and day_high >= target_1:
                    exit_price = target_1
                    exit_dt = df_daily.index[d]
                    exit_reason = "Target"
                    exited = True
                    break
                elif direction == "short" and day_low <= target_1:
                    exit_price = target_1
                    exit_dt = df_daily.index[d]
                    exit_reason = "Target"
                    exited = True
                    break
                    
            if not exited:
                # Time exit: close out on the final day's close
                final_d = min(i + max_holding_days, len(df_daily) - 1)
                exit_price = float(df_daily[close_col].iloc[final_d])
                exit_dt = df_daily.index[final_d]
                exit_reason = "Time Exit"
                
            # Simple position sizing: use 10% of starting capital per trade
            trade_capital = capital * 0.1
            quantity = max(1.0, np.floor(trade_capital / entry_price))
            capital_used = entry_price * quantity
            
            # Calculate raw return and net return after costs
            raw_ret = (exit_price - entry_price) / entry_price if direction == "long" else (entry_price - exit_price) / entry_price
            net_ret = raw_ret - (commission_pct + slippage_pct)
            
            profit = net_ret * capital_used
            profit_pct = net_ret * 100.0
            
            # Market Regime
            ema_200 = df_daily["EMA_200"].iloc[i]
            regime = "Bull" if not pd.isna(ema_200) and entry_price > ema_200 else "Bear"
            
            # Cost calculations
            brokerage = capital_used * commission_pct
            slippage = capital_used * slippage_pct
            
            # Construct Trade Record
            trade = TradeRecord(
                trade_id=f"T_{trade_id_counter:03d}",
                strategy_name="QuickTrade_Consensus",
                module_name="CombinedSystem",
                signal_name=f"Vote_{bullish_votes if direction=='long' else bearish_votes}",
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
                triggered_rules=triggered_rules,
                brokerage=brokerage,
                slippage=slippage,
                direction=direction,
            )
            
            trades.append(trade)
            trade_id_counter += 1

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
