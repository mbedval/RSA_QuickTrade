"""RSA QuickTrade — CLI entry point.

Orchestrates the full pipeline: universe → download → analyze → score → rank → report → charts.

Usage::

    python -m bsa_quicktrade scan
    python -m bsa_quicktrade analyze RELIANCE
    python -m bsa_quicktrade chart RELIANCE
    python -m bsa_quicktrade backtest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from bsa_quicktrade.core.cache import DataCache
from bsa_quicktrade.core.config import AppConfig, load_config
from bsa_quicktrade.core.constants import get_sector, to_yfinance_ticker
from bsa_quicktrade.core.logging_config import setup_logging
from bsa_quicktrade.core.models import AnalysisResult, StockAnalysis, StockData

console = Console()
logger = logging.getLogger(__name__)


# ── Analyzer Registry ───────────────────────────────────────────────────────

def _build_analyzers(config: AppConfig):
    """Instantiate all 12 analysis modules."""
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


# ── Pipeline Steps ──────────────────────────────────────────────────────────

def _download_all(config: AppConfig, cache: DataCache, tickers: list[str], filter_liquidity: bool = False):
    """Download all required data."""
    from bsa_quicktrade.data.downloader import DataDownloader
    from bsa_quicktrade.data.nse_data import NSEDataFetcher
    import logging
    
    # Suppress yfinance individual print errors
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    dl = DataDownloader(config, cache)
    nse = NSEDataFetcher(cache)

    console.print("[bold cyan]Phase 1/4: Downloading market data …[/]")

    daily = dl.download_daily(tickers)
    weekly = dl.download_weekly(tickers)

    # Download NIFTY index
    index_df = dl.download_index("^NSEI")

    # Filter by liquidity first to avoid slow option/delivery downloads for illiquid stocks
    if filter_liquidity:
        from bsa_quicktrade.data.universe import UniverseManager
        universe = UniverseManager(config)
        tickers = universe.filter_by_liquidity(tickers, daily)

    # Build StockData for each ticker
    stock_data: dict[str, StockData] = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("[dim]({task.completed}/{task.total})[/]"),
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching stock details", total=len(tickers))
        
        for t in tickers:
            if t not in daily or daily[t].empty:
                progress.advance(task)
                continue

            symbol = t.replace(".NS", "")
            progress.update(task, description=f"Fetching details for {symbol}")

            # Get company name
            info = dl.download_stock_info(t)
            company_name = info.get("longName", info.get("shortName", symbol))

            # Delivery data (best effort)
            delivery_df = nse.get_delivery_data(t, num_days=15)

            # Option chain (best effort)
            raw_oc = nse.get_option_chain(t)
            parsed_oc = nse.parse_option_chain(raw_oc) if raw_oc else None

            stock_data[t] = StockData(
                ticker=t,
                company_name=company_name,
                daily=daily[t],
                weekly=weekly.get(t, daily[t]),
                option_chain=parsed_oc,
                delivery_data=delivery_df,
                index_daily=index_df if not index_df.empty else None,
                sector=get_sector(t),
            )
            progress.advance(task)

    console.print(f"[green]✓ Downloaded data for {len(stock_data)} stocks[/]")
    
    # Consolidate failed/delisted stocks and print highlight message
    failed_tickers = [t.replace(".NS", "") for t in tickers if t not in daily or daily[t].empty]
    if failed_tickers:
        console.print()
        console.print(Panel(
            f"[bold yellow]⚠️ Information:[/] The following stock(s) were not found on NSE and may be delisted or invalid:\n"
            f"[bold cyan]{', '.join(sorted(failed_tickers))}[/]",
            border_style="yellow",
            title="[bold yellow]Delisted / Missing Stocks[/]"
        ))
        console.print()

    return stock_data


def _analyze_all(
    config: AppConfig,
    analyzers: list,
    stock_data: dict[str, StockData],
) -> list[StockAnalysis]:
    """Run all 12 analyzers on every stock and aggregate scores."""
    from bsa_quicktrade.scoring.ranking import RankingEngine

    ranking = RankingEngine(config)
    analyses: list[StockAnalysis] = []

    console.print("[bold cyan]Phase 2/4: Analyzing stocks …[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("[dim]({task.completed}/{task.total})[/]"),
    ) as progress:
        task = progress.add_task("Analyzing stocks", total=len(stock_data))

        for ticker, data in stock_data.items():
            symbol = ticker.replace(".NS", "")
            progress.update(task, description=f"Analyzing {symbol}")
            results: dict[str, AnalysisResult] = {}

            for analyzer in analyzers:
                try:
                    result = analyzer.analyze(data)
                    results[analyzer.name] = result
                except Exception as exc:
                    logger.debug("Analyzer %s failed on %s: %s", analyzer.name, ticker, exc)

            if results:
                analysis = ranking.aggregate(ticker, data, results)
                analyses.append(analysis)

            progress.advance(task)

    console.print(f"[green]✓ Analyzed {len(analyses)} stocks across {len(analyzers)} modules[/]")
    return analyses


def _rank_and_report(config: AppConfig, analyses: list[StockAnalysis]):
    """Rank stocks and generate report."""
    from bsa_quicktrade.scoring.ranking import RankingEngine
    from bsa_quicktrade.output.report import ReportGenerator

    console.print("[bold cyan]Phase 3/4: Ranking and scoring …[/]")

    ranking = RankingEngine(config)
    top_stocks = ranking.rank(analyses)

    console.print(f"[green]✓ Top {len(top_stocks)} stocks selected[/]")
    console.print()

    console.print("[bold cyan]Phase 4/4: Generating report …[/]")
    report = ReportGenerator(config)
    report.generate(top_stocks)

    return top_stocks


def _generate_charts(config: AppConfig, stock_data: dict[str, StockData], top_stocks: list[StockAnalysis]):
    """Generate charts for top-ranked stocks."""
    from bsa_quicktrade.output.visualization import ChartGenerator

    console.print("[bold cyan]Generating charts …[/]")
    chart_gen = ChartGenerator(config)

    for analysis in top_stocks:
        if analysis.ticker in stock_data:
            path = chart_gen.generate_chart(stock_data[analysis.ticker], analysis)
            if path:
                console.print(f"  [dim]Chart: {path}[/]")

            # OI chart if option data available
            sd = stock_data[analysis.ticker]
            if sd.option_chain:
                chart_gen.generate_oi_chart(analysis.ticker, sd.option_chain)


# ── CLI Commands ────────────────────────────────────────────────────────────

def cmd_scan(args: argparse.Namespace, config: AppConfig) -> None:
    """Full scan — download, analyze, rank, report."""
    from bsa_quicktrade.data.universe import UniverseManager

    cache = DataCache(
        directory=config.cache.directory,
        enabled=config.cache.enabled,
        daily_ttl_hours=config.cache.daily_ttl_hours,
        option_ttl_hours=config.cache.option_ttl_hours,
        delivery_ttl_hours=config.cache.delivery_ttl_hours,
    )

    try:
        # 1. Build universe
        universe = UniverseManager(config)
        tickers = universe.build_universe()

        # 2. Download
        stock_data = _download_all(config, cache, tickers, filter_liquidity=True)

        if not stock_data:
            console.print("[bold red]No stocks passed liquidity filters.[/]")
            return

        # 4. Analyze
        analyzers = _build_analyzers(config)
        analyses = _analyze_all(config, analyzers, stock_data)

        # 5. Rank & Report
        top_stocks = _rank_and_report(config, analyses)

        # 6. Charts
        if getattr(args, "charts", False) and top_stocks:
            _generate_charts(config, stock_data, top_stocks)

    finally:
        cache.close()


def cmd_analyze(args: argparse.Namespace, config: AppConfig) -> None:
    """Deep-dive analysis on a single stock."""
    ticker = args.ticker.upper()
    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"

    cache = DataCache(
        directory=config.cache.directory,
        enabled=config.cache.enabled,
    )

    try:
        stock_data = _download_all(config, cache, [ticker])
        if not stock_data:
            console.print(f"[bold red]Failed to download data for {ticker}[/]")
            return

        analyzers = _build_analyzers(config)
        analyses = _analyze_all(config, analyzers, stock_data)

        if analyses:
            from bsa_quicktrade.output.report import ReportGenerator
            report = ReportGenerator(config)
            ticker_clean = args.ticker.upper().replace(".NS", "")
            prefix = f"{ticker_clean.title()}_"
            report.generate(analyses, filename_prefix=prefix)

            _generate_charts(config, stock_data, analyses)
    finally:
        cache.close()


def cmd_chart(args: argparse.Namespace, config: AppConfig) -> None:
    """Generate chart for a stock."""
    ticker = args.ticker.upper()
    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"

    cache = DataCache(directory=config.cache.directory, enabled=config.cache.enabled)

    try:
        stock_data = _download_all(config, cache, [ticker])
        if ticker in stock_data:
            from bsa_quicktrade.output.visualization import ChartGenerator
            chart_gen = ChartGenerator(config)
            path = chart_gen.generate_chart(stock_data[ticker])
            if path:
                console.print(f"[green]Chart saved: {path}[/]")
    finally:
        cache.close()


def cmd_backtest(args: argparse.Namespace, config: AppConfig) -> None:
    """Run backtesting on a sample stock."""
    from bsa_quicktrade.backtesting.engine import BacktestEngine
    from rich.table import Table

    ticker = getattr(args, "ticker", "RELIANCE")
    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"

    cache = DataCache(directory=config.cache.directory, enabled=config.cache.enabled)

    try:
        stock_data = _download_all(config, cache, [ticker])
        if ticker not in stock_data:
            console.print(f"[red]No data for {ticker}[/]")
            return

        analyzers = _build_analyzers(config)
        bt = BacktestEngine(config)

        table = Table(title=f"Backtest Results — {ticker.replace('.NS', '')}", show_lines=True)
        table.add_column("Module", width=18)
        table.add_column("Trades", justify="center", width=8)
        table.add_column("Win Rate", justify="center", width=10)
        table.add_column("Avg Return", justify="center", width=10)
        table.add_column("Sharpe", justify="center", width=8)
        table.add_column("P/F", justify="center", width=8)
        table.add_column("Max DD", justify="center", width=10)
        table.add_column("Accuracy", justify="center", width=10)

        console.print("[bold cyan]Running backtests …[/]")
        for analyzer in analyzers:
            result = bt.backtest_module(analyzer, stock_data[ticker])
            wr_style = "green" if result.win_rate > 50 else "red"
            table.add_row(
                analyzer.name.replace("_", " ").title(),
                str(result.total_trades),
                f"[{wr_style}]{result.win_rate:.1f}%[/]",
                f"{result.avg_return_pct:+.3f}%",
                f"{result.sharpe_ratio:.2f}",
                f"{result.profit_factor:.2f}",
                f"{result.max_drawdown_pct:.2f}%",
                f"{result.accuracy:.1f}%",
            )

        # System backtest
        sys_result = bt.backtest_system(analyzers, stock_data[ticker])
        table.add_row(
            "[bold]SYSTEM[/]",
            f"[bold]{sys_result.total_trades}[/]",
            f"[bold]{sys_result.win_rate:.1f}%[/]",
            f"[bold]{sys_result.avg_return_pct:+.3f}%[/]",
            f"[bold]{sys_result.sharpe_ratio:.2f}[/]",
            f"[bold]{sys_result.profit_factor:.2f}[/]",
            f"[bold]{sys_result.max_drawdown_pct:.2f}%[/]",
            f"[bold]{sys_result.accuracy:.1f}%[/]",
        )

        console.print(table)

    finally:
        cache.close()


# ── Argument Parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsa-quicktrade",
        description="RSA QuickTrade — Institutional-Quality NSE Stock Screener",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--top", "-n", type=int, default=None,
        help="Number of top stocks to display",
    )
    parser.add_argument(
        "--min-volume", type=int, default=None,
        help="Minimum average daily volume",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # scan
    p_scan = sub.add_parser("scan", help="Full market scan — Top 10 intraday opportunities")
    p_scan.add_argument(
        "--charts", action="store_true", default=False,
        help="Generate visual charts for top ranked stocks",
    )

    # analyze
    p_analyze = sub.add_parser("analyze", help="Deep-dive analysis on a single stock")
    p_analyze.add_argument("ticker", help="NSE stock symbol (e.g. RELIANCE)")

    # chart
    p_chart = sub.add_parser("chart", help="Generate chart for a stock")
    p_chart.add_argument("ticker", help="NSE stock symbol")

    # backtest
    p_bt = sub.add_parser("backtest", help="Backtest scoring modules")
    p_bt.add_argument("ticker", nargs="?", default="RELIANCE", help="Stock to backtest on")

    return parser


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.top:
        config.scoring.top_n = args.top
    if args.min_volume:
        config.universe.min_avg_daily_volume = args.min_volume

    # Setup logging
    setup_logging(
        console_level=config.logging.console_level,
        file_level=config.logging.file_level,
        log_file=config.logging.log_file,
    )

    # Banner
    console.print(Panel(
        "[bold white]RSA QuickTrade[/] — Institutional-Quality NSE Stock Screener\n"
        f"[dim]Command: {args.command} │ "
        f"Modules: 12 │ "
        f"Top N: {config.scoring.top_n}[/]",
        border_style="cyan",
    ))
    console.print()

    start = time.time()

    # Dispatch
    cmd_map = {
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "chart": cmd_chart,
        "backtest": cmd_backtest,
    }
    cmd_map[args.command](args, config)

    elapsed = time.time() - start
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/]")


if __name__ == "__main__":
    main()
