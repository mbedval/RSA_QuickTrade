"""Command Line Interface (CLI) entry point for the Backtest Analytics Engine.

Provides commands to parse trade logs, calculate metrics, export reports,
and perform multi-strategy comparisons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bsa_quicktrade.analytics.exporter import export_to_csv, export_to_json
from bsa_quicktrade.analytics.performance import calculate_performance_metrics
from bsa_quicktrade.analytics.risk import calculate_risk_metrics
from bsa_quicktrade.analytics.report_generator import ReportGenerator
from bsa_quicktrade.analytics.trade_log import load_trade_log
from bsa_quicktrade.analytics.health import calculate_strategy_health

console = Console()


def main(argv: List[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        description="Backtest Analytics & Research Engine - Analysis CLI Tool"
    )
    
    parser.add_argument(
        "--log",
        type=str,
        help="Path to the standardized TradeLog file (CSV or JSON)."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="Backtest Strategy",
        help="Name of the strategy for reporting."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Starting capital for performance and drawdown calculations."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated reports and charts."
    )
    parser.add_argument(
        "--compare",
        action="append",
        help="Compare multiple logs in the format 'StrategyName:path_to_log' (e.g., --compare ModelA:logs/modela.csv)."
    )
    parser.add_argument(
        "--json",
        type=str,
        help="File path to save the metrics summary as JSON."
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="File path to save the metrics summary as CSV."
    )

    args = parser.parse_args(argv)

    # Handlers
    # A. Strategy Comparison Mode
    if args.compare:
        console.print(Panel("[bold purple]Backtest Analytics Engine: Strategy Comparison Mode[/]"))
        
        strategies = {}
        for comp_str in args.compare:
            if ":" not in comp_str:
                console.print(f"[bold red]Error: Invalid comparison format '{comp_str}'. Use Name:path.[/]")
                return 1
            name, path_str = comp_str.split(":", 1)
            try:
                records = load_trade_log(path_str)
                strategies[name] = records
                console.print(f"  [green]✓ Loaded {len(records)} trades for '{name}'[/]")
            except Exception as e:
                console.print(f"[bold red]Error loading '{path_str}': {e}[/]")
                return 1

        if len(strategies) < 2:
            console.print("[bold red]Error: At least two strategies are required for comparison.[/]")
            return 1

        try:
            generator = ReportGenerator(output_dir=args.output_dir)
            report_file = generator.generate_comparison_report(
                strategies_dict=strategies, starting_capital=args.capital, file_name="strategy_comparison.html"
            )
            console.print(f"\n[bold green]✓ Comparison Dashboard successfully saved to:[/] {report_file}")
            return 0
        except Exception as e:
            console.print(f"[bold red]Comparison failed: {e}[/]")
            return 1

    # B. Single Trade Log Mode
    if not args.log:
        parser.print_help()
        return 1

    console.print(Panel(f"[bold blue]Backtest Analytics Engine: Evaluating {args.strategy}[/]"))
    
    try:
        # 1. Load log
        records = load_trade_log(args.log)
        console.print(f"[green]✓ Successfully loaded {len(records)} completed trades[/]")
        
        if not records:
            console.print("[bold yellow]Warning: Log contains no trades. Aborting analysis.[/]")
            return 0

        # 2. Run calculations
        perf = calculate_performance_metrics(records, starting_capital=args.capital)
        risk = calculate_risk_metrics(records, starting_capital=args.capital)

        # 3. Print Quick Summary Table
        table = Table(title="Performance Summary KPIs", title_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Trades", str(perf.total_trades))
        table.add_row("Win Rate", f"{perf.win_rate:.2f}%")
        table.add_row("Net Profit", f"₹{perf.net_profit:,.2f}")
        table.add_row("CAGR", f"{perf.cagr:.2f}%")
        table.add_row("Sharpe Ratio", f"{risk.sharpe_ratio:.2f}")
        table.add_row("Max Drawdown %", f"{risk.drawdown.max_drawdown_pct:.2f}%")
        table.add_row("Profit Factor", f"{perf.profit_factor:.2f}")
        table.add_row("Payoff Ratio", f"{perf.payoff_ratio:.2f}")
        table.add_row("Expectancy", f"₹{perf.expectancy:,.2f}")
        
        console.print(table)
        console.print()

        # Compute Strategy Health Score
        health = calculate_strategy_health(records, starting_capital=args.capital)
        
        status_icon = "🔴" if health.status == "REJECT" else ("🟡" if health.status == "WARNING" else "🟢")
        status_style = "bold red" if health.status == "REJECT" else ("bold yellow" if health.status == "WARNING" else "bold green")
        
        health_text = (
            "========================================\n"
            "QuickTrade Strategy Health Score\n\n"
            f"[bold]{health.score} / 100[/]\n\n"
            "Status\n"
            f"{status_icon} [{status_style}]{health.status}[/]\n\n"
            "Reason\n"
            + "\n".join(health.reasons) + "\n\n"
            "Recommendation\n"
            f"{health.recommendation}\n"
            "========================================"
        )
        
        console.print(health_text)
        console.print()

        # 4. Generate HTML Report
        generator = ReportGenerator(output_dir=args.output_dir)
        report_file = generator.generate_html_report(
            records=records, strategy_name=args.strategy, starting_capital=args.capital, file_name="analytics_report.html"
        )
        console.print(f"[bold green]✓ Interactive HTML Dashboard saved to:[/] {report_file}")

        # 5. Export JSON/CSV if requested
        if args.json:
            export_to_json(records, args.json, args.capital)
            console.print(f"[green]✓ Metrics JSON exported to:[/] {args.json}")
        if args.csv:
            export_to_csv(records, args.csv, args.capital)
            console.print(f"[green]✓ Metrics CSV exported to:[/] {args.csv}")

        return 0

    except Exception as e:
        console.print(f"[bold red]Analysis execution failed: {e}[/]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
