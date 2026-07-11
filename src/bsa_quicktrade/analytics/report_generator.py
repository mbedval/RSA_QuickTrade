"""Professional HTML Report Generator for Backtest Analytics.

Compiles performance, risk, distribution, and attribution metrics,
along with Plotly interactive charts, into a premium glassmorphism dashboard.
Also supports cross-strategy comparison dashboards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.performance import calculate_performance_metrics
from bsa_quicktrade.analytics.distribution import calculate_distribution_metrics
from bsa_quicktrade.analytics.risk import calculate_risk_metrics
from bsa_quicktrade.analytics.regime import calculate_regime_analytics, calculate_volatility_analytics
from bsa_quicktrade.analytics.attribution import calculate_signal_attribution
from bsa_quicktrade.analytics.health import calculate_strategy_health
from bsa_quicktrade.analytics.visualization import HAS_PLOTLY, VisualizationEngine

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Premium glassmorphic styling system
HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-surface: rgba(17, 24, 39, 0.7);
            --bg-card: rgba(30, 41, 59, 0.45);
            --border-glass: rgba(255, 255, 255, 0.08);
            
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            
            --accent-success: #10b981;
            --accent-danger: #ef4444;
            --accent-warning: #f59e0b;
            --accent-primary: #6366f1;
            --accent-purple: #8b5cf6;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(139, 92, 246, 0.1) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border-glass);
            margin-bottom: 2rem;
        }

        .logo-title h1 {
            font-size: 2.2rem;
            background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-title p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .meta-badge {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        /* Metrics grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            backdrop-filter: blur(12px);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.3);
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }

        .metric-val.profit-positive { color: var(--accent-success); }
        .metric-val.profit-negative { color: var(--accent-danger); }

        /* Main layout */
        .dashboard-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            backdrop-filter: blur(16px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
        }

        .card h2 {
            font-size: 1.4rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--accent-primary);
            padding-left: 0.75rem;
            color: var(--text-primary);
        }

        /* Tables styling */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }

        th, td {
            padding: 0.8rem 1rem;
            border-bottom: 1px solid var(--border-glass);
        }

        th {
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge-success { background: rgba(16, 185, 129, 0.15); color: var(--accent-success); }
        .badge-danger { background: rgba(239, 68, 68, 0.15); color: var(--accent-danger); }
        .badge-primary { background: rgba(99, 102, 241, 0.15); color: #818cf8; }

        /* Chart container */
        .chart-box {
            background: rgba(15, 23, 42, 0.3);
            border-radius: 12px;
            padding: 0.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.03);
            overflow: hidden;
        }
    </style>
</head>
<body>
"""

HTML_FOOT = """
</body>
</html>
"""


class ReportGenerator:
    """Compiles trading strategy analytics into rich HTML reports."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis = VisualizationEngine(output_dir=self.output_dir / "charts")

    def generate_html_report(
        self, records: List[TradeRecord], strategy_name: str, starting_capital: float = 100000.0, file_name: str = "analytics_report.html"
    ) -> Path:
        """Calculate all metrics, generate charts, and render a single-page HTML dashboard."""
        if not records:
            raise ValueError("No trade records found to generate a report.")

        # 1. Calculate metrics
        perf = calculate_performance_metrics(records, starting_capital)
        risk = calculate_risk_metrics(records, starting_capital)
        dist = calculate_distribution_metrics(records)
        regimes = calculate_regime_analytics(records, starting_capital=starting_capital)
        vols = calculate_volatility_analytics(records, starting_capital=starting_capital)
        attr = calculate_signal_attribution(records)
        health = calculate_strategy_health(records, starting_capital)

        # 2. Generate charts (returns HTML strings for Plotly)
        eq_chart = self.vis.generate_equity_curve(records, starting_capital)
        dd_chart = self.vis.generate_drawdown_curve(records, starting_capital)
        heatmap_chart = self.vis.generate_monthly_heatmap(records)
        dist_chart = self.vis.generate_profit_distribution(records)
        regime_chart = self.vis.generate_regime_chart(regimes)

        # 3. Build HTML body
        report_title = f"{strategy_name} - Backtest Performance Report"
        
        # Determine Profit formatting
        profit_class = "profit-positive" if perf.net_profit >= 0 else "profit-negative"
        profit_sign = "+" if perf.net_profit >= 0 else ""

        html = []
        html.append(HTML_HEAD.replace("{title}", report_title))
        
        # Header
        html.append(f"""
        <header>
            <div class="logo-title">
                <h1>{strategy_name}</h1>
                <p>Institutional Grade Backtesting Analytics & Strategy Research</p>
            </div>
            <div class="meta-badge">
                Generated: {records[0].exit_date.strftime('%Y-%m-%d')} to {records[-1].exit_date.strftime('%Y-%m-%d')}
            </div>
        </header>
        """)

        # Determine status styling
        status_color = "var(--accent-success)"
        status_bg = "rgba(16, 185, 129, 0.15)"
        bullet = "✓"
        if health.status == "WARNING":
            status_color = "var(--accent-warning)"
            status_bg = "rgba(245, 158, 11, 0.15)"
            bullet = "⚠"
        elif health.status == "REJECT":
            status_color = "var(--accent-danger)"
            status_bg = "rgba(239, 68, 68, 0.15)"
            bullet = "✘"

        reasons_html = ""
        for r in health.reasons:
            reasons_html += f'<li style="margin-bottom: 0.35rem;"><span style="color: {status_color}; margin-right: 0.5rem;">{bullet}</span> {r}</li>'

        # Strategy Health Score block
        html.append(f"""
        <div class="card" style="margin-bottom: 2rem; border-left: 6px solid {status_color}; background: rgba(30, 41, 59, 0.45); padding: 1.5rem;">
            <h2 style="border-left: none; padding-left: 0; margin-bottom: 1rem; color: #a78bfa; font-family: 'Space Grotesk', sans-serif;">QuickTrade Strategy Health Score</h2>
            <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 2rem;">
                <div style="display: flex; align-items: center; gap: 2rem;">
                    <div style="font-size: 3.5rem; font-weight: 800; font-family: 'Space Grotesk', sans-serif; background: linear-gradient(135deg, #fff 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;">
                        {health.score} <span style="font-size: 1.5rem; color: var(--text-secondary);">/ 100</span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.25rem;">
                        <span style="background: {status_bg}; color: {status_color}; font-size: 1.1rem; padding: 0.4rem 1.2rem; border-radius: 8px; width: fit-content; text-align: center; font-weight: 700; letter-spacing: 0.05em;">
                            {health.status}
                        </span>
                        <span style="font-size: 0.95rem; color: var(--text-primary); font-weight: 600; margin-top: 0.25rem;">
                            Recommendation: {health.recommendation}
                        </span>
                    </div>
                </div>
                <div style="flex-grow: 1; max-width: 500px; background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                    <h4 style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Space Grotesk', sans-serif;">Evaluation Details</h4>
                    <ul style="list-style: none; padding-left: 0; margin: 0; font-size: 0.85rem; color: var(--text-primary);">
                        {reasons_html}
                    </ul>
                </div>
            </div>
        </div>
        """)

        # KPI Metrics Grid
        html.append(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-val {profit_class}">{profit_sign}₹{perf.net_profit:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAGR</div>
                <div class="metric-val">{perf.cagr:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-val" style="color: var(--accent-warning);">{perf.win_rate:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-val" style="color: var(--accent-primary);">{risk.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-val" style="color: var(--accent-danger);">{risk.drawdown.max_drawdown_pct:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-val">{perf.total_trades}</div>
            </div>
        </div>
        """)

        # Dashboard body columns
        html.append('<div class="dashboard-container">')

        # COLUMN 1: Detailed Tables
        html.append('<div class="left-col">')

        # 1. Performance Overview Table
        html.append(f"""
        <div class="card">
            <h2>Performance Metrics Overview</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Winning Trades</td><td><span class="badge badge-success">{perf.winning_trades}</span></td></tr>
                <tr><td>Losing Trades</td><td><span class="badge badge-danger">{perf.losing_trades}</span></td></tr>
                <tr><td>Gross Profit</td><td style="color: var(--accent-success);">₹{perf.gross_profit:,.2f}</td></tr>
                <tr><td>Gross Loss</td><td style="color: var(--accent-danger);">₹{perf.gross_loss:,.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{perf.profit_factor:.2f}</td></tr>
                <tr><td>Payoff Ratio</td><td>{perf.payoff_ratio:.2f}</td></tr>
                <tr><td>Expectancy</td><td>₹{perf.expectancy:,.2f}</td></tr>
                <tr><td>Average Win</td><td style="color: var(--accent-success);">₹{perf.average_win:,.2f}</td></tr>
                <tr><td>Average Loss</td><td style="color: var(--accent-danger);">₹{perf.average_loss:,.2f}</td></tr>
                <tr><td>Largest Win</td><td style="color: var(--accent-success);">₹{perf.largest_win:,.2f}</td></tr>
                <tr><td>Largest Loss</td><td style="color: var(--accent-danger);">₹{perf.largest_loss:,.2f}</td></tr>
            </table>
        </div>
        """)

        # 2. Risk Ratios Table
        html.append(f"""
        <div class="card">
            <h2>Risk & Drawdown Analytics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Sortino Ratio</td><td>{risk.sortino_ratio:.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{risk.calmar_ratio:.2f}</td></tr>
                <tr><td>MAR Ratio</td><td>{risk.mar_ratio:.2f}</td></tr>
                <tr><td>Recovery Factor</td><td>{risk.recovery_factor:.2f}</td></tr>
                <tr><td>Ulcer Index</td><td>{risk.ulcer_index:.2f}</td></tr>
                <tr><td>Max Drawdown (Absolute)</td><td style="color: var(--accent-danger);">₹{risk.drawdown.max_drawdown:,.2f}</td></tr>
                <tr><td>Longest Drawdown Duration</td><td>{risk.drawdown.longest_drawdown_days:.1f} days</td></tr>
                <tr><td>Average Time to Recovery</td><td>{risk.drawdown.time_to_recovery_days:.1f} days</td></tr>
            </table>
        </div>
        """)

        # 3. Streaks & Time-based Distribution
        html.append(f"""
        <div class="card">
            <h2>Trade Distribution & Streaks</h2>
            <table>
                <tr><th>Category</th><th>Value</th></tr>
                <tr><td>Consecutive Wins</td><td><span class="badge badge-success">{dist.consecutive_wins}</span></td></tr>
                <tr><td>Consecutive Losses</td><td><span class="badge badge-danger">{dist.consecutive_losses}</span></td></tr>
                <tr><td>Win Streak Avg Profit</td><td style="color: var(--accent-success);">₹{dist.win_streak_avg_profit:,.2f}</td></tr>
                <tr><td>Loss Streak Avg Loss</td><td style="color: var(--accent-danger);">₹{dist.loss_streak_avg_loss:,.2f}</td></tr>
                <tr><td>Average Holding Period</td><td>{dist.avg_holding_period:.2f} days</td></tr>
            </table>
        </div>
        """)

        # 4. Market Regimes
        html.append('<div class="card"><h2>Market Regime Performance</h2><table>')
        html.append("<tr><th>Regime</th><th>Trades</th><th>Win Rate</th><th>Net Profit</th><th>Drawdown</th></tr>")
        for reg_name, r_metrics in regimes.items():
            html.append(f"""
            <tr>
                <td><b>{reg_name}</b></td>
                <td>{r_metrics.total_trades}</td>
                <td>{r_metrics.win_rate:.1f}%</td>
                <td style="color: {'var(--accent-success)' if r_metrics.net_profit >=0 else 'var(--accent-danger)'};">₹{r_metrics.net_profit:,.2f}</td>
                <td style="color: var(--accent-danger);">{r_metrics.max_drawdown_pct:.1f}%</td>
            </tr>
            """)
        html.append("</table></div>")

        # 5. Signal Attributions
        html.append('<div class="card"><h2>Signal Attribution Analysis</h2><table>')
        html.append("<tr><th>Indicator/Rule</th><th>Triggers</th><th>Win Rate</th><th>Net Profit</th><th>False Sig Rate</th></tr>")
        for ind_name, ind_metrics in list(attr.indicators.items())[:8]:  # Top 8 indicators
            html.append(f"""
            <tr>
                <td><b>{ind_name}</b></td>
                <td>{ind_metrics.total_signals}</td>
                <td>{ind_metrics.win_rate:.1f}%</td>
                <td style="color: {'var(--accent-success)' if ind_metrics.net_profit >=0 else 'var(--accent-danger)'};">₹{ind_metrics.net_profit:,.2f}</td>
                <td style="color: var(--accent-danger);">{ind_metrics.false_signal_rate:.1f}%</td>
            </tr>
            """)
        html.append("</table>")
        html.append(f"""
        <div style="margin-top: 1.2rem; font-size: 0.85rem; color: var(--text-secondary);">
            <p><b>Best Rule Combination:</b> <span class="badge badge-success">{attr.best_combination}</span> ({attr.best_combination_win_rate:.1f}% Win Rate)</p>
            <p style="margin-top: 0.5rem;"><b>Worst Rule Combination:</b> <span class="badge badge-danger">{attr.worst_combination}</span> ({attr.worst_combination_win_rate:.1f}% Win Rate)</p>
        </div>
        """)
        html.append("</div>")

        html.append("</div>")  # Close left column

        # COLUMN 2: Charts
        html.append('<div class="right-col">')

        if eq_chart:
            html.append(f'<div class="card"><h2>Equity Performance Chart</h2><div class="chart-box">{eq_chart}</div></div>')
        if dd_chart:
            html.append(f'<div class="card"><h2>Drawdown Chart</h2><div class="chart-box">{dd_chart}</div></div>')
        if regime_chart:
            html.append(f'<div class="card"><h2>Regime Allocation Chart</h2><div class="chart-box">{regime_chart}</div></div>')
        if heatmap_chart:
            html.append(f'<div class="card"><h2>Monthly Return Heatmap Matrix</h2><div class="chart-box">{heatmap_chart}</div></div>')
        if dist_chart:
            html.append(f'<div class="card"><h2>Trade Profit/Loss Distribution</h2><div class="chart-box">{dist_chart}</div></div>')

        html.append("</div>")  # Close right column

        html.append("</div>")  # Close container
        html.append(HTML_FOOT)

        # Output file
        output_file = self.output_dir / file_name
        output_file.write_text("\n".join(html), encoding="utf-8")
        
        logger.info("Report successfully generated: %s", output_file)
        return output_file

    def generate_comparison_report(
        self, strategies_dict: Dict[str, List[TradeRecord]], starting_capital: float = 100000.0, file_name: str = "strategy_comparison.html"
    ) -> Path:
        """Generate a comparison report comparing multiple strategies side-by-side."""
        if not strategies_dict:
            raise ValueError("No strategy logs passed for comparison.")

        # Compute metrics for all strategies
        strategy_summaries = {}
        for name, records in strategies_dict.items():
            perf = calculate_performance_metrics(records, starting_capital)
            risk = calculate_risk_metrics(records, starting_capital)
            strategy_summaries[name] = {"perf": perf, "risk": risk}

        # Render comparison table
        report_title = "Trading Strategy Comparison Dashboard"
        
        html = []
        html.append(HTML_HEAD.replace("{title}", report_title))
        
        # Header
        html.append(f"""
        <header>
            <div class="logo-title">
                <h1>Strategy Comparison Dashboard</h1>
                <p>Side-by-side quantitative evaluation across {len(strategies_dict)} backtested models</p>
            </div>
            <div class="meta-badge">
                Strategies compared: {", ".join(strategies_dict.keys())}
            </div>
        </header>
        """)

        # Main Comparison Card
        html.append("""
        <div class="card">
            <h2>Comparative Metric Overview</h2>
            <table>
                <tr>
                    <th>Strategy Name</th>
                    <th>Total Trades</th>
                    <th>Win Rate</th>
                    <th>Net Profit</th>
                    <th>CAGR</th>
                    <th>Sharpe</th>
                    <th>Sortino</th>
                    <th>Max DD %</th>
                    <th>Profit Factor</th>
                    <th>Expectancy</th>
                </tr>
        """)

        for name, data in strategy_summaries.items():
            perf = data["perf"]
            risk = data["risk"]
            profit_style = "color: var(--accent-success);" if perf.net_profit >= 0 else "color: var(--accent-danger);"
            html.append(f"""
            <tr>
                <td><b>{name}</b></td>
                <td>{perf.total_trades}</td>
                <td>{perf.win_rate:.2f}%</td>
                <td style="{profit_style}">₹{perf.net_profit:,.2f}</td>
                <td>{perf.cagr:.2f}%</td>
                <td style="font-weight: 600; color: var(--accent-primary);">{risk.sharpe_ratio:.2f}</td>
                <td>{risk.sortino_ratio:.2f}</td>
                <td style="color: var(--accent-danger);">{risk.drawdown.max_drawdown_pct:.2f}%</td>
                <td>{perf.profit_factor:.2f}</td>
                <td>₹{perf.expectancy:,.2f}</td>
            </tr>
            """)

        html.append("</table></div>")

        # Visualizations (Comparative Equity Curves can go here)
        # For each strategy, plot its equity curve on the same Plotly chart
        if HAS_PLOTLY:
            fig = go.Figure()
            for name, records in strategies_dict.items():
                sorted_recs = sorted(records, key=lambda r: r.exit_date)
                eq = [starting_capital]
                current_eq = starting_capital
                for r in sorted_recs:
                    current_eq += r.net_profit
                    eq.append(current_eq)
                
                dates = [sorted_recs[0].entry_date] + [r.exit_date for r in sorted_recs]
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=eq,
                    mode="lines",
                    name=name
                ))
            fig.update_layout(
                title=dict(text="Comparative Equity Growth Chart", font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title="Equity (₹)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            
            html.append(f"""
            <div class="card">
                <h2>Equity Growth Comparison Chart</h2>
                <div class="chart-box">{chart_html}</div>
            </div>
            """)

        html.append(HTML_FOOT)

        # Output
        output_file = self.output_dir / file_name
        output_file.write_text("\n".join(html), encoding="utf-8")
        
        return output_file
