"""Visualization Engine for Backtest Analytics.

Generates professional Plotly interactive charts for equity curves, drawdowns,
distributions, attribution, and regime analysis, with Matplotlib static fallbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Try importing Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Try importing Matplotlib
try:
    import matplotlib
    # Use non-interactive backend for headless environments
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from bsa_quicktrade.analytics.models import TradeRecord

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Generates charts from completed trade logs using Plotly or Matplotlib."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Equity Curve ───────────────────────────────────────────────────────

    def generate_equity_curve(
        self, records: List[TradeRecord], starting_capital: float = 100000.0, use_plotly: bool = True
    ) -> Union[str, Path, None]:
        """Generate cumulative equity curve chart.
        
        Returns HTML string for Plotly, or Path to saved image file for Matplotlib.
        """
        if not records:
            return None

        # Prepare data
        sorted_recs = sorted(records, key=lambda r: r.exit_date)
        df = pd.DataFrame([{
            "Date": r.exit_date,
            "Profit": r.net_profit,
            "TradeID": r.trade_id
        } for r in sorted_recs])
        df["Cumulative Equity"] = starting_capital + df["Profit"].cumsum()
        
        # Insert initial starting capital point
        start_row = pd.DataFrame([{
            "Date": sorted_recs[0].entry_date - timedelta_safe(sorted_recs[0]),
            "Profit": 0.0,
            "TradeID": "Start",
            "Cumulative Equity": starting_capital
        }])
        df = pd.concat([start_row, df], ignore_index=True)

        if use_plotly and HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=df["Cumulative Equity"],
                mode="lines+markers",
                name="Equity Curve",
                line=dict(color="#10b981", width=2.5),
                marker=dict(size=6, color="#059669"),
                hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> ₹%{y:,.2f}<extra></extra>"
            ))
            fig.update_layout(
                title=dict(text="Equity Growth Curve", font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title="Equity (₹)",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            
        elif HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 5))
            plt.style.use("dark_background")
            plt.plot(df["Date"], df["Cumulative Equity"], color="#10b981", marker="o", linewidth=2)
            plt.title("Equity Growth Curve")
            plt.xlabel("Date")
            plt.ylabel("Equity (₹)")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            file_path = self.output_dir / "equity_curve.png"
            plt.savefig(file_path, transparent=True)
            plt.close()
            return file_path

        return None

    # ── 2. Drawdown Curve ─────────────────────────────────────────────────────

    def generate_drawdown_curve(self, records: List[TradeRecord], starting_capital: float = 100000.0, use_plotly: bool = True) -> Union[str, Path, None]:
        """Generate drawdown curve chart over time."""
        if not records:
            return None

        sorted_recs = sorted(records, key=lambda r: r.exit_date)
        equity = [starting_capital]
        current_eq = starting_capital
        for r in sorted_recs:
            current_eq += r.net_profit
            equity.append(current_eq)
            
        dates = [sorted_recs[0].entry_date] + [r.exit_date for r in sorted_recs]
        
        peaks = np.maximum.accumulate(equity)
        drawdowns_pct = [(peaks[i] - equity[i]) / peaks[i] * 100.0 for i in range(len(equity))]

        df = pd.DataFrame({
            "Date": dates,
            "Drawdown": drawdowns_pct
        })

        if use_plotly and HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=-df["Drawdown"],  # display drawdowns as negative
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.2)",
                name="Drawdown %",
                line=dict(color="#ef4444", width=2),
                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> -%{y:.2f}%<extra></extra>"
            ))
            fig.update_layout(
                title=dict(text="Historical Drawdown Curve", font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        elif HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 5))
            plt.style.use("dark_background")
            plt.fill_between(df["Date"], -df["Drawdown"], color="#ef4444", alpha=0.3)
            plt.plot(df["Date"], -df["Drawdown"], color="#ef4444", linewidth=1.5)
            plt.title("Historical Drawdown Curve")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            file_path = self.output_dir / "drawdown_curve.png"
            plt.savefig(file_path, transparent=True)
            plt.close()
            return file_path

        return None

    # ── 3. Monthly Return Heatmap ─────────────────────────────────────────────

    def generate_monthly_heatmap(self, records: List[TradeRecord], use_plotly: bool = True) -> Union[str, Path, None]:
        """Generate monthly returns heatmap matrix (Years vs Months)."""
        if not records:
            return None

        # Build monthly grid
        data = []
        for r in records:
            if r.exit_date:
                data.append({
                    "Year": r.exit_date.year,
                    "Month": r.exit_date.strftime("%b"),
                    "MonthNum": r.exit_date.month,
                    "Profit": r.net_profit
                })
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        # Group by Year/Month
        pivot_df = df.groupby(["Year", "Month", "MonthNum"])["Profit"].sum().reset_index()
        pivot_df = pivot_df.pivot(index="Year", columns="Month", values="Profit").fillna(0.0)
        
        # Sort months chronologically
        months_ordered = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        existing_months = [m for m in months_ordered if m in pivot_df.columns]
        pivot_df = pivot_df[existing_months]

        if use_plotly and HAS_PLOTLY:
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Month", y="Year", color="Profit (₹)"),
                x=pivot_df.columns,
                y=pivot_df.index.map(str),
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0.0
            )
            fig.update_layout(
                title=dict(text="Monthly Net Profit Heatmap (₹)", font=dict(size=16)),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        elif HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 4))
            plt.style.use("dark_background")
            plt.imshow(pivot_df, cmap="RdYlGn", aspect="auto")
            plt.colorbar(label="Profit (₹)")
            plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns)
            plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)
            plt.title("Monthly Net Profit Heatmap (₹)")
            plt.tight_layout()
            file_path = self.output_dir / "monthly_heatmap.png"
            plt.savefig(file_path, transparent=True)
            plt.close()
            return file_path

        return None

    # ── 4. Profit Distribution Histogram ──────────────────────────────────────

    def generate_profit_distribution(self, records: List[TradeRecord], use_plotly: bool = True) -> Union[str, Path, None]:
        """Generate profit/loss histogram distribution."""
        if not records:
            return None

        profits = [r.net_profit for r in records]

        if use_plotly and HAS_PLOTLY:
            fig = px.histogram(
                x=profits,
                nbins=30,
                color_discrete_sequence=["#3b82f6"],
                labels={"x": "Trade Net Profit (₹)", "y": "Frequency"}
            )
            fig.update_layout(
                title=dict(text="Trade Profit/Loss Distribution", font=dict(size=16)),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        elif HAS_MATPLOTLIB:
            plt.figure(figsize=(8, 4))
            plt.style.use("dark_background")
            plt.hist(profits, bins=25, color="#3b82f6", edgecolor="black", alpha=0.7)
            plt.title("Trade Profit/Loss Distribution")
            plt.xlabel("Trade Net Profit (₹)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            file_path = self.output_dir / "profit_distribution.png"
            plt.savefig(file_path, transparent=True)
            plt.close()
            return file_path

        return None

    # ── 5. Regime Performance Bar Chart ───────────────────────────────────────

    def generate_regime_chart(self, regime_data: Dict[str, any], use_plotly: bool = True) -> Union[str, Path, None]:
        """Generate performance bar chart per market regime.
        
        regime_data maps regime names -> RegimeMetrics objects.
        """
        if not regime_data:
            return None

        regimes = list(regime_data.keys())
        profits = [regime_data[r].net_profit for r in regimes]
        win_rates = [regime_data[r].win_rate for r in regimes]

        if use_plotly and HAS_PLOTLY:
            fig = go.Figure()
            # Net profit bars
            fig.add_trace(go.Bar(
                x=regimes,
                y=profits,
                name="Net Profit (₹)",
                marker_color="#8b5cf6",
                yaxis="y"
            ))
            # Win rate line
            fig.add_trace(go.Scatter(
                x=regimes,
                y=win_rates,
                name="Win Rate (%)",
                mode="lines+markers",
                marker=dict(size=8, color="#f59e0b"),
                line=dict(color="#f59e0b", width=2.5),
                yaxis="y2"
            ))
            fig.update_layout(
                title=dict(text="Performance by Market Regime", font=dict(size=16)),
                yaxis=dict(title="Net Profit (₹)"),
                yaxis2=dict(title="Win Rate (%)", overlaying="y", side="right", range=[0, 100]),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        elif HAS_MATPLOTLIB:
            fig, ax1 = plt.subplots(figsize=(8, 4))
            plt.style.use("dark_background")
            
            # Bars
            ax1.bar(regimes, profits, color="#8b5cf6", alpha=0.7, label="Net Profit (₹)")
            ax1.set_ylabel("Net Profit (₹)", color="#8b5cf6")
            ax1.tick_params(axis="y", labelcolor="#8b5cf6")
            
            # Line on dual axis
            ax2 = ax1.twinx()
            ax2.plot(regimes, win_rates, color="#f59e0b", marker="o", linewidth=2, label="Win Rate (%)")
            ax2.set_ylabel("Win Rate (%)", color="#f59e0b")
            ax2.tick_params(axis="y", labelcolor="#f59e0b")
            ax2.set_ylim(0, 100)
            
            plt.title("Performance by Market Regime")
            plt.tight_layout()
            file_path = self.output_dir / "regime_performance.png"
            plt.savefig(file_path, transparent=True)
            plt.close()
            return file_path

        return None


def timedelta_safe(r: TradeRecord) -> pd.Timedelta:
    """Safely calculate time delta or default to 1 day."""
    if r.exit_date and r.entry_date:
        return r.exit_date - r.entry_date
    return pd.Timedelta(days=1)
