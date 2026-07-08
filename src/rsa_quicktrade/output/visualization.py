"""Professional chart visualization using mplfinance.

Generates multi-panel candlestick charts with technical overlays,
volume, RSI, MACD, and pattern annotations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import numpy as np
import pandas as pd

from rsa_quicktrade.core.config import AppConfig
from rsa_quicktrade.core.models import StockAnalysis, StockData

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate professional multi-panel charts for analysed stocks."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.visualization
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_chart(
        self,
        data: StockData,
        analysis: StockAnalysis | None = None,
        last_n_days: int = 120,
    ) -> Path | None:
        """Generate a multi-panel chart for a single stock.

        Returns the path to the saved PNG, or None on error.
        """
        try:
            df_full = self._prepare_df(data.daily)
            if df_full.empty or len(df_full) < 10:
                logger.warning("Not enough data to chart %s", data.ticker)
                return None

            # Calculate indicators on the full dataset to avoid cold start / minimum length issues
            addplots_full = self._build_overlays(df_full, analysis)

            # Slice the main DataFrame to the last N days for plotting
            df = df_full.tail(last_n_days)

            # Slice all addplots to match the sliced main DataFrame
            addplots = []
            for ap in addplots_full:
                ap_data = ap.get("data")
                if isinstance(ap_data, (pd.Series, pd.DataFrame)):
                    ap["data"] = ap_data.loc[df.index]
                addplots.append(ap)

            # Support / Resistance horizontal lines
            hlines = self._build_hlines(analysis, df)

            # Custom style
            mc = mpf.make_marketcolors(
                up="#26a69a", down="#ef5350",
                edge={"up": "#26a69a", "down": "#ef5350"},
                wick={"up": "#26a69a", "down": "#ef5350"},
                volume={"up": "#26a69a80", "down": "#ef535080"},
            )
            style = mpf.make_mpf_style(
                base_mpf_style="nightclouds",
                marketcolors=mc,
                gridstyle="-",
                gridcolor="#2a2a2a",
                facecolor="#1a1a2e",
                edgecolor="#1a1a2e",
                figcolor="#1a1a2e",
                y_on_right=True,
            )

            # Title
            ticker_clean = data.ticker.replace(".NS", "")
            signal_str = ""
            if analysis:
                signal_str = f" — {analysis.signal.label} (Score: {analysis.overall_score:.0f})"
            title = f"{ticker_clean} | {data.company_name}{signal_str}"

            # Plot
            save_path = self.output_dir / f"{ticker_clean}_chart.png"

            kwargs: dict[str, Any] = {
                "type": "candle",
                "style": style,
                "title": title,
                "ylabel": "Price (₹)",
                "volume": True,
                "figsize": (self.cfg.width, self.cfg.height),
                "savefig": dict(fname=str(save_path), dpi=self.cfg.dpi, bbox_inches="tight"),
                "warn_too_much_data": 500,
            }

            if addplots:
                kwargs["addplot"] = addplots

            if hlines:
                kwargs["hlines"] = hlines

            mpf.plot(df, **kwargs)
            plt.close("all")

            logger.info("Chart saved: %s", save_path)
            return save_path

        except Exception as exc:
            logger.error("Chart generation failed for %s: %s", data.ticker, exc)
            return None

    def generate_all(
        self,
        stock_data: dict[str, StockData],
        analyses: list[StockAnalysis],
    ) -> list[Path]:
        """Generate charts for all ranked stocks."""
        analysis_map = {a.ticker: a for a in analyses}
        paths: list[Path] = []

        for ticker, sd in stock_data.items():
            if ticker in analysis_map:
                path = self.generate_chart(sd, analysis_map[ticker])
                if path:
                    paths.append(path)

        return paths

    # ── Internal Helpers ────────────────────────────────────────────────

    def _prepare_df(self, daily: pd.DataFrame, last_n: int | None = None) -> pd.DataFrame:
        """Prepare DataFrame for mplfinance — needs DatetimeIndex + OHLCV columns."""
        df = daily.copy()

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in df.columns]

        # Normalize column names
        rename_map = {}
        for col in df.columns:
            cl = str(col).lower()
            if "open" in cl:
                rename_map[col] = "Open"
            elif "high" in cl:
                rename_map[col] = "High"
            elif "low" in cl:
                rename_map[col] = "Low"
            elif "close" in cl:
                rename_map[col] = "Close"
            elif "volume" in cl:
                rename_map[col] = "Volume"
        df = df.rename(columns=rename_map)

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Take last N days if specified
        if last_n is not None:
            df = df.tail(last_n)
        return df

    def _build_overlays(
        self, df: pd.DataFrame, analysis: StockAnalysis | None,
    ) -> list:
        """Build mplfinance addplot overlays: EMAs, Bollinger Bands, RSI, MACD."""
        import pandas_ta_classic as ta  # type: ignore[import-untyped]

        addplots = []
        close = df["Close"]

        # EMAs
        ema_colors = {20: "#ffeb3b", 50: "#ff9800", 100: "#2196f3", 200: "#e91e63"}
        for period, color in ema_colors.items():
            ema = ta.ema(close, length=period)
            if ema is not None and not ema.isna().all():
                addplots.append(mpf.make_addplot(
                    ema, panel=0, color=color, width=1.0,
                    secondary_y=False,
                ))

        # Bollinger Bands
        bb = ta.bbands(close, length=20, std=2.0)
        if bb is not None:
            for col in bb.columns:
                cl = str(col).lower()
                if "bbu" in cl:
                    addplots.append(mpf.make_addplot(
                        bb[col], panel=0, color="#4fc3f788", width=0.8,
                        linestyle="--", secondary_y=False,
                    ))
                elif "bbl" in cl:
                    addplots.append(mpf.make_addplot(
                        bb[col], panel=0, color="#4fc3f788", width=0.8,
                        linestyle="--", secondary_y=False,
                    ))

        # RSI
        rsi = ta.rsi(close, length=14)
        if rsi is not None and not rsi.isna().all():
            addplots.append(mpf.make_addplot(
                rsi, panel=2, color="#ba68c8", width=1.2,
                ylabel="RSI",
            ))
            # Overbought / Oversold lines
            addplots.append(mpf.make_addplot(
                pd.Series(70, index=df.index), panel=2,
                color="#ef535050", width=0.5, linestyle="--",
            ))
            addplots.append(mpf.make_addplot(
                pd.Series(30, index=df.index), panel=2,
                color="#26a69a50", width=0.5, linestyle="--",
            ))

        # MACD
        macd_data = ta.macd(close)
        if macd_data is not None:
            macd_cols = macd_data.columns.tolist()
            if len(macd_cols) >= 3:
                macd_line = macd_data[macd_cols[0]]
                signal_line = macd_data[macd_cols[1]]
                histogram = macd_data[macd_cols[2]]

                addplots.append(mpf.make_addplot(
                    macd_line, panel=3, color="#2196f3", width=1.0,
                    ylabel="MACD",
                ))
                addplots.append(mpf.make_addplot(
                    signal_line, panel=3, color="#ff9800", width=1.0,
                ))

                # Histogram as bar chart
                hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram.fillna(0)]
                addplots.append(mpf.make_addplot(
                    histogram, panel=3, type="bar",
                    color=hist_colors, width=0.6,
                ))

        return addplots

    def _build_hlines(
        self, analysis: StockAnalysis | None, df: pd.DataFrame,
    ) -> dict | None:
        """Build horizontal lines for support/resistance."""
        if analysis is None:
            return None

        hlines_prices = []
        hlines_colors = []

        for s in analysis.support_levels[:3]:
            hlines_prices.append(s.price)
            hlines_colors.append("#26a69a")

        for r in analysis.resistance_levels[:3]:
            hlines_prices.append(r.price)
            hlines_colors.append("#ef5350")

        if not hlines_prices:
            return None

        return {
            "hlines": hlines_prices,
            "colors": hlines_colors,
            "linestyle": "--",
            "linewidths": 1.0,
        }

    # ── Option OI Chart ─────────────────────────────────────────────────

    def generate_oi_chart(
        self,
        ticker: str,
        option_data: dict[str, Any] | None,
    ) -> Path | None:
        """Generate a separate option OI chart (Call OI vs Put OI by strike)."""
        if not option_data or not option_data.get("records"):
            return None

        try:
            records = option_data["records"]
            strikes = [r["strike"] for r in records if r.get("ce_oi") or r.get("pe_oi")]
            ce_oi = [r.get("ce_oi", 0) for r in records if r.get("ce_oi") or r.get("pe_oi")]
            pe_oi = [r.get("pe_oi", 0) for r in records if r.get("ce_oi") or r.get("pe_oi")]

            if not strikes:
                return None

            # Focus on ATM ± 10 strikes
            underlying = option_data.get("underlying_value", 0)
            if underlying > 0:
                atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying))
                start = max(0, atm_idx - 10)
                end = min(len(strikes), atm_idx + 11)
                strikes = strikes[start:end]
                ce_oi = ce_oi[start:end]
                pe_oi = pe_oi[start:end]

            fig, ax = plt.subplots(figsize=(14, 6), facecolor="#1a1a2e")
            ax.set_facecolor("#1a1a2e")

            x = np.arange(len(strikes))
            width = 0.35

            ax.bar(x - width / 2, ce_oi, width, label="Call OI", color="#ef5350", alpha=0.8)
            ax.bar(x + width / 2, pe_oi, width, label="Put OI", color="#26a69a", alpha=0.8)

            ax.set_xlabel("Strike Price", color="white")
            ax.set_ylabel("Open Interest", color="white")
            ax.set_title(f"{ticker.replace('.NS', '')} — Option Chain OI", color="white", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(s)) for s in strikes], rotation=45, color="white", fontsize=8)
            ax.tick_params(colors="white")
            ax.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")
            ax.grid(axis="y", alpha=0.2)

            # Mark underlying price
            if underlying > 0 and min(strikes) <= underlying <= max(strikes):
                ax.axvline(
                    x=np.interp(underlying, strikes, x),
                    color="#ffeb3b", linestyle="--", linewidth=1.5, label="Spot",
                )

            save_path = self.output_dir / f"{ticker.replace('.NS', '')}_oi_chart.png"
            fig.savefig(save_path, dpi=self.cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            logger.info("OI chart saved: %s", save_path)
            return save_path

        except Exception as exc:
            logger.error("OI chart generation failed for %s: %s", ticker, exc)
            return None
