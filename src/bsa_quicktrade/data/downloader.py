"""Parallel data downloader — yfinance OHLCV, option chains, delivery data.

Downloads daily / weekly / hourly OHLCV for the full universe in batched
chunks using yfinance's built-in threading.  Option chain and delivery
data use separate NSE-specific fetchers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import yfinance as yf
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from bsa_quicktrade.core.cache import DataCache
from bsa_quicktrade.core.config import AppConfig

logger = logging.getLogger(__name__)


class DataDownloader:
    """Orchestrates all data downloads with caching and retry logic."""

    def __init__(self, config: AppConfig, cache: DataCache) -> None:
        self.cfg = config.data
        self.cache = cache

    # ── Public API ──────────────────────────────────────────────────────

    def download_ohlcv(
        self,
        tickers: list[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Download OHLCV for *tickers* in batched chunks.

        Returns a dict mapping each ticker to its DataFrame (or empty
        DataFrame on failure).
        """
        result: dict[str, pd.DataFrame] = {}
        to_download: list[str] = []

        # 1. Check cache
        cache_ns = f"ohlcv_{interval}_{period}"
        for t in tickers:
            cached = self.cache.get(cache_ns, t)
            if cached is not None:
                result[t] = cached
            else:
                to_download.append(t)

        if not to_download:
            logger.info("All %d tickers served from cache (%s)", len(tickers), interval)
            return result

        logger.info(
            "Downloading %d / %d tickers (%s, %s) …",
            len(to_download), len(tickers), interval, period,
        )

        # 2. Chunked download
        chunk_size = self.cfg.download_chunk_size
        chunks = [
            to_download[i: i + chunk_size]
            for i in range(0, len(to_download), chunk_size)
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"Downloading {interval} data", total=len(to_download),
            )
            for chunk_idx, chunk in enumerate(chunks):
                for attempt in range(1, self.cfg.max_retries + 1):
                    try:
                        data = yf.download(
                            tickers=chunk,
                            period=period,
                            interval=interval,
                            group_by="ticker",
                            auto_adjust=True,
                            threads=True,
                            progress=False,
                        )
                        break
                    except Exception as exc:
                        wait = self.cfg.retry_backoff_factor ** attempt
                        logger.debug(
                            "Chunk %d attempt %d failed: %s — retrying in %.1fs",
                            chunk_idx, attempt, exc, wait,
                        )
                        time.sleep(wait)
                else:
                    logger.debug("Chunk %d failed after %d retries", chunk_idx, self.cfg.max_retries)
                    data = pd.DataFrame()

                # Extract per-ticker DataFrames
                for t in chunk:
                    try:
                        if len(chunk) == 1:
                            df = data.copy()
                        else:
                            df = data[t].copy()
                        df = df.dropna(how="all")
                        if not df.empty:
                            result[t] = df
                            self.cache.set(cache_ns, t, df, ttl="daily")
                    except (KeyError, TypeError):
                        logger.debug("No data returned for %s", t)
                    progress.advance(task)

                # Rate limiting between chunks
                if chunk_idx < len(chunks) - 1:
                    time.sleep(self.cfg.download_delay_seconds)

        logger.info(
            "Download complete: %d / %d tickers succeeded (%s)",
            len(result), len(tickers), interval,
        )
        return result

    def download_daily(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        return self.download_ohlcv(tickers, period=self.cfg.daily_period, interval="1d")

    def download_weekly(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        return self.download_ohlcv(tickers, period=self.cfg.weekly_period, interval="1wk")

    def download_hourly(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        return self.download_ohlcv(tickers, period=self.cfg.hourly_period, interval="1h")

    def download_index(self, index_ticker: str = "^NSEI") -> pd.DataFrame:
        """Download benchmark index data."""
        cached = self.cache.get("index", index_ticker)
        if cached is not None:
            return cached
        try:
            df = yf.download(
                index_ticker,
                period=self.cfg.daily_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if not df.empty:
                self.cache.set("index", index_ticker, df, ttl="daily")
            return df
        except Exception as exc:
            logger.debug("Index download failed for %s: %s", index_ticker, exc)
            return pd.DataFrame()

    def download_stock_info(self, ticker: str) -> dict[str, Any]:
        """Get basic stock info (company name, etc.)."""
        cached = self.cache.get("info", ticker)
        if cached is not None:
            return cached
        try:
            info = yf.Ticker(ticker).info
            self.cache.set("info", ticker, info, ttl="daily")
            return info
        except Exception:
            return {}
