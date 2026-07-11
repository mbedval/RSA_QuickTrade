"""Sector Filter — Stage 2 of swing trading pipeline.

Ranks NSE sectors by momentum and returns the strongest performing sectors.
Only stocks from top-ranked sectors are eligible for swing trade entries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)

# NSE sector indices available via yfinance
SECTOR_INDICES: Dict[str, str] = {
    "Banking":       "^CNXBANK",
    "IT":            "^CNXIT",
    "FMCG":          "^CNXFMCG",
    "Auto":          "^CNXAUTO",
    "Pharma":        "^CNXPHARMA",
    "Energy":        "^CNXENERGY",
    "Metals":        "^CNXMETAL",
    "Realty":        "^CNXREALTY",
    "Infra":         "^CNXINFRA",
    "Media":         "^CNXMEDIA",
}

STARS = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"]


@dataclass
class SectorRanking:
    """Sector strength ranking output."""
    rankings: Dict[str, float]      # sector name → momentum score (%)
    top_sectors: List[str]          # names of top-K sectors
    ratings: Dict[str, str]         # sector name → star rating string
    allow_sectors: set[str]

    def is_allowed(self, sector: str) -> bool:
        """Check if a sector is in the approved list."""
        if not self.allow_sectors:
            return True  # if no data, allow all
        for allowed in self.allow_sectors:
            if allowed.lower() in sector.lower() or sector.lower() in allowed.lower():
                return True
        return False


class SectorFilter:
    """Rank NSE sector indices by 20-day momentum.

    Uses pre-downloaded sector data passed in as a dict of DataFrames.
    If no sector data is provided, returns a permissive SectorRanking.
    """

    def __init__(self, momentum_window: int = 20, top_k: int = 5) -> None:
        self.momentum_window = momentum_window
        self.top_k = top_k

    def evaluate(
        self,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> SectorRanking:
        """Rank sectors by momentum.

        Parameters
        ----------
        sector_data:
            Dict mapping sector name → OHLCV DataFrame.
            If None or empty, returns permissive ranking (all sectors allowed).
        """
        if not sector_data:
            return SectorRanking(
                rankings={},
                top_sectors=list(SECTOR_INDICES.keys()),
                ratings={s: STARS[2] for s in SECTOR_INDICES},
                allow_sectors=set(),  # empty means allow all
            )

        scores: Dict[str, float] = {}
        for sector_name, df in sector_data.items():
            close = self._get_close(df)
            if close is None or len(close) < self.momentum_window + 5:
                continue
            # Momentum = (current close / close N days ago - 1) × 100
            momentum = (float(close.iloc[-1]) / float(close.iloc[-self.momentum_window]) - 1.0) * 100.0
            scores[sector_name] = round(momentum, 2)

        if not scores:
            return SectorRanking(
                rankings={}, top_sectors=[],
                ratings={}, allow_sectors=set(),
            )

        # Rank and assign star ratings
        sorted_sectors = sorted(scores, key=lambda s: scores[s], reverse=True)
        n = len(sorted_sectors)
        ratings: Dict[str, str] = {}
        for rank_i, sector in enumerate(sorted_sectors):
            star_idx = min(4, int((n - rank_i - 1) / max(1, n - 1) * 4 + 0.5))
            ratings[sector] = STARS[star_idx]

        top_sectors = sorted_sectors[: self.top_k]
        allow_sectors = set(top_sectors)

        return SectorRanking(
            rankings=scores,
            top_sectors=top_sectors,
            ratings=ratings,
            allow_sectors=allow_sectors,
        )

    def _get_close(self, df: pd.DataFrame) -> Optional[pd.Series]:
        for col in df.columns:
            if str(col).lower() in {"close", "adj close"}:
                return df[col].dropna()
        return None
