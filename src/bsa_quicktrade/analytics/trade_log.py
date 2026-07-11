"""TradeLog loading, parsing, standardization, and exporting.

Standardizes raw trade files into type-safe TradeRecord structures.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from datetime import datetime
import pandas as pd

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.utils import parse_date, parse_dict, parse_float, parse_list

logger = logging.getLogger(__name__)

# Column name variants mapping to standard TradeRecord field names
COLUMN_MAPPINGS = {
    "trade_id": ["trade id", "trade_id", "id", "tradeid", "ticket"],
    "strategy_name": ["strategy name", "strategy_name", "strategy", "strat"],
    "module_name": ["module name", "module_name", "module"],
    "signal_name": ["signal name", "signal_name", "signal", "signal_type"],
    "entry_date": ["entry date", "entry_date", "entry date/time", "entry_time", "date", "entry"],
    "exit_date": ["exit date", "exit_date", "exit date/time", "exit_time", "exit"],
    "entry_price": ["entry price", "entry_price", "buy price", "buy_price", "entryprice"],
    "exit_price": ["exit price", "exit_price", "sell price", "sell_price", "exitprice"],
    "quantity": ["quantity", "quantity", "qty", "size", "units"],
    "capital_used": ["capital used", "capital_used", "capital", "cost"],
    "profit": ["profit", "profit", "pnl", "realized pnl", "realized_pnl", "gain"],
    "profit_pct": ["profit %", "profit_pct", "profit_percent", "pnl %", "pnl_pct", "gain %"],
    "stop_loss": ["stop loss", "stop_loss", "sl", "stop", "stop price"],
    "target_price": ["target price", "target_price", "target", "tp", "target price"],
    "risk": ["risk", "risk", "initial risk", "r"],
    "reward": ["reward", "reward", "potential reward"],
    "risk_reward_ratio": ["risk reward ratio", "risk_reward_ratio", "rr", "rrr", "risk/reward"],
    "r_multiple": ["r-multiple", "r_multiple", "r multiple", "r_mult", "rmultiple"],
    "holding_period": ["holding period", "holding_period", "duration", "days"],
    "atr": ["atr", "atr_value"],
    "adx": ["adx", "adx_value"],
    "rsi": ["rsi", "rsi_value"],
    "vwap": ["vwap", "vwap_value"],
    "volume": ["volume", "vol"],
    "market_breadth": ["market breadth", "market_breadth", "breadth"],
    "market_regime": ["market regime", "market_regime", "regime"],
    "exit_reason": ["exit reason", "exit_reason", "reason"],
    "entry_indicator_snapshot": ["entry indicator snapshot", "entry_indicator_snapshot", "entry_indicators"],
    "exit_indicator_snapshot": ["exit indicator snapshot", "exit_indicator_snapshot", "exit_indicators"],
    "triggered_rules": ["triggered rules", "triggered_rules", "rules"],
    "confidence_score": ["confidence score", "confidence_score", "confidence"],
    "brokerage": ["brokerage", "brokerage", "commission"],
    "taxes": ["taxes", "taxes", "tax"],
    "slippage": ["slippage"],
    "remarks": ["remarks", "note", "notes", "comment"],
    "direction": ["direction", "type", "position_type", "side"]
}


def _find_column(df_columns: List[str], target: str) -> str | None:
    """Find the column in df_columns that matches target or its variations."""
    target_lower = target.lower()
    df_cols_lower = [str(c).lower() for c in df_columns]
    
    # 1. Direct match or case-insensitive direct match
    if target_lower in df_cols_lower:
        idx = df_cols_lower.index(target_lower)
        return df_columns[idx]
        
    # 2. Try mappings
    mappings = COLUMN_MAPPINGS.get(target, [])
    for map_val in mappings:
        if map_val.lower() in df_cols_lower:
            idx = df_cols_lower.index(map_val.lower())
            return df_columns[idx]
            
    return None


def load_trade_log(file_path: Union[str, Path]) -> List[TradeRecord]:
    """Load a trade log file (CSV or JSON) and parse it into a list of TradeRecords.
    
    Performs data standardization, column mapping, and field validation.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Trade log file not found: {path}")

    # Standardize path suffix
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    elif suffix in (".csv", ".txt"):
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Must be CSV or JSON.")


def _load_json(path: Path) -> List[TradeRecord]:
    """Parse JSON trade log."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Standardize format: should be list of dicts
    records: List[Dict[str, Any]] = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Check if it has a key containing the trades list
        for key in ["trades", "records", "data"]:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        if not records:
            # Maybe the dict contains trades indexed by ID
            records = list(data.values())
    else:
        raise ValueError("Invalid JSON trade log format. Root must be a list or dictionary.")

    return _parse_raw_records(records)


def _load_csv(path: Path) -> List[TradeRecord]:
    """Parse CSV trade log using Pandas to handle raw structures."""
    try:
        # read_csv naturally handles delimiters and headers
        df = pd.read_csv(path)
    except Exception as e:
        # Fallback to python csv reader if pandas fails
        logger.warning("Pandas failed to read CSV, attempting standard library parser: %s", e)
        df = _load_csv_fallback(path)

    # Convert dataframe to records list, resolving column mappings
    records: List[Dict[str, Any]] = []
    df_cols = list(df.columns)
    
    # Generate field mapping dictionary
    field_to_col: Dict[str, str] = {}
    for standard_field in COLUMN_MAPPINGS.keys():
        col_name = _find_column(df_cols, standard_field)
        if col_name:
            field_to_col[standard_field] = col_name

    for _, row in df.iterrows():
        raw_rec: Dict[str, Any] = {}
        for std_field, mapped_col in field_to_col.items():
            val = row[mapped_col]
            # Replace pd.NA/NaN with None
            if pd.isna(val):
                raw_rec[std_field] = None
            else:
                raw_rec[std_field] = val
        records.append(raw_rec)

    return _parse_raw_records(records)


def _load_csv_fallback(path: Path) -> pd.DataFrame:
    """Helper to load CSV via standard library when Pandas has issues."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return pd.DataFrame(rows)


def _parse_raw_records(raw_records: List[Dict[str, Any]]) -> List[TradeRecord]:
    """Safely parses dictionaries of raw values into a validated list of TradeRecord objects."""
    trade_records: List[TradeRecord] = []
    
    for idx, rec in enumerate(raw_records):
        try:
            # Required fields validation
            # Entry Date
            entry_date_raw = rec.get("entry_date")
            if not entry_date_raw:
                logger.warning("Skipping record %d: Missing entry date", idx)
                continue
            entry_date = parse_date(entry_date_raw)

            # Exit Date
            exit_date_raw = rec.get("exit_date")
            if not exit_date_raw:
                logger.warning("Skipping record %d: Missing exit date", idx)
                continue
            exit_date = parse_date(exit_date_raw)

            # Entry / Exit Prices
            entry_price = parse_float(rec.get("entry_price"))
            exit_price = parse_float(rec.get("exit_price"))
            quantity = parse_float(rec.get("quantity", 1.0), default=1.0)
            
            # Auto-calculate capital used if missing
            capital_used = parse_float(rec.get("capital_used"))
            if not capital_used:
                capital_used = entry_price * quantity

            # Calculate direction
            direction = str(rec.get("direction", "long")).strip().lower()
            if direction not in ("long", "short"):
                # If direction is not specified, let's infer from signal name or profit
                sig_name = str(rec.get("signal_name", "")).lower()
                if "short" in sig_name or "bear" in sig_name:
                    direction = "short"
                else:
                    direction = "long"

            # Auto-calculate profit if missing
            profit = parse_float(rec.get("profit"), default=None)
            if profit is None:
                if direction == "long":
                    profit = (exit_price - entry_price) * quantity
                else:
                    profit = (entry_price - exit_price) * quantity

            # Auto-calculate profit pct if missing
            profit_pct = parse_float(rec.get("profit_pct"), default=None)
            if profit_pct is None:
                if capital_used > 0:
                    profit_pct = (profit / capital_used) * 100
                else:
                    profit_pct = 0.0

            # Auto-calculate risk/reward if missing
            stop_loss = parse_float(rec.get("stop_loss"), default=None)
            target_price = parse_float(rec.get("target_price"), default=None)
            risk = parse_float(rec.get("risk"), default=None)
            reward = parse_float(rec.get("reward"), default=None)
            
            if risk is None and stop_loss is not None:
                risk = abs(entry_price - stop_loss) * quantity
            if reward is None and target_price is not None:
                reward = abs(target_price - entry_price) * quantity

            risk_reward_ratio = parse_float(rec.get("risk_reward_ratio"), default=None)
            if risk_reward_ratio is None and risk and reward:
                risk_reward_ratio = reward / risk if risk > 0 else 0.0

            r_multiple = parse_float(rec.get("r_multiple"), default=None)
            if r_multiple is None and profit is not None and risk is not None and risk > 0:
                # If short, win is when exit_price < entry_price
                # R multiple is Profit / Risk_Amount
                r_multiple = profit / risk

            # Parse Indicator Snaps and rules
            entry_indicator_snapshot = parse_dict(rec.get("entry_indicator_snapshot"))
            exit_indicator_snapshot = parse_dict(rec.get("exit_indicator_snapshot"))
            triggered_rules = parse_list(rec.get("triggered_rules"))

            # Construct validated record
            record = TradeRecord(
                trade_id=str(rec.get("trade_id", f"T_{idx+1}")),
                strategy_name=str(rec.get("strategy_name", "DefaultStrategy")),
                module_name=str(rec.get("module_name", "DefaultModule")),
                signal_name=str(rec.get("signal_name", "DefaultSignal")),
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                capital_used=capital_used,
                profit=profit,
                profit_pct=profit_pct,
                stop_loss=stop_loss,
                target_price=target_price,
                risk=risk,
                reward=reward,
                risk_reward_ratio=risk_reward_ratio,
                r_multiple=r_multiple,
                holding_period=parse_float(rec.get("holding_period", 0.0)),
                atr=parse_float(rec.get("atr"), default=None),
                adx=parse_float(rec.get("adx"), default=None),
                rsi=parse_float(rec.get("rsi"), default=None),
                ema_values=parse_dict(rec.get("ema_values")),
                vwap=parse_float(rec.get("vwap"), default=None),
                volume=parse_float(rec.get("volume"), default=None),
                market_breadth=parse_float(rec.get("market_breadth"), default=None),
                market_regime=str(rec.get("market_regime", "Unknown")),
                exit_reason=str(rec.get("exit_reason", "Unknown")),
                entry_indicator_snapshot=entry_indicator_snapshot,
                exit_indicator_snapshot=exit_indicator_snapshot,
                triggered_rules=triggered_rules,
                confidence_score=parse_float(rec.get("confidence_score", 0.0)),
                brokerage=parse_float(rec.get("brokerage", 0.0)),
                taxes=parse_float(rec.get("taxes", 0.0)),
                slippage=parse_float(rec.get("slippage", 0.0)),
                remarks=str(rec.get("remarks", "")),
                direction=direction
            )
            
            trade_records.append(record)
            
        except Exception as e:
            logger.error("Error parsing row %d: %s", idx, e, exc_info=True)
            continue
            
    return trade_records


def save_trade_log(records: List[TradeRecord], file_path: Union[str, Path]) -> None:
    """Save a list of TradeRecords back into a CSV or JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Helper to serialize record into dictionary
    def rec_to_dict(r: TradeRecord) -> Dict[str, Any]:
        d = {}
        for field_name in r.__dataclass_fields__.keys():
            val = getattr(r, field_name)
            if isinstance(val, datetime):
                d[field_name] = val.isoformat()
            else:
                d[field_name] = val
        return d

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump([rec_to_dict(r) for r in records], f, indent=4)
            
    elif suffix in (".csv", ".txt"):
        if not records:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return
            
        fields = list(records[0].__dataclass_fields__.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in records:
                writer.writerow(rec_to_dict(r))
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Must be CSV or JSON.")
