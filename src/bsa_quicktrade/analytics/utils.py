"""Utility helpers for the Backtest Analytics & Research Engine.

Includes robust string parsing, date/time normalization, and numerical calculations.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def parse_date(value: Any) -> datetime:
    """Parse a date value from various formats into a datetime object.
    
    Supports:
    - datetime / date objects
    - strings in common formats: 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', 'DD-MM-YYYY', 'DD/MM/YYYY'
    - Unix timestamps
    """
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        return value.to_pydatetime()
    if hasattr(value, "date") and callable(getattr(value, "date")):
        d = value.date()
        return datetime(d.year, d.month, d.day)

    if isinstance(value, (int, float)):
        # Assume timestamp
        try:
            return datetime.fromtimestamp(value)
        except Exception as e:
            logger.warning("Failed to parse timestamp %s: %s", value, e)
            raise ValueError(f"Invalid timestamp: {value}") from e

    if not isinstance(value, str):
        raise ValueError(f"Unsupported date type: {type(value)}")

    val_str = value.strip()
    if not val_str:
        raise ValueError("Empty date string")

    # Try common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val_str, fmt)
        except ValueError:
            continue

    # Try isoformat
    try:
        return datetime.fromisoformat(val_str)
    except ValueError:
        pass

    raise ValueError(f"Could not parse date string: {value}")


def parse_float(value: Any, default: float = 0.0) -> float:
    """Safely parse a numerical float value, handling strings and formatting."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if not cleaned or cleaned.lower() in ("nan", "none", "null", ""):
            return default
        try:
            val = float(cleaned)
            # If original value had a percentage sign, we might want to scale it,
            # but usually it's cleaner to return it raw (e.g. 5.5 for 5.5%).
            # We let the caller decide or keep it raw here.
            return val
        except ValueError:
            logger.warning("Failed to parse float string: %s", value)
            return default

    return default


def parse_dict(value: Any) -> Dict[str, Any]:
    """Parse a dictionary structure from strings (JSON) or pass-through dict."""
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            return {}
        try:
            res = json.loads(val_str)
            if isinstance(res, dict):
                return res
            return {"value": res}
        except json.JSONDecodeError:
            # Maybe it's a key-value list like "a:1, b:2"
            try:
                pairs = [p.split(":") for p in val_str.split(",") if ":" in p]
                return {k.strip(): parse_float(v.strip()) for k, v in pairs}
            except Exception:
                return {"raw_string": val_str}
    return {"raw": str(value)}


def parse_list(value: Any) -> List[str]:
    """Parse a list of strings from JSON, comma-separated string, or pass-through list."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            return []
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                res = json.loads(val_str)
                if isinstance(res, list):
                    return [str(item) for item in res]
            except json.JSONDecodeError:
                pass
        # Fallback to comma separated
        return [item.strip() for item in val_str.split(",") if item.strip()]
    return [str(value)]
