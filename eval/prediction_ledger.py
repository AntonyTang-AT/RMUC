"""Append-only JSONL storage and maturity selection for forecasts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .forecast_schema import validate_forecast


def append_prediction(path, row) -> None:
    """Validate and append exactly one archived forecast record."""
    valid, errors = validate_forecast(row)
    if not valid:
        raise ValueError("invalid prediction: " + "; ".join(errors))
    try:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"prediction is not JSON serializable: {exc}") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def load_predictions(path) -> list[dict]:
    """Read archived rows without filtering malformed forecast records."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                rows.append({"_ledger_error": "blank JSONL line", "_line": line_number})
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_ledger_error": str(exc), "_line": line_number, "_raw": line.rstrip("\n")})
                continue
            rows.append(row if isinstance(row, dict) else {"_ledger_error": "JSONL value is not an object", "_line": line_number, "_raw": row})
    return rows


def mature_predictions(rows, now) -> list[dict]:
    """Return independent copies of forecasts whose validity time has arrived."""
    return [copy.deepcopy(row) for row in rows if isinstance(row, dict) and isinstance(row.get("valid_at"), (int, float)) and not isinstance(row["valid_at"], bool) and row["valid_at"] <= now]
