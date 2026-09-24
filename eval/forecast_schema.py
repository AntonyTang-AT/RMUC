"""Validation of the forecast records stored in the prediction ledger."""

from __future__ import annotations

import math
from collections.abc import Mapping


_REQUIRED = (
    "prediction_id", "game_id", "side", "issued_at", "valid_at", "horizon",
    "task", "entity_or_region", "target_schema_version", "input_snapshot_hash",
    "visibility_summary", "model_version", "calibration_version",
)
_TEXT = (
    "prediction_id", "task", "target_schema_version", "input_snapshot_hash",
    "model_version", "calibration_version",
)
_HORIZONS = (10, 30, 60, 120)


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_forecast(row) -> tuple[bool, list[str]]:
    """Return validity and explicit diagnostics without repairing any values."""
    if not isinstance(row, Mapping):
        return False, ["forecast must be a mapping"]

    errors = [f"missing {name}" for name in _REQUIRED if name not in row]
    for name in _TEXT:
        if name in row and (not isinstance(row[name], str) or not row[name].strip()):
            errors.append(f"{name} must be a nonempty string")
    if "side" in row and row["side"] not in ("红", "蓝"):
        errors.append("side must be 红 or 蓝")
    if "entity_or_region" in row and (
        isinstance(row["entity_or_region"], bool)
        or not isinstance(row["entity_or_region"], (str, int))
        or not str(row["entity_or_region"]).strip()
    ):
        errors.append("entity_or_region must be a nonempty string or integer")
    if "game_id" in row and (isinstance(row["game_id"], bool) or not isinstance(row["game_id"], (int, str)) or not str(row["game_id"]).strip()):
        errors.append("game_id must be an integer or nonempty string")
    if "visibility_summary" in row and not isinstance(row["visibility_summary"], Mapping):
        errors.append("visibility_summary must be a mapping")
    for name in ("issued_at", "valid_at", "horizon"):
        if name in row and not _finite_number(row[name]):
            errors.append(f"{name} must be finite")
    if all(_finite_number(row.get(name)) for name in ("issued_at", "valid_at", "horizon")):
        if row["valid_at"] <= row["issued_at"]:
            errors.append("valid_at must be after issued_at")
        if row["horizon"] <= 0 or not math.isclose(
            row["valid_at"] - row["issued_at"], row["horizon"], rel_tol=0, abs_tol=1e-6
        ):
            errors.append("horizon must equal valid_at minus issued_at")
        if row["horizon"] not in _HORIZONS:
            errors.append("horizon must be one of 10, 30, 60, or 120 seconds")

    probabilities = row.get("probabilities", row.get("probability_distribution"))
    quantiles = row.get("quantiles")
    if "probabilities" in row and "probability_distribution" in row:
        errors.append("provide only one categorical probability field")
    if probabilities is None and quantiles is None:
        errors.append("probabilities or quantiles required")
    if probabilities is not None and quantiles is not None:
        errors.append("provide probabilities or quantiles, not both")
    if probabilities is not None:
        labels = row.get("labels")
        labels_valid = (
            isinstance(labels, (list, tuple))
            and bool(labels)
            and all(isinstance(label, str) and bool(label) for label in labels)
        )
        if not labels_valid or len(set(labels)) != len(labels):
            errors.append("labels must be unique nonempty strings")
        if not isinstance(probabilities, Mapping) or not probabilities:
            errors.append("probabilities must be a nonempty mapping")
        else:
            if labels_valid and set(probabilities) != set(labels):
                errors.append("probabilities must include every declared label and no undeclared labels")
            values = list(probabilities.values())
            if any(not _finite_number(value) or not 0 <= value <= 1 for value in values):
                errors.append("probabilities must be finite numbers in [0, 1]")
            elif abs(sum(values) - 1.0) > 1e-6:
                errors.append("probabilities sum must equal 1 within 1e-6")

    if quantiles is not None:
        if not isinstance(quantiles, Mapping) or not quantiles:
            errors.append("quantiles must be a nonempty ordered mapping")
        else:
            levels = []
            values = []
            for level, value in quantiles.items():
                try:
                    number = float(level)
                except (TypeError, ValueError, OverflowError):
                    number = math.nan
                if not math.isfinite(number) or not 0 <= number <= 1:
                    errors.append("quantile levels must be finite and in [0, 1]")
                levels.append(number)
                values.append(value)
            if any(not left < right for left, right in zip(levels, levels[1:])):
                errors.append("quantile levels must be strictly ordered")
            if any(not _finite_number(value) for value in values):
                errors.append("quantile values must be finite")
            elif any(left > right for left, right in zip(values, values[1:])):
                errors.append("quantile values must be ordered")
        bounds = row.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any(not _finite_number(value) for value in bounds):
            errors.append("bounds must contain two finite numbers")
        elif bounds[0] > bounds[1]:
            errors.append("bounds must be ordered")
        elif isinstance(quantiles, Mapping) and all(_finite_number(value) for value in quantiles.values()) and any(
            not bounds[0] <= value <= bounds[1] for value in quantiles.values()
        ):
            errors.append("quantile values must lie within bounds")

    return not errors, errors
