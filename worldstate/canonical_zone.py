"""Canonical JSON zone geometry used by training and deployment.

The user-approved canonical map is stored in image coordinates (x right, y
 down). Replay/world-state positions use the 28m x 15m field frame with y up,
so queries flip y before testing polygons. The module deliberately keeps the
functional-zone graph and the canonical geometry as separate contracts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "out_m2" / "official" / "canonical_zone_adjacency.json"
REGION_PATH = ROOT / "out_m2" / "official" / "canonical_zone_regions.json"
FIELD_WIDTH_M = 28.0
FIELD_HEIGHT_M = 15.0


def _load() -> tuple[dict, dict]:
    geometry = json.loads(REGION_PATH.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8")) if GRAPH_PATH.exists() else {}
    return geometry, graph


_GEOMETRY, _PAYLOAD = _load()
_REGIONS = sorted(_GEOMETRY.get("regions", []), key=lambda row: row["zone_id"])
CANONICAL_ZONE_LABELS = tuple(row["zone_id"] for row in _REGIONS)
CANONICAL_ZONE_DIM = len(CANONICAL_ZONE_LABELS)
_INDEX = {label: i for i, label in enumerate(CANONICAL_ZONE_LABELS)}

# The adjacency JSON has richer rule metadata; the fallback is useful when a
# clean checkout only has canonical_zone_regions.json.
_META = {row["zone_id"]: dict(row) for row in _REGIONS}
for row in _PAYLOAD.get("regions", []):
    _META.setdefault(row["zone_id"], {}).update(row)
_EDGES = _PAYLOAD.get("physical_adjacency", [])
_NEIGHBORS: dict[str, set[str]] = {label: set() for label in CANONICAL_ZONE_LABELS}
for edge in _EDGES:
    if edge.get("rule", {}).get("general") == "blocked":
        continue
    a, b = edge.get("from"), edge.get("to")
    if a in _NEIGHBORS and b in _NEIGHBORS:
        _NEIGHBORS[a].add(b)
        _NEIGHBORS[b].add(a)

# Load polygon coordinates in field metres, converting the canonical image y
# (down) into the world-state y (up) at query time.
_POLYGONS: dict[str, np.ndarray] = {}
for row in _REGIONS:
    points = np.asarray(row.get("polygon_field_m", []), dtype=np.float32)
    if len(points) >= 3:
        points = points.copy()
        points[:, 1] = FIELD_HEIGHT_M - points[:, 1]
        _POLYGONS[row["zone_id"]] = points


def zone_index(zone_id: str | None) -> int:
    return _INDEX.get(str(zone_id), -1)


def zone_metadata(zone_id: str | None) -> dict | None:
    return _META.get(str(zone_id))


def neighbors(zone_id: str | None) -> tuple[str, ...]:
    return tuple(sorted(_NEIGHBORS.get(str(zone_id), ())))


def _point_in_polygon(x: float, y: float, points: np.ndarray) -> bool:
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def locate_zone(x_m: float, y_m: float) -> str | None:
    """Return the smallest canonical polygon containing a world-state point."""
    x, y = float(x_m), float(y_m)
    candidates = []
    nearest = []
    for zone_id, polygon in _POLYGONS.items():
        contour = polygon.astype(np.float32).reshape(-1, 1, 2)
        signed_distance = float(cv2.pointPolygonTest(contour, (x, y), True))
        nearest.append((signed_distance, zone_id))
        if signed_distance >= 0.0:
            area = float((_META[zone_id] or {}).get("area_px", 1e12))
            candidates.append((area, zone_id))
    if not candidates:
        # The canonical polygons retain narrow construction gaps between
        # neighboring cells. Snap only a small gap to the nearest polygon;
        # points clearly outside the field still return None.
        signed_distance, zone_id = max(nearest, default=(-1e9, None))
        return zone_id if signed_distance >= -0.25 else None
    return min(candidates)[1]


def access_code(zone_id: str | None) -> float:
    """Numeric planning prior for a model feature: allowed=1, conditional=.5, forbidden=0."""
    row = zone_metadata(zone_id) or {}
    access = row.get("ground_access")
    return {"allowed": 1.0, "conditional": 0.5, "conditional_interaction": 0.5,
            "interaction_only": 0.5, "forbidden": 0.0}.get(access, 0.0)


def zone_features(position: Iterable[float] | None) -> tuple[float, float, float, float]:
    """Return compact geometry features for one observed entity."""
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return 0.0, 0.0, 0.0, 0.0
    zone_id = locate_zone(float(position[0]), float(position[1]))
    idx = zone_index(zone_id)
    if idx < 0:
        return 0.0, 0.0, 0.0, 0.0
    degree = len(neighbors(zone_id))
    is_base = float((zone_metadata(zone_id) or {}).get("planning_class") == "base")
    return (
        float(idx / max(1, CANONICAL_ZONE_DIM - 1)),
        float(min(1.0, degree / 8.0)),
        float(access_code(zone_id)),
        is_base,
    )


def is_legal_transition(from_zone: str | None, to_zone: str | None) -> bool:
    if not from_zone or not to_zone:
        return False
    return str(to_zone) == str(from_zone) or str(to_zone) in _NEIGHBORS.get(str(from_zone), set())


__all__ = [
    "CANONICAL_ZONE_LABELS", "CANONICAL_ZONE_DIM", "FIELD_WIDTH_M", "FIELD_HEIGHT_M",
    "zone_index", "zone_metadata", "neighbors", "locate_zone", "zone_features",
    "is_legal_transition",
]
