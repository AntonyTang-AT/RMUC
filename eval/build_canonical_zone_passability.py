"""Build a rule-aware adjacency graph from the canonical JSON polygons.

The canonical polygons remain the geometry source of truth. Adjacency is inferred
from polygon proximity, then filtered with the RMUC planning passability policy.
Outputs include a white map and an original-background overlay with arrows.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from shapely.ops import nearest_points

ROOT = Path(r"F:\Desktop\学校\rm考核项目")
REGION_JSON = ROOT / "out_m2" / "official" / "canonical_zone_regions.json"
BACKGROUND = Path(r"F:\documents\Tencent Files\2820916363\nt_qq\nt_data\Pic\2026-09\Ori\50f788026ad4944f212aebc17658b3a7.jpg")
OUT_DIR = ROOT / "out_m2" / "official"
MAP_OUT = OUT_DIR / "canonical_zone_passability_arrows.png"
OVERLAY_OUT = OUT_DIR / "canonical_zone_passability_arrows_overlay.png"
GRAPH_OUT = OUT_DIR / "canonical_zone_adjacency.json"

# Reuse the stable label renderer from the previous deterministic map script.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import render_canonical_zones_from_json as renderer  # noqa: E402

from worldstate.functional_zone import assign_functional_zone  # noqa: E402

# Arrow colors in BGR.
ARROW_BIDIR = (30, 150, 30)       # green: normal two-way ground route
ARROW_CONDITIONAL = (0, 150, 225) # orange: highland / slope conditional
ARROW_FACILITY = (35, 105, 195)   # amber: enter/exit facility for interaction
ARROW_ROLE = (175, 95, 45)        # muted blue: role-specific one-way policy
BASE_MARK = (90, 90, 90)
TOP_MARGIN = 132
BLOCKED = (120, 120, 120)

RESTRICTED_WORDS = ("基地", "前哨", "装配", "补给", "停机坪", "能量机关", "堡垒", "雷达")
HIGH_WORDS = ("高地", "飞坡")
ROAD_WORDS = ("公路", "隧道", "中路")


def read_cv(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_cv(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise IOError(path)
    encoded.tofile(str(path))


def classify_zone(name: str) -> tuple[str, str, str]:
    """Return (planning class, ground access, air access)."""
    if name == "场地未知区":
        return "unknown", "forbidden", "forbidden"
    if "基地" in name:
        return "base", "forbidden", "conditional"
    if any(word in name for word in RESTRICTED_WORDS):
        return "restricted_module", "interaction_only", "conditional"
    if any(word in name for word in HIGH_WORDS):
        return "highland", "conditional", "conditional"
    if any(word in name for word in ROAD_WORDS):
        return "road", "allowed", "allowed"
    if name == "场地开放区":
        return "free_ground", "allowed", "allowed"
    return "free_ground", "allowed", "allowed"


def build_adjacency(regions: list[dict], tol: float = 10.0, min_overlap: float = 700.0) -> list[dict]:
    geoms = {r["zone_id"]: Polygon(r["polygon_px"]) for r in regions}
    pairs: list[dict] = []
    ids = [r["zone_id"] for r in regions]
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            a, b = geoms[a_id], geoms[b_id]
            distance = float(a.boundary.distance(b.boundary))
            if distance > 2 * tol + 1:
                continue
            overlap = float(a.buffer(tol).intersection(b.buffer(tol)).area)
            if overlap < min_overlap:
                continue
            p1, p2 = nearest_points(a.boundary, b.boundary)
            pairs.append({
                "from": a_id,
                "to": b_id,
                "boundary_distance_px": round(distance, 3),
                "buffer_overlap_px2": round(overlap, 1),
                "contact_px": [[round(p1.x, 1), round(p1.y, 1)], [round(p2.x, 1), round(p2.y, 1)]],
            })
    return pairs


def rule_edges(pair: dict, meta: dict[str, dict]) -> dict:
    a, b = meta[pair["from"]], meta[pair["to"]]
    ca, cb = a["planning_class"], b["planning_class"]
    # Generic ground robot graph. Facilities are not ordinary transit cells,
    # but a robot may enter/leave its own facility footprint for deployment,
    # defense or interaction. We represent that as a conditional bidirectional
    # endpoint edge rather than silently dropping the edge.
    allowed_classes = {"free_ground", "road", "highland"}
    if ca in ("unknown", "base") or cb in ("unknown", "base"):
        general = "blocked"
        arrow = "none"
    elif ca in allowed_classes and cb in allowed_classes:
        if ca == "highland" or cb == "highland":
            general = "bidirectional_conditional"
            arrow = "bidirectional"
        else:
            general = "bidirectional"
            arrow = "bidirectional"
    else:
        # A facility endpoint is traversable only when it belongs to the same
        # field side as the adjacent region. Enemy facilities remain tactical
        # targets and are not generic destinations.
        ra, rb = meta[pair["from"]]["semantic_zone"], meta[pair["to"]]["semantic_zone"]
        same_side_a = (ra.startswith("红") and a["side"] == "red") or (ra.startswith("蓝") and a["side"] == "blue")
        same_side_b = (rb.startswith("红") and b["side"] == "red") or (rb.startswith("蓝") and b["side"] == "blue")
        if (ca == "restricted_module" and same_side_a) or (cb == "restricted_module" and same_side_b):
            general = "bidirectional_facility_access"
            arrow = "bidirectional"
        else:
            general = "blocked"
            arrow = "none"

    # Role policy is intentionally separate from physical passability. The
    # project prior keeps engineers/sentries on their own side. A directed
    # return edge is recorded when a robot is already outside its preferred side.
    unit_dirs = {}
    for unit in ("英雄", "步兵3", "步兵4", "工程", "哨兵", "空中"):
        for team, prefix in (("红", "red"), ("蓝", "blue")):
            def legal(src: dict, dst: dict) -> bool:
                if unit == "空中":
                    return dst["planning_class"] != "unknown"
                if dst["planning_class"] == "base":
                    return False
                if dst["planning_class"] == "restricted_module":
                    # Ground robots can enter their own facility footprint for
                    # interaction; enemy facilities are not ordinary targets.
                    if team == "红":
                        if not dst["semantic_zone"].startswith("红"):
                            return False
                    elif team == "蓝":
                        if not dst["semantic_zone"].startswith("蓝"):
                            return False
                elif dst["planning_class"] not in allowed_classes:
                    return False
                if unit in ("工程", "哨兵"):
                    if team == "红" and dst["side"] not in ("red",):
                        return False
                    if team == "蓝" and dst["side"] not in ("blue",):
                        return False
                return True
            fwd = legal(a, b)
            rev = legal(b, a)
            unit_dirs[f"{team}{unit}"] = "bidirectional" if fwd and rev else ("forward" if fwd else ("reverse" if rev else "blocked"))

    return {"general": general, "arrow": arrow, "unit_directions": unit_dirs}


def _segment_from_centroids(pair: dict, metadata: dict[str, dict], scale: float = 1.0):
    """Return a short segment around the edge midpoint, avoiding vertex hubs."""
    a = np.asarray(metadata[pair["from"]]["centroid_px"], dtype=np.float32)
    b = np.asarray(metadata[pair["to"]]["centroid_px"], dtype=np.float32)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length <= 1:
        return None
    unit = vec / length
    half = min(28.0 * scale, max(10.0 * scale, 0.15 * length))
    mid = (a + b) * 0.5
    return mid - unit * half, mid + unit * half, unit


def _draw_edge_arrow(canvas: np.ndarray, start: np.ndarray, end: np.ndarray,
                     color: tuple[int, int, int], thickness: int, both: bool = True) -> None:
    s = tuple(np.round(start).astype(int))
    e = tuple(np.round(end).astype(int))
    cv2.arrowedLine(canvas, s, e, color, thickness, cv2.LINE_AA, tipLength=0.20)
    if both:
        cv2.arrowedLine(canvas, e, s, color, thickness, cv2.LINE_AA, tipLength=0.20)


def _draw_zone_base(canvas: np.ndarray, points: np.ndarray, label_xy: tuple[int, int]) -> None:
    """Visually mark a base footprint as a no-entry structure."""
    renderer.alpha_fill(canvas, points, BASE_MARK, 0.18)
    cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, BASE_MARK, 2, cv2.LINE_AA)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
    hatch = np.zeros_like(canvas)
    cv2.line(hatch, (x, y), (x + w, y + h), BASE_MARK, 2, cv2.LINE_AA)
    cv2.line(hatch, (x + w, y), (x, y + h), BASE_MARK, 2, cv2.LINE_AA)
    inside = mask > 0
    canvas[inside] = np.where(hatch[inside] > 0, hatch[inside], canvas[inside])


def _add_base_text(image: np.ndarray, base_rows: list[tuple[int, int]]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    f = renderer.font(13, bold=True)
    for x, y in base_rows:
        text = "禁入"
        box = draw.textbbox((0, 0), text, font=f)
        draw.text((x - (box[2] - box[0]) // 2, y + 12), text, font=f, fill=(55, 55, 55))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _add_top_margin(image: np.ndarray, overlay: bool = False) -> np.ndarray:
    """Move the field down so the legend never covers the map."""
    h, w = image.shape[:2]
    margin = np.full((h + TOP_MARGIN, w, 3), 255, np.uint8)
    margin[TOP_MARGIN:, :] = image
    return margin


def _alpha_fill_expanded(canvas: np.ndarray, points: np.ndarray,
                         color: tuple[int, int, int], alpha: float,
                         expand_px: int = 2) -> None:
    """Bleed the fill a few pixels so polygon approximation gaps stay narrow."""
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    if expand_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    idx = mask > 0
    color_arr = np.empty_like(canvas)
    color_arr[:] = color
    canvas[idx] = (alpha * color_arr[idx] + (1.0 - alpha) * canvas[idx]).astype(np.uint8)


def make_base(payload: dict, regions: list[dict], metadata: dict[str, dict], pairs: list[dict]) -> np.ndarray:
    w, h = payload["line_map_size_px"]
    canvas = np.full((h, w, 3), 255, np.uint8)
    for r in regions:
        points = np.asarray(r["polygon_px"], dtype=np.int32)
        color = renderer.COLORS_BGR[r["side"]]
        _alpha_fill_expanded(canvas, points, color, 0.055, expand_px=2)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        if metadata[r["zone_id"]]["planning_class"] == "base":
            _draw_zone_base(canvas, points, tuple(map(round, metadata[r["zone_id"]]["centroid_px"])))
    for pair in pairs:
        if pair["rule"]["arrow"] == "none":
            continue
        segment = _segment_from_centroids(pair, metadata)
        if segment is None:
            continue
        start, end, unit = segment
        if pair["rule"]["general"] == "bidirectional_conditional":
            color = ARROW_CONDITIONAL
        elif pair["rule"]["general"] == "bidirectional_facility_access":
            color = ARROW_FACILITY
        else:
            color = ARROW_BIDIR
        _draw_edge_arrow(canvas, start, end, color, 2, both=True)
        # A thin single-headed lane exposes the engineer/sentinel tactical
        # prior when only the return direction is legal. It is deliberately
        # offset from the physical double arrow and is not a physical wall.
        one_way = []
        for key, direction in pair["rule"]["unit_directions"].items():
            if key.endswith("工程") or key.endswith("哨兵"):
                if direction in ("forward", "reverse"):
                    one_way.append((key, direction))
        if one_way:
            normal = np.array([-unit[1], unit[0]], dtype=np.float32) * 8.0
            rs, re = start + normal, end + normal
            if all(direction == "reverse" for _, direction in one_way):
                rs, re = re, rs
            _draw_edge_arrow(canvas, rs, re, ARROW_ROLE, 1, both=False)
    # Draw labels last so arrows never overwrite zone IDs.
    _, label_rows = renderer.render_regions(payload)
    canvas = renderer.add_text_labels(canvas, label_rows, scale=1.0, show_legend=False)
    base_rows = [tuple(map(round, metadata[r["zone_id"]]["centroid_px"])) for r in regions if metadata[r["zone_id"]]["planning_class"] == "base"]
    return _add_base_text(canvas, base_rows)


def make_overlay(background: np.ndarray, payload: dict, regions: list[dict], metadata: dict[str, dict], pairs: list[dict]) -> np.ndarray:
    bh, bw = background.shape[:2]
    lw, lh = payload["line_map_size_px"]
    sx, sy = bw / float(lw), bh / float(lh)
    canvas = background.copy()
    for r in regions:
        points = np.asarray(r["polygon_px"], np.float32)
        scaled = np.round(points * np.array([sx, sy], np.float32)).astype(np.int32)
        color = renderer.COLORS_BGR[r["side"]]
        _alpha_fill_expanded(canvas, scaled, color, 0.075, expand_px=2)
        cv2.polylines(canvas, [scaled.reshape(-1, 1, 2)], True, color, max(2, round(3 * min(sx, sy))), cv2.LINE_AA)
        if metadata[r["zone_id"]]["planning_class"] == "base":
            _draw_zone_base(canvas, scaled, tuple(np.round(np.asarray(metadata[r["zone_id"]]["centroid_px"]) * np.array([sx, sy])).astype(int)))
    for pair in pairs:
        if pair["rule"]["arrow"] == "none":
            continue
        segment = _segment_from_centroids(pair, metadata, scale=min(sx, sy))
        if segment is None:
            continue
        start, end, unit = segment
        start = start * np.array([sx, sy], np.float32)
        end = end * np.array([sx, sy], np.float32)
        unit = unit / np.linalg.norm(unit * np.array([sx, sy], np.float32))
        if pair["rule"]["general"] == "bidirectional_conditional":
            color = ARROW_CONDITIONAL
        elif pair["rule"]["general"] == "bidirectional_facility_access":
            color = ARROW_FACILITY
        else:
            color = ARROW_BIDIR
        _draw_edge_arrow(canvas, start, end, color, max(2, round(2.5 * min(sx, sy))), both=True)
        one_way = []
        for key, direction in pair["rule"]["unit_directions"].items():
            if key.endswith("工程") or key.endswith("哨兵"):
                if direction in ("forward", "reverse"):
                    one_way.append((key, direction))
        if one_way:
            normal = np.array([-unit[1], unit[0]], dtype=np.float32) * (10.0 * min(sx, sy))
            rs, re = start + normal, end + normal
            if all(direction == "reverse" for _, direction in one_way):
                rs, re = re, rs
            _draw_edge_arrow(canvas, rs, re, ARROW_ROLE, max(1, round(2 * min(sx, sy))), both=False)
    _, label_rows = renderer.render_regions(payload)
    for row in label_rows:
        row["label_px"] = [round(row["label_px"][0] * sx), round(row["label_px"][1] * sy)]
    canvas = renderer.add_text_labels(canvas, label_rows, scale=min(sx, sy), show_legend=False)
    base_rows = [tuple(np.round(np.asarray(metadata[r["zone_id"]]["centroid_px"]) * np.array([sx, sy])).astype(int)) for r in regions if metadata[r["zone_id"]]["planning_class"] == "base"]
    return _add_base_text(canvas, base_rows)


def add_legend(image: np.ndarray, overlay: bool = False) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    f = renderer.font(14 if not overlay else 16, bold=False)
    x, y = 18, 16
    rows = [
        (ARROW_BIDIR, "↔  普通地面/公路：双向可通行"),
        (ARROW_CONDITIONAL, "↔  高地/坡：双向但需车型与底盘条件"),
        (ARROW_FACILITY, "↔  设施进出：仅限所属方交互/防守"),
        (ARROW_ROLE, "→  工程/哨兵己方侧策略限制：仅方向提示"),
        ((90, 90, 90), "—  设施/场外/未知：不作为普通穿行区"),
    ]
    width = 360 if not overlay else 420
    height = 24 + 22 * len(rows)
    draw.rectangle((8, 8, width, 8 + height), fill=(255, 255, 255), outline=(120, 120, 120), width=1)
    for bgr, text in rows:
        rgb = (bgr[2], bgr[1], bgr[0])
        draw.line((x, y + 8, x + 38, y + 8), fill=rgb, width=4)
        if text.startswith("↔"):
            draw.polygon([(x+5,y+8),(x+12,y+4),(x+12,y+12)], fill=rgb)
            draw.polygon([(x+33,y+8),(x+26,y+4),(x+26,y+12)], fill=rgb)
        elif text.startswith("→"):
            draw.polygon([(x+33,y+8),(x+25,y+3),(x+25,y+13)], fill=rgb)
        draw.text((x + 48, y), text, font=f, fill=(35, 35, 35))
        y += 22
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    payload = json.loads(REGION_JSON.read_text(encoding="utf-8"))
    regions = payload["regions"]
    metadata: dict[str, dict] = {}
    for r in regions:
        x, y_down = r["centroid_field_m"]
        # functional_zone.py uses y from bottom; canonical JSON uses y from top.
        semantic = assign_functional_zone(float(x), 15.0 - float(y_down))
        planning_class, ground_access, air_access = classify_zone(semantic)
        metadata[r["zone_id"]] = {
            "zone_id": r["zone_id"], "side": r["side"], "centroid_px": r["centroid_px"],
            "centroid_field_m": r["centroid_field_m"], "semantic_zone": semantic,
            "planning_class": planning_class, "ground_access": ground_access, "air_access": air_access,
        }
    raw_pairs = build_adjacency(regions)
    pairs = []
    for pair in raw_pairs:
        pair = dict(pair)
        pair["rule"] = rule_edges(pair, metadata)
        pairs.append(pair)
    base = make_base(payload, regions, metadata, pairs)
    base = add_legend(_add_top_margin(base))
    background = read_cv(BACKGROUND)
    overlay = make_overlay(background, payload, regions, metadata, pairs)
    overlay = add_legend(_add_top_margin(overlay, overlay=True), overlay=True)
    write_cv(MAP_OUT, base)
    write_cv(OVERLAY_OUT, overlay)
    graph = {
        "schema": "rmuc2026.canonical_zone_adjacency.v1",
        "geometry_source": str(REGION_JSON),
        "rules_sources": [str(ROOT / "docs" / "field_map" / "passability_map.json"), str(ROOT / "docs" / "field_map" / "functional_zones.md"), str(ROOT / "worldstate" / "functional_zone.py")],
        "coordinate_convention": payload["coordinate_convention"],
        "adjacency_method": {"boundary_tolerance_px": 10.0, "minimum_buffer_overlap_px2": 700.0, "nested_polygons_excluded": False, "note": "nested facility footprints are retained as interaction/entry neighbors"},
        "interpretation": {
            "bidirectional": "普通地面机器人在两个区域之间可双向通行",
            "bidirectional_conditional": "连接高地/坡，需按机器人底盘、离地间隙和现场安全裕度检查",
            "bidirectional_facility_access": "所属方机器人可进出设施区域进行部署、防守或规则交互；不表示设施可作为普通穿行走廊",
            "blocked": "不作为普通穿行边；设施只保留交互/占用语义，场外和未知默认禁止",
            "role_policy": "工程和哨兵的己方侧限制是战术先验，不是物理单向墙；单位方向写入 unit_directions",
        },
        "region_count": len(regions),
        "regions": list(metadata.values()),
        "physical_adjacency": pairs,
    }
    GRAPH_OUT.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print(json.dumps({"map": str(MAP_OUT), "overlay": str(OVERLAY_OUT), "graph": str(GRAPH_OUT), "regions": len(regions), "edges": len(pairs), "general": Counter(p["rule"]["general"] for p in pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
