"""Render the canonical field zones directly from canonical_zone_regions.json.

The JSON polygons are the source of truth. No image segmentation or generative
model is used here; the output is deterministic and suitable for later geometry
lookups in the replay/forecast pipeline.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"F:\Desktop\学校\rm考核项目")
REGION_JSON = ROOT / "out_m2" / "official" / "canonical_zone_regions.json"
BACKGROUND = Path(r"F:\documents\Tencent Files\2820916363\nt_qq\nt_data\Pic\2026-09\Ori\50f788026ad4944f212aebc17658b3a7.jpg")
OUT_DIR = ROOT / "out_m2" / "official"
MATH_OUT = OUT_DIR / "canonical_zone_math_labeled.png"
OVERLAY_OUT = OUT_DIR / "canonical_zone_math_labeled_overlay.png"
LABEL_JSON = OUT_DIR / "canonical_zone_labels.json"
LABEL_CSV = OUT_DIR / "canonical_zone_labels.csv"

FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

# BGR for OpenCV drawing and RGB for PIL text.
COLORS_BGR = {
    "red": (30, 55, 238),
    "purple": (196, 40, 232),
    "blue": (238, 86, 28),
}
COLORS_RGB = {k: (v[2], v[1], v[0]) for k, v in COLORS_BGR.items()}
SIDE_NAME = {"red": "红侧", "purple": "中央", "blue": "蓝侧"}
SIDE_SHORT = {"red": "R", "purple": "C", "blue": "B"}


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


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    chosen = FONT_BOLD if bold and FONT_BOLD.exists() else FONT
    return ImageFont.truetype(str(chosen), size=size)


def label_anchor(points: np.ndarray, fallback: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    """Choose a point safely inside a polygon using a distance transform."""
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0.0:
        return fallback
    yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(xx), int(yy)


def alpha_fill(canvas: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    layer = np.zeros_like(canvas)
    cv2.fillPoly(layer, [polygon.reshape(-1, 1, 2)], color)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 255)
    idx = mask > 0
    canvas[idx] = (alpha * layer[idx] + (1.0 - alpha) * canvas[idx]).astype(np.uint8)


def add_text_labels(image: np.ndarray, label_rows: list[dict], scale: float = 1.0, show_legend: bool = True) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font = font(max(14, round(17 * scale)), bold=True)
    side_font = font(max(12, round(13 * scale)), bold=False)
    tiny_font = font(max(11, round(11 * scale)), bold=False)
    # Keep the map labels stable and short: Zxx plus side. Exact coordinates live in JSON/CSV.
    for row in label_rows:
        x, y = row["label_px"]
        zid = row["zone_id"]
        side = row["side_name"]
        text = f"{zid}"
        # Tiny/nested cells get an ID-only label to avoid text collisions.
        tiny = row["area_px"] < 9000 or row["zone_id"] in {"Z09", "Z10", "Z50", "Z51"}
        local_font = font(max(11, round(13 * scale)) if tiny else max(14, round(17 * scale)), bold=True)
        bbox = draw.textbbox((0, 0), text, font=local_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # A compact opaque backing makes labels readable over either white or textured map.
        pad_x = max(3, round(3 * scale))
        pad_y = max(2, round(2 * scale))
        left, top = x - tw // 2 - pad_x, y - th - round(8 * scale) - pad_y
        right, bottom = x + tw // 2 + pad_x, y - round(8 * scale) + pad_y
        fill = (255, 255, 255, 228) if image.shape[2] == 3 else (255, 255, 255, 228)
        draw.rounded_rectangle((left, top, right, bottom), radius=max(2, round(2 * scale)), fill=fill, outline=(35, 35, 35), width=max(1, round(scale)))
        draw.text((x - tw // 2, y - th - round(8 * scale)), text, font=local_font, fill=COLORS_RGB[row["side"]])
        if not tiny:
            sub = f"{SIDE_SHORT[row['side']]} · {side}"
            sb = draw.textbbox((0, 0), sub, font=side_font)
            sw = sb[2] - sb[0]
            draw.text((x - sw // 2, y + round(1 * scale)), sub, font=side_font, fill=COLORS_RGB[row["side"]])

    if show_legend:
        # A small fixed legend explains the IDs without hiding the map geometry.
        legend_x, legend_y = round(18 * scale), round(16 * scale)
        legend = "区域编号来自 canonical_zone_regions.json"
        draw.text((legend_x, legend_y), legend, font=tiny_font, fill=(35, 35, 35))
        legend2 = "R=红侧   C=中央   B=蓝侧"
        draw.text((legend_x, legend_y + round(17 * scale)), legend2, font=tiny_font, fill=(70, 70, 70))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_regions(payload: dict) -> tuple[np.ndarray, list[dict]]:
    w, h = payload["line_map_size_px"]
    canvas = np.full((h, w, 3), 255, np.uint8)
    label_rows: list[dict] = []
    # Stable per-side indices make a later human review easier than relying only on global order.
    side_order = {k: 0 for k in SIDE_NAME}
    ordered = sorted(payload["regions"], key=lambda r: int(r["zone_id"][1:]))
    for region in ordered:
        side = region["side"]
        side_order[side] += 1
        points = np.asarray(region["polygon_px"], dtype=np.int32)
        color = COLORS_BGR[side]
        alpha_fill(canvas, points, color, 0.075)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, color, 4, cv2.LINE_AA)
        cx, cy = region["centroid_px"]
        anchor = label_anchor(points, (round(cx), round(cy)), w, h)
        # Two facility footprints contain an inner polygon. Move the outer-cell
        # labels into a free side pocket so the IDs remain readable.
        label_overrides = {
            "Z09": (156, 105),
            "Z51": (1670, 803),
        }
        if region["zone_id"] in label_overrides:
            anchor = label_overrides[region["zone_id"]]
        label_rows.append({
            "zone_id": region["zone_id"],
            "side": side,
            "side_name": SIDE_NAME[side],
            "side_index": side_order[side],
            "label_px": list(anchor),
            "centroid_px": region["centroid_px"],
            "centroid_field_m": region["centroid_field_m"],
            "area_px": region["area_px"],
            "bbox_px": region["bbox_px"],
            "polygon_px": region["polygon_px"],
            "polygon_field_m": region["polygon_field_m"],
        })
    canvas = add_text_labels(canvas, label_rows, scale=1.0)
    return canvas, label_rows


def make_overlay(background: np.ndarray, payload: dict, label_rows: list[dict]) -> np.ndarray:
    bh, bw = background.shape[:2]
    lw, lh = payload["line_map_size_px"]
    overlay = background.copy()
    # Scale the same mathematical polygons into the source-image coordinate system.
    sx, sy = bw / float(lw), bh / float(lh)
    for region in payload["regions"]:
        points = np.asarray(region["polygon_px"], np.float32)
        scaled = np.round(points * np.array([sx, sy], np.float32)).astype(np.int32)
        color = COLORS_BGR[region["side"]]
        alpha_fill(overlay, scaled, color, 0.13)
        cv2.polylines(overlay, [scaled.reshape(-1, 1, 2)], True, color, max(4, round(5 * min(sx, sy))), cv2.LINE_AA)
    scaled_rows = []
    for row in label_rows:
        r = dict(row)
        r["label_px"] = [round(row["label_px"][0] * sx), round(row["label_px"][1] * sy)]
        scaled_rows.append(r)
    return add_text_labels(overlay, scaled_rows, scale=min(sx, sy))


def main() -> None:
    payload = json.loads(REGION_JSON.read_text(encoding="utf-8"))
    math_map, rows = render_regions(payload)
    background = read_cv(BACKGROUND)
    overlay = make_overlay(background, payload, rows)
    write_cv(MATH_OUT, math_map)
    write_cv(OVERLAY_OUT, overlay)
    LABEL_JSON.write_text(json.dumps({
        "source_region_json": str(REGION_JSON),
        "coordinate_convention": payload["coordinate_convention"],
        "field_scale_m": payload["field_scale_m"],
        "region_count": len(rows),
        "regions": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with LABEL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["zone_id", "side", "side_name", "side_index", "centroid_x_m", "centroid_y_m", "area_px", "label_x_px", "label_y_px"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "zone_id": row["zone_id"],
                "side": row["side"],
                "side_name": row["side_name"],
                "side_index": row["side_index"],
                "centroid_x_m": row["centroid_field_m"][0],
                "centroid_y_m": row["centroid_field_m"][1],
                "area_px": row["area_px"],
                "label_x_px": row["label_px"][0],
                "label_y_px": row["label_px"][1],
            })
    print(json.dumps({"math": str(MATH_OUT), "overlay": str(OVERLAY_OUT), "labels": str(LABEL_JSON), "csv": str(LABEL_CSV), "region_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
