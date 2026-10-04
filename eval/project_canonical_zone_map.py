"""Project the canonical white line map onto the original background and extract cells."""

from pathlib import Path
import json

import cv2
import numpy as np


LINE_MAP = Path(r"F:\Desktop\download.png")
BACKGROUND = Path(r"F:\documents\Tencent Files\2820916363\nt_qq\nt_data\Pic\2026-09\Ori\50f788026ad4944f212aebc17658b3a7.jpg")
OVERLAY = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\canonical_zone_overlay.png")
REGION_IMAGE = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\canonical_zone_regions.png")
REGION_JSON = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\canonical_zone_regions.json")


def read(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return {
        "red": (((h <= 8) | (h >= 170)) & (s >= 120) & (v >= 110)),
        "blue": ((h >= 95) & (h <= 123) & (s >= 120) & (v >= 100)),
        "purple": ((h >= 125) & (h <= 169) & (s >= 90) & (v >= 100)),
    }


def main() -> None:
    line = read(LINE_MAP, cv2.IMREAD_COLOR)
    background = read(BACKGROUND, cv2.IMREAD_COLOR)
    bh, bw = background.shape[:2]
    lh, lw = line.shape[:2]
    masks = color_masks(line)

    # Overlay the canonical lines on the original image using normalized map
    # coordinates; the original field remains visible below the lines.
    overlay = background.copy()
    colors = {"red": (0, 0, 255), "blue": (255, 70, 0), "purple": (220, 35, 255)}
    for name, mask in masks.items():
        resized = cv2.resize(mask.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
        color_layer = np.zeros_like(background)
        color_layer[:] = colors[name]
        overlay[resized] = cv2.addWeighted(background[resized], 0.22, color_layer[resized], 0.78, 0)
    write(OVERLAY, overlay)

    # Treat all colored boundaries as walls and flood-fill from the outer frame.
    boundary = np.logical_or.reduce(list(masks.values())).astype(np.uint8) * 255
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    boundary = cv2.dilate(boundary, np.ones((2, 2), np.uint8), iterations=1)
    free = (boundary == 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(free, 8)
    border_labels = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))

    region_canvas = np.full((lh, lw, 3), 255, np.uint8)
    regions = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 250 or label in border_labels:
            continue
        cell = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(2.0, 0.006 * perimeter), True)
        points = approx[:, 0, :].astype(int).tolist()
        x, y = centroids[label]
        if x < 0.34 * lw:
            side = "red"
        elif x > 0.66 * lw:
            side = "blue"
        else:
            side = "purple"
        regions.append({
            "side": side,
            "area_px": area,
            "centroid_px": [round(float(x), 2), round(float(y), 2)],
            "bbox_px": [int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP]), int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])],
            "polygon_px": points,
            "centroid_norm": [round(float(x / (lw - 1)), 6), round(float(y / (lh - 1)), 6)],
            "polygon_norm": [[round(px / (lw - 1), 6), round(py / (lh - 1), 6)] for px, py in points],
            "centroid_field_m": [round(float(28.0 * x / (lw - 1)), 4), round(float(15.0 * y / (lh - 1)), 4)],
            "polygon_field_m": [[round(float(28.0 * px / (lw - 1)), 4), round(float(15.0 * py / (lh - 1)), 4)] for px, py in points],
        })
        cv2.drawContours(region_canvas, [approx], -1, colors[side], 5)

    regions.sort(key=lambda r: (r["centroid_px"][1], r["centroid_px"][0]))
    for idx, region in enumerate(regions, start=1):
        region["zone_id"] = f"Z{idx:02d}"
    payload = {
        "source_line_map": str(LINE_MAP),
        "background": str(BACKGROUND),
        "line_map_size_px": [lw, lh],
        "background_size_px": [bw, bh],
        "coordinate_convention": "origin=(0,0) top-left; x right; y down; normalized coordinates divide by (width-1,height-1)",
        "field_scale_m": {"width": 28.0, "height": 15.0},
        "region_count": len(regions),
        "regions": regions,
    }
    REGION_JSON.parent.mkdir(parents=True, exist_ok=True)
    REGION_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write(REGION_IMAGE, region_canvas)
    print(json.dumps({"overlay": str(OVERLAY), "region_image": str(REGION_IMAGE), "region_json": str(REGION_JSON), "region_count": len(regions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
