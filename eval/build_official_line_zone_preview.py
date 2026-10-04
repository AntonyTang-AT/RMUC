"""Overlay the functional-zone proposal on an official-derived white line map."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from worldstate.functional_zone import ZONE_POLYGONS, zone_centroid  # noqa: E402


BASE = REPO / "out_m2" / "official" / "official_white_line_draft.png"
OUT = REPO / "out_m2" / "official" / "official_white_line_zones_v1.png"
FIELD_BOX = (30, 38, 1900, 1045)
WIDTH_M, HEIGHT_M = 28.0, 15.0


def to_px(point, size):
    x, y = float(point[0]), float(point[1])
    left, top, right, bottom = FIELD_BOX
    return (round(left + x / WIDTH_M * (right - left)),
            round(bottom - y / HEIGHT_M * (bottom - top)))


def main():
    base = Image.open(BASE).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    font = ImageFont.truetype(font_path, 17)
    title = ImageFont.truetype(font_path, 24)

    central = {"中央北隧道", "中央南隧道", "中央高地"}
    roads = {"红上公路", "红下公路", "红中路", "蓝上公路", "蓝下公路", "蓝中路"}
    colors = {
        "red": (220, 38, 38, 220), "blue": (25, 105, 205, 220),
        "central": (113, 67, 189, 230), "road": (189, 116, 15, 220),
        "module": (15, 115, 95, 210),
    }

    def colour(zone):
        if zone in central:
            return colors["central"]
        if zone in roads:
            return colors["road"]
        if zone.startswith("红"):
            return colors["red"]
        if zone.startswith("蓝"):
            return colors["blue"]
        return colors["module"]

    # Draw primary movement regions first. Special modules get an outline only,
    # so the image communicates one primary location plus optional capabilities.
    primary = [
        "中央高地", "中央北隧道", "中央南隧道",
        "红上公路", "红中路", "红下公路", "蓝上公路", "蓝中路", "蓝下公路",
        "红基地", "蓝基地", "红前哨区", "蓝前哨区", "红梯形高地区", "蓝梯形高地区",
        "红上装配区", "红下装配区", "蓝上装配区", "蓝下装配区",
    ]
    for zone in primary:
        poly = [to_px(p, base.size) for p in ZONE_POLYGONS[zone]]
        c = colour(zone)
        draw.polygon(poly, fill=(*c[:3], 22), outline=c, width=4)

    modules = [z for z in ZONE_POLYGONS if z not in primary]
    for zone in modules:
        poly = [to_px(p, base.size) for p in ZONE_POLYGONS[zone]]
        c = colour(zone)
        draw.line(poly + [poly[0]], fill=(*c[:3], 190), width=3, joint="curve")

    # Only central adjacency arrows are shown; the full directed graph remains
    # in functional_zone.py and will be reviewed against the official drawing.
    def arrow(a, b):
        a, b = to_px(a, base.size), to_px(b, base.size)
        c = colors["central"]
        draw.line((a, b), fill=c, width=5)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        tip = 14
        left = (b[0] - tip * math.cos(ang - 0.45), b[1] - tip * math.sin(ang - 0.45))
        right = (b[0] - tip * math.cos(ang + 0.45), b[1] - tip * math.sin(ang + 0.45))
        draw.polygon((b, left, right), fill=c)

    for left, centre, right in (
        ("红上公路", "中央北隧道", "蓝上公路"),
        ("红下公路", "中央南隧道", "蓝下公路"),
        ("红中路", "中央高地", "蓝中路"),
    ):
        arrow(zone_centroid(left), zone_centroid(centre))
        arrow(zone_centroid(centre), zone_centroid(right))

    labels = {
        "中央高地": "中央高地", "中央北隧道": "中央北隧道", "中央南隧道": "中央南隧道",
        "红上公路": "红上公路", "红中路": "红中路", "红下公路": "红下公路",
        "蓝上公路": "蓝上公路", "蓝中路": "蓝中路", "蓝下公路": "蓝下公路",
        "红基地": "红基地", "蓝基地": "蓝基地", "红前哨区": "红前哨区", "蓝前哨区": "蓝前哨区",
        "红梯形高地区": "红梯形高地", "蓝梯形高地区": "蓝梯形高地",
        "红上装配区": "红上装配区", "红下装配区": "红下装配区",
        "蓝上装配区": "蓝上装配区", "蓝下装配区": "蓝下装配区",
    }
    for zone, text in labels.items():
        x, y = to_px(zone_centroid(zone), base.size)
        c = colour(zone)
        box = draw.textbbox((x, y), text, font=font, anchor="mm")
        pad = 4
        draw.rounded_rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
                               radius=4, fill=(255, 255, 255, 225), outline=c, width=2)
        draw.text((x, y), text, font=font, fill=(20, 25, 35, 255), anchor="mm")

    draw.rounded_rectangle((25, 20, 580, 100), radius=9, fill=(255, 255, 255, 238), outline=(20, 30, 50, 255), width=2)
    draw.text((45, 38), "官方线稿底图 · 功能区预览", font=title, fill=(18, 30, 58, 255))
    draw.text((45, 76), "主区域用填充表示，特殊模块只作叠加标记", font=font, fill=(50, 60, 75, 255))
    result = Image.alpha_composite(base, layer).convert("RGB")
    result.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
