"""Render the current fine-grained functional-zone proposal on the RM map."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from worldstate.functional_zone import ZONE_POLYGONS, zone_centroid


SOURCE = Path(r"F:\Desktop\50F788026AD4944F212AEBC17658B3A7.jpg")
OUT = REPO / "out_m2" / "field_map" / "functional_zone_preview_v3_fine.png"
WIDTH_M, HEIGHT_M = 28.0, 15.0


def px(point, size):
    x, y = point
    width, height = size
    return (round(float(x) / WIDTH_M * width), round((HEIGHT_M - float(y)) / HEIGHT_M * height))


def main():
    image = Image.open(SOURCE).convert("RGBA")
    image = ImageEnhance.Brightness(image).enhance(0.55)
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    fonts = r"C:\Windows\Fonts\msyh.ttc"
    label_font = ImageFont.truetype(fonts, 18)
    title_font = ImageFont.truetype(fonts, 25)

    central = {"中央北隧道", "中央南隧道", "中央高地"}
    roads = {"红上公路", "红下公路", "红中路", "蓝上公路", "蓝下公路", "蓝中路"}
    modules = {"红基地", "蓝基地", "红前哨区", "蓝前哨区", "红梯形高地区", "蓝梯形高地区",
               "红上装配区", "红下装配区", "蓝上装配区", "蓝下装配区", "红停机坪", "蓝停机坪",
               "红补给区", "蓝补给区", "红能量机关区", "蓝能量机关区"}
    colors = {
        "red": (239, 68, 68, 220), "blue": (37, 145, 235, 220),
        "central": (157, 103, 231, 235), "road": (249, 181, 56, 220),
        "module": (241, 115, 46, 205),
    }

    def group(zone):
        if zone in central:
            return "central"
        if zone in roads:
            return "road"
        if zone in modules:
            return "red" if zone.startswith("红") else "blue"
        return "module"

    # Central geometry is drawn first so the actual north/south tunnel and the
    # central highland remain visually distinct from the side roads.
    order = list(central) + list(roads) + [z for z in ZONE_POLYGONS if z not in central and z not in roads]
    for zone in order:
        polygon = [px(point, image.size) for point in ZONE_POLYGONS[zone]]
        colour = colors[group(zone)]
        alpha_fill = 22 if zone not in central else 35
        draw.polygon(polygon, fill=(*colour[:3], alpha_fill), outline=colour, width=4)

    # Explicit central movement corridors, with arrows rather than dashed boxes.
    def arrow(a, b, colour):
        a, b = px(a, image.size), px(b, image.size)
        draw.line((a, b), fill=colour, width=5)
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        tip = 15
        left = (b[0] - tip * math.cos(angle - 0.45), b[1] - tip * math.sin(angle - 0.45))
        right = (b[0] - tip * math.cos(angle + 0.45), b[1] - tip * math.sin(angle + 0.45))
        draw.polygon((b, left, right), fill=colour)

    for left, centre, right in (
        ("红上公路", "中央北隧道", "蓝上公路"),
        ("红下公路", "中央南隧道", "蓝下公路"),
    ):
        arrow(zone_centroid(left), zone_centroid(centre), colors["central"])
        arrow(zone_centroid(centre), zone_centroid(right), colors["central"])
    arrow(zone_centroid("红中路"), zone_centroid("中央高地"), colors["central"])
    arrow(zone_centroid("中央高地"), zone_centroid("蓝中路"), colors["central"])

    def short_name(zone):
        return {
            "中央北隧道": "中央北隧道", "中央南隧道": "中央南隧道", "中央高地": "中央高地",
            "红上公路": "红上公路", "红下公路": "红下公路", "红中路": "红中路",
            "蓝上公路": "蓝上公路", "蓝下公路": "蓝下公路", "蓝中路": "蓝中路",
            "红梯形高地区": "红梯形高地", "蓝梯形高地区": "蓝梯形高地",
            "红上装配区": "红上装配", "红下装配区": "红下装配",
            "蓝上装配区": "蓝上装配", "蓝下装配区": "蓝下装配",
        }.get(zone, zone)

    # Only label zones that are useful for online position description. Small
    # interaction modules remain outlined but are labelled with a compact tag.
    label_zones = [
        "红基地", "红上装配区", "红下装配区", "红前哨区", "红梯形高地区", "红上公路", "红下公路", "红中路",
        "蓝基地", "蓝上装配区", "蓝下装配区", "蓝前哨区", "蓝梯形高地区", "蓝上公路", "蓝下公路", "蓝中路",
        "中央北隧道", "中央南隧道", "中央高地", "红能量机关区", "蓝能量机关区",
    ]
    for zone in label_zones:
        centre = px(zone_centroid(zone), image.size)
        colour = colors[group(zone)]
        text = short_name(zone)
        bbox = draw.textbbox(centre, text, font=label_font, anchor="mm")
        pad = 5
        draw.rounded_rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                               radius=5, fill=(12, 18, 32, 205), outline=colour, width=2)
        draw.text(centre, text, font=label_font, fill=(255, 255, 255, 245), anchor="mm")

    # Header and legend.
    draw.rounded_rectangle((24, 24, 520, 105), radius=10, fill=(12, 18, 32, 220), outline=(255, 255, 255, 160), width=2)
    draw.text((44, 45), "细分功能区预览 v3", font=title_font, fill="white")
    draw.text((44, 80), "中央北/南隧道、中央高地、三条侧路分别建模", font=label_font, fill=(230, 235, 245, 245))
    legend = [("红方区域", colors["red"]), ("蓝方区域", colors["blue"]), ("道路/通道", colors["road"]), ("中央功能区", colors["central"])]
    x, y = 30, height - 45
    for text, colour in legend:
        draw.line((x, y, x + 32, y), fill=colour, width=6)
        draw.text((x + 42, y), text, font=label_font, fill="white", anchor="lm")
        x += 175

    Image.alpha_composite(image, layer).convert("RGB").save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
