"""Build a first-pass functional-zone preview over the supplied RM map.

The preview deliberately separates primary non-overlapping regions from internal
landmarks. It is a review artifact; coordinates should only be promoted to the
runtime map after rule/trajectory validation.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"F:\Desktop\50F788026AD4944F212AEBC17658B3A7.jpg")
OUT = ROOT / "out_m2" / "field_map" / "functional_zone_preview_v2.png"


def mirror(points, width):
    return [(width - x, y) for x, y in points]


def main():
    image = Image.open(SOURCE).convert("RGBA")
    width, height = image.size
    # Keep the map readable while making the reviewed boundaries stand out.
    image = ImageEnhance.Brightness(image).enhance(0.62)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    red = (239, 68, 68, 210)
    blue = (37, 145, 235, 210)
    neutral = (141, 103, 219, 210)
    neutral_fill = (141, 103, 219, 36)
    red_fill = (239, 68, 68, 34)
    blue_fill = (37, 145, 235, 34)
    dark = (15, 23, 42, 235)

    # Primary polygons: each side has four regions. These are deliberately
    # closed and share edges; resource pads are drawn as landmarks below.
    red_base = [(0, 360), (208, 360), (208, 758), (0, 758)]
    red_top = [(208, 0), (845, 0), (845, 145), (710, 265), (365, 265), (365, 360), (208, 360)]
    red_mid = [(208, 758), (365, 758), (365, 806), (495, 806), (528, 895), (688, 895),
               (720, 850), (720, 740), (694, 700), (694, 389), (710, 265),
               (365, 265), (365, 360), (208, 360)]
    red_bottom = [(0, 758), (208, 758), (365, 758), (365, 806), (495, 806), (528, 895),
                  (688, 895), (720, 850), (720, 1024), (0, 1024)]
    # Central neutral spine closes the transition between the two sides.
    central = [(845, 0), (1075, 0), (1190, 130), (1190, 250), (1110, 410),
               (1075, 650), (1075, 850), (1075, 1024), (720, 1024),
               (720, 850), (720, 740), (694, 700), (694, 389), (710, 265), (845, 145)]
    blue_base = mirror(red_base, width)
    blue_top = mirror(red_top, width)
    blue_mid = mirror(red_mid, width)
    blue_bottom = mirror(red_bottom, width)

    # Draw neutral underlays first, then side regions. Visible boundaries are
    # therefore a planar partition even where the source map has markings.
    draw.polygon(central, fill=neutral_fill, outline=neutral, width=7)
    for polygon in (red_top, red_base, red_mid, red_bottom):
        draw.polygon(polygon, fill=red_fill, outline=red, width=7)
    for polygon in (blue_top, blue_base, blue_mid, blue_bottom):
        draw.polygon(polygon, fill=blue_fill, outline=blue, width=7)

    # Internal landmarks are not extra primary regions. Their thin outlines
    # prevent accidental overlap in the primary zoning graph.
    red_pad = [(375, 444), (525, 444), (557, 493), (535, 580), (405, 580), (375, 530)]
    blue_pad = mirror(red_pad, width)
    for polygon, colour in ((red_pad, red), (blue_pad, blue)):
        draw.line(polygon + [polygon[0]], fill=colour, width=4, joint="curve")

    # Arrowed adjacency hints: the runtime graph will use the same directed
    # relations, while the exact path remains a separate motion layer.
    def arrow(start, end, colour):
        draw.line([start, end], fill=colour, width=5)
        import math
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        size = 16
        left = (end[0] - size * math.cos(angle - 0.45), end[1] - size * math.sin(angle - 0.45))
        right = (end[0] - size * math.cos(angle + 0.45), end[1] - size * math.sin(angle + 0.45))
        draw.polygon([end, left, right], fill=colour)

    arrow((710, 270), (760, 320), neutral)
    arrow((1110, 320), (1160, 270), neutral)
    arrow((720, 740), (780, 790), neutral)
    arrow((1075, 790), (1130, 740), neutral)

    # Labels and legend.
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 19)

    def label(x, y, text, colour, anchor="mm"):
        box = draw.textbbox((x, y), text, font=font, anchor=anchor)
        pad_x, pad_y = 12, 7
        expanded = (box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y)
        draw.rounded_rectangle(expanded, radius=8, fill=dark, outline=colour, width=3)
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255), anchor=anchor)

    label(102, 560, "红方基地", red)
    label(520, 115, "红方上侧区", red)
    label(530, 690, "红方中侧区", red)
    label(330, 925, "红方下侧区", red)
    label(960, 360, "中央通道", neutral)
    label(960, 650, "中央高地", neutral)
    label(1810, 560, "蓝方基地", blue)
    label(1400, 115, "蓝方上侧区", blue)
    label(1390, 690, "蓝方中侧区", blue)
    label(1590, 925, "蓝方下侧区", blue)

    # Landmark captions are intentionally smaller and do not define topology.
    for x, y, text, colour in ((465, 515, "资源台（地标）", red), (1455, 515, "资源台（地标）", blue)):
        draw.text((x, y), text, font=small, fill=(255, 255, 255, 235), anchor="mm",
                  stroke_width=2, stroke_fill=(15, 23, 42, 220))

    legend_x, legend_y = 30, 30
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 430, legend_y + 90), radius=12,
                           fill=(15, 23, 42, 210), outline=(255, 255, 255, 180), width=2)
    draw.text((legend_x + 18, legend_y + 20), "主分区预览 v2", font=font, fill="white", anchor="lm")
    draw.line((legend_x + 20, legend_y + 58, legend_x + 70, legend_y + 58), fill=red, width=7)
    draw.text((legend_x + 82, legend_y + 58), "红方相对区域", font=small, fill="white", anchor="lm")
    draw.line((legend_x + 230, legend_y + 58, legend_x + 280, legend_y + 58), fill=blue, width=7)
    draw.text((legend_x + 292, legend_y + 58), "蓝方相对区域", font=small, fill="white", anchor="lm")

    result = Image.alpha_composite(image, overlay).convert("RGB")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
