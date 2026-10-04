"""Render a new topology-first zone proposal on the official line map.

This intentionally does not reuse the old red/blue semantic polygons. Zones are
numbered by movement topology; team ownership and interaction modules are
separate attributes in the eventual runtime schema.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out_m2" / "official" / "official_white_line_draft.png"
OUT = ROOT / "out_m2" / "official" / "topology_zone_preview_v1.png"


def mirror(points, width=1920):
    return [(width - x, y) for x, y in points]


def main():
    image = Image.open(BASE).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    label_font = ImageFont.truetype(font_path, 17)
    title_font = ImageFont.truetype(font_path, 25)

    # New topology-first partition. These are neutral IDs; no zone is named
    # after red/blue ownership and special objects are not separate polygons.
    left = {
        "Z01": ("外侧上端", [(30, 40), (250, 40), (250, 260), (30, 260)]),
        "Z02": ("上侧内场", [(250, 40), (650, 40), (650, 160), (550, 300), (250, 300)]),
        "Z03": ("上侧斜坡", [(250, 300), (550, 300), (680, 180), (730, 260), (600, 430), (250, 430)]),
        "Z04": ("中侧外廊", [(30, 260), (250, 260), (250, 430), (380, 450), (520, 600), (360, 670), (30, 670)]),
        "Z05": ("前哨环廊", [(250, 430), (520, 430), (580, 540), (520, 650), (250, 650)]),
        "Z06": ("外侧下端", [(30, 670), (250, 670), (250, 1045), (30, 1045)]),
        "Z07": ("下侧内场", [(250, 650), (520, 650), (680, 820), (650, 1045), (250, 1045)]),
        "Z08": ("中央西侧接近段", [(520, 600), (680, 560), (760, 700), (820, 800), (680, 820), (520, 650)]),
    }
    zones = dict(left)
    for key, (name, polygon) in left.items():
        zones["M" + key[1:]] = ("镜像" + name, mirror(polygon))

    central = {
        "C01": ("中央北通道", [(650, 40), (1270, 40), (1270, 180), (1120, 280), (800, 280), (650, 180)]),
        "C02": ("中央北西入口", [(650, 180), (800, 280), (760, 400), (680, 430), (600, 320)]),
        "C03": ("中央北东入口", [(1270, 180), (1320, 320), (1240, 430), (1160, 400), (1120, 280)]),
        "C04": ("中央高地西环", [(680, 430), (800, 300), (960, 300), (960, 550), (820, 700), (700, 680)]),
        "C05": ("中央高地东环", [(960, 300), (1120, 300), (1240, 430), (1240, 680), (1100, 800), (960, 800), (960, 550)]),
        "C06": ("中央南西入口", [(560, 680), (700, 680), (820, 800), (820, 880), (650, 930), (560, 820)]),
        "C07": ("中央南东入口", [(1360, 680), (1220, 680), (1100, 800), (1100, 880), (1270, 930), (1360, 820)]),
        "C08": ("中央南通道", [(650, 930), (820, 880), (1100, 880), (1270, 930), (1270, 1045), (650, 1045)]),
    }
    zones.update(central)

    palette = {
        "outer": (120, 145, 170, 190), "inner": (35, 145, 125, 190),
        "approach": (221, 147, 36, 205), "central": (118, 78, 188, 220),
    }

    def zone_colour(key):
        if key.startswith("C"):
            return palette["central"]
        if key in {"Z03", "Z08", "M03", "M08"}:
            return palette["approach"]
        if key in {"Z05", "M05"}:
            return palette["inner"]
        return palette["outer"]

    # Draw central cells first and side cells second, with only shared edges in
    # common. This makes the map read as a topological partition rather than a
    # set of team-colored ownership blobs.
    for key, (_, polygon) in zones.items():
        c = zone_colour(key)
        draw.polygon(polygon, fill=(*c[:3], 35), outline=c, width=4)

    # Show legal transition corridors only at actual interfaces.
    arrows = [((650, 180), (800, 280)), ((1270, 180), (1120, 280)),
              ((700, 680), (820, 800)), ((1220, 680), (1100, 800)),
              ((760, 400), (820, 500)), ((1160, 400), (1100, 500))]
    for start, end in arrows:
        draw.line((start, end), fill=(118, 78, 188, 230), width=5)

    for key, (name, polygon) in zones.items():
        if key.startswith("M"):
            display = key + "\n" + name.replace("镜像", "")
        else:
            display = key + "\n" + name
        x = sum(p[0] for p in polygon) // len(polygon)
        y = sum(p[1] for p in polygon) // len(polygon)
        box = draw.multiline_textbbox((x, y), display, font=label_font, anchor="mm", align="center")
        pad = 4
        c = zone_colour(key)
        draw.rounded_rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
                               radius=5, fill=(255, 255, 255, 235), outline=c, width=2)
        draw.multiline_text((x, y), display, font=label_font, fill=(20, 25, 35, 255),
                            anchor="mm", align="center", spacing=1)

    draw.rounded_rectangle((25, 20, 650, 105), radius=9, fill=(255, 255, 255, 240), outline=(20, 30, 50, 255), width=2)
    draw.text((45, 38), "全新拓扑分区预览 v1", font=title_font, fill=(18, 30, 58, 255))
    draw.text((45, 78), "按通行连接与障碍边界划分；红蓝阵营、资源台、能量机关另作属性", font=label_font, fill=(50, 60, 75, 255))
    legend = [("外侧/内场", palette["outer"]), ("入口/接近段", palette["approach"]), ("中央单元", palette["central"]), ("环廊", palette["inner"])]
    x, y = 35, 1015
    for text, c in legend:
        draw.line((x, y, x + 32, y), fill=c, width=7)
        draw.text((x + 42, y), text, font=label_font, fill=(20, 25, 35, 255), anchor="lm")
        x += 210

    Image.alpha_composite(image, layer).convert("RGB").save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
