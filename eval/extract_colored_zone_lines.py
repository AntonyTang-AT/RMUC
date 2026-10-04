from pathlib import Path

import cv2
import numpy as np


SRC = Path(r"F:\Desktop\Codex 图像 2026年9月26日 16_31_15.png")
OUT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_white.png")


def main() -> None:
    # cv2.imread on Windows can fail for non-ASCII paths; decode bytes explicitly.
    raw = np.fromfile(str(SRC), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(SRC)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # The manually edited zone strokes are the most saturated colored pixels.
    # Keep a little antialiased edge by including bright pixels near each hue.
    red = (((h <= 8) | (h >= 170)) & (s >= 105) & (v >= 110))
    blue = ((h >= 95) & (h <= 135) & (s >= 95) & (v >= 95))
    purple = ((h >= 130) & (h <= 169) & (s >= 70) & (v >= 90))

    # Remove isolated colored noise while preserving the continuous strokes.
    masks = {"red": red, "blue": blue, "purple": purple}
    kernel = np.ones((3, 3), np.uint8)
    cleaned = {}
    for name, mask in masks.items():
        m = (mask.astype(np.uint8) * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.dilate(m, np.ones((2, 2), np.uint8), iterations=1)
        # Keep only stroke-sized connected components. This removes tiny
        # colored artifacts from the original robot/field markings.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m, dtype=bool)
        for component in range(1, n):
            if stats[component, cv2.CC_STAT_AREA] >= 35:
                keep[labels == component] = True
        cleaned[name] = keep

    # Paint a pure white canvas and use the dominant source color for each mask.
    out = np.full_like(image, 255)
    out[cleaned["red"]] = (0, 0, 255)
    out[cleaned["blue"]] = (255, 80, 0)
    out[cleaned["purple"]] = (220, 40, 255)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".png", out)[1]
    encoded.tofile(str(OUT))
    print(OUT)


if __name__ == "__main__":
    main()
