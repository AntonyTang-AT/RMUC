"""Extract the manually drawn red/purple/blue zone boundaries as clean math strokes.

The source image is treated as an observation only.  Color masks are cleaned,
skeletonized, and rendered with a fixed stroke width on a white canvas.  This
keeps the output independent of the original background texture.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


SRC = Path(r"F:\Desktop\Codex 图像 2026年9月26日 16_31_15.png")
OUT_PNG = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_math.png")
OUT_NPZ = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_math_masks.npz")


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def clean_mask(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    """Close small gaps, remove noise, then reduce to a one-pixel skeleton."""
    m = (mask.astype(np.uint8) * 255)
    # Closing joins antialiased breaks at intersections without filling regions.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m, dtype=np.uint8)
    for component in range(1, n):
        if stats[component, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == component] = 255
    return skeletonize(keep > 0)


def main() -> None:
    image = load_image(SRC)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    masks = {
        "red": (((h <= 8) | (h >= 170)) & (s >= 120) & (v >= 120)),
        # Keep the hue bands disjoint: the edited purple is around H=130-140,
        # while the blue strokes are around H=105-115.
        "blue": ((h >= 95) & (h <= 121) & (s >= 110) & (v >= 100)),
        "purple": ((h >= 125) & (h <= 169) & (s >= 90) & (v >= 100)),
    }
    skeletons = {name: clean_mask(mask) for name, mask in masks.items()}

    # Render all mathematically cleaned strokes with one consistent width.
    output = np.full_like(image, 255)
    colors = {
        "red": (0, 0, 255),
        "blue": (255, 70, 0),
        "purple": (220, 35, 255),
    }
    # Draw purple first so red/blue retain their edge identity at junctions.
    for name in ("purple", "red", "blue"):
        stroke = (skeletons[name].astype(np.uint8) * 255)
        stroke = cv2.dilate(stroke, np.ones((7, 7), np.uint8), iterations=1)
        output[stroke > 0] = colors[name]

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", output)[1].tofile(str(OUT_PNG))
    np.savez_compressed(OUT_NPZ, **{k: v.astype(np.uint8) for k, v in skeletons.items()})
    print(OUT_PNG)
    print(OUT_NPZ)


if __name__ == "__main__":
    main()
