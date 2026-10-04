"""Mathematically enforce 180-degree center symmetry on the line map."""

from pathlib import Path
import json

import cv2
import numpy as np


SRC = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_model_white_bluebase.png")
OUT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_center_symmetric.png")
REPORT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_center_symmetry_report.json")


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def rotate180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2)


def main() -> None:
    image = read_image(SRC)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red = (((h <= 8) | (h >= 170)) & (s >= 150) & (v >= 120))
    blue = ((h >= 98) & (h <= 123) & (s >= 130) & (v >= 120))
    purple = ((h >= 140) & (h <= 169) & (s >= 100) & (v >= 120))

    # Keep the extracted strokes clean but do not change their topology.
    kernel = np.ones((3, 3), np.uint8)
    def clean(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return m > 0

    red, blue, purple = map(clean, (red, blue, purple))

    # 180-degree symmetry with team-color exchange: R(x) = B(rot(x)).
    red_s = red | rotate180(blue)
    blue_s = blue | rotate180(red)
    purple_s = purple | rotate180(purple)

    # A red/blue overlap cannot be assigned both colors at one pixel.  Treat
    # such exact intersections as public purple junctions; this is symmetric.
    overlap = red_s & blue_s
    purple_s |= overlap
    red_s &= ~overlap
    blue_s &= ~overlap

    output = np.full_like(image, 255)
    output[purple_s] = (220, 35, 255)
    output[red_s] = (0, 0, 255)
    output[blue_s] = (255, 70, 0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", output)[1].tofile(str(OUT))

    report = {
        "source": str(SRC),
        "output": str(OUT),
        "symmetry": "180-degree center rotation",
        "red_blue_mismatch_pixels": int(np.count_nonzero(red_s != rotate180(blue_s))),
        "blue_red_mismatch_pixels": int(np.count_nonzero(blue_s != rotate180(red_s))),
        "purple_mismatch_pixels": int(np.count_nonzero(purple_s != rotate180(purple_s))),
        "red_pixels": int(red_s.sum()),
        "blue_pixels": int(blue_s.sum()),
        "purple_pixels": int(purple_s.sum()),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
