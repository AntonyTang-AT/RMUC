"""Rebuild the lower half from the upper half under exact center symmetry."""

from pathlib import Path
import json

import cv2
import numpy as np


SRC = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_model_white_bluebase.png")
OUT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_center_symmetric_clean.png")
REPORT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_center_symmetric_clean_report.json")


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def rot180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2)


def main() -> None:
    image = read_image(SRC)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h <= 8) | (h >= 170)) & (s >= 150) & (v >= 120))
    blue = ((h >= 98) & (h <= 123) & (s >= 130) & (v >= 120))
    purple = ((h >= 140) & (h <= 169) & (s >= 100) & (v >= 120))

    # A small close removes antialias gaps before selecting the canonical half.
    k = np.ones((3, 3), np.uint8)
    def clean(x: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx((x.astype(np.uint8) * 255), cv2.MORPH_CLOSE, k) > 0

    red, blue, purple = map(clean, (red, blue, purple))
    height = image.shape[0]
    mid = height // 2
    top = np.zeros_like(red)
    top[:mid, :] = True

    # Use only the upper half as the canonical geometry.  Its 180-degree
    # counterpart becomes the lower half, with red and blue exchanged.
    red_top = red & top
    blue_top = blue & top
    purple_top = purple & top
    red_out = red_top | rot180(blue_top)
    blue_out = blue_top | rot180(red_top)
    purple_out = purple_top | rot180(purple_top)

    # Any exact center overlap is a public purple junction.
    overlap = red_out & blue_out
    purple_out |= overlap
    red_out &= ~overlap
    blue_out &= ~overlap

    output = np.full_like(image, 255)
    output[purple_out] = (220, 35, 255)
    output[red_out] = (0, 0, 255)
    output[blue_out] = (255, 70, 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", output)[1].tofile(str(OUT))

    report = {
        "source": str(SRC),
        "output": str(OUT),
        "canonical_half": "upper half; lower reconstructed by 180-degree rotation",
        "red_blue_mismatch_pixels": int(np.count_nonzero(red_out != rot180(blue_out))),
        "blue_red_mismatch_pixels": int(np.count_nonzero(blue_out != rot180(red_out))),
        "purple_mismatch_pixels": int(np.count_nonzero(purple_out != rot180(purple_out))),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
