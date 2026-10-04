"""Repair small gaps and rebuild a clean exact center-symmetric line map."""

from pathlib import Path
import json

import cv2
import numpy as np


SRC = Path(r"C:\Users\AntonyTang\.codex\generated_images\01a0d26c-e553-7562-9bc2-455ceecdbfaa\exec-07adcd39-df35-45e0-a484-fc13eb5ea667.png")
OUT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_repaired_center_symmetric.png")
REPORT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_repaired_center_symmetric_report.json")


def read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def rot180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2)


def close_and_filter(mask: np.ndarray) -> np.ndarray:
    # Bridge small editing gaps so polygon borders meet at vertices.
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= 60:
            keep[labels == idx] = 255
    return keep > 0


def main() -> None:
    image = read_image(SRC)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    masks = {
        "red": (((h <= 10) | (h >= 170)) & (s >= 145) & (v >= 120)),
        "blue": ((h >= 100) & (h <= 125) & (s >= 145) & (v >= 120)),
        "purple": ((h >= 140) & (h <= 165) & (s >= 145) & (v >= 120)),
    }
    masks = {name: close_and_filter(mask) for name, mask in masks.items()}

    # Canonical upper half; generate the lower half by exact center rotation.
    mid = image.shape[0] // 2
    top = np.zeros_like(masks["red"])
    top[:mid, :] = True
    red_top = masks["red"] & top
    blue_top = masks["blue"] & top
    purple_top = masks["purple"] & top
    red = red_top | rot180(blue_top)
    blue = blue_top | rot180(red_top)
    purple = purple_top | rot180(purple_top)

    # Symmetric junction policy: public purple takes precedence over a red/blue
    # collision created at the exact center of rotation.
    overlap = red & blue
    purple |= overlap
    red &= ~overlap
    blue &= ~overlap

    # Close only the reconstructed center seam.  The kernel is symmetric, so
    # this cannot introduce a new asymmetry while it joins split endpoints.
    seam_kernel = np.ones((5, 5), np.uint8)
    red = cv2.morphologyEx((red.astype(np.uint8) * 255), cv2.MORPH_CLOSE, seam_kernel) > 0
    blue = cv2.morphologyEx((blue.astype(np.uint8) * 255), cv2.MORPH_CLOSE, seam_kernel) > 0
    purple = cv2.morphologyEx((purple.astype(np.uint8) * 255), cv2.MORPH_CLOSE, seam_kernel) > 0

    output = np.full_like(image, 255)
    output[purple] = (220, 35, 255)
    output[red] = (0, 0, 255)
    output[blue] = (255, 70, 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", output)[1].tofile(str(OUT))

    report = {
        "source": str(SRC),
        "output": str(OUT),
        "repair": "7x7 gap closure + upper-half 180-degree reconstruction",
        "red_blue_mismatch_pixels": int(np.count_nonzero(red != rot180(blue))),
        "blue_red_mismatch_pixels": int(np.count_nonzero(blue != rot180(red))),
        "purple_mismatch_pixels": int(np.count_nonzero(purple != rot180(purple))),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
