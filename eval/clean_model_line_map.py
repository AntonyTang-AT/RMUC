from pathlib import Path

import cv2
import numpy as np


SRC = Path(r"C:\Users\AntonyTang\.codex\generated_images\01a0d26c-e553-7562-9bc2-455ceecdbfaa\exec-b64cd8ba-a18b-4277-a259-9817567a0f47.png")
OUT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_model_white_bluebase.png")


def read_rgba(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[2] != 4:
        raise RuntimeError(f"expected RGBA model output: {path}")
    bgr = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    return np.clip(bgr * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)


def clean(mask: np.ndarray, min_area: int = 45) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == idx] = 255
    return keep > 0


def main() -> None:
    image = read_rgba(SRC)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Correct the right-base octagon before classification.  The model output
    # occasionally copies the left red base color into this symmetric slot.
    right_base = (np.indices(h.shape)[1] >= 1450) & (np.indices(h.shape)[1] <= 1665) & (np.indices(h.shape)[0] >= 350) & (np.indices(h.shape)[0] <= 570)
    red_in_base = (((h <= 8) | (h >= 170)) & (s >= 90) & (v >= 90) & right_base)
    image[red_in_base] = (255, 70, 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    masks = {
        "red": (((h <= 8) | (h >= 170)) & (s >= 100) & (v >= 100)),
        "blue": ((h >= 95) & (h <= 121) & (s >= 100) & (v >= 95)),
        "purple": ((h >= 125) & (h <= 169) & (s >= 75) & (v >= 95)),
    }
    masks = {name: clean(mask) for name, mask in masks.items()}

    output = np.full_like(image, 255)
    colors = {"red": (0, 0, 255), "blue": (255, 70, 0), "purple": (220, 35, 255)}
    for name in ("purple", "red", "blue"):
        output[masks[name]] = colors[name]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", output)[1].tofile(str(OUT))
    print(OUT)


if __name__ == "__main__":
    main()
