from pathlib import Path

from PIL import Image


CURRENT = Path(r"C:\Users\AntonyTang\.codex\generated_images\01a0d26c-e553-7562-9bc2-455ceecdbfaa\exec-8805d22f-0752-43f8-9849-4ad2a0c3943a.png")
REFERENCE = Path(r"C:\Users\AntonyTang\.codex\generated_images\01a0d26c-e553-7562-9bc2-455ceecdbfaa\exec-07adcd39-df35-45e0-a484-fc13eb5ea667.png")
OUT = Path(r"F:\Desktop\学校\rm考核项目\out_m2\official\zone_lines_central_restored.png")


def main() -> None:
    base = Image.open(CURRENT).convert("RGB")
    ref = Image.open(REFERENCE).convert("RGB").resize(base.size, Image.Resampling.NEAREST)
    width, height = base.size
    left = int(width / 3)
    right = int(width * 2 / 3)
    # Replace only the central third; outer red/blue topology stays byte-for-byte
    # from the current repaired version.
    base.paste(ref.crop((left, 0, right, height)), (left, 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
