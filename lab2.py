from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class GrayWeights:
    r: float = 0.299
    g: float = 0.587
    b: float = 0.114


def clamp_u8(value: int) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def to_grayscale(rgb: Image.Image, weights: GrayWeights = GrayWeights()) -> Image.Image:
    """
    Convert an RGB image to 8-bit grayscale (mode 'L') without using Image.convert().
    Gray = r*w.r + g*w.g + b*w.b.
    """
    if rgb.mode != "RGB":
        rgb = rgb.convert("RGB")
    width, height = rgb.size
    src = rgb.tobytes()  # RGBRGB...

    out = bytearray(width * height)
    wr, wg, wb = weights.r, weights.g, weights.b
    for i in range(width * height):
        base = i * 3
        r = src[base]
        g = src[base + 1]
        b = src[base + 2]
        out[i] = clamp_u8(int(r * wr + g * wg + b * wb + 0.5))

    return Image.frombytes("L", (width, height), bytes(out))


def integral_image_u8(gray: bytes, width: int, height: int) -> list[int]:
    """
    Summed-area table over a grayscale byte buffer.
    Returns an array of size (height+1)*(width+1) in row-major order.
    """
    stride = width + 1
    sat = [0] * ((height + 1) * (width + 1))
    for y in range(height):
        row_sum = 0
        row_offset = y * width
        sat_row = (y + 1) * stride
        sat_prev = y * stride
        for x in range(width):
            row_sum += gray[row_offset + x]
            sat[sat_row + (x + 1)] = sat[sat_prev + (x + 1)] + row_sum
    return sat


def rect_sum(sat: Sequence[int], width: int, x0: int, y0: int, x1: int, y1: int) -> int:
    """
    Sum in [x0, x1) x [y0, y1) using summed-area table.
    """
    stride = width + 1
    a = sat[y0 * stride + x0]
    b = sat[y0 * stride + x1]
    c = sat[y1 * stride + x0]
    d = sat[y1 * stride + x1]
    return d - b - c + a


def adaptive_threshold_mean(
    gray_img: Image.Image,
    window_size: int = 5,
    offset: int = 0,
) -> Image.Image:
    """
    Adaptive binarization by local mean:
      T(x,y) = mean(window) - offset
      out = 255 if gray >= T else 0
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and >= 3")
    if gray_img.mode != "L":
        raise ValueError("adaptive_threshold_mean expects an 'L' (grayscale) image")

    width, height = gray_img.size
    gray = gray_img.tobytes()
    sat = integral_image_u8(gray, width, height)
    r = window_size // 2

    out = bytearray(width * height)
    for y in range(height):
        y0 = max(0, y - r)
        y1 = min(height, y + r + 1)
        for x in range(width):
            x0 = max(0, x - r)
            x1 = min(width, x + r + 1)
            area = (x1 - x0) * (y1 - y0)
            s = rect_sum(sat, width, x0, y0, x1, y1)
            mean = s // area
            t = mean - offset
            out[y * width + x] = 255 if gray[y * width + x] >= t else 0

    return Image.frombytes("L", (width, height), bytes(out))


def hstack(images: Sequence[Image.Image], bg: int = 255) -> Image.Image:
    widths, heights = zip(*(im.size for im in images))
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h), (bg, bg, bg))
    x = 0
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


def generate_samples(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    # 1) "Contour map" style
    w, h = 640, 420
    im = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(im)
    for i, color in enumerate([(30, 30, 30), (60, 60, 60), (90, 90, 90)]):
        step = 20 + i * 10
        for y in range(20, h - 20, step):
            d.arc((20, y - 80, w - 20, y + 80), start=0, end=180, fill=color, width=2)
    for x in range(40, w - 40, 80):
        d.line((x, 30, x + 30, h - 30), fill=(20, 20, 20), width=3)
    p = out_dir / "sample_contours.png"
    im.save(p)
    paths.append(p)

    # 2) "X-ray" style (soft gradients + structures)
    w, h = 640, 420
    im = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(h):
        g = int(20 + 160 * (1 - abs((y - h / 2) / (h / 2))))
        d.line((0, y, w, y), fill=(g, g + 10, g + 20))
    for i in range(10):
        x0 = 60 + i * 50
        d.ellipse((x0, 60, x0 + 160, 360), outline=(220, 235, 255), width=3)
    p = out_dir / "sample_xray.png"
    im.save(p)
    paths.append(p)

    # 3) "Cartoon" style (flat colors + edges)
    w, h = 640, 420
    im = Image.new("RGB", (w, h), (120, 190, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 280, w, h), fill=(80, 200, 120))
    d.polygon([(60, 280), (220, 140), (380, 280)], fill=(255, 180, 120), outline=(30, 30, 30))
    d.polygon([(280, 280), (440, 120), (600, 280)], fill=(255, 210, 140), outline=(30, 30, 30))
    d.ellipse((430, 40, 560, 170), fill=(255, 245, 120), outline=(30, 30, 30), width=3)
    for x in range(40, w, 80):
        d.rectangle((x, 300, x + 30, 410), fill=(120, 70, 40))
        d.ellipse((x - 20, 240, x + 60, 320), fill=(30, 140, 60), outline=(20, 60, 30))
    p = out_dir / "sample_cartoon.png"
    im.save(p)
    paths.append(p)

    # 4) "Fingerprint" style
    w, h = 420, 420
    im = Image.new("RGB", (w, h), (235, 235, 235))
    d = ImageDraw.Draw(im)
    cx, cy = w // 2, h // 2
    for r in range(20, 200, 6):
        bbox = (cx - r, cy - r, cx + r, cy + r)
        start = 200
        end = 520
        d.arc(bbox, start=start, end=end, fill=(40, 40, 40), width=2)
    for y in range(80, 340, 22):
        d.arc((80, y - 40, 340, y + 40), start=0, end=180, fill=(70, 70, 70), width=1)
    p = out_dir / "sample_fingerprint.png"
    im.save(p)
    paths.append(p)

    # 5) Unevenly lit text page
    w, h = 900, 520
    base = Image.new("RGB", (w, h), (245, 245, 235))
    d = ImageDraw.Draw(base)
    # add uneven illumination
    illum = Image.new("L", (w, h), 0)
    di = ImageDraw.Draw(illum)
    di.ellipse((-200, -150, 700, 600), fill=180)
    di.ellipse((250, -200, 1200, 700), fill=110)
    di.rectangle((0, 0, w, h), outline=0)
    base = Image.composite(base, Image.new("RGB", (w, h), (210, 210, 210)), illum)
    d = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    text = (
        "Пример страницы с неравномерной засветкой.\n"
        "Adaptive threshold должен помочь отделить текст от фона.\n"
        "Строка 3: 0123456789, ABCDEFGHIJ.\n"
        "Строка 4: быстрый тест бинаризации."
    )
    y = 60
    for line in text.splitlines():
        d.text((60, y), line, fill=(30, 30, 30), font=font)
        y += 40
    for i in range(12):
        d.text((60, y + i * 28), f"Строка {5+i}: пример текста для порога.", fill=(40, 40, 40), font=font)
    p = out_dir / "sample_text_uneven.png"
    base.save(p)
    paths.append(p)

    return paths


def process_one(
    input_path: Path,
    out_dir: Path,
    weights: GrayWeights,
    window_size: int,
    offset: int,
) -> tuple[Path, Path, Path]:
    rgb = Image.open(input_path)
    gray = to_grayscale(rgb, weights=weights)

    out_dir.mkdir(parents=True, exist_ok=True)
    gray_path = out_dir / f"{input_path.stem}_gray.bmp"
    gray.save(gray_path, format="BMP")

    binary = adaptive_threshold_mean(gray, window_size=window_size, offset=offset)
    bin_path = out_dir / f"{input_path.stem}_bin.bmp"
    binary.save(bin_path, format="BMP")

    compare = hstack([rgb.convert("RGB"), gray.convert("RGB"), binary.convert("RGB")])
    comp_path = out_dir / f"{input_path.stem}_compare.png"
    compare.save(comp_path, format="PNG")

    return gray_path, bin_path, comp_path


def iter_images(input_dir: Path) -> Iterable[Path]:
    exts = {".png", ".bmp"}
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Lab 2: grayscale + adaptive mean thresholding (no built-in grayscale/binarize)."
    )
    ap.add_argument("--input", type=Path, default=Path("input"), help="Input folder with .png/.bmp")
    ap.add_argument("--output", type=Path, default=Path("output"), help="Output folder")
    ap.add_argument("--generate-samples", action="store_true", help="Generate synthetic PNG samples into input folder")
    ap.add_argument(
        "--download-slavcorpora",
        type=str,
        default=None,
        help="Download images from slavcorpora (UUID or a URL containing it) into input/ as PNG.",
    )
    ap.add_argument("--limit", type=int, default=3, help="How many images to download (when downloading)")
    ap.add_argument("--window", type=int, default=5, help="Odd window size for adaptive mean threshold (variant: 5)")
    ap.add_argument("--offset", type=int, default=5, help="Offset subtracted from local mean (higher -> more white)")
    ap.add_argument("--weights", type=str, default="0.299,0.587,0.114", help="Grayscale weights: r,g,b")
    ap.add_argument(
        "--export-previews",
        action="store_true",
        help="Also export *_gray.png and *_bin.png for README preview.",
    )
    return ap.parse_args(argv)


def extract_uuid(text: str) -> str | None:
    import re

    m = re.search(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        text,
    )
    return m.group(0).lower() if m else None


def list_slavcorpora_keys(sample_uuid: str) -> list[str]:
    url = "https://www.slavcorpora.ru/images/?" + urllib.parse.urlencode({"prefix": sample_uuid})
    xml_bytes = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=30,
    ).read()
    root = ET.fromstring(xml_bytes)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [c.findtext("s3:Key", default="", namespaces=ns) for c in root.findall("s3:Contents", ns)]
    return [k for k in keys if k]


def pick_slavcorpora_images(keys: Sequence[str], limit: int) -> list[str]:
    images = [k for k in keys if k.lower().endswith(".jpeg") and "manifest" not in k.lower()]

    def rank(name: str) -> tuple[int, str]:
        lower = name.lower()
        if lower.endswith("-2.jpeg"):
            tier = 0
        elif lower.endswith("-1.jpeg"):
            tier = 1
        else:
            tier = 2
        return (tier, lower)

    images_sorted = sorted(images, key=rank)

    picked: list[str] = []
    seen_page: set[str] = set()
    for k in images_sorted:
        page_key = k
        if "/image-" in k:
            page_key = k.split("/image-", 1)[1]
        page_key = page_key.split(".jpeg", 1)[0].split("-", 1)[0]
        if page_key in seen_page:
            continue
        picked.append(k)
        seen_page.add(page_key)
        if len(picked) >= limit:
            break
    return picked


def download_slavcorpora_to_input(sample_uuid_or_url: str, input_dir: Path, limit: int) -> list[Path]:
    sample_uuid = extract_uuid(sample_uuid_or_url)
    if not sample_uuid:
        raise SystemExit("Can't extract UUID from --download-slavcorpora value")

    keys = list_slavcorpora_keys(sample_uuid)
    picked = pick_slavcorpora_images(keys, limit=limit)
    if not picked:
        raise SystemExit("No images found for this UUID in /images bucket listing")

    input_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for i, key in enumerate(picked, start=1):
        url = f"https://www.slavcorpora.ru/images/{urllib.parse.quote(key, safe='/')}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=120).read()
        img = Image.open(io.BytesIO(data))
        rgb = img.convert("RGB")
        out_path = input_dir / f"slavcorpora_{sample_uuid[:8]}_{i}.png"
        rgb.save(out_path, format="PNG")
        out_paths.append(out_path)

    meta_path = input_dir / f"slavcorpora_{sample_uuid[:8]}_meta.json"
    meta = {"uuid": sample_uuid, "downloaded_keys": picked}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_paths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input
    output_dir: Path = args.output

    try:
        wr, wg, wb = (float(x.strip()) for x in str(args.weights).split(","))
    except Exception:
        raise SystemExit("--weights must be like '0.299,0.587,0.114'")
    weights = GrayWeights(r=wr, g=wg, b=wb)

    if args.download_slavcorpora:
        downloaded = download_slavcorpora_to_input(
            args.download_slavcorpora,
            input_dir=input_dir,
            limit=int(args.limit),
        )
        print(f"Downloaded {len(downloaded)} images to {input_dir}")

    if args.generate_samples:
        generated = generate_samples(input_dir)
        print(f"Generated {len(generated)} samples in {input_dir}")

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        raise SystemExit(f"No .png/.bmp images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in images:
        gray_path, bin_path, comp_path = process_one(
            p,
            out_dir=output_dir,
            weights=weights,
            window_size=args.window,
            offset=args.offset,
        )
        print(f"{p.name} -> {os.fspath(gray_path)}, {os.fspath(bin_path)}, {os.fspath(comp_path)}")

        if args.export_previews:
            Image.open(gray_path).convert("RGB").save(output_dir / f"{p.stem}_gray.png", format="PNG")
            Image.open(bin_path).convert("RGB").save(output_dir / f"{p.stem}_bin.png", format="PNG")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
