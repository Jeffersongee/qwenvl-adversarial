"""
Scene-coherent adversarial text renderer.

Four styles:
  decal   — vinyl sticker, perspective-sheared, semi-transparent, drop shadow
  badge   — chrome emblem look, small, placed on lower body panel
  painted — stencil/spray-paint, no background, directly on surface
  sticker — white label, rounded corners, peel shadow

All styles use PIL AFFINE shear (not QUAD — avoids stripe artifact)
and standard alpha-composite blending.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter

IMPACT      = "/System/Library/Fonts/Supplemental/Impact.ttf"
VERDANA_B   = "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"

PLACEMENT_ANCHORS = {
    "center":     (0.50, 0.50),
    "top":        (0.50, 0.12),
    "bottom":     (0.50, 0.88),
    "badge":      (0.70, 0.55),   # mid-body panel, away from grille
    "windshield": (0.50, 0.28),
    "hood":       (0.50, 0.35),
    "door":       (0.30, 0.55),   # door panel (upper half of lower bbox)
}

STYLE_FOR_PLACEMENT = {
    "badge":      "badge",
    "windshield": "sticker",
    "hood":       "painted",
    "door":       "decal",
    "center":     "decal",
    "top":        "painted",
    "bottom":     "decal",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    path = IMPACT if bold else VERDANA_B
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _fit_font(text: str, target_w: int, max_h: int,
              bold: bool = True) -> tuple[ImageFont.FreeTypeFont, int, int]:
    lo, hi = 10, max(11, max_h)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        f = _load_font(mid, bold)
        bb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=f)
        if bb[2] - bb[0] <= target_w:
            lo = mid
        else:
            hi = mid
    font = _load_font(lo, bold)
    bb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    return font, bb[2] - bb[0], bb[3] - bb[1]


def _luminance(r, g, b) -> float:
    return 0.2126*(r/255) + 0.7152*(g/255) + 0.0722*(b/255)


def _sample_bg(img: Image.Image, cx: int, cy: int, r: int = 60
               ) -> tuple[int, int, int]:
    x0 = max(0, cx - r); y0 = max(0, cy - r)
    x1 = min(img.width, cx + r); y1 = min(img.height, cy + r)
    region = np.array(img.crop((x0, y0, x1, y1)).convert("RGB"))
    return tuple(int(v) for v in region.reshape(-1, 3).mean(axis=0))


def _contrast_fg(bg: tuple) -> tuple[int, int, int]:
    return (245, 245, 245) if _luminance(*bg) < 0.45 else (15, 15, 15)


def _affine_shear(tile: Image.Image,
                  shear_x: float = 0.06,
                  shear_y: float = 0.0) -> Image.Image:
    """
    Mild horizontal shear via AFFINE transform.
    src_x = dst_x + shear_x * dst_y
    src_y = dst_y + shear_y * dst_x
    Expands canvas to avoid clipping.
    """
    w, h = tile.size
    extra_x = int(abs(shear_x) * h)
    extra_y = int(abs(shear_y) * w)
    new_w = w + extra_x
    new_h = h + extra_y

    # offset so content stays inside canvas
    ox = extra_x if shear_x > 0 else 0
    oy = extra_y if shear_y > 0 else 0

    # AFFINE data: (a, b, c, d, e, f)
    # src_x = a*dst_x + b*dst_y + c
    # src_y = d*dst_x + e*dst_y + f
    data = (1, -shear_x, shear_x * h - ox,
            -shear_y, 1, shear_y * w - oy)

    return tile.transform(
        (new_w, new_h),
        Image.AFFINE,
        data,
        resample=Image.BICUBIC,
    )


def _drop_shadow(base: Image.Image, tile: Image.Image,
                 tx: int, ty: int,
                 offset: int = 5, blur: int = 8,
                 alpha: int = 110) -> Image.Image:
    """Composite a blurred shadow under tile onto base."""
    sh_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask = tile.split()[3] if tile.mode == "RGBA" else tile.convert("L")
    shadow = Image.new("RGBA", tile.size, (0, 0, 0, alpha))
    shadow.putalpha(mask if tile.mode == "RGBA" else
                    Image.new("L", tile.size, alpha))
    sh_layer.paste(shadow, (tx + offset, ty + offset))
    sh_layer = sh_layer.filter(ImageFilter.GaussianBlur(blur))
    result = base.convert("RGBA")
    result.alpha_composite(sh_layer)
    return result.convert("RGB")


def _paste_tile(base: Image.Image, tile: Image.Image,
                cx: int, cy: int) -> tuple[Image.Image, int, int]:
    """Alpha-composite tile centred at (cx, cy), clamped to image bounds."""
    tw, th = tile.size
    tx = max(0, min(cx - tw // 2, base.width  - tw))
    ty = max(0, min(cy - th // 2, base.height - th))
    result = base.convert("RGBA")
    if tile.mode != "RGBA":
        tile = tile.convert("RGBA")
    result.alpha_composite(tile, dest=(tx, ty))
    return result.convert("RGB"), tx, ty


# ── four style renderers ──────────────────────────────────────────────────────

def _style_decal(text, bbox_px, img, placement):
    """Vinyl sticker: coloured bg, white/black text, sheared, semi-transparent."""
    x0, y0, x1, y1 = bbox_px
    bw, bh = x1 - x0, y1 - y0
    ax, ay = PLACEMENT_ANCHORS[placement]
    cx = x0 + int(ax * bw)
    cy = y0 + int(ay * bh)

    bg = _sample_bg(img, cx, cy, r=80)
    fg = _contrast_fg(bg)

    font, tw, th = _fit_font(text, int(0.55 * bw), int(0.30 * bh))
    pad = max(10, th // 4)

    tile = Image.new("RGBA", (tw + pad*2, th + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, tile.width-1, tile.height-1], fill=bg + (230,))
    draw.text((pad, pad), text, fill=fg + (255,), font=font)

    tile = _affine_shear(tile, shear_x=0.06, shear_y=0.0)
    result = _drop_shadow(img, tile, *_paste_tile(img, tile, cx, cy)[1:],
                          offset=4, blur=7, alpha=100)
    return _paste_tile(result, tile, cx, cy)


def _style_badge(text, bbox_px, img, placement):
    """Chrome badge: silver text, no background, slight shear."""
    x0, y0, x1, y1 = bbox_px
    bw, bh = x1 - x0, y1 - y0
    ax, ay = PLACEMENT_ANCHORS[placement]
    cx = x0 + int(ax * bw)
    cy = y0 + int(ay * bh)

    bg = _sample_bg(img, cx, cy, r=60)
    # Always use high-contrast fg for badge readability
    fg = (230, 230, 235) if _luminance(*bg) < 0.5 else (20, 20, 25)
    outline = (30, 30, 30, 180) if _luminance(*bg) < 0.5 else (200, 200, 200, 180)

    font, tw, th = _fit_font(text, int(0.45 * bw), int(0.22 * bh))
    pad = max(8, th // 4)

    tile = Image.new("RGBA", (tw + pad*2, th + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    # Dark outline for depth
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
        draw.text((pad+dx, pad+dy), text, fill=outline, font=font)
    draw.text((pad, pad), text, fill=fg+(250,), font=font)

    tile = _affine_shear(tile, shear_x=0.04, shear_y=0.0)
    result = _drop_shadow(img, tile, *_paste_tile(img, tile, cx, cy)[1:],
                          offset=2, blur=4, alpha=80)
    return _paste_tile(result, tile, cx, cy)


def _style_painted(text, bbox_px, img, placement):
    """Spray-painted stencil: no background, slightly noisy alpha edges."""
    x0, y0, x1, y1 = bbox_px
    bw, bh = x1 - x0, y1 - y0
    ax, ay = PLACEMENT_ANCHORS[placement]
    cx = x0 + int(ax * bw)
    cy = y0 + int(ay * bh)

    bg = _sample_bg(img, cx, cy, r=80)
    fg = _contrast_fg(bg)

    font, tw, th = _fit_font(text, int(0.60 * bw), int(0.28 * bh))
    pad = max(6, th // 5)

    tile = Image.new("RGBA", (tw + pad*2, th + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((pad, pad), text, fill=fg + (220,), font=font)

    # Spray noise on alpha channel
    arr = np.array(tile, dtype=np.float32)
    noise = np.random.normal(0, 12, arr.shape[:2])
    arr[..., 3] = np.clip(arr[..., 3] + noise, 0, 255)
    tile = Image.fromarray(arr.astype(np.uint8))

    tile = _affine_shear(tile, shear_x=0.05, shear_y=0.0)
    return _paste_tile(img, tile, cx, cy)


def _style_sticker(text, bbox_px, img, placement):
    """White label sticker: rounded rect, dark text, peel-shadow corner."""
    x0, y0, x1, y1 = bbox_px
    bw, bh = x1 - x0, y1 - y0
    ax, ay = PLACEMENT_ANCHORS[placement]
    cx = x0 + int(ax * bw)
    cy = y0 + int(ay * bh)

    font, tw, th = _fit_font(text, int(0.50 * bw), int(0.25 * bh), bold=False)
    pad = max(12, th // 3)
    r = max(6, pad // 2)

    tile = Image.new("RGBA", (tw + pad*2, th + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    draw.rounded_rectangle([0, 0, tile.width-1, tile.height-1],
                            radius=r, fill=(252, 252, 250, 245))
    draw.rounded_rectangle([0, 0, tile.width-1, tile.height-1],
                            radius=r, outline=(170, 170, 170, 200),
                            width=max(2, pad//6))
    draw.text((pad, pad), text, fill=(18, 18, 18, 255), font=font)
    # Peel curl shadow
    cs = max(10, pad)
    draw.polygon([(tile.width-cs, tile.height),
                  (tile.width,    tile.height-cs),
                  (tile.width,    tile.height)],
                 fill=(60, 60, 60, 130))

    tile = _affine_shear(tile, shear_x=0.03, shear_y=0.0)
    result = _drop_shadow(img, tile, *_paste_tile(img, tile, cx, cy)[1:],
                          offset=5, blur=9, alpha=95)
    return _paste_tile(result, tile, cx, cy)


STYLE_FNS = {
    "decal":   _style_decal,
    "badge":   _style_badge,
    "painted": _style_painted,
    "sticker": _style_sticker,
}


# ── public API ────────────────────────────────────────────────────────────────

def render_scene_coherent(
    image_path: str,
    text: str,
    placement: str,
    bbox: list[float],
    style: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x0 = int(bbox[0]*w); y0 = int(bbox[1]*h)
    x1 = int(bbox[2]*w); y1 = int(bbox[3]*h)
    bbox_px = (x0, y0, x1, y1)

    chosen = style or STYLE_FOR_PLACEMENT.get(placement, "decal")
    result, _, _ = STYLE_FNS[chosen](text, bbox_px, img, placement)

    if output_path is None:
        p = Path(image_path)
        safe = text[:14].replace(" ", "_").replace("/", "_")
        output_path = str(
            p.parent / "results" /
            f"{p.stem}_sc_{chosen}_{placement}_{safe}{p.suffix}"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=92)
    return output_path


if __name__ == "__main__":
    import sys
    img   = sys.argv[1]
    text  = sys.argv[2]
    place = sys.argv[3] if len(sys.argv) > 3 else "badge"
    sty   = sys.argv[4] if len(sys.argv) > 4 else None
    bbox  = [float(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 \
            else [0.05, 0.4, 0.55, 0.82]
    print(render_scene_coherent(img, text, place, bbox, sty))
