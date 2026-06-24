"""
QA gate — assess mask quality and route images to success or review folder.
Ported from test_BiRefNet.py.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class MaskQuality:
    transparent_ratio: float
    foreground_ratio: float
    touches_border_ratio: float
    components: int
    largest_component_ratio: float
    status: str   # "success" | "review"
    reason: str


def assess_mask(alpha: np.ndarray) -> MaskQuality:
    hard = alpha >= 128
    h, w = hard.shape
    area = h * w

    foreground_ratio = float(hard.mean())
    transparent_ratio = 1.0 - foreground_ratio

    border = np.concatenate([hard[0, :], hard[-1, :], hard[:, 0], hard[:, -1]])
    touches_border_ratio = float(border.mean())

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), connectivity=8)
    components = max(0, num_labels - 1)
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if components else 0
    largest_component_ratio = largest_area / area if area else 0.0

    status, reason = "success", "ok"
    if foreground_ratio < 0.015:
        status, reason = "review", "almost everything became transparent"
    elif foreground_ratio > 0.995:
        status, reason = "review", "almost nothing was removed"
    elif touches_border_ratio > 0.92:
        status, reason = "review", "foreground touches most image borders"
    elif components > 25 and largest_component_ratio < 0.15:
        status, reason = "review", "mask is fragmented"

    return MaskQuality(
        transparent_ratio=round(transparent_ratio, 4),
        foreground_ratio=round(foreground_ratio, 4),
        touches_border_ratio=round(touches_border_ratio, 4),
        components=components,
        largest_component_ratio=round(largest_component_ratio, 4),
        status=status,
        reason=reason,
    )


def quality_score(q: MaskQuality) -> float:
    """
    Numeric score for comparing two MaskQuality results. Higher is better.

    A "success" mask always outscores a "review" mask (10-point bonus).
    Within the same status, scores reflect:
      - foreground_ratio: ideally ~0.35; extremes (too empty/too full) are penalised
      - touches_border_ratio: lower is better
      - largest_component_ratio: higher means the product is one coherent blob
    """
    status_bonus = 10.0 if q.status == "success" else 0.0
    fg = q.foreground_ratio
    fg_score = 1.0 - min(1.0, abs(fg - 0.35) / 0.35)
    border_score = 1.0 - q.touches_border_ratio
    return status_bonus + fg_score + border_score + q.largest_component_ratio


def save_cutout(
    original_rgba: Image.Image,
    alpha: np.ndarray,
    destination: Path,
    background: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Save a true background-removed RGBA PNG.

    This version builds a brand-new canvas:
      * RGB starts as a uniform neutral matte (``background``, default black);
      * original colours are copied back ONLY on foreground pixels (alpha > 0),
        which keeps soft edges and feathering intact;
      * everywhere the model marked transparent, RGB is the matte — the original
        background is physically destroyed, not merely hidden.

    """
    # Original colours as a contiguous (H, W, 3) uint8 array. convert("RGB")
    # is a no-op-cost guard that also normalises odd source modes (P/LA/RGB).
    rgb = np.asarray(original_rgba.convert("RGB"))
    src_h, src_w = rgb.shape[:2]

    # Normalise alpha to a 2-D uint8 map matching the source resolution.
    alpha = np.asarray(alpha)
    if alpha.ndim != 2:
        alpha = alpha.reshape(alpha.shape[0], alpha.shape[1])
    if alpha.dtype != np.uint8:
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    if alpha.shape != (src_h, src_w):
        raise ValueError(
            f"alpha shape {alpha.shape} does not match image (H, W)=({src_h}, {src_w})"
        )

    # Fresh canvas: fully transparent + neutral matte. Starting from zeros means
    # every transparent pixel is black/0 (no residual background) by default.
    out = np.zeros((src_h, src_w, 4), dtype=np.uint8)
    if any(background):
        out[..., 0], out[..., 1], out[..., 2] = background

    # Copy ORIGINAL RGB only where the product exists. ``out[..., :3]`` is a
    # view into ``out``, so this boolean-indexed assignment writes through to the
    # canvas. Background pixels keep the matte; their original colour is gone.
    foreground = alpha > 0
    out[..., :3][foreground] = rgb[foreground]

    # Straight (non-premultiplied) alpha preserved exactly — soft edges, feather
    # and semi-transparent boundaries are untouched.
    out[..., 3] = alpha

    Image.fromarray(out, mode="RGBA").save(destination, "PNG", optimize=True)


def verify_hidden_background_removed(
    png_path,
    background: tuple[int, int, int] = (0, 0, 0),
) -> dict:
    """
    Audit a saved PNG for recoverable background RGB in transparent regions.

    A "clean" cutout has the matte colour (``background``) in every fully
    transparent pixel, so ``convert("RGB")`` cannot resurrect the original scene.

    Returns a dict:
        clean                 True if no background RGB leaks behind alpha == 0.
        mode                  PIL mode of the file.
        has_alpha             whether the file carries an alpha channel.
        transparent_pixels    count of fully transparent (alpha == 0) pixels.
        leaked_pixels         transparent pixels whose RGB != matte.
        leak_ratio            leaked_pixels / transparent_pixels.
        max_leak_value        largest |RGB - matte| seen in transparent region.
    """
    img = Image.open(png_path)
    mode = img.mode

    # No alpha channel → nothing is "hidden"; the file is a flat opaque image.
    if mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return {
            "clean": True,
            "mode": mode,
            "has_alpha": False,
            "transparent_pixels": 0,
            "leaked_pixels": 0,
            "leak_ratio": 0.0,
            "max_leak_value": 0,
        }

    arr = np.asarray(img.convert("RGBA"))
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]

    matte = np.array(background, dtype=np.int16)
    transparent = alpha == 0
    n_transparent = int(transparent.sum())

    if n_transparent == 0:
        return {
            "clean": True,
            "mode": mode,
            "has_alpha": True,
            "transparent_pixels": 0,
            "leaked_pixels": 0,
            "leak_ratio": 0.0,
            "max_leak_value": 0,
        }

    # Per-pixel deviation from the matte inside the transparent region.
    deviation = np.abs(rgb[transparent] - matte).max(axis=1)
    leaked = deviation > 0
    n_leaked = int(leaked.sum())

    return {
        "clean": n_leaked == 0,
        "mode": mode,
        "has_alpha": True,
        "transparent_pixels": n_transparent,
        "leaked_pixels": n_leaked,
        "leak_ratio": round(n_leaked / n_transparent, 6),
        "max_leak_value": int(deviation.max()) if n_leaked else 0,
    }
