"""
Orchestrator — full pipeline for a single image.

Merge of Retail_AI_Training/test.py (YOLO-seg) + test_BiRefNet.py (BiRefNet).

Flow
----
1. Read image (Unicode-safe BGR).
2. ONE YOLO-seg inference -> {mask (semantic prior), box (crop hint)}.
3a. If product found:
      crop -> BiRefNet -> refine (guided/GrabCut/solidify/feather) -> back-project
      -> FUSE with YOLO mask (semantic-gated matting).
3b. If nothing found:
      full-image BiRefNet -> refine.   (no rejection — universal coverage)
4. QA gate (assess_mask).
5. If QA fails, keep the best of {fused, BiRefNet-only, YOLO-mask-only} by
   quality_score — never reject for a single detector's miss.
6. Leak-proof compose + save to output/ (success) or review/ (uncertain).

Fallbacks instead of hard failures at every step.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SystemConfig
from .fusion import fuse_alpha
from .image_io import read_bgr, save_original_to_review
from .inference import predict_alpha
from .models import yolo_segment
from .postprocessing import refine_alpha, smooth_alpha
from .quality import MaskQuality, assess_mask, quality_score, save_cutout

# BGR<->RGB without importing cv2 at module top (kept local for clarity)
import cv2


def _new_result(path: Path) -> dict:
    return {
        "input": str(path),
        "status": None,      # success | review | error
        "reason": None,
        "routing": None,     # fused | birefnet_crop | fullimg_fallback | yolo_mask_only 
        "quality": None,
        "output": None,
    }


def process_one_image(
    path: Path,
    yolo_model,
    birefnet_model,
    transform,
    cfg: SystemConfig,
) -> dict:
    result = _new_result(path)

    # ── 1. Read image ─────────────────────────────────────────────────────────
    bgr = read_bgr(path)
    if bgr is None:
        result["status"] = "review"
        result["reason"] = "unreadable"
        result["routing"] = "none"
        return result

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_full = Image.fromarray(rgb)
    original_rgba = pil_full.convert("RGBA")
    h, w = bgr.shape[:2]
    use_fp16 = cfg.use_fp16 and cfg.device == "cuda"

    # ── 2. Single YOLO-seg inference (box + mask) ─────────────────────────────
    seg = yolo_segment(yolo_model, bgr, cfg)
    yolo_mask = seg["mask"]
    box = seg["box"]

    primary = None
    routing = None
    birefnet_full = None   # BiRefNet-only candidate (for fallback scoring)

    # ── 3. Background removal ─────────────────────────────────────────────────
    try:
        if box is not None:
            x1, y1, x2, y2 = box
            crop_pil = Image.fromarray(rgb[y1:y2, x1:x2])
            raw_crop = predict_alpha(birefnet_model, crop_pil, transform, cfg.birefnet_size, cfg.device, use_fp16)
            refined_crop = refine_alpha(crop_pil, raw_crop, cfg)

            birefnet_full = np.zeros((h, w), dtype=np.uint8)
            birefnet_full[y1:y2, x1:x2] = refined_crop

            if yolo_mask is not None and cfg.enable_fusion:
                primary = fuse_alpha(birefnet_full, yolo_mask, cfg)
                routing = "fused"
            else:
                primary = birefnet_full
                routing = "birefnet_crop"
        else:
            # No YOLO detection -> full-image BiRefNet (NO rejection)
            raw_full = predict_alpha(birefnet_model, pil_full, transform, cfg.birefnet_size, cfg.device, use_fp16)
            primary = refine_alpha(pil_full, raw_full, cfg)
            birefnet_full = primary
            routing = "fullimg_fallback"

    except Exception as e:
        # BiRefNet failed — fall back to the YOLO mask if we have one
        if yolo_mask is not None:
            primary = smooth_alpha(yolo_mask.copy(), cfg.edge_blur)
            routing = "yolo_mask_only"
            result["reason"] = f"birefnet_error_used_yolo: {e}"
        else:
            result["status"] = "review"
            result["reason"] = f"birefnet_error_no_yolo: {e}"
            result["routing"] = "none"
            if not cfg.dry_run:
                save_original_to_review(bgr, Path(cfg.review_dir) / f"{path.stem}.png")
            return result

    quality: MaskQuality = assess_mask(primary)

    # ── 4/5. Fallback cascade if QA not satisfied ─────────────────────────────
    if quality.status != "success":
        candidates = [(primary, quality, routing)]

        # BiRefNet-only (un-gated) — useful when fusion over-clipped
        if birefnet_full is not None and routing == "fused":
            q_bire = assess_mask(birefnet_full)
            candidates.append((birefnet_full, q_bire, "birefnet_alt"))

        # YOLO mask alone — useful when BiRefNet was the weak link
        if yolo_mask is not None:
            ym = smooth_alpha(yolo_mask.copy(), cfg.edge_blur)
            q_yolo = assess_mask(ym)
            candidates.append((ym, q_yolo, "yolo_mask_alt"))

        primary, quality, routing = max(candidates, key=lambda c: quality_score(c[1]))

    # ── Empty-alpha guard — never write a blank cutout silently ──────────────
    if int(np.count_nonzero(primary > 127)) == 0:
        result["status"] = "review"
        result["reason"] = "empty_alpha"
        result["routing"] = routing
        result["quality"] = asdict(quality)
        if not cfg.dry_run:
            save_original_to_review(bgr, Path(cfg.review_dir) / f"{path.stem}.png")
        return result

    result["routing"] = routing
    result["quality"] = asdict(quality)

    if cfg.dry_run:
        result["status"] = quality.status
        if result["reason"] is None:
            result["reason"] = quality.reason
        return result

    # ── 6. Leak-proof compose + save ──────────────────────────────────────────
    if quality.status == "success":
        dest_dir = Path(cfg.output_dir)
        result["status"] = "success"
    else:
        dest_dir = Path(cfg.review_dir)
        result["status"] = "review"
        if result["reason"] is None:
            result["reason"] = quality.reason

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{path.stem}.png"
    save_cutout(original_rgba, primary, dest_path)   # leak-proof (black canvas)
    result["output"] = str(dest_path)

    if cfg.save_masks:
        Image.fromarray(primary, mode="L").save(dest_dir / f"{path.stem}_mask.png", "PNG")

    return result
