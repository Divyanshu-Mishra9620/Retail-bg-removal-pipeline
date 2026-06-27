"""
Orchestrator — decides which ORIGINAL script's flow to run per image.

It does NOT do any image processing itself. Both flows below call the exact
original functions (via `originals`), reproducing each source script step for
step so the saved cutout is identical to running that script alone.

Per-image decision (user-approved): "Script A -> Script B if needed"
  1. PRIMARY  : test_BiRefNet.py flow (EXIF load -> YOLO crop box -> BiRefNet
                -> refine_alpha -> back-project -> assess_mask).
  2. If QA != success: FALLBACK to test.py flow (cv2 read -> YOLO-seg ->
     mask_from_result -> smooth_alpha). If it yields a mask, that output is
     used (exactly test.py's result). Otherwise the BiRefNet result is kept
     and routed to review.

No fusion, no candidate-blending, no novel algorithms.
"""

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import SystemConfig
from .image_io import read_bgr, save_original_to_review
from . import originals as O
from .quality import save_cutout


def _birefnet_flow(path: Path, yolo_model, birefnet_model, transform, cfg: SystemConfig):
    """
    Exact reproduction of test_BiRefNet.py:process_one (without file writing).
    Returns (original_rgba, full_alpha, quality, routing).
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        original_rgba = image.convert("RGBA")
        image_rgb = original_rgba.convert("RGB")

    img_np = np.array(image_rgb)
    orig_w, orig_h = image_rgb.size
    use_fp16 = cfg.device == "cuda" and cfg.use_fp16

    yolo_box = None
    if yolo_model is not None:
        yolo_box = O.yolo_detect(yolo_model, img_np, cfg.yolo_conf, cfg.yolo_pad)

    if yolo_box is not None:
        x1, y1, x2, y2 = yolo_box
        crop_rgb = image_rgb.crop((x1, y1, x2, y2))
        raw_alpha_crop = O.predict_alpha(
            birefnet_model, crop_rgb, transform, cfg.birefnet_size, cfg.device, use_fp16
        )
        refined_crop = O.refine_alpha(crop_rgb, raw_alpha_crop, cfg)
        full_alpha = np.zeros((orig_h, orig_w), dtype=np.uint8)
        full_alpha[y1:y2, x1:x2] = refined_crop
        routing = "birefnet_yolo_crop"
    else:
        raw_alpha = O.predict_alpha(
            birefnet_model, image_rgb, transform, cfg.birefnet_size, cfg.device, use_fp16
        )
        full_alpha = O.refine_alpha(image_rgb, raw_alpha, cfg)
        routing = "birefnet_fullimage"

    quality = O.assess_mask(full_alpha)
    return original_rgba, full_alpha, quality, routing


def _yolo_seg_flow(path: Path, yolo_model, cfg: SystemConfig):
    """
    Exact reproduction of Retail_AI_Training/test.py:process_one (without file
    writing). Returns (original_rgba, alpha) or None if no mask was detected.
    """
    if yolo_model is None:
        return None
    bgr = read_bgr(path)
    if bgr is None:
        return None

    height, width = bgr.shape[:2]
    results = yolo_model.predict(
        bgr, imgsz=cfg.seg_imgsz, conf=cfg.seg_conf, iou=cfg.seg_iou, verbose=False
    )
    alpha = O.mask_from_result(results[0], width, height, cfg.seg_min_mask_area)
    if alpha is None:
        return None

    alpha = O.smooth_alpha(alpha, cfg.edge_blur)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_rgba = Image.fromarray(rgb).convert("RGBA")
    return original_rgba, alpha


def process_one_image(path: Path, yolo_model, birefnet_model, transform, cfg: SystemConfig) -> dict:
    result = {
        "input": str(path),
        "status": None,      # success | review | error
        "reason": None,
        "routing": None,
        "quality": None,
        "output": None,
    }

    # ── PRIMARY: BiRefNet flow ────────────────────────────────────────────────
    try:
        original_rgba, full_alpha, quality, routing = _birefnet_flow(
            path, yolo_model, birefnet_model, transform, cfg
        )
    except Exception as e:
        # BiRefNet couldn't run at all → try Script B, else review the original
        try:
            seg = _yolo_seg_flow(path, yolo_model, cfg)
        except Exception:
            seg = None
        if seg is not None:
            seg_rgba, seg_alpha = seg
            result.update(status="success", routing="yolo_seg_fallback",
                          reason="birefnet_error_used_yolo_seg")
            if not cfg.dry_run:
                _save(seg_rgba, seg_alpha, Path(cfg.output_dir), path, cfg, result)
            return result
        result.update(status="review", routing="none", reason=f"birefnet_error: {e}")
        if not cfg.dry_run:
            bgr = read_bgr(path)
            if bgr is not None:
                save_original_to_review(bgr, Path(cfg.review_dir) / f"{path.stem}.png")
        return result

    # ── Script A succeeded QA → use BiRefNet result ───────────────────────────
    if quality.status == "success":
        result.update(status="success", routing=routing, quality=asdict(quality))
        if not cfg.dry_run:
            _save(original_rgba, full_alpha, Path(cfg.output_dir), path, cfg, result)
        return result

    # ── Script A flagged review → try Script B (test.py YOLO-seg) ─────────────
    seg = _yolo_seg_flow(path, yolo_model, cfg)
    if seg is not None:
        seg_rgba, seg_alpha = seg
        result.update(status="success", routing="yolo_seg_fallback",
                      reason=f"birefnet_review({quality.reason})_used_yolo_seg",
                      quality=asdict(O.assess_mask(seg_alpha)))
        if not cfg.dry_run:
            _save(seg_rgba, seg_alpha, Path(cfg.output_dir), path, cfg, result)
        return result

    # ── Both insufficient → keep BiRefNet result, route to review ─────────────
    result.update(status="review", routing=routing + "_review",
                  reason=quality.reason, quality=asdict(quality))
    if not cfg.dry_run:
        _save(original_rgba, full_alpha, Path(cfg.review_dir), path, cfg, result)
    return result


def _save(original_rgba, alpha, dest_dir: Path, path: Path, cfg: SystemConfig, result: dict):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{path.stem}.png"
    save_cutout(original_rgba, alpha, dest_path)   # leak-proof compositor
    result["output"] = str(dest_path)
    if cfg.save_masks:
        Image.fromarray(np.asarray(alpha), mode="L").save(dest_dir / f"{path.stem}_mask.png", "PNG")
