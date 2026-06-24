"""
Model loaders + YOLO-seg inference.

Each model is loaded ONCE and reused for the whole batch.

best.pt is a 1-class YOLOv8-seg model ('product'). yolo_segment() runs a single
inference and returns BOTH:
  - a semantic union mask  (fusion prior — "where is the product")
  - a padded crop box       (BiRefNet crop hint)
so the box+mask the two source files each needed come from ONE inference.

mask_union_from_result() is the mask-extraction logic generalised to also expose
the union mask for fusion.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_yolo_cache = None


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_yolo(weights: str):
    """Load YOLO-seg once; cached. Returns None if unavailable (graceful)."""
    global _yolo_cache
    if _yolo_cache is not None:
        return _yolo_cache
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[WARN] ultralytics not installed — YOLO-seg disabled (BiRefNet-only mode).")
        return None
    p = Path(weights)
    if not p.exists():
        print(f"[WARN] YOLO weights not found at '{weights}' — BiRefNet-only mode.")
        return None
    _yolo_cache = YOLO(str(p))
    _yolo_cache.overrides["verbose"] = False
    print(f"[INFO] YOLO-seg loaded: {weights}  (task={_yolo_cache.task}, classes={list(_yolo_cache.names.values())})")
    return _yolo_cache


def load_birefnet(model_id: str, device: str, use_fp16: bool, local_files_only: bool = True):
    print(f"[INFO] Loading BiRefNet: {model_id}")
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_files_only
    )
    model.to(device).eval()
    if use_fp16 and device == "cuda":
        model.half()
    return model


def build_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── YOLO-seg inference ────────────────────────────────────────────────────────

def mask_union_from_result(result, width: int, height: int, min_mask_area: int) -> np.ndarray | None:
    """
    Union of YOLO instance masks (>= min_mask_area) at full image resolution.
    Returns uint8 0/255, or None.
    """
    if result.masks is None or len(result.masks) == 0:
        return None

    masks = result.masks.data.cpu().numpy()
    alpha = np.zeros((height, width), dtype=np.uint8)

    for mask in masks:
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        candidate = (resized * 255).astype(np.uint8)
        if np.count_nonzero(candidate > 127) >= min_mask_area:
            alpha = np.maximum(alpha, candidate)

    if np.count_nonzero(alpha > 127) < min_mask_area:
        return None
    return alpha


def _box_from_mask(mask: np.ndarray, pad: float, width: int, height: int):
    """Tight bounding box of a binary mask, expanded by `pad` fraction, clamped."""
    ys, xs = np.where(mask > 127)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(width, x2 + px + 1)
    y2 = min(height, y2 + py + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_from_detections(result, pad: float, width: int, height: int):
    """Fallback: padded box of the best-confidence detection (if masks absent)."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best = int(result.boxes.conf.argmax())
    x1, y1, x2, y2 = result.boxes.xyxy[best].cpu().tolist()
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(width, int(x2 + px))
    y2 = min(height, int(y2 + py))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def yolo_segment(yolo_model, image_bgr: np.ndarray, cfg) -> dict:
    """
    Single YOLO-seg inference → {'mask': uint8|None, 'box': (x1,y1,x2,y2)|None}.

    mask : semantic union mask (fusion prior)
    box  : padded crop box (BiRefNet hint) — derived from the mask when present,
           else from the best detection box.

    Both come from ONE model.predict() call — no redundant inference.
    """
    if yolo_model is None:
        return {"mask": None, "box": None}

    results = yolo_model.predict(
        image_bgr,
        imgsz=cfg.yolo_imgsz,
        conf=cfg.yolo_conf,
        iou=cfg.yolo_iou,
        verbose=False,
    )
    r = results[0]
    h, w = image_bgr.shape[:2]

    mask = mask_union_from_result(r, w, h, cfg.yolo_min_mask_area)
    box = _box_from_mask(mask, cfg.yolo_pad, w, h) if mask is not None else None
    if box is None:
        box = _box_from_detections(r, cfg.yolo_pad, w, h)

    return {"mask": mask, "box": box}
