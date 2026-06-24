"""
Alpha mask refinement pipeline.

Pipeline order (called by refine_alpha):
  1. guided_filter_alpha      — edge-aware alpha smoothing
  2. grabcut_refine_alpha     — boundary lock (includes skin/hand suppression)
  3. remove_small_components  — drop noise blobs
  4. fill_internal_holes      — plug holes inside the silhouette
  5. solidify_alpha           — morphological close + contour fill
  6. trim_and_feather         — thin border erosion + Gaussian blur
"""

import math

import cv2
import numpy as np
from PIL import Image


# ── 1. Guided filter ─────────────────────────────────────────────────────────

def guided_filter_alpha(image_rgb: Image.Image, alpha: np.ndarray, radius: int = 8, eps: float = 1e-4) -> np.ndarray:
    if not (hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")):
        return alpha
    guide = (
        cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    )
    src = alpha.astype(np.float32) / 255.0
    refined = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=radius, eps=eps)
    return np.clip(refined * 255.0, 0, 255).astype(np.uint8)


# ── 2. GrabCut refinement ────────────────────────────────────────────────────

def border_connected_skin_mask(image_rgb: Image.Image, alpha: np.ndarray) -> np.ndarray:
    """
    Detect skin-coloured regions that touch the image border and have low alpha
    overlap — these are background hands that leaked into the foreground mask.
    Returns a boolean mask (True = suppress these pixels).
    """
    image_np = np.asarray(image_rgb)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)

    hsv_skin = (
        (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 25)
        & (hsv[:, :, 1] <= 230)
        & (hsv[:, :, 2] >= 55)
    )
    ycrcb_skin = (
        (ycrcb[:, :, 1] >= 133)
        & (ycrcb[:, :, 1] <= 180)
        & (ycrcb[:, :, 2] >= 70)
        & (ycrcb[:, :, 2] <= 145)
    )
    skin = (hsv_skin & ycrcb_skin).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(skin, dtype=bool)

    h, w = skin.shape
    suppress = np.zeros_like(skin, dtype=bool)
    strong_alpha = alpha >= 220

    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        if area < max(64, int(h * w * 0.001)):
            continue
        touches_border = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        if not touches_border:
            continue
        component = labels == label_id
        strong_overlap = float((component & strong_alpha).sum()) / float(area)
        if strong_overlap < 0.45:
            suppress |= component

    return suppress


def grabcut_refine_alpha(
    image_rgb: Image.Image,
    alpha: np.ndarray,
    grabcut_iters: int = 5,
    hand_suppression: bool = True,
) -> np.ndarray:
    image_np = np.asarray(image_rgb)
    h, w = alpha.shape
    if h < 20 or w < 20:
        return alpha

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[alpha <= 8] = cv2.GC_BGD
    mask[alpha >= 160] = cv2.GC_PR_FGD
    mask[alpha >= 225] = cv2.GC_FGD

    margin = max(2, int(round(min(h, w) * 0.035)))
    border = np.zeros((h, w), dtype=bool)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    mask[border & (alpha < 180)] = cv2.GC_BGD

    if np.count_nonzero(mask == cv2.GC_FGD) < max(32, int(h * w * 0.003)):
        cx1, cy1 = int(w * 0.30), int(h * 0.30)
        cx2, cy2 = int(w * 0.70), int(h * 0.70)
        center = np.zeros((h, w), dtype=bool)
        center[cy1:cy2, cx1:cx2] = True
        mask[center & ~border & (mask != cv2.GC_FGD)] = cv2.GC_PR_FGD
        inner = np.zeros((h, w), dtype=bool)
        inner[int(h * 0.40):int(h * 0.60), int(w * 0.40):int(w * 0.60)] = True
        mask[inner & ~border] = cv2.GC_FGD

    if hand_suppression:
        skin_bg = border_connected_skin_mask(image_rgb, alpha)
        mask[skin_bg & (alpha < 230)] = cv2.GC_BGD

    try:
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image_np, mask, None, bg_model, fg_model, max(1, grabcut_iters), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return alpha

    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    return np.where(fg, np.maximum(alpha, 220), 0).astype(np.uint8)


# ── 3. Remove small components ───────────────────────────────────────────────

def remove_small_components(alpha: np.ndarray, mode: str = "all", min_area_ratio: float = 0.001) -> np.ndarray:
    hard = (alpha >= 16).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if num_labels <= 1:
        return alpha

    image_area = alpha.shape[0] * alpha.shape[1]
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(num_labels, dtype=bool)

    if mode == "largest":
        keep[1 + int(np.argmax(areas))] = True
    else:
        min_area = max(16, int(image_area * min_area_ratio))
        for label_id, area in enumerate(areas, start=1):
            if area >= min_area:
                keep[label_id] = True

    return np.where(keep[labels], alpha, 0).astype(np.uint8)


# ── 4. Fill internal holes ───────────────────────────────────────────────────

def fill_internal_holes(alpha: np.ndarray) -> np.ndarray:
    hard_mask = (alpha >= 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(alpha)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return np.where(filled == 255, np.maximum(alpha, 255), alpha)


# ── 5. Solidify ──────────────────────────────────────────────────────────────

def solidify_alpha(alpha: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    hard = (alpha >= 128).astype(np.uint8) * 255
    if not np.any(hard):
        return alpha

    h, w = hard.shape
    close_px = max(3, min(21, int(round(min(h, w) * 0.018))))
    if close_px % 2 == 0:
        close_px += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return alpha

    image_area = h * w
    largest_area = max(cv2.contourArea(c) for c in contours)
    min_area = max(32, image_area * min_area_ratio, largest_area * 0.08)
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not kept:
        kept = [max(contours, key=cv2.contourArea)]

    filled = np.zeros_like(hard)
    cv2.drawContours(filled, kept, -1, 255, thickness=cv2.FILLED)
    return np.where(filled == 255, np.maximum(alpha, 255), 0).astype(np.uint8)


# ── 6. Trim and feather ──────────────────────────────────────────────────────

def trim_and_feather(alpha: np.ndarray, edge_trim: int = 1, feather: float = 0.8) -> np.ndarray:
    if edge_trim <= 0 and feather <= 0:
        return alpha
    hard = (alpha >= 128).astype(np.uint8) * 255
    if edge_trim > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_trim * 2 + 1, edge_trim * 2 + 1))
        hard = cv2.erode(hard, kernel, iterations=1)
        alpha = np.minimum(alpha, hard)
    if feather > 0:
        radius = max(1, int(math.ceil(feather * 2)))
        kernel_size = radius * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), feather)
    return np.clip(alpha, 0, 255).astype(np.uint8)


# ── YOLO-mask smoothing (fallback path) ──────────────────────────────────────

def smooth_alpha(alpha: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """
    Fast mask cleanup 
    Used on the raw YOLO-seg mask when it is the chosen alpha (fallback path).

    Binary threshold -> morphological close+open (5x5 ellipse) -> Gaussian blur.
    """
    _, alpha = cv2.threshold(alpha, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    if blur_size > 0:
        blur_size = max(1, blur_size | 1)
        alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
    return alpha


# ── Orchestrated refinement ──────────────────────────────────────────────────

def refine_alpha(image_rgb: Image.Image, raw_alpha: np.ndarray, cfg) -> np.ndarray:
    """
    Run the full refinement pipeline in order.
    cfg is a SystemConfig instance.
    """
    alpha = raw_alpha
    if cfg.guided_filter:
        alpha = guided_filter_alpha(image_rgb, alpha)
    if cfg.grabcut_refine:
        alpha = grabcut_refine_alpha(image_rgb, alpha, cfg.grabcut_iters, cfg.hand_suppression)
    alpha = remove_small_components(alpha, mode=cfg.component_mode)
    alpha = fill_internal_holes(alpha)
    if cfg.solidify:
        alpha = solidify_alpha(alpha)
    alpha = trim_and_feather(alpha, edge_trim=cfg.edge_trim, feather=cfg.feather)
    return alpha
