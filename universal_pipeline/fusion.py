"""
Semantic-gated matting — fuse the YOLO-seg mask with the BiRefNet alpha.

The two models are complementary:
  - YOLO-seg (best.pt) knows WHERE the product is (semantic, trained on retail
    products), but its edges are coarse (160x160 prototype masks upsampled).
  - BiRefNet knows the EXACT EDGE (1024^2 matting, guided filter, feather), but
    it is generic saliency — it can chase glare or drop low-contrast product.

Fusion takes the best of both:
  GATE  — zero any BiRefNet alpha that falls OUTSIDE the (dilated) YOLO mask.
          Kills BiRefNet false positives (shelf edges, hands, background items).
          Dilation gives BiRefNet's finer edge room so the coarse YOLO boundary
          never clips the real product.
  FILL  — where the (eroded, confident) YOLO interior says "product" but
          BiRefNet alpha is near-zero, restore it. Rescues low-contrast product
          BiRefNet drops (brown meat, white-on-white).

Result: BiRefNet edge quality, YOLO semantic correctness.
"""

import cv2
import numpy as np

from .postprocessing import trim_and_feather


def _frac_kernel(shape, frac: float):
    """Odd elliptical kernel sized as a fraction of the image's smaller side."""
    h, w = shape
    k = max(1, int(round(min(h, w) * frac)))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fuse_alpha(birefnet_alpha: np.ndarray, yolo_mask: np.ndarray, cfg) -> np.ndarray:
    """
    Combine a refined full-size BiRefNet alpha with the YOLO-seg semantic mask.

    birefnet_alpha : uint8 (H, W) — refined BiRefNet alpha, back-projected to full size
    yolo_mask      : uint8 (H, W) — YOLO union mask (0/255)
    cfg            : SystemConfig

    Returns a fused uint8 alpha (H, W).
    """
    h, w = birefnet_alpha.shape
    m_bin = (yolo_mask > 127).astype(np.uint8) * 255

    # ── GATE: keep BiRefNet alpha only inside the dilated YOLO region ────────
    gate = cv2.dilate(m_bin, _frac_kernel((h, w), cfg.gate_dilate_frac), iterations=1)
    fused = np.where(gate > 0, birefnet_alpha, 0).astype(np.uint8)

    # ── FILL: restore confident YOLO interior that BiRefNet dropped ──────────
    core = cv2.erode(m_bin, _frac_kernel((h, w), cfg.fill_erode_frac), iterations=1)
    fill_region = (core > 0) & (fused < cfg.fill_min_alpha)
    fused[fill_region] = cfg.fill_value

    # ── Light feather to smooth fill seams and the gate boundary ─────────────
    if cfg.fusion_feather > 0:
        fused = trim_and_feather(fused, edge_trim=0, feather=cfg.fusion_feather)

    return fused
