"""
Model loading (orchestration only).

The actual inference/refinement functions live in the original scripts and are
re-exported here from `originals`. This module only adds the production concern
of loading each model ONCE and caching it for the whole batch.
"""

from pathlib import Path

# Re-export the exact original loaders/builders (no reimplementation).
from .originals import build_transform, load_birefnet  # noqa: F401

_yolo_cache = None


def load_yolo(weights: str):
    """
    Load the YOLO-seg model once and cache it. Returns None (gracefully) if
    ultralytics is missing or the weights file is absent — the pipeline then
    runs BiRefNet on full images (test_BiRefNet.py's own no-YOLO behavior).
    """
    global _yolo_cache
    if _yolo_cache is not None:
        return _yolo_cache
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[WARN] ultralytics not installed — YOLO disabled (BiRefNet full-image mode).")
        return None
    p = Path(weights)
    if not p.exists():
        print(f"[WARN] YOLO weights not found at '{weights}' — BiRefNet full-image mode.")
        return None
    _yolo_cache = YOLO(str(p))
    _yolo_cache.overrides["verbose"] = False
    print(f"[INFO] YOLO loaded: {weights}  (task={_yolo_cache.task}, classes={list(_yolo_cache.names.values())})")
    return _yolo_cache
