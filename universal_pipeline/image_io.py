"""
Unicode-safe decode (np.fromfile + cv2.imdecode) so non-ASCII filenames
(e.g. Spanish category names) load correctly on Windows.
"""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(source_dir: str) -> list[Path]:
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"Input folder not found: {source}")
    return sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_bgr(path: Path) -> np.ndarray | None:
    """Unicode-safe BGR decode. Returns None for empty/corrupt files."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_original_to_review(bgr: np.ndarray, dest: Path) -> None:
    """Copy an unprocessable original into the review folder (Unicode-safe)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", bgr)
    if ok:
        encoded.tofile(str(dest))
