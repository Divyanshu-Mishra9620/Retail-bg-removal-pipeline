import torch
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """
    Configuration for the merged YOLO-seg + BiRefNet pipeline.

    best.pt is a 1-class YOLOv8-seg model. A single inference yields BOTH a
    bounding box (BiRefNet crop hint) AND a semantic mask (fusion prior).
    """

    # ── I/O ──────────────────────────────────────────────────────────────────
    input_dir: str = "./input_images"
    output_dir: str = "./output_clean"
    review_dir: str = "./output_review"
    manifest_path: str = "./pipeline_manifest.jsonl"

    # ── YOLO-seg (best.pt) ─────────────────────────────
    yolo_weights: str = "./best.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    yolo_imgsz: int = 640
    yolo_min_mask_area: int = 500   # min pixels for a YOLO instance mask to count
    yolo_pad: float = 0.15          # fractional padding around crop box

    # ── BiRefNet —──────────────────────────────
    birefnet_model: str = "ZhengPeng7/BiRefNet_dynamic"
    birefnet_size: int = 1024
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True
    allow_model_download: bool = False

    # ── Fusion (semantic-gated matting) ──────────────────────────────────────
    enable_fusion: bool = True
    gate_dilate_frac: float = 0.02   # dilate YOLO mask before gating BiRefNet alpha
    fill_erode_frac: float = 0.06    # erode YOLO mask to find confident interior
    fill_min_alpha: int = 40         # if BiRefNet alpha < this inside YOLO core, fill it
    fill_value: int = 255            # alpha value used when filling a YOLO-confident hole
    fusion_feather: float = 0.8      # light feather to smooth fill seams / gate boundary

    # ── BiRefNet refinement — full chain, max quality ─────
    guided_filter: bool = True
    grabcut_refine: bool = True      
    grabcut_iters: int = 5
    hand_suppression: bool = True
    component_mode: str = "all"      # "all" | "largest"
    solidify: bool = True
    edge_trim: int = 1
    feather: float = 0.8

    # ── YOLO-mask fallback smoothing ──────────────────
    edge_blur: int = 5

    # ── Misc ──────────────────────────────────────────────────────────────────
    save_masks: bool = False
    overwrite: bool = False
    dry_run: bool = False
