import torch
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """
    Configuration for the merged pipeline.

    All values are the ORIGINAL script defaults so each flow reproduces its
    source exactly. The attribute names in the BiRefNet section deliberately
    match the argparse names that test_BiRefNet.py's refine_alpha() /
    grabcut_refine_alpha() read off `args` — so this object can be passed
    straight through to those original functions as their `args`.

    Sources:
      - test_BiRefNet.py            -> BiRefNet matting flow (primary)
      - Retail_AI_Training/test.py  -> YOLO-seg flow (fallback)
    """

    # ── I/O ──────────────────────────────────────────────────────────────────
    input_dir: str = "./input_images"
    output_dir: str = "./output_clean"
    review_dir: str = "./output_review"
    manifest_path: str = "./pipeline_manifest.jsonl"

    # ── Shared model weights ──────────────────────────────────────────────────
    yolo_weights: str = "./best.pt"

    # ── BiRefNet flow (primary) — test_BiRefNet.py defaults ──────────────────
    birefnet_model: str = "ZhengPeng7/BiRefNet_dynamic"
    birefnet_size: int = 1024                # test_BiRefNet --size
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True                    # test_BiRefNet: fp16 unless --no-fp16
    allow_model_download: bool = False
    yolo_conf: float = 0.15                  # test_BiRefNet --yolo-conf (crop-box detection)
    yolo_pad: float = 0.15                   # test_BiRefNet --yolo-pad
    # refinement chain — names match test_BiRefNet.py argparse (read as args.* by refine_alpha)
    guided_filter: bool = True
    grabcut_refine: bool = True
    grabcut_iters: int = 5
    hand_suppression: bool = True
    component_mode: str = "all"
    solidify: bool = True
    edge_trim: int = 1
    feather: float = 0.8

    # ── YOLO-seg flow (fallback) — Retail_AI_Training/test.py defaults ───────
    seg_conf: float = 0.25                   # test.py --conf
    seg_iou: float = 0.7                     # test.py --iou
    seg_imgsz: int = 640                     # test.py --imgsz
    seg_min_mask_area: int = 500             # test.py --min-mask-area
    edge_blur: int = 5                       # test.py --edge-blur (smooth_alpha)

    # ── Misc (orchestration only) ─────────────────────────────────────────────
    save_masks: bool = False
    overwrite: bool = False
    dry_run: bool = False
