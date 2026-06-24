"""
Universal Retail Product Background Removal Pipeline
=====================================================
best.pt is a 1-class YOLOv8-seg model. A SINGLE inference per image yields both
a semantic mask (fusion prior) and a crop box (BiRefNet hint). BiRefNet refines
the edges inside the product region; the YOLO mask gates out false positives and
fills low-contrast product BiRefNet drops ("semantic-gated matting").

No image is rejected for a single detector's miss — every stage has a fallback.

Usage
-----
  python run_pipeline.py --input ./input_images --output ./output_clean
  python run_pipeline.py --input ./input_images --dry-run
  python run_pipeline.py --input ./input_images --allow-model-download   # first run
  python run_pipeline.py --input ./input_images --no-fusion              # disable fusion
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

from universal_pipeline.config import SystemConfig
from universal_pipeline.image_io import iter_images
from universal_pipeline.models import build_transform, load_birefnet, load_yolo
from universal_pipeline.orchestrator import process_one_image


def parse_args() -> SystemConfig:
    p = argparse.ArgumentParser(
        description="Universal retail product background removal (YOLO-seg + BiRefNet).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--input", default="./input_images", help="Folder of input product images.")
    p.add_argument("--output", default="./output_clean", help="Folder for accepted PNG cutouts.")
    p.add_argument("--review", default="./output_review", help="Folder for QA-flagged / fallback cutouts.")
    p.add_argument("--manifest", default="./pipeline_manifest.jsonl", help="Path for JSONL run manifest.")

    # ── YOLO-seg ──────────────────────────────────────────────────────────────
    p.add_argument("--yolo-model", default="./best.pt", help="Path to best.pt (YOLO-seg weights).")
    p.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    p.add_argument("--yolo-iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--min-mask-area", type=int, default=500, help="Min pixels for a YOLO mask to count.")
    p.add_argument("--yolo-pad", type=float, default=0.15, help="Fractional padding around crop box.")

    # ── BiRefNet ──────────────────────────────────────────────────────────────
    p.add_argument("--model", default="ZhengPeng7/BiRefNet_dynamic", help="HuggingFace BiRefNet model ID.")
    p.add_argument("--size", type=int, default=1024, help="BiRefNet inference square size.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16 inference on CUDA.")
    p.add_argument("--allow-model-download", action="store_true", help="Allow HuggingFace download (first run).")

    # ── Fusion ────────────────────────────────────────────────────────────────
    p.add_argument("--fusion", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable semantic-gated matting (gate + fill). --no-fusion to disable.")
    p.add_argument("--gate-dilate-frac", type=float, default=0.02)
    p.add_argument("--fill-erode-frac", type=float, default=0.06)
    p.add_argument("--fill-min-alpha", type=int, default=40)
    p.add_argument("--fusion-feather", type=float, default=0.8)

    # ── BiRefNet refinement (full chain — max quality) ───────────────────────
    p.add_argument("--guided-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grabcut-refine", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grabcut-iters", type=int, default=5)
    p.add_argument("--hand-suppression", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--component-mode", choices=["all", "largest"], default="all")
    p.add_argument("--solidify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--edge-trim", type=int, default=1)
    p.add_argument("--feather", type=float, default=0.8)
    p.add_argument("--edge-blur", type=int, default=5, help="Smoothing for YOLO-mask fallback path.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Re-process images that already have output.")
    p.add_argument("--dry-run", action="store_true", help="Analyse and report without writing files.")

    a = p.parse_args()

    return SystemConfig(
        input_dir=a.input, output_dir=a.output, review_dir=a.review, manifest_path=a.manifest,
        yolo_weights=a.yolo_model, yolo_conf=a.yolo_conf, yolo_iou=a.yolo_iou,
        yolo_imgsz=a.yolo_imgsz, yolo_min_mask_area=a.min_mask_area, yolo_pad=a.yolo_pad,
        birefnet_model=a.model, birefnet_size=a.size, device=a.device,
        use_fp16=not a.no_fp16, allow_model_download=a.allow_model_download,
        enable_fusion=a.fusion, gate_dilate_frac=a.gate_dilate_frac,
        fill_erode_frac=a.fill_erode_frac, fill_min_alpha=a.fill_min_alpha,
        fusion_feather=a.fusion_feather,
        guided_filter=a.guided_filter, grabcut_refine=a.grabcut_refine,
        grabcut_iters=a.grabcut_iters, hand_suppression=a.hand_suppression,
        component_mode=a.component_mode, solidify=a.solidify,
        edge_trim=a.edge_trim, feather=a.feather, edge_blur=a.edge_blur,
        save_masks=a.save_masks, overwrite=a.overwrite, dry_run=a.dry_run,
    )


def main():
    cfg = parse_args()

    if not cfg.dry_run:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.review_dir).mkdir(parents=True, exist_ok=True)

    try:
        image_files = iter_images(cfg.input_dir)
    except FileNotFoundError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*64}")
    print(f"  Universal Background Removal — YOLO-seg + BiRefNet")
    print(f"{'='*64}")
    print(f"  Input        : {cfg.input_dir}  ({len(image_files)} images)")
    print(f"  Output       : {cfg.output_dir}")
    print(f"  Review       : {cfg.review_dir}")
    print(f"  YOLO-seg     : {cfg.yolo_weights}  (conf={cfg.yolo_conf}, iou={cfg.yolo_iou})")
    print(f"  BiRefNet     : {cfg.birefnet_model}  (size={cfg.birefnet_size})")
    print(f"  Fusion       : {cfg.enable_fusion}   GrabCut: {cfg.grabcut_refine}")
    print(f"  Device       : {cfg.device}  fp16={cfg.use_fp16 and cfg.device == 'cuda'}")
    print(f"  Dry-run      : {cfg.dry_run}")
    print(f"{'='*64}\n")

    if not image_files:
        print("[WARN] No images found. Exiting.")
        return

    # ── Load both models ONCE ─────────────────────────────────────────────────
    yolo_model = load_yolo(cfg.yolo_weights)
    if yolo_model is None:
        print("[WARN] Running WITHOUT YOLO-seg — every image uses full-image BiRefNet.")

    try:
        birefnet_model = load_birefnet(
            cfg.birefnet_model, cfg.device, use_fp16=cfg.use_fp16,
            local_files_only=not cfg.allow_model_download,
        )
    except Exception as e:
        print(f"[FATAL] Failed to load BiRefNet: {e}")
        print("        If this is a first run, add --allow-model-download")
        sys.exit(1)

    transform = build_transform(cfg.birefnet_size)

    counts = {
        "skipped": 0, "success": 0, "review": 0, "error": 0,
        # routing
        "fused": 0, "birefnet_crop": 0, "fullimg_fallback": 0,
        "yolo_mask_only": 0, "yolo_mask_alt": 0, "birefnet_alt": 0, "none": 0,
    }

    manifest_path = Path(cfg.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for path in tqdm(image_files, desc="Processing", unit="img"):

            if not cfg.overwrite and not cfg.dry_run:
                out_file = Path(cfg.output_dir) / f"{path.stem}.png"
                rev_file = Path(cfg.review_dir) / f"{path.stem}.png"
                if out_file.exists() or rev_file.exists():
                    counts["skipped"] += 1
                    continue

            try:
                result = process_one_image(path, yolo_model, birefnet_model, transform, cfg)
            except Exception as exc:
                counts["error"] += 1
                tqdm.write(f"  [ERROR] {path.name}: {exc}")
                traceback.print_exc()
                manifest.write(json.dumps({"input": str(path), "status": "error", "reason": str(exc)}, ensure_ascii=True) + "\n")
                manifest.flush()
                continue

            status = result.get("status", "error")
            counts[status] = counts.get(status, 0) + 1
            routing = result.get("routing")
            if routing in counts:
                counts[routing] += 1

            if status == "review":
                tqdm.write(f"  [REVIEW] {path.name} -> {result.get('reason', '')} ({routing})")

            manifest.write(json.dumps(result, ensure_ascii=True) + "\n")
            manifest.flush()

    print(f"\n{'='*64}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*64}")
    print(f"  Skipped (already done) : {counts['skipped']}")
    print(f"  Accepted (clean PNG)   : {counts['success']}")
    print(f"  Needs review           : {counts['review']}")
    print(f"  Errors                 : {counts['error']}")
    print(f"  ── Routing breakdown ──────────────────────────")
    print(f"  Fused (YOLO+BiRefNet)  : {counts['fused']}")
    print(f"  BiRefNet on crop       : {counts['birefnet_crop']}")
    print(f"  Full-image fallback    : {counts['fullimg_fallback']}")
    print(f"  YOLO mask (best/alt)   : {counts['yolo_mask_only'] + counts['yolo_mask_alt']}")
    print(f"  BiRefNet alt (fallback): {counts['birefnet_alt']}")
    print(f"  Manifest               : {cfg.manifest_path}")
    print(f"{'='*64}\n")

    if cfg.dry_run:
        print("  [DRY-RUN] No files were written.")


if __name__ == "__main__":
    main()
