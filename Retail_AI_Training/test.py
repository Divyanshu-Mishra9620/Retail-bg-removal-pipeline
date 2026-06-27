import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(source_dir):
    source = Path(source_dir)
    return sorted(
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_bgr(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_rgba(path, rgba):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, "PNG")


def smooth_alpha(alpha, blur_size=5):
    _, alpha = cv2.threshold(alpha, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)

    if blur_size > 0:
        blur_size = max(1, blur_size | 1)
        alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
    return alpha


def mask_from_result(result, width, height, min_mask_area):
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


def process_one(model, image_path, output_path, args):
    bgr = read_bgr(image_path)
    if bgr is None:
        return {"status": "failed", "reason": "unreadable"}

    height, width = bgr.shape[:2]
    results = model.predict(
        bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    alpha = mask_from_result(results[0], width, height, args.min_mask_area)
    if alpha is None:
        return {"status": "failed", "reason": "no_mask_detected"}

    alpha = smooth_alpha(alpha, args.edge_blur)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Leak-proof compose: np.dstack([rgb, alpha]) keeps the FULL original RGB
    # (incl. background) in the colour channels — only hidden by alpha, and
    # recovered by any alpha-dropping op (JPG export / convert("RGB")). Instead
    # start from a black canvas and copy original RGB only where the product is
    # (alpha > 0); transparent pixels are truly zeroed. Alpha (incl. the soft
    # blurred edge) is preserved exactly.
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    fg = alpha > 0
    rgba[..., :3][fg] = rgb[fg]
    rgba[..., 3] = alpha
    save_rgba(output_path, rgba)

    return {
        "status": "processed",
        "reason": "ok",
        "output": str(output_path),
        "mask_pixels": int(np.count_nonzero(alpha > 127)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process refine2_review_yolo images into transparent PNG cutouts."
    )
    parser.add_argument("--source", default=r"d:\bg_remove\Retail_AI_Training\input_test_images")
    parser.add_argument(
        "--output", default=r"d:\bg_remove\Retail_AI_Training\refined_review_output"
    )
    parser.add_argument(
        "--review", default=r"d:\bg_remove\Retail_AI_Training\output_refined_review"
    )
    parser.add_argument("--weights", default=r"d:\bg_remove\Retail_AI_Training\best.pt")
    parser.add_argument(
        "--report", default=r"d:\bg_remove\Retail_AI_Training\output_refined_report.json"
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--edge-blur", type=int, default=5)
    parser.add_argument("--min-mask-area", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    review = Path(args.review)
    weights = Path(args.weights)

    if not source.exists():
        raise SystemExit(f"Source folder does not exist: {source}")
    if not weights.exists():
        raise SystemExit(f"YOLO weights do not exist: {weights}")

    output.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    images = iter_images(source)
    records = []
    processed = skipped = failed = 0

    for image_path in tqdm(images, desc="Processing refine2"):
        output_path = output / f"{image_path.stem}_nobg.png"
        if output_path.exists() and not args.overwrite:
            records.append(
                {
                    "input": str(image_path),
                    "output": str(output_path),
                    "status": "skipped",
                    "reason": "already_exists",
                }
            )
            skipped += 1
            continue

        result = process_one(model, image_path, output_path, args)
        record = {"input": str(image_path), **result}
        records.append(record)

        if result["status"] == "processed":
            processed += 1
        else:
            failed += 1
            review_target = review / image_path.name
            bgr = read_bgr(image_path)
            if bgr is not None:
                ok, encoded = cv2.imencode(image_path.suffix, bgr)
                if ok:
                    encoded.tofile(str(review_target))

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "output": str(output),
        "review": str(review),
        "weights": str(weights),
        "settings": vars(args),
        "summary": {
            "total": len(images),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
        },
        "records": records,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"Total:     {len(images)}")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Output:    {output}")
    print(f"Review:    {review}")
    print(f"Report:    {args.report}")


if __name__ == "__main__":
    main()
