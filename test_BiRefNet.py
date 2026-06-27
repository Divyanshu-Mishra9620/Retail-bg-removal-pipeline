import argparse
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class MaskQuality:
    transparent_ratio: float
    foreground_ratio: float
    touches_border_ratio: float
    components: int
    largest_component_ratio: float
    status: str
    reason: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch background removal with YOLO pre-detection + BiRefNet and production-style QA."
    )
    parser.add_argument("--input", default="./input_images", help="Input image folder.")
    parser.add_argument(
        "--output", default="./results_bi_yolo", help="Folder for accepted PNG cutouts."
    )
    parser.add_argument(
        "--review",
        default="./review_birefnet",
        help="Folder for suspicious PNG cutouts.",
    )
    parser.add_argument(
        "--model",
        default="ZhengPeng7/BiRefNet_dynamic",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network checks/downloads. By default cached files are used only.",
    )
    parser.add_argument(
        "--yolo-model",
        default="./best.pt",
        help="Path to your custom YOLO best.pt weights.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--yolo-pad",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--size", type=int, default=1024, help="BiRefNet inference square size."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--no-fp16", action="store_true", help="Disable FP16 inference on CUDA."
    )
    parser.add_argument(
        "--component-mode",
        choices=["all", "largest"],
        default="all",
    )
    parser.add_argument("--edge-trim", type=int, default=1)
    parser.add_argument("--feather", type=float, default=0.8)
    parser.add_argument(
        "--guided-filter", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--grabcut-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use YOLO-crop/alpha-guided GrabCut to rescue same-color products and reject background hands.",
    )
    parser.add_argument(
        "--grabcut-iters",
        type=int,
        default=5,
        help="GrabCut refinement iterations.",
    )
    parser.add_argument(
        "--hand-suppression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat border-connected skin-colored regions as background during GrabCut.",
    )
    parser.add_argument(
        "--solidify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill the final product silhouette after refinement.",
    )
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yolo(yolo_model_path, device):
    try:
        from ultralytics import YOLO

        path = Path(yolo_model_path)
        if not path.exists():
            print(
                f"[WARN] YOLO model not found at {yolo_model_path}. Running without YOLO pre-detection."
            )
            return None
        model = YOLO(str(path))
        print(f"[INFO] YOLO model loaded: {yolo_model_path}")
        return model
    except ImportError:
        print("[WARN] ultralytics not installed. Running without YOLO pre-detection.")
        return None


def yolo_detect(yolo_model, image_rgb_np, conf_threshold, pad_ratio):
    results = yolo_model(image_rgb_np, imgsz=640, conf=conf_threshold, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().tolist()

    h, w = image_rgb_np.shape[:2]
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    return x1, y1, x2, y2


def list_images(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def load_model(model_id, device, use_fp16, local_files_only=True):
    print(f"[INFO] Loading BiRefNet model: {model_id}")
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_files_only
    )
    model.to(device)
    if use_fp16:
        model.half()
    model.eval()
    return model


def build_transform(size):
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def extract_prediction(output):
    if isinstance(output, torch.Tensor):
        pred = output
    elif isinstance(output, dict):
        pred = None
        for key in ("logits", "pred", "out", "last_hidden_state"):
            if key in output and output[key] is not None:
                pred = output[key]
                break
        if pred is None:
            raise KeyError(f"Could not find prediction tensor in keys: {list(output)}")
    elif isinstance(output, (list, tuple)):
        pred = output[-1]
    else:
        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    if isinstance(pred, (list, tuple)):
        pred = pred[-1]

    if pred.ndim == 4:
        pred = pred[:, :1, :, :]
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported prediction shape: {tuple(pred.shape)}")
    return pred


def predict_alpha(model, image_rgb, transform, size, device, use_fp16):
    tensor = transform(image_rgb).unsqueeze(0).to(device)
    if use_fp16:
        tensor = tensor.half()

    with torch.inference_mode():
        output = model(tensor)
        pred = extract_prediction(output)
        pred = F.interpolate(
            pred,
            size=(image_rgb.height, image_rgb.width),
            mode="bilinear",
            align_corners=False,
        )
        alpha = torch.sigmoid(pred)[0, 0].float().cpu().numpy()

    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def guided_filter_alpha(image_rgb, alpha, radius=8, eps=1e-4):
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        return alpha
    guide = (
        cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
        / 255.0
    )
    source = alpha.astype(np.float32) / 255.0
    refined = cv2.ximgproc.guidedFilter(guide=guide, src=source, radius=radius, eps=eps)
    return np.clip(refined * 255.0, 0, 255).astype(np.uint8)


def remove_small_components(alpha, mode="all", min_area_ratio=0.001):
    hard = (alpha >= 16).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hard, connectivity=8
    )
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


def fill_internal_holes(alpha):
    hard_mask = (alpha >= 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(alpha)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    return np.where(filled_mask == 255, np.maximum(alpha, 255), alpha)


def border_connected_skin_mask(image_rgb, alpha):
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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skin, connectivity=8
    )
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


def grabcut_refine_alpha(image_rgb, alpha, args):
    if not args.grabcut_refine:
        return alpha

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
        promote_to_prob_fg = center & ~border & (mask != cv2.GC_FGD)
        mask[promote_to_prob_fg] = cv2.GC_PR_FGD
        inner = np.zeros((h, w), dtype=bool)
        ix1, iy1 = int(w * 0.40), int(h * 0.40)
        ix2, iy2 = int(w * 0.60), int(h * 0.60)
        inner[iy1:iy2, ix1:ix2] = True
        mask[inner & ~border] = cv2.GC_FGD

    if args.hand_suppression:
        skin_bg = border_connected_skin_mask(image_rgb, alpha)
        mask[skin_bg & (alpha < 230)] = cv2.GC_BGD

    try:
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            image_np,
            mask,
            None,
            bg_model,
            fg_model,
            max(1, args.grabcut_iters),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return alpha

    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    refined = np.where(fg, np.maximum(alpha, 220), 0).astype(np.uint8)
    return refined


def solidify_alpha(alpha, min_area_ratio=0.001):
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


def trim_and_feather(alpha, edge_trim=1, feather=0.8):
    if edge_trim <= 0 and feather <= 0:
        return alpha
    hard = (alpha >= 128).astype(np.uint8) * 255
    if edge_trim > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (edge_trim * 2 + 1, edge_trim * 2 + 1)
        )
        hard = cv2.erode(hard, kernel, iterations=1)
        alpha = np.minimum(alpha, hard)
    if feather > 0:
        radius = max(1, int(math.ceil(feather * 2)))
        kernel_size = radius * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), feather)
    return np.clip(alpha, 0, 255).astype(np.uint8)


def refine_alpha(image_rgb, raw_alpha, args):
    alpha = raw_alpha
    if args.guided_filter:
        alpha = guided_filter_alpha(image_rgb, alpha)
    alpha = grabcut_refine_alpha(image_rgb, alpha, args)
    alpha = remove_small_components(alpha, mode=args.component_mode)
    alpha = fill_internal_holes(alpha)
    if args.solidify:
        alpha = solidify_alpha(alpha)
    alpha = trim_and_feather(alpha, edge_trim=args.edge_trim, feather=args.feather)
    return alpha


def assess_mask(alpha):
    hard = alpha >= 128
    h, w = hard.shape
    area = h * w
    foreground_ratio = float(hard.mean())
    transparent_ratio = 1.0 - foreground_ratio

    border = np.concatenate([hard[0, :], hard[-1, :], hard[:, 0], hard[:, -1]])
    touches_border_ratio = float(border.mean())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hard.astype(np.uint8), connectivity=8
    )
    components = max(0, num_labels - 1)
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if components else 0
    largest_component_ratio = largest_area / area if area else 0.0

    status = "success"
    reason = "ok"

    if foreground_ratio < 0.015:
        status, reason = "review", "almost everything became transparent"
    elif foreground_ratio > 0.995:
        status, reason = "review", "almost nothing was removed"
    elif touches_border_ratio > 0.92:
        status, reason = "review", "foreground touches most image borders"
    elif components > 25 and largest_component_ratio < 0.15:
        status, reason = "review", "mask is fragmented"

    return MaskQuality(
        transparent_ratio=round(transparent_ratio, 4),
        foreground_ratio=round(foreground_ratio, 4),
        touches_border_ratio=round(touches_border_ratio, 4),
        components=components,
        largest_component_ratio=round(largest_component_ratio, 4),
        status=status,
        reason=reason,
    )


def save_cutout(original_rgba, alpha, destination, background=(0, 0, 0)):
    """
    Save a true background-removed RGBA PNG (rebuild from scratch).

    The old version did original_rgba.copy() + putalpha(alpha), which only
    overwrites the alpha channel and leaves the ORIGINAL background pixels in
    the RGB channels. Those pixels are invisible while alpha is honoured, but
    any operation that drops alpha -- e.g. Image.open(png).convert("RGB") --
    recovers them intact, so the "removed" background reappears. That is a real
    data leak for a training dataset.

    This version builds a brand-new canvas:
      * RGB starts as a uniform neutral matte (background, default black);
      * original colours are copied back ONLY on foreground pixels (alpha > 0),
        which keeps soft/anti-aliased edges and feathering intact;
      * everywhere the model marked transparent, RGB is the matte -- the
        original background is physically destroyed, not merely hidden.

    Only the final export changes; alpha (incl. feather/soft edges) is preserved
    bit-for-bit, so QA, YOLO and BiRefNet behaviour are unaffected.
    """
    # Original colours as a contiguous (H, W, 3) uint8 array. convert("RGB")
    # also normalises odd source modes (P/LA/RGB).
    rgb = np.asarray(original_rgba.convert("RGB"))
    src_h, src_w = rgb.shape[:2]

    # Normalise alpha to a 2-D uint8 map matching the source resolution.
    alpha = np.asarray(alpha)
    if alpha.ndim != 2:
        alpha = alpha.reshape(alpha.shape[0], alpha.shape[1])
    if alpha.dtype != np.uint8:
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    if alpha.shape != (src_h, src_w):
        raise ValueError(
            f"alpha shape {alpha.shape} does not match image (H, W)=({src_h}, {src_w})"
        )

    # Fresh canvas: fully transparent + neutral matte. Starting from zeros means
    # every transparent pixel is black/0 (no residual background) by default.
    out = np.zeros((src_h, src_w, 4), dtype=np.uint8)
    if any(background):
        out[..., 0], out[..., 1], out[..., 2] = background

    # Copy ORIGINAL RGB only where the product exists. out[..., :3] is a view
    # into out, so this boolean-indexed assignment writes through to the canvas.
    # Background pixels keep the matte; their original colour is gone.
    foreground = alpha > 0
    out[..., :3][foreground] = rgb[foreground]

    # Straight (non-premultiplied) alpha preserved exactly -- soft edges,
    # feather and semi-transparent boundaries are untouched.
    out[..., 3] = alpha

    Image.fromarray(out, mode="RGBA").save(destination, "PNG", optimize=True)


def verify_hidden_background_removed(png_path, background=(0, 0, 0)):
    """
    Audit a saved PNG for recoverable background RGB in transparent regions.

    A "clean" cutout has the matte colour (background) in every fully
    transparent pixel, so convert("RGB") cannot resurrect the original scene.
    A leaky cutout (the old putalpha behaviour) keeps arbitrary original colours
    behind alpha == 0.

    Returns a dict: clean, mode, has_alpha, transparent_pixels, leaked_pixels,
    leak_ratio, max_leak_value.
    """
    img = Image.open(png_path)
    mode = img.mode

    # No alpha channel -> nothing is "hidden"; the file is a flat opaque image.
    if mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return {
            "clean": True, "mode": mode, "has_alpha": False,
            "transparent_pixels": 0, "leaked_pixels": 0,
            "leak_ratio": 0.0, "max_leak_value": 0,
        }

    arr = np.asarray(img.convert("RGBA"))
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]

    matte = np.array(background, dtype=np.int16)
    transparent = alpha == 0
    n_transparent = int(transparent.sum())

    if n_transparent == 0:
        return {
            "clean": True, "mode": mode, "has_alpha": True,
            "transparent_pixels": 0, "leaked_pixels": 0,
            "leak_ratio": 0.0, "max_leak_value": 0,
        }

    # Per-pixel deviation from the matte inside the transparent region.
    deviation = np.abs(rgb[transparent] - matte).max(axis=1)
    leaked = deviation > 0
    n_leaked = int(leaked.sum())

    return {
        "clean": n_leaked == 0, "mode": mode, "has_alpha": True,
        "transparent_pixels": n_transparent, "leaked_pixels": n_leaked,
        "leak_ratio": round(n_leaked / n_transparent, 6),
        "max_leak_value": int(deviation.max()) if n_leaked else 0,
    }


def process_one(path, birefnet_model, transform, yolo_model, args):
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            original_rgba = image.convert("RGBA")
            image_rgb = original_rgba.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Corrupted or unreadable image file: {e}")

    img_np = np.array(image_rgb)
    orig_w, orig_h = image_rgb.size

    yolo_box = None
    routing = "birefnet_only"

    if yolo_model is not None:
        yolo_box = yolo_detect(yolo_model, img_np, args.yolo_conf, args.yolo_pad)
        if yolo_box is not None:
            routing = "yolo_guided"

    if yolo_box is not None:
        x1, y1, x2, y2 = yolo_box
        crop_rgb = image_rgb.crop((x1, y1, x2, y2))
        raw_alpha_crop = predict_alpha(
            birefnet_model,
            crop_rgb,
            transform,
            args.size,
            args.device,
            use_fp16=(args.device == "cuda" and not args.no_fp16),
        )
        refined_crop = refine_alpha(crop_rgb, raw_alpha_crop, args)

        full_alpha = np.zeros((orig_h, orig_w), dtype=np.uint8)
        full_alpha[y1:y2, x1:x2] = refined_crop
    else:
        raw_alpha = predict_alpha(
            birefnet_model,
            image_rgb,
            transform,
            args.size,
            args.device,
            use_fp16=(args.device == "cuda" and not args.no_fp16),
        )
        full_alpha = refine_alpha(image_rgb, raw_alpha, args)

    quality = quality = assess_mask(full_alpha)

    target_root = (
        Path(args.output) if quality.status == "success" else Path(args.review)
    )
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / f"{path.stem}.png"
    save_cutout(original_rgba, full_alpha, destination)

    if args.save_masks:
        Image.fromarray(full_alpha, mode="L").save(
            target_root / f"{path.stem}_mask.png", "PNG"
        )

    return destination, quality, routing


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.review).mkdir(parents=True, exist_ok=True)

    explicit_results_folder = Path("./results_bi_yolo")
    explicit_results_folder.mkdir(parents=True, exist_ok=True)

    image_files = list_images(args.input)
    print(f"Found {len(image_files)} images to process.")
    if not image_files:
        return

    use_fp16 = args.device == "cuda" and not args.no_fp16
    birefnet_model = load_model(
        args.model,
        args.device,
        use_fp16,
        local_files_only=not args.allow_model_download,
    )
    transform = build_transform(args.size)
    yolo_model = load_yolo(args.yolo_model, args.device)

    manifest_path = Path(args.output).parent / "birefnet_manifest.jsonl"
    success_count = 0
    review_count = 0
    error_count = 0
    yolo_guided_count = 0
    fallback_count = 0
    skipped_count = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for path in tqdm(image_files, desc="Processing", unit="img"):
            output_file = Path(args.output) / f"{path.stem}.png"
            review_file = Path(args.review) / f"{path.stem}.png"
            explicit_check_file = explicit_results_folder / f"{path.stem}.png"

            if not args.overwrite and (
                output_file.exists()
                or review_file.exists()
                or explicit_check_file.exists()
            ):
                skipped_count += 1
                continue

            try:
                destination, quality, routing = process_one(
                    path, birefnet_model, transform, yolo_model, args
                )
                if quality.status == "success":
                    success_count += 1
                else:
                    review_count += 1

                if routing == "yolo_guided":
                    yolo_guided_count += 1
                else:
                    fallback_count += 1

                manifest.write(
                    json.dumps(
                        {
                            "input": str(path),
                            "output": str(destination),
                            "routing": routing,
                            "quality": asdict(quality),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                manifest.flush()

            except Exception as exc:
                error_count += 1
                tqdm.write(f"[ERROR] {path.name}: {exc}")
                traceback.print_exc()
                manifest.write(
                    json.dumps(
                        {"input": str(path), "status": "error", "error": str(exc)},
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                manifest.flush()

    print("\n" + "=" * 45)
    print(" PIPELINE SUMMARY")
    print("=" * 45)
    print(f" Skipped (Already Done) : {skipped_count}")
    print(f" Newly Accepted         : {success_count}")
    print(f" Needs review           : {review_count}")
    print(f" Errors                 : {error_count}")
    print(f" YOLO-guided crops      : {yolo_guided_count}")
    print(f" Full-image fallback    : {fallback_count}")
    print(f" Manifest               : {manifest_path}")
    print("=" * 45)


if __name__ == "__main__":
    main()
