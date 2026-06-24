# Retail Product Background Removal Pipeline

A production-grade background removal pipeline for retail product images. It fuses two complementary models — **YOLOv8-seg** (semantic mask) and **BiRefNet** (detail-preserving alpha matting) — to produce clean, transparent-background PNG cutouts at scale.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture Deep-Dive](#architecture-deep-dive)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Models Setup](#models-setup)
6. [Usage](#usage)
7. [CLI Reference](#cli-reference)
8. [Output & Manifest](#output--manifest)
9. [Module Reference](#module-reference)
10. [Design Decisions](#design-decisions)

---

## How It Works

The pipeline runs every input image through a **4-stage process**:

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — YOLO-seg (best.pt)                           │
│  Single inference → semantic union mask + padded bbox   │
└──────────────────────┬──────────────────────────────────┘
                       │ box + mask
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — BiRefNet (ZhengPeng7/BiRefNet_dynamic)       │
│  Runs on the YOLO crop (not full image) for speed       │
│  Outputs a high-res alpha matte at 1024×1024            │
└──────────────────────┬──────────────────────────────────┘
                       │ refined alpha (back-projected)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — Alpha Refinement Chain                       │
│  Guided filter → GrabCut → component cleanup →          │
│  hole fill → solidify → trim & feather                  │
└──────────────────────┬──────────────────────────────────┘
                       │ cleaned alpha
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4 — Semantic-Gated Matting (Fusion)              │
│  GATE: zero BiRefNet alpha outside dilated YOLO mask    │
│  FILL: restore product pixels BiRefNet dropped          │
└──────────────────────┬──────────────────────────────────┘
                       │ fused alpha
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5 — QA Gate + Fallback Cascade                   │
│  assess_mask() → route to output/ or review/            │
│  If QA fails: pick best of {fused, BiRefNet, YOLO-only} │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
             output_clean/  or  output_review/
```

**No image is ever rejected** — every stage has a fallback so even partial detections produce some output.

---

## Architecture Deep-Dive

### Why Two Models?

| Model | Strength | Weakness |
|---|---|---|
| **YOLOv8-seg** (`best.pt`) | Knows *what* is a product (trained on retail data); fast; gives a semantic bounding box | Coarse 160×160 prototype masks upsampled to full resolution — edges are blocky |
| **BiRefNet** | Sub-pixel edge quality (1024² matting); handles hair, glass, fur | Generic saliency — can chase glare, shelf edges, or drop low-contrast product (brown meat, white-on-white) |

Fusion takes **BiRefNet's edge quality** and **YOLO's semantic correctness**.

---

### Stage 1 — YOLO-seg Inference

`models.yolo_segment()` runs a **single** `model.predict()` call and extracts:

- **Semantic union mask** — union of all instance masks above `--min-mask-area` pixels (used as fusion prior)
- **Padded crop box** — tight bounding box of the union mask, expanded by `--yolo-pad` fraction (used as BiRefNet input hint)

A single inference gives both, with zero redundant computation.

Fallback: if YOLO weights are missing or ultralytics is not installed, the pipeline continues in **BiRefNet-only mode** (full-image inference).

---

### Stage 2 — BiRefNet Inference

`inference.predict_alpha()` runs BiRefNet on the YOLO crop (or full image if no detection):

1. Resize crop to `--size × --size` (default 1024)
2. Normalise with ImageNet mean/std
3. Forward pass (FP16 on CUDA for speed)
4. Sigmoid → resize back to crop dimensions → uint8 alpha

The refined crop alpha is then **back-projected** onto a full-image zero canvas.

BiRefNet handles heterogeneous output formats (tensor, dict, list/tuple) via `extract_prediction()` — it works with any variant of the BiRefNet family on HuggingFace.

---

### Stage 3 — Alpha Refinement Chain

`postprocessing.refine_alpha()` runs six operations in order:

| Step | Function | What it does |
|---|---|---|
| 1 | `guided_filter_alpha` | Edge-aware smoothing — aligns soft alpha edges to the RGB gradient (requires `opencv-contrib`) |
| 2 | `grabcut_refine_alpha` | Runs OpenCV GrabCut seeded from the BiRefNet alpha, then suppresses border-connected skin regions (hand suppression) |
| 3 | `remove_small_components` | Drops noise blobs below `0.1%` of image area; or keeps only the largest blob (`--component-mode largest`) |
| 4 | `fill_internal_holes` | Fills holes inside the product silhouette using contour fill |
| 5 | `solidify_alpha` | Morphological close + contour fill — removes partial transparency in the product core |
| 6 | `trim_and_feather` | 1-pixel border erosion + Gaussian blur — softens hard edges for compositing |

Each step can be individually disabled via CLI flags (e.g. `--no-guided-filter`, `--no-grabcut-refine`).

---

### Stage 4 — Semantic-Gated Matting (Fusion)

`fusion.fuse_alpha()` combines the refined BiRefNet alpha with the raw YOLO mask:

```
GATE operation
──────────────
1. Dilate YOLO mask by --gate-dilate-frac (fraction of image's shorter side)
2. Zero any BiRefNet alpha that falls OUTSIDE the dilated region
→ Kills BiRefNet false positives (shelf edges, reflections, background items)
   The dilation gives BiRefNet's finer edge room so the coarse YOLO boundary
   never clips real product pixels.

FILL operation
──────────────
1. Erode YOLO mask by --fill-erode-frac to find a confident product interior
2. Where the interior says "product" BUT fused alpha < --fill-min-alpha, set alpha = 255
→ Rescues low-contrast product that BiRefNet dropped
   (e.g. dark brown product on dark background, white product on white background)

Final: light feather (--fusion-feather) smooths the seam between filled and gated regions
```

Disable fusion entirely with `--no-fusion` (useful for debugging individual models).

---

### Stage 5 — QA Gate & Fallback Cascade

`quality.assess_mask()` checks four conditions:

| Check | Threshold | Reason flagged |
|---|---|---|
| `foreground_ratio < 0.015` | Almost everything is transparent | Model removed too much |
| `foreground_ratio > 0.995` | Almost nothing was removed | Background removal failed completely |
| `touches_border_ratio > 0.92` | Product fills most of the image border | Possible full-background image |
| `components > 25 AND largest < 15%` | Mask is fragmented | Noise, not a product |

If QA fails, the pipeline scores all available candidates (fused alpha, BiRefNet-only, YOLO-mask-only) using `quality_score()` and picks the highest-scoring one instead of discarding the image.

`quality_score()` = `10 × (status == "success") + fg_score + border_score + largest_component_ratio`

where `fg_score` is highest near `foreground_ratio ≈ 0.35` (typical product fill on a clean background).

---

### Routing Modes

Each image's result records one of these routing labels in the manifest:

| Routing | Meaning |
|---|---|
| `fused` | YOLO + BiRefNet fusion (best quality path) |
| `birefnet_crop` | BiRefNet on YOLO crop, no fusion (YOLO found box but no mask) |
| `fullimg_fallback` | No YOLO detection — BiRefNet on full image |
| `yolo_mask_only` | BiRefNet crashed — raw YOLO mask used |
| `birefnet_alt` | QA fallback: un-gated BiRefNet (fusion over-clipped) |
| `yolo_mask_alt` | QA fallback: YOLO mask scored better than fused result |
| `none` | Complete failure (both models failed) |

---

## Project Structure

```
bg_remove/
├── run_pipeline.py          # Entry point — CLI, model loading, batch loop
├── best.pt                  # YOLOv8-seg weights (1-class: product)
├── requirements.txt
├── README.md
│
└── universal_pipeline/
    ├── __init__.py
    ├── config.py            # SystemConfig dataclass (all hyperparameters)
    ├── models.py            # load_yolo, load_birefnet, yolo_segment
    ├── inference.py         # predict_alpha (BiRefNet forward pass)
    ├── postprocessing.py    # refine_alpha chain (guided filter, grabcut, etc.)
    ├── fusion.py            # fuse_alpha (semantic-gated matting)
    ├── quality.py           # assess_mask, quality_score, save_cutout
    └── image_io.py          # iter_images, read_bgr (Unicode-safe)
```

**Runtime directories** (created automatically, not committed to git):
```
input_images/          # Drop your product images here
output_clean/          # Accepted cutouts (RGBA PNG, transparent background)
output_review/         # QA-flagged images that need human review
pipeline_manifest.jsonl  # Per-image log: status, routing, quality metrics
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- A CUDA-capable GPU is strongly recommended (NVIDIA, with driver ≥ 525)
- CPU mode works but is ~15-50× slower

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd bg_remove

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install PyTorch with CUDA support (adjust cu121 to your driver version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only machines:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt
```

> **Important:** `opencv-contrib-python` includes the `ximgproc` module needed for the guided filter step. Do **not** install `opencv-python` alongside it — they conflict.

---

## Models Setup

### YOLO-seg (`best.pt`)

`best.pt` is a custom-trained YOLOv8-seg model (1 class: `product`). Place it in the project root (or point to it with `--yolo-model`).

If `best.pt` is missing, the pipeline automatically falls back to **BiRefNet-only mode** with a warning — no crash.

### BiRefNet

BiRefNet is downloaded automatically from HuggingFace on first run:

```bash
# Add --allow-model-download on the very first run
python run_pipeline.py --input ./input_images --allow-model-download
```

On subsequent runs, omit the flag — the model is loaded from the local HuggingFace cache (`~/.cache/huggingface/`).

The default model is `ZhengPeng7/BiRefNet_dynamic` (dynamic resolution variant). Change it with `--model`.

---

## Usage

### Basic run

```bash
python run_pipeline.py --input ./input_images --output ./output_clean
```

### First run (download BiRefNet)

```bash
python run_pipeline.py --input ./input_images --allow-model-download
```

### Dry run (analyse without writing files)

```bash
python run_pipeline.py --input ./input_images --dry-run
```

### CPU-only machine

```bash
python run_pipeline.py --input ./input_images --device cpu --no-fp16
```

### Disable fusion (BiRefNet-only output)

```bash
python run_pipeline.py --input ./input_images --no-fusion
```

### Keep only the largest detected component

```bash
python run_pipeline.py --input ./input_images --component-mode largest
```

### Reprocess already-completed images

```bash
python run_pipeline.py --input ./input_images --overwrite
```

---

## CLI Reference

### I/O

| Flag | Default | Description |
|---|---|---|
| `--input` | `./input_images` | Folder containing product images |
| `--output` | `./output_clean` | Destination for accepted cutouts |
| `--review` | `./output_review` | Destination for QA-flagged cutouts |
| `--manifest` | `./pipeline_manifest.jsonl` | Per-image JSON log |

### YOLO-seg

| Flag | Default | Description |
|---|---|---|
| `--yolo-model` | `./best.pt` | Path to YOLOv8-seg weights |
| `--yolo-conf` | `0.25` | Confidence threshold |
| `--yolo-iou` | `0.7` | NMS IoU threshold |
| `--yolo-imgsz` | `640` | Inference image size |
| `--min-mask-area` | `500` | Minimum pixels for a YOLO instance mask to count |
| `--yolo-pad` | `0.15` | Fractional padding around the crop box (0.15 = 15%) |

### BiRefNet

| Flag | Default | Description |
|---|---|---|
| `--model` | `ZhengPeng7/BiRefNet_dynamic` | HuggingFace model ID |
| `--size` | `1024` | BiRefNet inference square size |
| `--device` | auto (cuda/cpu) | Inference device |
| `--no-fp16` | — | Disable FP16 on CUDA (use FP32) |
| `--allow-model-download` | — | Allow HuggingFace download on first run |

### Fusion (Semantic-Gated Matting)

| Flag | Default | Description |
|---|---|---|
| `--fusion` / `--no-fusion` | enabled | Enable/disable semantic-gated matting |
| `--gate-dilate-frac` | `0.02` | YOLO mask dilation before gating (fraction of image shorter side) |
| `--fill-erode-frac` | `0.06` | YOLO mask erosion to find confident interior |
| `--fill-min-alpha` | `40` | BiRefNet alpha threshold below which a YOLO-confident pixel is filled |
| `--fusion-feather` | `0.8` | Gaussian sigma to smooth fill/gate seams |

### Alpha Refinement

| Flag | Default | Description |
|---|---|---|
| `--guided-filter` / `--no-guided-filter` | enabled | Edge-aware alpha smoothing |
| `--grabcut-refine` / `--no-grabcut-refine` | enabled | GrabCut boundary locking |
| `--grabcut-iters` | `5` | GrabCut EM iterations |
| `--hand-suppression` / `--no-hand-suppression` | enabled | Suppress border-connected skin regions |
| `--component-mode` | `all` | `all` = keep all blobs above threshold; `largest` = keep only the biggest |
| `--solidify` / `--no-solidify` | enabled | Morphological close + contour fill |
| `--edge-trim` | `1` | Pixels to erode from the hard mask boundary |
| `--feather` | `0.8` | Gaussian sigma for final edge softening |
| `--edge-blur` | `5` | Smoothing kernel for YOLO-mask fallback path |

### Misc

| Flag | Default | Description |
|---|---|---|
| `--save-masks` | — | Also save grayscale alpha masks alongside cutouts |
| `--overwrite` | — | Re-process images that already have output |
| `--dry-run` | — | Run pipeline analysis without writing any files |

---

## Output & Manifest

### Output files

- **`output_clean/<name>.png`** — RGBA PNG with transparent background. RGB data in transparent pixels is replaced with black (not just hidden), so `convert("RGB")` cannot reveal the original background.
- **`output_review/<name>.png`** — Same format, but QA flagged these for human inspection.
- **`output_clean/<name>_mask.png`** (with `--save-masks`) — Grayscale alpha map.

### Manifest (`pipeline_manifest.jsonl`)

Each line is a JSON record for one image:

```json
{
  "input": "input_images/product_001.jpg",
  "status": "success",
  "reason": "ok",
  "routing": "fused",
  "quality": {
    "transparent_ratio": 0.6823,
    "foreground_ratio": 0.3177,
    "touches_border_ratio": 0.04,
    "components": 1,
    "largest_component_ratio": 0.3177,
    "status": "success",
    "reason": "ok"
  },
  "output": "output_clean/product_001.png"
}
```

`status` is one of: `success`, `review`, `error`.

`routing` is one of: `fused`, `birefnet_crop`, `fullimg_fallback`, `yolo_mask_only`, `birefnet_alt`, `yolo_mask_alt`, `none`.

---

## Module Reference

| Module | Key exports | Responsibility |
|---|---|---|
| `config.py` | `SystemConfig` | Single dataclass holding every hyperparameter; constructed from CLI args |
| `models.py` | `load_yolo`, `load_birefnet`, `build_transform`, `yolo_segment` | One-time model loading + YOLO inference |
| `inference.py` | `predict_alpha` | BiRefNet forward pass → uint8 alpha at input resolution |
| `postprocessing.py` | `refine_alpha`, `smooth_alpha` | Full 6-step alpha refinement; `smooth_alpha` for YOLO-only fallback |
| `fusion.py` | `fuse_alpha` | Semantic-gated matting (GATE + FILL) |
| `quality.py` | `assess_mask`, `quality_score`, `save_cutout` | QA metrics, scorer for fallback selection, leak-proof PNG writer |
| `image_io.py` | `iter_images`, `read_bgr` | Unicode-safe image discovery and decode (handles non-ASCII filenames on Windows) |
| `orchestrator.py` | `process_one_image` | Wires all stages together; handles all fallback paths; returns manifest record |

---

## Design Decisions

**Single YOLO inference per image.** Both the semantic mask (fusion prior) and the crop box (BiRefNet hint) come from the same `model.predict()` call — there is no second forward pass.

**Crop then BiRefNet, not full-image BiRefNet.** Running BiRefNet on the YOLO crop (typically 30–60% of the image) reduces compute significantly and also helps BiRefNet focus on the product rather than the full scene.

**Graceful degradation instead of hard failures.** The pipeline never raises an uncaught exception to abort a batch. Missing YOLO weights → BiRefNet-only. BiRefNet crash → YOLO-mask-only. QA fail → best-scored fallback. This makes large batch runs reliable without manual supervision.

**Leak-proof PNG output.** `save_cutout()` builds the output on a fresh black canvas and copies original RGB only to foreground pixels. This physically destroys background pixel data — `convert("RGB")` on the output cannot reveal the original scene.

**Unicode-safe I/O.** `read_bgr()` uses `np.fromfile + cv2.imdecode` instead of `cv2.imread` to handle non-ASCII filenames (Spanish product names, Chinese category labels, etc.) correctly on Windows.
