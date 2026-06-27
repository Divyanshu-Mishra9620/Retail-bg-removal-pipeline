# Retail Product Background Removal Pipeline

A production orchestrator that turns two proven background-removal scripts into one automated, resumable, batch pipeline for retail product images — **without changing their image-processing behavior**.

The pipeline does **not** reimplement any segmentation, refinement, QA, or compositing logic. It imports the original functions verbatim and only adds production concerns: load-once models, per-image routing, resumable batching, a JSON manifest, and leak-proof PNG output. The result for any given image is identical to running the original script that produced it.

---

## Table of Contents

1. [Source of Truth](#source-of-truth)
2. [How It Works](#how-it-works)
3. [Why Two Scripts](#why-two-scripts)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Models Setup](#models-setup)
7. [Usage](#usage)
8. [CLI Reference](#cli-reference)
9. [Output & Manifest](#output--manifest)
10. [Module Reference](#module-reference)
11. [Design Decisions](#design-decisions)

---

## Source of Truth

The two original scripts are the canonical implementations. The pipeline imports their functions and calls them unchanged:

| Original script | Role in pipeline | Functions reused (verbatim) |
|---|---|---|
| `test_BiRefNet.py` | **Primary** — BiRefNet matting flow | `predict_alpha`, `refine_alpha` (guided filter → GrabCut → component cleanup → hole fill → solidify → trim/feather), `assess_mask`, `MaskQuality`, `build_transform`, `load_model`, `yolo_detect` |
| `Retail_AI_Training/test.py` | **Fallback** — YOLO-seg flow | `mask_from_result`, `smooth_alpha`, `read_bgr` |

`universal_pipeline/originals.py` loads both scripts as modules and re-exports these symbols. Every threshold, kernel size, blur radius, GrabCut iteration count, ImageNet normalization, and resize dimension is therefore exactly the original value — there is no second copy to drift.

---

## How It Works

Each image is routed through one original script's exact flow. "Script A → Script B if needed":

```
Input Image
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  PRIMARY — test_BiRefNet.py flow (exact reproduction)         │
│  EXIF-correct load → YOLO detection box (+pad) → crop →       │
│  BiRefNet @1024² → refine_alpha → back-project → assess_mask  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                ┌───────────┴────────────┐
          QA = success               QA = review
                │                         │
                ▼                         ▼
          output_clean/      ┌──────────────────────────────────┐
                             │  FALLBACK — test.py flow (exact)  │
                             │  cv2 read → YOLO-seg → mask_from_  │
                             │  result → smooth_alpha            │
                             └───────────────┬───────────────────┘
                                             │
                              ┌──────────────┴───────────────┐
                         mask found                     no mask
                              │                              │
                              ▼                              ▼
                        output_clean/                 output_review/
                     (exactly test.py's          (BiRefNet review result
                       output)                     kept for inspection)
```

- **No fusion, no blending.** Each saved cutout is exactly what one original script would produce.
- **No hard failures.** If BiRefNet errors, the YOLO-seg flow is tried; if both are insufficient, the image is routed to review rather than dropped.
- **Resumable.** Images that already have an output are skipped (use `--overwrite` to redo).

---

## Why Two Scripts

| Script / model | Strength | Weakness |
|---|---|---|
| `test_BiRefNet.py` — **BiRefNet** matting | Sub-pixel edge quality (1024² matte, guided filter, feather); handles fine/soft edges | Generic saliency — can struggle on cluttered or same-color scenes |
| `test.py` — **YOLOv8-seg** (`best.pt`, 1 class: `product`) | Trained on retail products — knows *what* a product is; robust where saliency fails | Coarse 160² prototype masks upsampled → blocky edges |

BiRefNet runs first for its edge quality. When its QA gate flags a result as suspect, the semantically-trained YOLO-seg flow takes over.

---

## Project Structure

```
bg_remove/
├── run_pipeline.py            # Entry point — CLI, load-once models, batch loop, manifest
├── test_BiRefNet.py           # ORIGINAL script — BiRefNet flow (source of truth)
├── Retail_AI_Training/
│   └── test.py                # ORIGINAL script — YOLO-seg flow (source of truth)
├── best.pt                    # YOLOv8-seg weights (1 class: product)
├── requirements.txt
├── README.md
│
└── universal_pipeline/
    ├── __init__.py
    ├── originals.py           # Loads both originals; re-exports their functions verbatim
    ├── config.py              # SystemConfig — all params at original defaults
    ├── models.py              # load_yolo (cached) + re-exported build_transform/load_birefnet
    ├── quality.py             # assess_mask/MaskQuality (from original) + leak-proof save_cutout
    ├── image_io.py            # iter_images, read_bgr, save_original_to_review (Unicode-safe)
    └── orchestrator.py        # Per-image routing: BiRefNet flow → YOLO-seg fallback
```

**Runtime directories** (auto-created, git-ignored):
```
input_images/            # Drop your product images here
output_clean/            # Accepted cutouts (RGBA PNG, transparent background)
output_review/           # QA-flagged images for human review
pipeline_manifest.jsonl  # Per-image log: status, routing, quality metrics
```

---

## Installation

### Prerequisites
- Python 3.10+
- A CUDA-capable NVIDIA GPU is strongly recommended (CPU works but is ~15–50× slower)

### Steps
```bash
git clone <repo-url>
cd bg_remove

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# PyTorch with CUDA (match cu121 to your driver; use /whl/cpu for CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

> **Important:** the pipeline must run in this environment. `ultralytics` (YOLO) lives here; the system Python likely does not have it, and without it the YOLO-seg fallback is disabled. Always invoke the venv interpreter explicitly (see Usage).

> `opencv-contrib-python` provides the `ximgproc` guided filter used by `refine_alpha`. Do not install `opencv-python` alongside it — they conflict.

---

## Models Setup

### YOLO-seg (`best.pt`)
Custom-trained YOLOv8-seg model, 1 class `product`. Keep it in the project root (or pass `--yolo-model`). If it is missing or `ultralytics` is unavailable, BiRefNet runs on full images and the YOLO-seg fallback is skipped — with a warning, no crash.

### BiRefNet
Downloaded from HuggingFace on first run (`ZhengPeng7/BiRefNet_dynamic`). Add `--allow-model-download` the first time; afterwards it loads from the local cache.

---

## Usage

Run with the **venv interpreter** (so YOLO is available):

```powershell
# Windows
.\venv\Scripts\python.exe run_pipeline.py --input ./input_images --output ./output_clean --review ./output_review
```
```bash
# macOS / Linux
./venv/bin/python run_pipeline.py --input ./input_images --output ./output_clean --review ./output_review
```

First run (download BiRefNet weights):
```powershell
.\venv\Scripts\python.exe run_pipeline.py --input ./input_images --allow-model-download
```

Dry run (analyse, write nothing):
```powershell
.\venv\Scripts\python.exe run_pipeline.py --input ./input_images --dry-run
```

CPU-only:
```powershell
.\venv\Scripts\python.exe run_pipeline.py --input ./input_images --device cpu --no-fp16
```

The run is resumable — stop with Ctrl-C and rerun; completed images are skipped. Use `--overwrite` to reprocess.

---

## CLI Reference

### I/O
| Flag | Default | Description |
|---|---|---|
| `--input` | `./input_images` | Folder of input images |
| `--output` | `./output_clean` | Accepted cutouts |
| `--review` | `./output_review` | QA-flagged cutouts |
| `--manifest` | `./pipeline_manifest.jsonl` | Per-image JSON log |

### Shared model
| Flag | Default | Description |
|---|---|---|
| `--yolo-model` | `./best.pt` | YOLOv8-seg weights (used by both flows) |

### BiRefNet flow (primary) — `test_BiRefNet.py` defaults
| Flag | Default | Description |
|---|---|---|
| `--model` | `ZhengPeng7/BiRefNet_dynamic` | HuggingFace model ID |
| `--size` | `1024` | BiRefNet inference square size |
| `--device` | auto | `cuda` or `cpu` |
| `--no-fp16` | — | Disable FP16 on CUDA |
| `--allow-model-download` | — | Allow HuggingFace download on first run |
| `--yolo-conf` | `0.15` | YOLO confidence for the crop box |
| `--yolo-pad` | `0.15` | Fractional padding around the crop box |
| `--guided-filter` / `--no-guided-filter` | enabled | Edge-aware alpha smoothing |
| `--grabcut-refine` / `--no-grabcut-refine` | enabled | GrabCut boundary locking + hand suppression |
| `--grabcut-iters` | `5` | GrabCut iterations |
| `--hand-suppression` / `--no-hand-suppression` | enabled | Suppress border-connected skin regions |
| `--component-mode` | `all` | `all` or `largest` |
| `--solidify` / `--no-solidify` | enabled | Morphological close + contour fill |
| `--edge-trim` | `1` | Border erosion (pixels) |
| `--feather` | `0.8` | Final edge Gaussian sigma |

### YOLO-seg flow (fallback) — `test.py` defaults
| Flag | Default | Description |
|---|---|---|
| `--seg-conf` | `0.25` | YOLO-seg confidence |
| `--seg-iou` | `0.7` | YOLO-seg NMS IoU |
| `--seg-imgsz` | `640` | YOLO-seg inference size |
| `--min-mask-area` | `500` | Min pixels for a YOLO-seg mask to count |
| `--edge-blur` | `5` | `smooth_alpha` blur radius |

### Misc
| Flag | Default | Description |
|---|---|---|
| `--save-masks` | — | Also save grayscale alpha masks |
| `--overwrite` | — | Reprocess images that already have output |
| `--dry-run` | — | Analyse without writing files |

---

## Output & Manifest

**Output files**
- `output_clean/<name>.png` — RGBA PNG, transparent background. Transparent pixels are written as black on a fresh canvas (leak-proof), so `convert("RGB")` cannot recover the original background.
- `output_review/<name>.png` — same format; QA flagged for human inspection.
- `output_clean/<name>_mask.png` — grayscale alpha (with `--save-masks`).

**Manifest** (`pipeline_manifest.jsonl`) — one JSON record per image:
```json
{
  "input": "input_images/product_001.jpg",
  "status": "success",
  "reason": null,
  "routing": "birefnet_yolo_crop",
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

`status` ∈ `success` | `review` | `error`.

`routing` values:

| Routing | Meaning |
|---|---|
| `birefnet_yolo_crop` | BiRefNet on the YOLO crop (primary path) |
| `birefnet_fullimage` | No YOLO detection — BiRefNet on the full image |
| `yolo_seg_fallback` | BiRefNet QA failed → test.py YOLO-seg output used |
| `birefnet_yolo_crop_review` / `birefnet_fullimage_review` | BiRefNet result kept and routed to review (fallback also insufficient) |
| `none` | BiRefNet errored and no YOLO-seg mask available → original copied to review |

---

## Module Reference

| Module | Key exports | Responsibility |
|---|---|---|
| `originals.py` | re-exports of all original functions | Loads `test_BiRefNet.py` + `Retail_AI_Training/test.py` as modules; single source of truth |
| `config.py` | `SystemConfig` | All parameters at original defaults; attribute names match what `test_BiRefNet.py`'s `refine_alpha` reads off `args` |
| `models.py` | `load_yolo`, `build_transform`, `load_birefnet` | Load each model once; build_transform/load_birefnet re-exported from the original |
| `quality.py` | `assess_mask`, `MaskQuality`, `save_cutout`, `verify_hidden_background_removed` | Original QA gate + leak-proof PNG writer + leak auditor |
| `image_io.py` | `iter_images`, `read_bgr`, `save_original_to_review` | Unicode-safe discovery/decode (handles non-ASCII filenames on Windows) |
| `orchestrator.py` | `process_one_image` | Per-image routing only — reproduces each original flow, no image processing of its own |
| `run_pipeline.py` | `main` | CLI, load-once models, batch loop, skip/resume, manifest, summary |

---

## Design Decisions

**Orchestrate, don't reimplement.** All image processing comes from the original scripts via `originals.py`. The merged code only decides which flow to run and handles batching/IO. This guarantees output parity with the originals and removes any risk of a rewritten filter silently changing quality.

**BiRefNet primary, YOLO-seg fallback.** BiRefNet gives the best edges; its own QA gate (`assess_mask`) decides when a result is suspect, at which point the semantically-trained YOLO-seg flow produces the cutout instead.

**Models loaded once.** YOLO and BiRefNet are loaded a single time and reused for the whole batch. (YOLO is invoked per flow at its source script's own confidence — 0.15 for the BiRefNet crop box, 0.25 for the YOLO-seg fallback — matching each original exactly.)

**Graceful degradation.** Missing YOLO weights or `ultralytics` → BiRefNet full-image mode. BiRefNet crash → YOLO-seg flow. Both insufficient → review folder. A batch never aborts on one bad image.

**Leak-proof PNG output.** `save_cutout()` composites onto a fresh black canvas and copies original RGB only to foreground pixels, physically destroying background data (matching `test.py`'s approach). `verify_hidden_background_removed()` can audit any output for residual background.

**Unicode-safe I/O.** `read_bgr()` uses `np.fromfile + cv2.imdecode` so non-ASCII filenames (e.g. Spanish category names) decode correctly on Windows.
