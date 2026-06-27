import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # d:\bg_remove


def _load(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Original script not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load 
_birefnet_src = _load("_orig_birefnet", _ROOT / "test_BiRefNet.py")
_yolo_seg_src = _load("_orig_yolo_seg", _ROOT / "Retail_AI_Training" / "test.py")


# ── BiRefNet flow ─────────────────────
predict_alpha = _birefnet_src.predict_alpha
extract_prediction = _birefnet_src.extract_prediction
refine_alpha = _birefnet_src.refine_alpha          # reads args.guided_filter, .grabcut_refine, etc.
assess_mask = _birefnet_src.assess_mask
MaskQuality = _birefnet_src.MaskQuality
build_transform = _birefnet_src.build_transform
load_birefnet = _birefnet_src.load_model
yolo_detect = _birefnet_src.yolo_detect            # detection box (+pad) for the crop hint

# ── YOLO-seg flow ───────────
mask_from_result = _yolo_seg_src.mask_from_result
smooth_alpha = _yolo_seg_src.smooth_alpha
read_bgr = _yolo_seg_src.read_bgr
