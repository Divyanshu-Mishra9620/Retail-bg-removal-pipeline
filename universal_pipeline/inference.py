"""
BiRefNet inference — forward pass + alpha extraction.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def extract_prediction(output) -> torch.Tensor:
    """Normalise heterogeneous BiRefNet output formats into a (1,1,H,W) tensor."""
    if isinstance(output, torch.Tensor):
        pred = output
    elif isinstance(output, dict):
        pred = None
        for key in ("logits", "pred", "out", "last_hidden_state"):
            if key in output and output[key] is not None:
                pred = output[key]
                break
        if pred is None:
            raise KeyError(f"Cannot find prediction tensor in output keys: {list(output)}")
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


def predict_alpha(
    model,
    image_rgb: Image.Image,
    transform,
    size: int,
    device: str,
    use_fp16: bool,
) -> np.ndarray:
    """
    Run BiRefNet on a PIL RGB image.
    Returns a uint8 alpha map [0, 255] at the same resolution as image_rgb.
    """
    tensor = transform(image_rgb).unsqueeze(0).to(device)
    if use_fp16 and device == "cuda":
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
