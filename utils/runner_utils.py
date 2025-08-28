import json
import logging
from contextlib import nullcontext
from typing import Dict, Tuple, Any, ContextManager, Optional

import torch


logger = logging.getLogger(__name__)


def load_crime_mapping(config_path: str) -> Dict[str, Any]:
    """Load crime mapping using a config.json path that contains crime_mapping_path.

    Args:
        config_path: Path to config.json

    Returns:
        Parsed crime mapping dictionary
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # Support nested (new) and flat (legacy) formats
    mapping_path = (
        cfg.get("paths", {}).get("crime_mapping_path")
        or cfg.get("crime_mapping_path")
    )
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping


def build_stride_and_frames(
    *, window_size: float, stride_percent: float, target_fps: float, model_frames_override: Optional[int]
) -> Tuple[float, int]:
    """Compute stride in seconds and frames per window given overrides.

    Returns:
        (stride_seconds, target_frames)
    """
    stride_seconds = float(window_size) * float(stride_percent)
    if model_frames_override is not None and int(model_frames_override) > 0:
        target_frames = int(model_frames_override)
    else:
        target_frames = int(float(window_size) * float(target_fps))
    if target_frames < 1:
        target_frames = 1
    return stride_seconds, target_frames


def aggregate_predictions_max(predictions_per_window: list[dict]) -> Dict[str, float]:
    """Aggregate per-window predictions by max into a single video-level dict."""
    video_level: Dict[str, float] = {}
    for window_result in predictions_per_window:
        for crime, prob in window_result.get("predictions", {}).items():
            prev = video_level.get(crime, 0.0)
            if prob > prev:
                video_level[crime] = prob
    return video_level


def get_device(device: Optional[torch.device] = None) -> torch.device:
    """Return a torch device; enable cudnn.benchmark when CUDA."""
    if device is not None:
        return device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return dev


def autocast_if_cuda(enabled: bool, device: torch.device) -> ContextManager:
    """Return autocast context manager for CUDA when enabled; otherwise nullcontext."""
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


