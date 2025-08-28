import os
from typing import Optional


__all__ = [
    "build_crimes_scores_out_path",
    "build_crimes_results_out_path",
    "build_crimes_thresholds_out_path",
    "resolve_config_path",
    "dataset_tag_from",
]


def build_crimes_scores_out_path(
    *,
    backbone: str,
    window_size: float,
    stride_percent: float,
    target_fps: float,
    model_frames: int,
    dataset_tag: str = "default",
):
    """
    results/crimes/scores/{dataset_tag}/{backbone}/(model_frames|fps)/
      - if model_frames > 0 → subdir 'model_frames', filename: window_{ws}_stride_{sp}.csv
      - else → subdir 'fps', filename: window_{ws}_stride_{sp}_fps_{target_fps}.csv
    """
    base_dir = os.path.join(
        "results", "crimes", "scores", str(dataset_tag), str(backbone).lower()
    )
    if model_frames and int(model_frames) > 0:
        subdir = os.path.join(base_dir, "model_frames")
        fname = f"window_{float(window_size)}_stride_{float(stride_percent)}.csv"
    else:
        subdir = os.path.join(base_dir, "fps")
        fname = f"window_{float(window_size)}_stride_{float(stride_percent)}_fps_{float(target_fps)}.csv"
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, fname)


def build_crimes_results_out_path(
    *,
    backbone: str,
    window_size: float,
    stride_percent: float,
    target_fps: float,
    model_frames: int,
    dataset_tag: str = "default",
):
    """
    results/crimes/{dataset_tag}/{backbone}/(model_frames|fps)/runs/
      - if model_frames > 0 → subdir 'model_frames/runs', filename: window_{ws}_stride_{sp}.json
      - else → subdir 'fps/runs', filename: window_{ws}_stride_{sp}_fps_{target_fps}.json
    """
    base_dir = os.path.join(
        "results", "crimes", str(dataset_tag), str(backbone).lower()
    )
    if model_frames and int(model_frames) > 0:
        subdir = os.path.join(base_dir, "model_frames", "runs")
        fname = f"window_{float(window_size)}_stride_{float(stride_percent)}.json"
    else:
        subdir = os.path.join(base_dir, "fps", "runs")
        fname = f"window_{float(window_size)}_stride_{float(stride_percent)}_fps_{float(target_fps)}.json"
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, fname)


def build_crimes_thresholds_out_path(
    *,
    backbone: str,
    window_size: float,
    stride_percent: float,
    target_fps: float,
    model_frames: int,
    dataset_tag: str = "default",
):
    """
    thresholds/{dataset_tag}/{backbone}/(model_frames|fps)/
      - if model_frames > 0 → thresholds_window_{ws}_stride_{sp}.json
      - else → thresholds_window_{ws}_stride_{sp}_fps_{target_fps}.json
    """
    base_dir = os.path.join(
        "thresholds", str(dataset_tag), str(backbone).lower()
    )
    if model_frames and int(model_frames) > 0:
        subdir = os.path.join(base_dir, "model_frames")
        fname = f"thresholds_window_{float(window_size)}_stride_{float(stride_percent)}.json"
    else:
        subdir = os.path.join(base_dir, "fps")
        fname = f"thresholds_window_{float(window_size)}_stride_{float(stride_percent)}_fps_{float(target_fps)}.json"
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, fname)


def resolve_config_path(config_path: Optional[str] = None) -> str:
    """Return provided config path or default to package's config.json.

    The default resolves to .../config.json regardless of CWD.
    """
    if config_path and str(config_path).strip():
        return str(config_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "config.json")


def dataset_tag_from(path: str) -> str:
    """Return dataset tag derived from a file path (basename without extension)."""
    base = os.path.basename(str(path))
    name, _ext = os.path.splitext(base)
    return name or "default"


