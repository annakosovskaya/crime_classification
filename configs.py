from dataclasses import dataclass
from typing import Any, List


@dataclass
class RuntimeConfig:
    window_size: float
    stride_percent: float
    target_fps: float
    model_frames: int
    backbone: str
    use_batched: bool
    batch_size: int
    num_workers: int
    prefetch_factor: int
    amp_enabled: bool


@dataclass
class InferenceContext:
    model: Any
    transform: Any
    kinetics_labels: List[str]
    device: Any
    config_path: str
    videos_base_path: str


