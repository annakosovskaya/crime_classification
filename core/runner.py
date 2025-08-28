import os

import argparse
import logging
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image

from core.video_processor import VideoProcessor, SlidingWindowProcessor
from utils.runner_utils import (
    load_crime_mapping,
    build_stride_and_frames,
    aggregate_predictions_max,
    get_device,
    autocast_if_cuda,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_crime_mapping(filepath):
    """
    Parses the crime mapping from a JSON file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def infer_actions_for_clip(
    window_frames,
    model,
    transform,
    kinetics_labels,
    device
):
    """
    Perform per-window action inference with the given model and transform.

    Args:
        window_frames: List of numpy frames in BGR.
        model: Torch video model.
        transform: Preprocessing transform for single frame.
        kinetics_labels: List of class labels.
        device: Torch device.

    Returns:
        List of (label, probability) for top-5 classes.
    """
    if not window_frames:
        return []

    # The new window generation and frame extraction logic guarantee the correct
    # number of frames, so no padding or trimming is needed.

    # Convert frames from BGR (OpenCV default) to RGB before creating PIL Image
    window_frames_rgb = [Image.fromarray(frame[:, :, ::-1]) for frame in window_frames]
    
    # Apply transform to each frame and stack
    clip = torch.stack([transform(frame) for frame in window_frames_rgb])
    
    # Permute from (T, C, H, W) to (C, T, H, W) and add batch dimension
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(clip)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)
        top5_prob, top5_indices = torch.topk(probabilities, 5)

    return [(kinetics_labels[i], prob.item()) for i, prob in zip(top5_indices, top5_prob)]

def process_video_serial(
    video_path,
    config_path,
    window_size,
    stride_percent,
    aggregation_method,
    target_fps: float = 1.0,
    log_window_predictions: bool = False,
    *,
    model: torch.nn.Module = None,
    transform = None,
    kinetics_labels: list = None,
    device: torch.device = None,
    model_frames_override: Optional[int] = None
):
    """
    Process a single video sequentially and aggregate crime probabilities.

    Args:
        video_path: Absolute path to video file.
        config_path: Path to config.json with crime mapping path.
        window_size: Sliding window size in seconds.
        stride_percent: Stride as fraction of window size.
        aggregation_method: 'max' or 'none'.
        target_fps: Target FPS for frame sampling inside window.
        log_window_predictions: Log per-window predictions if True.
        model, transform, kinetics_labels, device, model_frames_override: Optional overrides.

    Returns:
        Aggregated predictions (dict) or list per window when aggregation_method='none'.
    """
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return

    crime_mapping = load_crime_mapping(config_path)

    device = get_device(device)
    if (model is None) or (transform is None) or (kinetics_labels is None):
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights).eval().to(device)
        kinetics_labels = weights.meta["categories"]
        video_transform = weights.transforms()
        transform = transforms.Compose([
            transforms.Resize(video_transform.resize_size, antialias=True),
            transforms.CenterCrop(video_transform.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=video_transform.mean, std=video_transform.std),
        ])

    video_processor = VideoProcessor()
    video_info = video_processor.get_video_info(video_path)
    if not video_info:
        return {}

    stride_seconds, target_frames = build_stride_and_frames(
        window_size=window_size,
        stride_percent=stride_percent,
        target_fps=target_fps,
        model_frames_override=model_frames_override,
    )
    
    all_window_predictions = []
    sliding_window_processor = SlidingWindowProcessor(video_processor)
    window_data_list = list(sliding_window_processor.process_video_windows(
        video_info,
        window_size_seconds=window_size,
        stride_seconds=stride_seconds,
        min_window_size_seconds=1.0,
        model_frames=target_frames
    ))

    for i, (window_info, current_frames) in enumerate(window_data_list):

        # Get action probabilities from the real model
        action_probabilities = infer_actions_for_clip(
            list(current_frames),  # Pass a copy to avoid modification
            model,
            transform,
            kinetics_labels,
            device
        )

        # Calculate crime probabilities for the current window
        crime_evidence_sum = defaultdict(float)
        for action, action_prob in action_probabilities:
            if action in crime_mapping:
                for crime_info in crime_mapping[action]:
                    crime = crime_info["crime"]
                    w_i = crime_info["prob"]
                    s_i = action_prob * w_i
                    crime_evidence_sum[crime] += s_i

        # Aggregate probabilities using the saturating exponent formula with beta=1.0
        final_crime_probabilities = {
            crime: 1 - np.exp(-1.0 * s_sum)
            for crime, s_sum in crime_evidence_sum.items()
        }

        all_window_predictions.append({
            "start_time": window_info.start_time,
            "end_time": window_info.end_time,
            "predictions": final_crime_probabilities
        })
        
        if log_window_predictions:
            logger.info(f"Window {window_info.window_id}:")
            for crime, prob in final_crime_probabilities.items():
                logger.info(f"  - {crime}: {prob:.4f}")

    # Aggregate predictions across all windows
    if not all_window_predictions:
        return []

    if aggregation_method == 'none':
        return all_window_predictions

    if aggregation_method == 'max':
        return aggregate_predictions_max(all_window_predictions)
    
    raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def process_video_batched(
    video_path: str,
    config_path: str,
    window_size: float,
    stride_percent: float,
    aggregation_method: str,
    *,
    target_fps: float = 1.0,
    batch_size: int = 8,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    amp_enabled: bool = True,
    log_window_predictions: bool = False,
    model: torch.nn.Module = None,
    transform = None,
    kinetics_labels: list = None,
    device: torch.device = None,
    model_frames_override: Optional[int] = None
):
    """
    Process a video using Dataset/DataLoader, batched inference and optional AMP.

    Args:
        video_path, config_path, window_size, stride_percent, aggregation_method: See serial path.
        target_fps, batch_size, num_workers, prefetch_factor, amp_enabled: Dataloader/inference options.
        log_window_predictions, model, transform, kinetics_labels, device, model_frames_override: Optional overrides.

    Returns:
        Aggregated predictions (dict) or list per window when aggregation_method='none'.
    """
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return

    crime_mapping = load_crime_mapping(config_path)

    device = get_device(device)

    if (model is None) or (transform is None) or (kinetics_labels is None):
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights).eval().to(device)
        kinetics_labels = weights.meta["categories"]
        video_transform = weights.transforms()
        transform = transforms.Compose([
            transforms.Resize(video_transform.resize_size, antialias=True),
            transforms.CenterCrop(video_transform.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=video_transform.mean, std=video_transform.std),
        ])
    video_processor = VideoProcessor()
    video_info = video_processor.get_video_info(video_path)
    if not video_info:
        return {}

    stride_seconds = window_size * stride_percent
    # Compute target frames from override or (window_size, target_fps)
    target_frames = int(model_frames_override) if (model_frames_override is not None and int(model_frames_override) > 0) else int(window_size * target_fps)
    if target_frames < 1:
        target_frames = 1

    sliding_window_processor = SlidingWindowProcessor(video_processor)
    windows = video_processor.generate_sliding_windows(
        video_info,
        window_size_seconds=window_size,
        stride_seconds=stride_seconds,
        min_window_size_seconds=1.0,
    )

    class WindowsDataset(Dataset):
        def __init__(self):
            self.windows = windows

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index):
            window_info = self.windows[index]

            frames = video_processor.extract_frames_from_window(
                video_info, window_info, model_frames=target_frames
            )
            pil_frames = [Image.fromarray(frame[:, :, ::-1]) for frame in frames]
            clip = torch.stack([transform(frame) for frame in pil_frames])  # (T, C, H, W)
            clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
            return clip, window_info

    def collate_fn(batch):
        # batch: List[Tuple[Tensor(C,T,H,W), WindowInfo]]
        clips, winfos = zip(*batch)
        clips_tensor = torch.stack(clips, dim=0)  # (N, C, T, H, W)
        return {"clips": clips_tensor, "windows": list(winfos)}

    pin_memory = device.type == "cuda"
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    dataset = WindowsDataset()
    dataloader = DataLoader(dataset, **loader_kwargs)

    all_window_predictions = []

    autocast_enabled = bool(amp_enabled and device.type == "cuda")

    with torch.inference_mode():
        for batch in dataloader:
            clips = batch["clips"].to(device, non_blocking=True)
            winfos = batch["windows"]

            with autocast_if_cuda(autocast_enabled, device):
                logits = model(clips)

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)

            # For each window in the batch, compute crime probabilities and log
            for i in range(clips.shape[0]):
                action_probabilities = [
                    (kinetics_labels[int(cls_idx)], float(prob))
                    for cls_idx, prob in zip(top5_indices[i], top5_prob[i])
                ]

                crime_evidence_sum = defaultdict(float)
                for action, action_prob in action_probabilities:
                    if action in crime_mapping:
                        for crime_info in crime_mapping[action]:
                            crime = crime_info["crime"]
                            w_i = crime_info["prob"]
                            s_i = action_prob * w_i
                            crime_evidence_sum[crime] += s_i

                final_crime_probabilities = {
                    crime: 1 - np.exp(-1.0 * s_sum)
                    for crime, s_sum in crime_evidence_sum.items()
                }

                w = winfos[i]
                all_window_predictions.append({
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "predictions": final_crime_probabilities
                })

                if log_window_predictions:
                    logger.info(f"Window {w.window_id}:")
                    for crime, prob in final_crime_probabilities.items():
                        logger.info(f"  - {crime}: {prob:.4f}")

            # Free ASAP
            del clips, logits, probabilities, top5_prob, top5_indices
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not all_window_predictions:
        return []

    if aggregation_method == 'none':
        return all_window_predictions

    if aggregation_method == 'max':
        video_level_predictions = defaultdict(float)
        for window_result in all_window_predictions:
            for crime, prob in window_result['predictions'].items():
                video_level_predictions[crime] = max(video_level_predictions[crime], prob)
        return dict(video_level_predictions)

    raise ValueError(f"Unknown aggregation method: {aggregation_method}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to detect crimes using a classic approach.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument("--window_size", type=float, default=2.0, help="Size of the sliding window in seconds.")
    parser.add_argument("--stride_percent", type=float, default=0.5, help="Stride as a percentage of window size (e.g., 0.5 for 50% overlap).")
    # min_window_size removed; using fixed 1.0
    parser.add_argument("--target_fps", type=float, default=1.0, help="Desired sampling FPS inside each window; target_frames = int(window_size * target_fps).")
    parser.add_argument("--aggregation_method", type=str, default="max", help="Method to aggregate window predictions ('max', 'none').")
    # Batched path flags
    parser.add_argument("--use_batched", action="store_true", help="Use batched dataloader-based processing.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for window clips.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor per worker.")
    parser.add_argument("--amp_enabled", type=int, default=1, choices=[0, 1], help="Enable AMP (1) or disable (0) for CUDA.")
    parser.add_argument("--log_window_predictions", action="store_true", help="Log per-window predictions.")
    args = parser.parse_args()

    if args.use_batched:
        video_predictions = process_video_batched(
            video_path=args.video_path,
            config_path=args.config_path,
            window_size=args.window_size,
            stride_percent=args.stride_percent,
            
            aggregation_method=args.aggregation_method,
            target_fps=args.target_fps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            amp_enabled=bool(args.amp_enabled),
            log_window_predictions=bool(args.log_window_predictions),
        )
    else:
        video_predictions = process_video_serial(
            args.video_path,
            args.config_path,
            args.window_size,
            args.stride_percent,
            
            args.aggregation_method,
            target_fps=args.target_fps,
            log_window_predictions=bool(args.log_window_predictions),
        )

    print(json.dumps(video_predictions, indent=4))
