import os
import logging
from typing import Dict, List, Tuple, Any, Set

from collections import defaultdict


def _normalize_category_for_path(category: str) -> str:
    return 'Normal' if category and category.lower() == 'normal' else category


def score_dataset(
    *,
    videos: List[Dict[str, Any]],
    aggregation_method: str,
    ctx,
    runtime,
    process_video,
    process_video_batched_dataloader,
    collect_aggregates: bool = False,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], List[Dict[str, Any]], Set[str]]:
    """
    Iterate videos, run model inference via provided process functions, and
    collect per-class predictions/ground-truth with per-video score rows.
    """
    all_predictions: Dict[str, List[float]] = defaultdict(list) if collect_aggregates else {}
    all_ground_truth: Dict[str, List[int]] = defaultdict(list) if collect_aggregates else {}
    scores_rows: List[Dict[str, Any]] = []
    all_crimes_seen: Set[str] = set()

    for idx, video_data in enumerate(videos):
        video_filename = video_data["video_path"]
        true_category = video_data["category"]

        category_for_path = _normalize_category_for_path(true_category)
        video_path = os.path.join(ctx.videos_base_path, category_for_path, os.path.basename(video_filename))

        logging.info(f"Processing video {idx+1}/{len(videos)}: {video_path}")

        try:
            if runtime.use_batched:
                predictions = process_video_batched_dataloader(
                    video_path=video_path,
                    config_path=ctx.config_path,
                    window_size=runtime.window_size,
                    stride_percent=runtime.stride_percent,
                    
                    target_fps=runtime.target_fps,
                    aggregation_method=aggregation_method,
                    batch_size=runtime.batch_size,
                    num_workers=runtime.num_workers,
                    prefetch_factor=runtime.prefetch_factor,
                    amp_enabled=runtime.amp_enabled,
                    model=ctx.model,
                    transform=ctx.transform,
                    kinetics_labels=ctx.kinetics_labels,
                    device=ctx.device,
                    model_frames_override=(runtime.model_frames or None),
                )
            else:
                predictions = process_video(
                    video_path=video_path,
                    config_path=ctx.config_path,
                    window_size=runtime.window_size,
                    stride_percent=runtime.stride_percent,
                    
                    target_fps=runtime.target_fps,
                    aggregation_method=aggregation_method,
                    model=ctx.model,
                    transform=ctx.transform,
                    kinetics_labels=ctx.kinetics_labels,
                    device=ctx.device,
                    model_frames_override=(runtime.model_frames or None),
                )

            if predictions is None:
                logging.warning(f"Skipping video due to processing error: {video_filename}")
                continue

            if collect_aggregates:
                all_crimes = set(all_predictions.keys()) | set(predictions.keys())
                for crime in all_crimes:
                    all_predictions[crime].append(predictions.get(crime, 0))
                    all_ground_truth[crime].append(1 if crime == true_category else 0)

            try:
                row = {"video_path": video_path, "category": true_category}
                for k, v in predictions.items():
                    row[k] = float(v)
                scores_rows.append(row)
                all_crimes_seen.update(predictions.keys())
            except Exception:
                pass

        except Exception as e:
            logging.error(f"Error processing video {video_filename}: {e}")
            continue

    return all_predictions, all_ground_truth, scores_rows, all_crimes_seen


