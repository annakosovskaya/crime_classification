import argparse
import json
import logging
import torch

import os
from collections import defaultdict
import csv
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import warnings
from sklearn.exceptions import UndefinedMetricWarning


from core.runner import process_video_serial as process_video, process_video_batched

from utils.paths import (
    build_crimes_scores_out_path,
    build_crimes_results_out_path,
    build_crimes_thresholds_out_path,
    resolve_config_path,
    dataset_tag_from,
)
from models import load_backbone
from utils.metrics import (
    compute_auc,
    apply_threshold,
    compute_precision_recall,
    select_thresholds_for_classes,
)
from utils.io import write_scores_csv, write_results_json, append_validation_row
from utils.scoring import score_dataset
from configs import RuntimeConfig, InferenceContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def generate_scores_csv(*, videos_path: str, runtime: RuntimeConfig, ctx: InferenceContext):
    """
    Compute per-video scores on a dataset JSON and write a standard CSV.

    Args:
        videos_path: Path to dataset JSON with video entries.
        runtime: Windowing/inference runtime configuration.
        ctx: Initialized inference context with model, transforms and device.

    Returns:
        Path to the generated scores CSV.
    """
    with open(videos_path, 'r') as f:
        videos = json.load(f)

    _, _, scores_rows, all_crimes_seen = score_dataset(
        videos=videos,
        aggregation_method='max',
        ctx=ctx,
        runtime=runtime,
        process_video=process_video,
        process_video_batched_dataloader=process_video_batched,
        collect_aggregates=False,
    )

    dataset_tag = dataset_tag_from(videos_path)
    out_csv = build_crimes_scores_out_path(
        backbone=runtime.backbone,
        window_size=runtime.window_size,
        stride_percent=runtime.stride_percent,
        target_fps=runtime.target_fps,
        model_frames=runtime.model_frames,
        dataset_tag=dataset_tag,
    )
    crimes_sorted = sorted(all_crimes_seen)
    write_scores_csv(out_csv=out_csv, rows=scores_rows, crimes_sorted=crimes_sorted)
    logging.info(f"Saved crimes scores-only table to {out_csv} (N={len(scores_rows)})")
    return out_csv


def tune_thresholds_from_scores(*, scores_csv: str, thr_min: float, thr_max: float, thr_steps: int, runtime: RuntimeConfig):
    """
    Sweep class-wise thresholds on an existing scores CSV and write best thresholds JSON.

    Args:
        scores_csv: Path to scores CSV produced by generate_scores_csv.
        thr_min: Minimum threshold value to test.
        thr_max: Maximum threshold value to test.
        thr_steps: Number of steps between min and max.
        runtime: Runtime configuration (used for output path layout).

    Returns:
        Dict with per-class best metrics, macro metrics, thresholds path and sweep rows.
    """
    """Read scores CSV, sweep thresholds per-class, write thresholds JSON, return summary.

    This function does NOT recompute scores.
    """
    if not os.path.exists(scores_csv):
        logging.error(f"Scores CSV not found: {scores_csv}")
        return {}

    # Load from CSV
    all_predictions = defaultdict(list)
    all_ground_truth = defaultdict(list)
    with open(scores_csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        crime_columns = [c for c in fieldnames if c not in ("video_path", "category")]
        for row in reader:
            true_category = row.get("category", "")
            for crime in crime_columns:
                try:
                    score = float(row.get(crime, 0.0))
                except Exception:
                    score = 0.0
                all_predictions[crime].append(score)
                all_ground_truth[crime].append(1 if crime == true_category else 0)

    thr_values = np.linspace(thr_min, thr_max, num=max(2, int(thr_steps)))
    crimes_sorted = sorted(all_predictions.keys())
    class_to_true = {c: np.array(all_ground_truth[c]) for c in crimes_sorted}
    class_to_pred = {c: np.array(all_predictions[c]) for c in crimes_sorted}
    per_class_best, sweep_rows = select_thresholds_for_classes(
        class_to_true=class_to_true,
        class_to_pred=class_to_pred,
        thr_values=thr_values,
    )

    macro_f1 = float(np.mean([v["f1"] for v in per_class_best.values()])) if per_class_best else 0.0
    macro_p = float(np.mean([v["precision"] for v in per_class_best.values()])) if per_class_best else 0.0
    macro_r = float(np.mean([v["recall"] for v in per_class_best.values()])) if per_class_best else 0.0

    # Extract dataset tag from the scores CSV path structure
    # scores_csv format: results/crimes/scores/{dataset_tag}/{backbone}/fps/window_*.csv
    path_parts = scores_csv.split('/')
    if len(path_parts) >= 4 and path_parts[0] == 'results' and path_parts[1] == 'crimes' and path_parts[2] == 'scores':
        dataset_tag = path_parts[3]
    else:
        dataset_tag = dataset_tag_from(scores_csv)
    thr_out_path = build_crimes_thresholds_out_path(
        backbone=runtime.backbone,
        window_size=runtime.window_size,
        stride_percent=runtime.stride_percent,
        target_fps=runtime.target_fps,
        model_frames=runtime.model_frames,
        dataset_tag=dataset_tag,
    )
    with open(thr_out_path, "w") as f:
        json.dump({k: v["threshold"] for k, v in per_class_best.items()}, f, indent=2)
    logging.info(f"Saved tuned per-class thresholds to {thr_out_path}")

    return {
        "per_class_best": per_class_best,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "thresholds_out": thr_out_path,
        "sweep_rows": sweep_rows,
    }


def evaluate_scores(*, scores_csv: str, thresholds_json: str, runtime: RuntimeConfig, default_threshold: float = 0.5):
    """
    Evaluate per-class metrics from a saved scores CSV and thresholds JSON.

    Args:
        scores_csv: Path to scores CSV.
        thresholds_json: Path to thresholds JSON.
        runtime: Runtime configuration (used for output path layout).
        default_threshold: Fallback threshold if class-specific is missing.

    Returns:
        None. Writes results JSON to standard path.
    """
    """
    Evaluate per-class metrics from a saved scores CSV and thresholds JSON,
    then save metrics JSON to the standard runs path.
    """
    if not os.path.exists(scores_csv):
        logging.error(f"Scores CSV not found: {scores_csv}")
        return
    if not os.path.exists(thresholds_json):
        logging.error(f"Thresholds JSON not found: {thresholds_json}")
        return

    with open(thresholds_json, "r") as f:
        per_class_thresholds = json.load(f)

    all_predictions = defaultdict(list)
    all_ground_truth = defaultdict(list)

    with open(scores_csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Columns: video_path, category, <crime...>
        crime_columns = [c for c in fieldnames if c not in ("video_path", "category")]
        for row in reader:
            true_category = row.get("category", "")
            for crime in crime_columns:
                try:
                    score = float(row.get(crime, 0.0))
                except Exception:
                    score = 0.0
                all_predictions[crime].append(score)
                all_ground_truth[crime].append(1 if crime == true_category else 0)

    # Compute metrics exactly as in run_pipeline
    results = {}
    for crime in sorted(all_predictions.keys()):
        y_true = np.array(all_ground_truth[crime])
        y_pred = np.array(all_predictions[crime])

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = "N/A (only one class present)"

        crime_thr = default_threshold
        if crime in per_class_thresholds:
            try:
                crime_thr = float(per_class_thresholds[crime])
            except Exception:
                crime_thr = default_threshold
        y_bin = y_pred > crime_thr

        if np.sum(y_bin) == 0:
            precision, recall = 0.0, 0.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                precision, recall, _, _ = precision_recall_fscore_support(
                    y_true, y_bin, average='binary', zero_division=0
                )

        results[crime] = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "threshold_used": crime_thr,
        }

    # Extract dataset tag from the scores CSV path structure
    # scores_csv format: results/crimes/scores/{dataset_tag}/{backbone}/fps/window_*.csv
    path_parts = scores_csv.split('/')
    if len(path_parts) >= 4 and path_parts[0] == 'results' and path_parts[1] == 'crimes' and path_parts[2] == 'scores':
        dataset_tag = path_parts[3]
    else:
        dataset_tag = dataset_tag_from(scores_csv)
    out_path = build_crimes_results_out_path(
        backbone=runtime.backbone,
        window_size=runtime.window_size,
        stride_percent=runtime.stride_percent,
        target_fps=runtime.target_fps,
        model_frames=runtime.model_frames,
        dataset_tag=dataset_tag,
    )
    write_results_json(out_path=out_path, results=results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classic pipeline - simple CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scores = sub.add_parser("scores", help="Generate scores CSV from dataset JSON")
    p_scores.add_argument("videos_json", type=str, nargs="?", default="videos/selected_videos.json")

    p_tune = sub.add_parser("tune", help="Tune thresholds from an existing scores CSV")
    p_tune.add_argument("scores_csv", type=str)

    p_eval = sub.add_parser("eval", help="Evaluate metrics from scores CSV and thresholds JSON")
    p_eval.add_argument("scores_csv", type=str)
    p_eval.add_argument("thresholds_json", type=str)

    p_cfg = sub.add_parser("config", help="Print active config path and contents")

    args = parser.parse_args()

    # Load config defaults
    cfg_path = resolve_config_path()
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    # basic validation
    # Backward-compatible accessors
    def cfg_path_get(k):
        return cfg.get("paths", {}).get(k) or cfg.get(k)
    def cfg_rt_get(k, default=None):
        return (cfg.get("runtime", {}) or {}).get(k, cfg.get(k, default))

    required_paths = ["crime_mapping_path", "videos_base_path"]
    for k in required_paths:
        if not cfg_path_get(k):
            logging.error(f"Missing required key in config.json: paths.{k}")
            raise SystemExit(1)
    logging.info(
        f"[CONFIG] ws={cfg_rt_get('window_size', 2.0)} sp={cfg_rt_get('stride_percent', 0.5)} tfps={cfg_rt_get('target_fps', 1.6)} "
        f"backbone={cfg_rt_get('backbone', 'r3d18')} mf={cfg_rt_get('model_frames', 0)} "
        f"batched={bool(cfg_rt_get('use_batched', True))} bs={cfg_rt_get('batch_size', 4)} nw={cfg_rt_get('num_workers', 4)}"
    )
    def cfg_get(k, d):
        return cfg_rt_get(k, d)
    def cfg_tune_get(k, d):
        return (cfg.get("tuning", {}) or {}).get(k, d)

    window_size = cfg_get("window_size", 30.0)
    stride_percent = cfg_get("stride_percent", 1.0)
    target_fps = cfg_get("target_fps", 1.0)
    backbone = cfg_get("backbone", "r3d18")
    model_frames = int(cfg_get("model_frames", 0))
    use_batched = bool(cfg_get("use_batched", True))
    batch_size = int(cfg_get("batch_size", 16))
    num_workers = int(cfg_get("num_workers", 4))
    prefetch_factor = int(cfg_get("prefetch_factor", 2))
    amp_enabled = bool(cfg_get("amp_enabled", True))

    # Build RuntimeConfig and InferenceContext once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, kinetics_labels = load_backbone(backbone=cfg_get("backbone", "r3d18"), device=device)
    runtime = RuntimeConfig(
        window_size=cfg_get("window_size", 30.0),
        stride_percent=cfg_get("stride_percent", 1.0),
        target_fps=cfg_get("target_fps", 1.0),
        model_frames=int(cfg_get("model_frames", 0)),
        backbone=cfg_get("backbone", "r3d18"),
        use_batched=bool(cfg_get("use_batched", True)),
        batch_size=int(cfg_get("batch_size", 16)),
        num_workers=int(cfg_get("num_workers", 4)),
        prefetch_factor=int(cfg_get("prefetch_factor", 2)),
        amp_enabled=bool(cfg_get("amp_enabled", True)),
    )
    ctx = InferenceContext(
        model=model,
        transform=transform,
        kinetics_labels=kinetics_labels,
        device=device,
        config_path=cfg_path,
        videos_base_path=cfg_path_get("videos_base_path"),
    )

    if args.cmd == "scores":
        out_csv = generate_scores_csv(
            videos_path=args.videos_json,
            runtime=runtime,
            ctx=ctx,
        )
        logging.info(f"[SCORES] Saved to {out_csv}")
    elif args.cmd == "tune":
        thr_min = cfg_tune_get("thr_min", 0.0)
        thr_max = cfg_tune_get("thr_max", 1.0)
        thr_steps = cfg_tune_get("thr_steps", 201)
        tune_thresholds_from_scores(
            scores_csv=args.scores_csv,
            thr_min=float(thr_min),
            thr_max=float(thr_max),
            thr_steps=int(thr_steps),
            runtime=runtime,
        )
    elif args.cmd == "eval":
        evaluate_scores(
            scores_csv=args.scores_csv,
            thresholds_json=args.thresholds_json,
            runtime=runtime,
            default_threshold=cfg_get("default_threshold", 0.5),
        )
    elif args.cmd == "config":
        print(cfg_path)
        print(json.dumps(cfg, indent=2))


