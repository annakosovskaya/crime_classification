"""Utility modules for classic_approach."""

from .paths import (
    build_crimes_scores_out_path,
    build_crimes_results_out_path,
    build_crimes_thresholds_out_path,
    resolve_config_path,
    dataset_tag_from,
)
from .metrics import (
    compute_auc,
    apply_threshold,
    compute_precision_recall,
    select_thresholds_for_classes,
)
from .io import write_scores_csv, write_results_json, append_validation_row
from .runner_utils import (
    load_crime_mapping,
    build_stride_and_frames,
    aggregate_predictions_max,
    get_device,
    autocast_if_cuda,
)

__all__ = [
    # paths
    "build_crimes_scores_out_path",
    "build_crimes_results_out_path",
    "build_crimes_thresholds_out_path",
    "resolve_config_path",
    "dataset_tag_from",
    # metrics
    "compute_auc",
    "apply_threshold",
    "compute_precision_recall",
    "select_thresholds_for_classes",
    # io
    "write_scores_csv",
    "write_results_json",
    "append_validation_row",
    # runner utils
    "load_crime_mapping",
    "build_stride_and_frames",
    "aggregate_predictions_max",
    "get_device",
    "autocast_if_cuda",
]


