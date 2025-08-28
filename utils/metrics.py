from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import warnings
from sklearn.exceptions import UndefinedMetricWarning


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Compute ROC-AUC if both classes are present, else return None."""
    unique = np.unique(y_true)
    if len(unique) > 1:
        return float(roc_auc_score(y_true, y_pred))
    return None


def apply_threshold(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize predictions by threshold (> threshold)."""
    return (y_pred > float(threshold)).astype(int)


def compute_precision_recall(y_true: np.ndarray, y_bin: np.ndarray) -> Tuple[float, float]:
    """Compute precision and recall with safe handling of empty positive predictions."""
    if np.sum(y_bin) == 0:
        return 0.0, 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_bin, average='binary', zero_division=0
        )
    return float(precision), float(recall)


def sweep_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thr_values: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    """
    Sweep thresholds and return best metrics plus per-threshold rows.

    Returns:
        best: {threshold, precision, recall, f1}
        rows: list of {threshold, precision, recall, f1}
    """
    best_thr: Optional[float] = None
    best_f1: float = -1.0
    best_p: float = 0.0
    best_r: float = 0.0
    rows: List[Dict[str, float]] = []

    for thr in thr_values:
        y_bin = apply_threshold(y_pred, float(thr))
        p, r = compute_precision_recall(y_true, y_bin)
        f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)
        rows.append({
            "threshold": float(thr),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        })
        if f1 > best_f1:
            best_f1 = float(f1)
            best_p = float(p)
            best_r = float(r)
            best_thr = float(thr)

    best = {
        "threshold": best_thr,
        "precision": best_p,
        "recall": best_r,
        "f1": best_f1,
    }
    return best, rows


def select_thresholds_for_classes(
    *,
    class_to_true: Dict[str, np.ndarray],
    class_to_pred: Dict[str, np.ndarray],
    thr_values: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """For each class, sweep thresholds and choose the best by F1.

    Returns:
        per_class_best: {class: {threshold, precision, recall, f1}}
        sweep_rows: list of rows with class, threshold, precision, recall, f1
    """
    per_class_best: Dict[str, Dict[str, float]] = {}
    sweep_rows: List[Dict[str, float]] = []
    for crime in sorted(class_to_pred.keys()):
        y_true = class_to_true[crime]
        y_pred = class_to_pred[crime]
        best, rows = sweep_thresholds(y_true, y_pred, thr_values)
        per_class_best[crime] = best  # type: ignore[assignment]
        for r in rows:
            sweep_rows.append({
                "class": crime,
                **r,  # threshold, precision, recall, f1
            })
    return per_class_best, sweep_rows


