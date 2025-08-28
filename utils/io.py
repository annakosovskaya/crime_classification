import csv
import json
import logging
import os
from typing import List, Dict, Any, Iterable


def write_scores_csv(
    *,
    out_csv: str,
    rows: Iterable[Dict[str, Any]],
    crimes_sorted: List[str],
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    list_rows = list(rows)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "category", *crimes_sorted])
        for r in list_rows:
            row_vals = [r.get("video_path", ""), r.get("category", "")]
            row_vals.extend([r.get(c, 0.0) for c in crimes_sorted])
            writer.writerow(row_vals)
    logging.info(f"[scores csv] wrote {len(list_rows)} rows → {out_csv}")


def write_results_json(*, out_path: str, results: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"[results json] saved → {out_path}")


def append_validation_row(
    *,
    out_csv: str,
    header: List[str],
    row: List[Any],
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


