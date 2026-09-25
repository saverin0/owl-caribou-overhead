"""Point-detection metrics for the locked CAH split.

Matching is greedy over the closest admissible pairs, the rule the upstream
evaluator applies, so the numbers stay comparable with it. An optimal
assignment would score a little differently on crowded patches, so it is
deliberately not used here: the point is comparability, not the best possible
number.

Two properties matter more than they look:

* **Every patch counts, including the 755 with no animals.** They cannot
  contribute true positives, only false ones, and they are the entire basis of
  the empty-patch false-positive rate. Evaluating only annotated patches would
  flatter every model.

* **Counting error is computed per patch, then aggregated.** Aggregate count
  bias is reported separately because over- and under-counting cancel in a sum
  while they accumulate in a mean absolute error. A model can be excellent at
  totals and poor at any individual patch.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import GroundTruth, mosaic_from_patch_name


@dataclass(frozen=True)
class PointMetrics:
    """Detection and counting quality at one matching radius."""

    radius: float
    patches: int
    gt_points: int
    predicted_points: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    count_mae: float
    count_rmse: float
    mean_count_bias: float
    aggregate_count_bias_percent: float
    empty_patches: int
    empty_patch_false_positive_rate: float
    empty_patch_mean_false_count: float

    def as_row(self, **extra) -> dict:
        return {**extra, **asdict(self)}


def match_pairs(
    truth: np.ndarray,
    predicted: np.ndarray,
    radius: float,
) -> list[tuple[int, int]]:
    """One-to-one ``(prediction, annotation)`` index pairs within ``radius``.

    Greedy, closest pair first. Each prediction and each annotation can be used
    once. Ties are resolved by the sort, which is stable, so the result does
    not depend on input order beyond genuinely equal distances.
    """
    if len(truth) == 0 or len(predicted) == 0:
        return []

    distance = np.linalg.norm(predicted[:, None, :] - truth[None, :, :], axis=2)
    candidates = np.argwhere(distance <= radius)
    if candidates.size == 0:
        return []

    order = np.argsort(distance[candidates[:, 0], candidates[:, 1]], kind="stable")
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for index in order:
        p, g = int(candidates[index, 0]), int(candidates[index, 1])
        if p in used_pred or g in used_truth:
            continue
        used_pred.add(p)
        used_truth.add(g)
        pairs.append((p, g))
    return pairs


def match_points(
    truth: np.ndarray,
    predicted: np.ndarray,
    radius: float,
) -> int:
    """Count one-to-one matches within ``radius``; see :func:`match_pairs`."""
    return len(match_pairs(truth, predicted, radius))


def evaluate(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
) -> PointMetrics:
    """Score one model over every patch in ``patch_names``.

    ``predictions`` maps a patch name to an ``(N, 2)`` array of x, y in patch
    pixels. A patch absent from it is treated as having no detections, not as
    unevaluated — that is what keeps the empty patches in the denominator.
    """
    total_tp = total_fp = total_fn = 0
    errors: list[int] = []
    actuals: list[int] = []
    empty_predicted: list[int] = []

    for name in patch_names:
        actual = truth.for_patch(name)
        predicted = predictions.get(name)
        if predicted is None:
            predicted = np.empty((0, 2), dtype=np.float64)

        tp = match_points(actual, predicted, radius)
        total_tp += tp
        total_fp += len(predicted) - tp
        total_fn += len(actual) - tp

        errors.append(len(predicted) - len(actual))
        actuals.append(len(actual))
        if len(actual) == 0:
            empty_predicted.append(len(predicted))

    error = np.asarray(errors, dtype=np.float64)
    actual_total = int(sum(actuals))
    empty = np.asarray(empty_predicted, dtype=np.float64)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return PointMetrics(
        radius=float(radius),
        patches=len(patch_names),
        gt_points=actual_total,
        predicted_points=total_tp + total_fp,   # over patch_names only, so it equals tp + fp
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        count_mae=float(np.abs(error).mean()) if error.size else 0.0,
        count_rmse=float(math.sqrt(np.square(error).mean())) if error.size else 0.0,
        mean_count_bias=float(error.mean()) if error.size else 0.0,
        aggregate_count_bias_percent=(
            float(100.0 * error.sum() / actual_total) if actual_total else float("nan")
        ),
        empty_patches=int(empty.size),
        empty_patch_false_positive_rate=float((empty > 0).mean()) if empty.size else 0.0,
        empty_patch_mean_false_count=float(empty.mean()) if empty.size else 0.0,
    )


def load_predictions(csv_path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Read a detections table this project wrote back into arrays.

    Only patches with at least one detection appear as keys; the per-image
    table records the rest. :func:`evaluate` treats an absent patch as zero
    detections, so the two agree.
    """
    from .io import read_csv

    collected: dict[str, list[tuple[float, float]]] = {}
    for row in read_csv(csv_path):
        name = str(row["images"]).strip()
        collected.setdefault(name, []).append((float(row["x"]), float(row["y"])))
    return {
        name: np.asarray(points, dtype=np.float64).reshape(-1, 2)
        for name, points in collected.items()
    }


def count_table(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
) -> list[dict]:
    """Per-patch actual vs predicted counts, for diagnostics and plots.

    The mosaic comes from the patch filename, as everywhere else, so empty
    patches and background-only mosaics are labelled too (see
    :func:`owlcaribou.data.verify_mosaic_rule`).
    """
    rows = []
    for name in patch_names:
        actual = len(truth.for_patch(name))
        predicted = len(predictions.get(name, ()))
        rows.append(
            {
                "images": name,
                "actual": actual,
                "predicted": predicted,
                "error": predicted - actual,
                "mosaic": mosaic_from_patch_name(name),
            }
        )
    return rows


def score_radii(
    predictions_by_model: Mapping[str, Mapping[str, np.ndarray]],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radii: Iterable[float] = (20.0, 40.0),
) -> list[dict]:
    """Score every model at every matching radius, in the order given.

    A radius is a matching tolerance, not a confidence threshold; the
    confidence sweep lives in :mod:`owlcaribou.sweep`.
    """
    radii = tuple(radii)  # iterated once per model, so a generator must not run dry
    rows: list[dict] = []
    for model, predictions in predictions_by_model.items():
        for radius in radii:
            rows.append(evaluate(predictions, truth, patch_names, radius).as_row(model=model))
    return rows
