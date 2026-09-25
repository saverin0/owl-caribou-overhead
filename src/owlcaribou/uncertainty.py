"""Confidence intervals by resampling source mosaics.

Patches cut from one mosaic share a flight, light, altitude, ground cover and
herd density, so they are not independent observations. Treating the 2,607
patches as independent would give intervals far narrower than the evidence
supports. The honest unit of replication is the mosaic, so this resamples the
43 mosaics with replacement — a cluster bootstrap — and recomputes every metric
from pooled counts.

Every metric reported is either a ratio of sums (precision, recall, count
bias, empty-patch rates) or a mean over patches (MAE, RMSE). Each mosaic can
therefore be reduced once to a handful of totals, and a bootstrap replicate is
just a weighted sum of those totals: thousands of replicates cost one matrix
product.

State these caveats with any interval produced here:

* 43 clusters, 26 of them containing animals, is not many. Percentile
  intervals from a cluster bootstrap with this few clusters run somewhat
  narrow. Read them as honest but optimistic, not exact.
* Mosaics range from 1 patch to 311. Large mosaics dominate each replicate,
  exactly as they dominate the point estimate.
* Model comparisons use the *same* resampled mosaics for every model (a paired
  bootstrap). That is what makes a difference between two models meaningful:
  both are scored on identical draws of the survey.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .data import GroundTruth
from .metrics import match_points

# Columns of the per-patch statistics matrix, in order.
#
# ``tp`` counts matched animals and ``hits`` matched detections. Over whole
# patches they are the same number, one per matched pair. They differ only in
# the interior analysis (:func:`owlcaribou.failures.interior_statistics`),
# where a pair may have one end inside the border band and the other outside:
# recall is then counted over interior animals and precision over interior
# detections, so each side needs its own count.
STAT_COLUMNS = (
    "tp",
    "fp",
    "fn",
    "actual",
    "predicted",
    "abs_error",
    "sq_error",
    "error",
    "empty",
    "empty_with_detection",
    "empty_false_count",
    "patches",
    "hits",
)
# Column index by name, for code that reads the matrix.
COLUMN = {name: index for index, name in enumerate(STAT_COLUMNS)}

METRICS = (
    "precision",
    "recall",
    "f1_score",
    "count_mae",
    "count_rmse",
    "mean_count_bias",
    "aggregate_count_bias_percent",
    "empty_patch_false_positive_rate",
    "empty_patch_mean_false_count",
)

# How "better" is judged, used only to phrase comparisons. Signed bias metrics
# are better the closer they sit to zero, so neither direction wins by itself.
LOWER_IS_BETTER = {
    "count_mae",
    "count_rmse",
    "empty_patch_false_positive_rate",
    "empty_patch_mean_false_count",
}
CLOSER_TO_ZERO_IS_BETTER = {
    "mean_count_bias",
    "aggregate_count_bias_percent",
}


def patch_row(
    *,
    tp: int,
    fp: int,
    fn: int,
    actual: int,
    predicted: int,
    empty: bool,
    hits: int | None = None,
) -> np.ndarray:
    """One patch's statistics in :data:`STAT_COLUMNS` order.

    ``actual`` and ``predicted`` are the counts being scored on this patch;
    ``empty`` says whether the patch is empty ground, which is what the
    empty-patch metrics divide by. ``hits`` defaults to ``tp``.
    """
    error = predicted - actual
    return np.array(
        (
            tp,
            fp,
            fn,
            actual,
            predicted,
            abs(error),
            error * error,
            error,
            1.0 if empty else 0.0,
            1.0 if empty and predicted > 0 else 0.0,
            float(predicted) if empty else 0.0,
            1.0,
            tp if hits is None else hits,
        ),
        dtype=np.float64,
    )


def patch_statistics(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
) -> np.ndarray:
    """One row of summable statistics per patch, columns as :data:`STAT_COLUMNS`.

    Matching is done once per patch here, so resampling never repeats it.
    """
    rows = np.zeros((len(patch_names), len(STAT_COLUMNS)), dtype=np.float64)
    empty_prediction = np.empty((0, 2), dtype=np.float64)

    for row, name in enumerate(patch_names):
        actual = truth.for_patch(name)
        predicted = predictions.get(name, empty_prediction)
        tp = match_points(actual, predicted, radius)
        n_actual, n_predicted = len(actual), len(predicted)
        rows[row] = patch_row(
            tp=tp,
            fp=n_predicted - tp,
            fn=n_actual - tp,
            actual=n_actual,
            predicted=n_predicted,
            empty=n_actual == 0,
        )
    return rows


def group_totals(
    statistics: np.ndarray,
    groups: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Sum patch statistics within each group. Labels are returned sorted."""
    if len(groups) != len(statistics):
        raise ValueError(f"{len(groups)} group labels for {len(statistics)} patches")
    labels = sorted(set(groups))
    index = {label: position for position, label in enumerate(labels)}
    totals = np.zeros((len(labels), statistics.shape[1]), dtype=np.float64)
    np.add.at(totals, [index[g] for g in groups], statistics)
    return labels, totals


def metrics_from_totals(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Turn pooled totals into metrics. Works on one row or a stack of replicates.

    A replicate that happens to draw no empty patches, or no animals, has an
    undefined value for the metrics that divide by those; it becomes NaN and
    is excluded from the percentiles rather than counted as zero.
    """
    t = np.atleast_2d(np.asarray(totals, dtype=np.float64))
    c = {name: t[:, i] for name, i in COLUMN.items()}

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = c["hits"] / (c["hits"] + c["fp"])   # over detections
        recall = c["tp"] / (c["tp"] + c["fn"])          # over animals
        f1 = 2 * precision * recall / (precision + recall)
        out = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "count_mae": c["abs_error"] / c["patches"],
            "count_rmse": np.sqrt(c["sq_error"] / c["patches"]),
            "mean_count_bias": c["error"] / c["patches"],
            "aggregate_count_bias_percent": 100.0 * c["error"] / c["actual"],
            "empty_patch_false_positive_rate": c["empty_with_detection"] / c["empty"],
            "empty_patch_mean_false_count": c["empty_false_count"] / c["empty"],
        }
    return out


def bootstrap_weights(n_groups: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """How often each group is drawn in each replicate, shape ``(n_boot, n_groups)``.

    Drawing ``n_groups`` groups with replacement is exactly a multinomial count
    over the groups, which avoids materialising the draws themselves.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot).astype(
        np.float64
    )


@dataclass
class BootstrapResult:
    """Point estimates, intervals and the replicates behind them."""

    labels: list[str]
    n_boot: int
    confidence: float
    seed: int
    point: dict[str, dict[str, float]] = field(default_factory=dict)
    low: dict[str, dict[str, float]] = field(default_factory=dict)
    high: dict[str, dict[str, float]] = field(default_factory=dict)
    replicates: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    undefined: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        return list(self.point)

    def table(self) -> list[dict]:
        """One row per model and metric, ready to write as CSV."""
        rows = []
        for model in self.models:
            for metric in METRICS:
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "estimate": self.point[model][metric],
                        "ci_low": self.low[model][metric],
                        "ci_high": self.high[model][metric],
                        "confidence": self.confidence,
                        "undefined_replicates": self.undefined[model][metric],
                    }
                )
        return rows

    def difference(self, a: str, b: str, metric: str) -> dict[str, float]:
        """Paired comparison of ``a`` minus ``b`` on identical resampled mosaics.

        ``share_a_better`` is the fraction of replicates in which ``a`` beats
        ``b`` on this metric: lower wins for error metrics, closer to zero wins
        for signed bias, higher wins for everything else.
        """
        da = self.replicates[a][metric]
        db = self.replicates[b][metric]
        diff = da - db
        valid = np.isfinite(diff)
        alpha = (1.0 - self.confidence) / 2.0
        if metric in CLOSER_TO_ZERO_IS_BETTER:
            better = np.abs(da[valid]) < np.abs(db[valid])
        elif metric in LOWER_IS_BETTER:
            better = diff[valid] < 0
        else:
            better = diff[valid] > 0
        return {
            "metric": metric,
            "a": a,
            "b": b,
            "estimate": self.point[a][metric] - self.point[b][metric],
            "ci_low": float(np.quantile(diff[valid], alpha)),
            "ci_high": float(np.quantile(diff[valid], 1.0 - alpha)),
            "share_a_better": float(better.mean()),
            "replicates": int(valid.sum()),
        }


def cluster_bootstrap(
    statistics_by_model: Mapping[str, np.ndarray],
    groups: Sequence[str],
    *,
    n_boot: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile intervals for every metric, resampling whole groups.

    All models share one set of resampling weights, so any two can be compared
    replicate by replicate with :meth:`BootstrapResult.difference`.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if not statistics_by_model:
        raise ValueError("no models to bootstrap")

    # Callers score every model over ``patch_names`` in one order, so row i of
    # each matrix is the same patch; only the row count is checked here, by
    # group_totals, which rejects a matrix that does not match ``groups``.
    labels: list[str] = sorted(set(groups))
    totals_by_model: dict[str, np.ndarray] = {}
    for model, statistics in statistics_by_model.items():
        totals_by_model[model] = group_totals(statistics, groups)[1]

    weights = bootstrap_weights(len(labels), n_boot, seed)
    alpha = (1.0 - confidence) / 2.0
    result = BootstrapResult(labels=labels, n_boot=n_boot, confidence=confidence, seed=seed)

    for model, totals in totals_by_model.items():
        point = metrics_from_totals(totals.sum(axis=0))
        replicate = metrics_from_totals(weights @ totals)

        result.point[model] = {m: float(point[m][0]) for m in METRICS}
        result.replicates[model] = replicate
        result.low[model], result.high[model], result.undefined[model] = {}, {}, {}
        for metric in METRICS:
            values = replicate[metric]
            finite = values[np.isfinite(values)]
            result.undefined[model][metric] = int(values.size - finite.size)
            result.low[model][metric] = float(np.quantile(finite, alpha))
            result.high[model][metric] = float(np.quantile(finite, 1.0 - alpha))

    return result


def per_group_metrics(
    statistics: np.ndarray,
    groups: Sequence[str],
) -> list[dict]:
    """Metrics computed inside each group, to show how much groups differ."""
    labels, totals = group_totals(statistics, groups)
    per_group = metrics_from_totals(totals)
    rows = []
    for position, label in enumerate(labels):
        row = {
            "group": label,
            "patches": int(totals[position, COLUMN["patches"]]),
            "annotated_points": int(totals[position, COLUMN["actual"]]),
            "empty_patches": int(totals[position, COLUMN["empty"]]),
        }
        row.update({metric: float(per_group[metric][position]) for metric in METRICS})
        rows.append(row)
    return rows
