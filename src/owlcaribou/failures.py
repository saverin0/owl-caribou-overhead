"""Where a model goes wrong, and why.

Every point on every patch gets a label. A detection is either a hit or one of
five kinds of false alarm; an annotated animal is either found or one of three
kinds of miss. The kinds are geometric facts about the patch, not guesses about
the model, so they can be counted and then checked by eye:

False alarms, tested in this order:

* ``empty ground`` - the patch has no annotated animals at all.
* ``edge`` - within :data:`EDGE_MARGIN` px of the patch border. Usually an
  animal cut by the tile border whose annotation lies in the neighbouring
  patch, which this patch's ground truth cannot know about. Tested before
  proximity, because inspection showed that border detections near an
  annotated animal were mostly a second animal, not a double count.
* ``double hit`` - within :data:`NEAR` px of an animal that another detection
  already matched. One animal, counted twice.
* ``placement off`` - within :data:`NEAR` px of an animal that stayed unmatched:
  the detector found it but put the point too far away to count.
* ``away from animals`` - none of the above. Despite the name, most inspected
  cases were real animals the ground truth does not list; see below.

Misses, tested in this order:

* ``edge`` - the animal is within :data:`EDGE_MARGIN` px of the border, so part
  of it may be cut off by the tile boundary.
* ``touching`` - another annotated animal is within :data:`TOUCHING` px; at
  this spacing two animals can merge into one peak.
* ``clear view`` - none of the above.

The categories describe where errors happen, not whether they are errors.
OWL-D's output was inspected at full resolution, twelve errors of each kind
spread over as many mosaics as the kind allows (twelve, except false alarms on
empty ground: 16 in all, from eight mosaics). Edge false alarms were real caribou cut
by the tile border in about 11 of 12; false alarms away from animals were real
caribou missing from the annotations in about 8 of 12; false alarms on empty
ground were mostly genuine mistakes (dark mud on snow, pale patches on mud).
Misses in clear view had no single cause. Scored against the ground truth as
given, the real animals count as false alarms, so measured precision is a lower
bound and the penalty grows with a model's recall. The headline metrics still
score against the ground truth as given; :func:`crops` checks any single case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .data import PATCH_SIZE, GroundTruth, mosaic_from_patch_name
from .metrics import match_pairs
from .uncertainty import COLUMN, STAT_COLUMNS, metrics_from_totals, patch_row

EDGE_MARGIN = 16.0
NEAR = 40.0
TOUCHING = 12.0

HIT = "hit"
FOUND = "found"
FALSE_ALARM_KINDS = ("empty ground", "double hit", "placement off", "edge", "away from animals")
MISS_KINDS = ("edge", "touching", "clear view")


def near_edge(points: np.ndarray, margin: float = EDGE_MARGIN, size: int = PATCH_SIZE) -> np.ndarray:
    """Boolean per point: inside the border band of width ``margin``."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    upper = size - 1 - margin
    return np.any((points < margin) | (points > upper), axis=1)


@dataclass
class PatchOutcome:
    """Every annotated and detected point on one patch, labelled."""

    name: str
    mosaic: str
    truth: np.ndarray
    truth_kind: list[str]
    predicted: np.ndarray
    predicted_kind: list[str]

    @property
    def tp(self) -> int:
        return self.truth_kind.count(FOUND)

    @property
    def fn(self) -> int:
        return len(self.truth_kind) - self.tp

    @property
    def fp(self) -> int:
        return len(self.predicted_kind) - self.predicted_kind.count(HIT)

    def misses(self, kind: str | None = None) -> int:
        if kind is None:
            return self.fn
        return self.truth_kind.count(kind)

    def false_alarms(self, kind: str | None = None) -> int:
        if kind is None:
            return self.fp
        return self.predicted_kind.count(kind)


def classify(name: str, truth: np.ndarray, predicted: np.ndarray, radius: float) -> PatchOutcome:
    """Label each point on one patch."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 2)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    pairs = match_pairs(truth, predicted, radius)
    matched_pred = {p for p, _ in pairs}
    matched_truth = {g for _, g in pairs}

    truth_kind: list[str] = []
    edge_truth = near_edge(truth)
    for index in range(len(truth)):
        if index in matched_truth:
            truth_kind.append(FOUND)
            continue
        others = np.delete(truth, index, axis=0)
        touching = len(others) > 0 and np.linalg.norm(others - truth[index], axis=1).min() <= TOUCHING
        if edge_truth[index]:
            truth_kind.append("edge")
        elif touching:
            truth_kind.append("touching")
        else:
            truth_kind.append("clear view")

    predicted_kind: list[str] = []
    edge_pred = near_edge(predicted)
    for index in range(len(predicted)):
        if index in matched_pred:
            predicted_kind.append(HIT)
            continue
        if len(truth) == 0:
            predicted_kind.append("empty ground")
            continue
        # The border band is tested before proximity: inspected at full
        # resolution, most "double hits" inside the band were a second real
        # animal cut by the tile border, not one animal counted twice.
        if edge_pred[index]:
            predicted_kind.append("edge")
            continue
        distance = np.linalg.norm(truth - predicted[index], axis=1)
        nearest = int(distance.argmin())
        if distance[nearest] <= NEAR:
            predicted_kind.append("double hit" if nearest in matched_truth else "placement off")
        else:
            predicted_kind.append("away from animals")

    return PatchOutcome(
        name=name,
        mosaic=mosaic_from_patch_name(name),
        truth=truth,
        truth_kind=truth_kind,
        predicted=predicted,
        predicted_kind=predicted_kind,
    )


def analyse(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
) -> list[PatchOutcome]:
    empty = np.empty((0, 2), dtype=np.float64)
    return [
        classify(name, truth.for_patch(name), predictions.get(name, empty), radius)
        for name in patch_names
    ]


def breakdown(outcomes: Iterable[PatchOutcome]) -> dict[str, dict[str, int]]:
    """How many errors of each kind, across all patches."""
    outcomes = list(outcomes)
    return {
        "false_alarms": {k: sum(o.false_alarms(k) for o in outcomes) for k in FALSE_ALARM_KINDS},
        "misses": {k: sum(o.misses(k) for o in outcomes) for k in MISS_KINDS},
    }


def _scores(outcomes: Sequence[PatchOutcome]) -> dict[str, float]:
    tp = sum(o.tp for o in outcomes)
    fp = sum(o.fp for o in outcomes)
    fn = sum(o.fn for o in outcomes)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall and np.isfinite(precision + recall) else float("nan"))
    return {"patches": len(outcomes), "animals": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1_score": f1}


def resolution(outcome: PatchOutcome) -> str:
    """Ground sample distance recorded in the mosaic name, e.g. ``50 mm``."""
    found = re.search(r"_(\d+)mm_mosaic", outcome.mosaic)
    return f"{found.group(1)} mm" if found else "unknown"


def density(outcome: PatchOutcome) -> str:
    n = len(outcome.truth)
    if n == 0:
        return "0 (empty)"
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 20:
        return "6-20"
    return "21+"


DENSITY_ORDER = ("0 (empty)", "1", "2-5", "6-20", "21+")


def by_group(
    outcomes: Sequence[PatchOutcome],
    key: Callable[[PatchOutcome], str],
    order: Sequence[str] | None = None,
) -> list[dict]:
    """Detection scores within each group of patches.

    Descriptive only. Groups such as resolution follow the survey's flight
    plan, so they differ in location and herd as well; a gap between groups is
    not by itself evidence that the grouping variable causes it.
    """
    grouped: dict[str, list[PatchOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(key(outcome), []).append(outcome)
    labels = [g for g in (order or sorted(grouped)) if g in grouped]
    return [{"group": label, **_scores(grouped[label])} for label in labels]


def edge_concentration(
    outcomes: Sequence[PatchOutcome],
    margins: Iterable[float] = (8.0, 16.0, 32.0),
) -> list[dict]:
    """Share of all animals vs share of missed animals inside each border band."""
    animals = [o.truth for o in outcomes if len(o.truth)]
    missed = [o.truth[[k != FOUND for k in o.truth_kind]] for o in outcomes if o.fn]
    detections = [o.predicted for o in outcomes if len(o.predicted)]
    wrong = [o.predicted[[k != HIT for k in o.predicted_kind]] for o in outcomes if o.fp]
    stack = lambda parts: np.concatenate(parts) if parts else np.empty((0, 2))  # noqa: E731
    animals, missed, detections, wrong = map(stack, (animals, missed, detections, wrong))

    rows = []
    for margin in margins:
        share = lambda pts: float(near_edge(pts, margin).mean()) if len(pts) else float("nan")  # noqa: E731
        rows.append({
            "margin_px": margin,
            "animals_in_band": share(animals),
            "misses_in_band": share(missed),
            "detections_in_band": share(detections),
            "false_alarms_in_band": share(wrong),
        })
    return rows


def _patch_rows(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Whole-patch and interior statistics from one matching per patch, plus
    the number of matched pairs that straddle the band line.

    The whole-patch rows are exactly what
    :func:`owlcaribou.uncertainty.patch_statistics` returns; computing both
    from the same pairs halves the matching work of a sweep that needs both.
    See :func:`interior_statistics` for the interior rows.
    """
    full = np.zeros((len(patch_names), len(STAT_COLUMNS)), dtype=np.float64)
    rows = np.zeros((len(patch_names), len(STAT_COLUMNS)), dtype=np.float64)
    empty_prediction = np.empty((0, 2), dtype=np.float64)
    straddling = 0

    for row, name in enumerate(patch_names):
        actual = truth.for_patch(name)
        predicted = predictions.get(name, empty_prediction)
        # Match on the whole patch, exactly as the headline metrics do.
        pairs = match_pairs(actual, predicted, radius)
        matched = len(pairs)
        full[row] = patch_row(
            tp=matched,
            fp=len(predicted) - matched,
            fn=len(actual) - matched,
            actual=len(actual),
            predicted=len(predicted),
            empty=len(actual) == 0,
        )

        band_truth = near_edge(actual, margin)
        band_pred = near_edge(predicted, margin)
        matched_truth = np.zeros(len(actual), dtype=bool)
        matched_pred = np.zeros(len(predicted), dtype=bool)
        for p, g in pairs:
            matched_pred[p] = True
            matched_truth[g] = True
            if band_pred[p] != band_truth[g]:
                straddling += 1

        interior_truth = ~band_truth
        interior_pred = ~band_pred
        n_actual = int(interior_truth.sum())
        n_predicted = int(interior_pred.sum())
        tp = int((interior_truth & matched_truth).sum())    # interior animals found
        hits = int((interior_pred & matched_pred).sum())    # interior detections that found one
        rows[row] = patch_row(
            tp=tp,
            fp=n_predicted - hits,
            fn=n_actual - tp,
            actual=n_actual,
            predicted=n_predicted,
            empty=len(actual) == 0,   # empty ground means no animals anywhere on the patch
            hits=hits,
        )
    return full, rows, straddling


def interior_statistics(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
    margin: float = EDGE_MARGIN,
) -> np.ndarray:
    """Per-patch statistics scoring only the interior of each patch.

    A sensitivity check for the tiling, not a replacement for the headline.
    Matching is done on the whole patch first, then the border band is left
    out of the *counting*:

    * recall is taken over interior animals only - an interior animal counts
      as found wherever the detection that matched it sits;
    * precision is taken over interior detections only - an interior detection
      counts as a hit wherever the animal it matched sits.

    Trimming the band out of both point sets *before* matching, the obvious
    alternative, manufactures errors: a matched pair with one end inside the
    band and the other outside loses its partner and becomes an invented
    error, a miss or a false alarm depending on which end sits in the band,
    that no threshold can undo. Matching first keeps every pair the headline
    found, so the interior numbers differ from the full-patch numbers only
    because of what happens in the band.

    Empty ground keeps its full-patch meaning, a patch with no animals at all,
    so the empty-patch metrics describe the same 755 patches as the headline.
    The count columns (actual, predicted, the errors, the empty-patch false
    counts) are over interior points too. The returned matrix has the columns
    of :data:`owlcaribou.uncertainty.STAT_COLUMNS` and feeds the cluster
    bootstrap unchanged.
    """
    return _patch_rows(predictions, truth, patch_names, radius, margin)[1]


def full_and_interior_statistics(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
    margin: float = EDGE_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """``(whole-patch, interior)`` statistics from one matching per patch.

    The first matrix equals :func:`owlcaribou.uncertainty.patch_statistics`,
    the second :func:`interior_statistics`, each bit for bit.
    """
    full, interior, _ = _patch_rows(predictions, truth, patch_names, radius, margin)
    return full, interior


def interior_scores(
    predictions: Mapping[str, np.ndarray],
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
    margin: float = EDGE_MARGIN,
) -> dict[str, float]:
    """Pooled interior scores (see :func:`interior_statistics`).

    ``tp`` counts interior animals found and ``hits`` interior detections
    that found an animal; they differ by the pairs that straddle the band
    line. ``straddling_pairs`` is how many matched pairs had one end in the
    band and the other outside. Each such pair loses its partner under
    trim-then-rematch, so it is an upper bound on the misses plus false
    alarms that estimator would have invented (re-matching after trimming can
    absorb some).
    """
    _, rows, straddling = _patch_rows(predictions, truth, patch_names, radius, margin)
    totals = rows.sum(axis=0)
    pooled = metrics_from_totals(totals)
    return {
        "patches": len(patch_names),
        "animals": int(totals[COLUMN["actual"]]),
        "detections": int(totals[COLUMN["predicted"]]),
        "tp": int(totals[COLUMN["tp"]]),
        "hits": int(totals[COLUMN["hits"]]),
        "fp": int(totals[COLUMN["fp"]]),
        "fn": int(totals[COLUMN["fn"]]),
        "precision": float(pooled["precision"][0]),
        "recall": float(pooled["recall"][0]),
        "f1_score": float(pooled["f1_score"][0]),
        "straddling_pairs": straddling,
    }


def worst(
    outcomes: Sequence[PatchOutcome],
    count: Callable[[PatchOutcome], int],
    n: int = 8,
) -> list[PatchOutcome]:
    """The ``n`` patches scoring highest on ``count``, ties broken by name."""
    ranked = sorted((o for o in outcomes if count(o) > 0), key=lambda o: (-count(o), o.name))
    return ranked[:n]


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------

STYLE = {
    FOUND: dict(marker="o", facecolors="none", edgecolors="#1b9e77", s=110, linewidths=1.4),
    "miss": dict(marker="o", facecolors="none", edgecolors="#d95f02", s=260, linewidths=2.6),
    HIT: dict(marker="+", color="#1b9e77", s=60, linewidths=1.2),
    "false": dict(marker="x", color="#e7298a", s=150, linewidths=2.6),
}


def draw(axis, image: np.ndarray, outcome: PatchOutcome, margin: float = EDGE_MARGIN) -> None:
    """One patch with every point marked and the border band outlined."""
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    axis.imshow(image)
    size = image.shape[0]
    axis.add_patch(Rectangle((margin, margin), size - 1 - 2 * margin, size - 1 - 2 * margin,
                             fill=False, linestyle="--", linewidth=0.8, edgecolor="white", alpha=0.7))

    kinds = np.array(outcome.truth_kind)
    if len(outcome.truth):
        found = kinds == FOUND
        if found.any():
            axis.scatter(*outcome.truth[found].T, **STYLE[FOUND])
        if (~found).any():
            axis.scatter(*outcome.truth[~found].T, **STYLE["miss"])

    kinds = np.array(outcome.predicted_kind)
    if len(outcome.predicted):
        hit = kinds == HIT
        if hit.any():
            axis.scatter(*outcome.predicted[hit].T, **STYLE[HIT])
        if (~hit).any():
            axis.scatter(*outcome.predicted[~hit].T, **STYLE["false"])

    short = outcome.name.replace("CAH_20220630_", "").replace("_post_processed_seamline_", " ")
    axis.set_title(f"{short}\n{len(outcome.truth)} animals | missed {outcome.fn} | "
                   f"false {outcome.fp}", fontsize=7)
    axis.set_xlim(0, size - 1)
    axis.set_ylim(size - 1, 0)
    axis.axis("off")


def error_points(
    outcomes: Sequence[PatchOutcome],
    kind: str,
    of: str = "false_alarm",
    n: int = 12,
) -> list[tuple[PatchOutcome, np.ndarray]]:
    """Up to ``n`` individual errors of one kind, spread across mosaics.

    Takes one error from each mosaic in turn, then goes round again, with at
    most two per patch. Filename order would cluster the picks in whichever
    mosaic sorts first, and a sample from one flight cannot say whether a
    pattern holds across the survey. Deterministic, so reruns match.
    """
    by_mosaic: dict[str, list[tuple[PatchOutcome, np.ndarray]]] = {}
    for outcome in sorted(outcomes, key=lambda o: o.name):
        points, kinds = ((outcome.predicted, outcome.predicted_kind) if of == "false_alarm"
                         else (outcome.truth, outcome.truth_kind))
        matching = [point for point, label in zip(points, kinds) if label == kind][:2]
        by_mosaic.setdefault(outcome.mosaic, []).extend((outcome, point) for point in matching)

    queues = [by_mosaic[mosaic] for mosaic in sorted(by_mosaic)]
    picked: list[tuple[PatchOutcome, np.ndarray]] = []
    depth = 0
    while len(picked) < n and any(depth < len(queue) for queue in queues):
        for queue in queues:
            if depth < len(queue) and len(picked) < n:
                picked.append(queue[depth])
        depth += 1
    return picked


def crops(
    picks: Sequence[tuple[PatchOutcome, np.ndarray]],
    image_dir: str | Path,
    title: str,
    path: str | Path | None = None,
    half: int = 48,
    columns: int = 6,
):
    """Zoomed views around individual errors, at full pixel resolution.

    Thumbnails of a whole patch hide what matters here: whether a "false alarm"
    sits on half an animal cut by the border, or on an animal the ground truth
    does not list. Area outside the patch is drawn black and the border yellow.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if not picks:
        return None
    rows = int(np.ceil(len(picks) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.9 * columns, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    cache: dict[str, np.ndarray] = {}
    for axis, (outcome, (x, y)) in zip(axes, picks):
        if outcome.name not in cache:
            with Image.open(Path(image_dir) / outcome.name) as handle:
                cache[outcome.name] = np.asarray(handle.convert("RGB"))
        image = cache[outcome.name]
        last = image.shape[0] - 1
        x0, x1 = int(max(0, x - half)), int(min(last, x + half))
        y0, y1 = int(max(0, y - half)), int(min(last, y + half))
        axis.set_facecolor("black")
        axis.imshow(image[y0:y1 + 1, x0:x1 + 1], interpolation="nearest",
                    extent=(x0 - 0.5, x1 + 0.5, y1 + 0.5, y0 - 0.5))

        # every labelled point in view, so the context is visible too
        for point, label in zip(outcome.truth, outcome.truth_kind):
            style = STYLE[FOUND] if label == FOUND else STYLE["miss"]
            axis.scatter([point[0]], [point[1]], **style)
        for point, label in zip(outcome.predicted, outcome.predicted_kind):
            style = STYLE[HIT] if label == HIT else STYLE["false"]
            axis.scatter([point[0]], [point[1]], **style)

        for edge in (-0.5, last + 0.5):
            axis.axvline(edge, color="yellow", linewidth=1.2)
            axis.axhline(edge, color="yellow", linewidth=1.2)
        axis.set_xlim(x - half, x + half)
        axis.set_ylim(y + half, y - half)
        short = outcome.name.replace("CAH_20220630_", "").replace("_post_processed_seamline_", " ")
        axis.set_title(f"{short}\nat ({x:.0f}, {y:.0f})", fontsize=6.5)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(picks):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=130)
    return figure


def gallery(
    outcomes: Sequence[PatchOutcome],
    image_dir: str | Path,
    title: str,
    path: str | Path | None = None,
    columns: int = 4,
):
    """Draw a grid of patches with a shared legend; returns the figure."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if not outcomes:
        return None
    rows = int(np.ceil(len(outcomes) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, outcome in zip(axes, outcomes):
        with Image.open(Path(image_dir) / outcome.name) as handle:
            draw(axis, np.asarray(handle.convert("RGB")), outcome)
    for axis in axes[len(outcomes):]:
        axis.axis("off")

    legend = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#1b9e77", markersize=9, label="animal, found"),
        Line2D([], [], marker="+", linestyle="none", color="#1b9e77", markersize=9,
               label="detection that found it"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#d95f02", markeredgewidth=2.4, markersize=12, label="animal, MISSED"),
        Line2D([], [], marker="x", linestyle="none", color="#e7298a", markeredgewidth=2.4,
               markersize=10, label="FALSE alarm"),
        Line2D([], [], linestyle="--", color="grey", label=f"{EDGE_MARGIN:.0f} px border band"),
    ]
    wide = columns >= 4
    figure.legend(handles=legend, loc="lower center", ncol=5 if wide else 2,
                  fontsize=9, frameon=False)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0.04 if wide else 0.12, 1, 0.97))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=130)
    return figure
