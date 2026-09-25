"""Threshold sweeps from stored candidates.

The headline numbers come from one operating point: keep a local maximum if it
is at least 30% of its patch's own peak (``adapt_ts``), and drop the patch
entirely if that peak is under 0.1 (``neg_ts``). That rule was fixed before the
test data were seen, and it stays the headline. This module describes what the
rest of the trade-off looks like, which a finished detections table cannot
show because it has already discarded everything below the cut.

Two families of curves are traced:

* **relative** - vary ``adapt_ts`` with ``neg_ts`` held at the headline 0.1:
  how sensitive the results are to the rule the models were evaluated with;
* **absolute** - keep every local maximum scoring at least ``t``, with no
  per-patch rule: the conventional precision-recall curve, summarised as
  average precision.

**This is description, not selection.** CAH 2022 is the locked test domain, so
a threshold chosen here by its score would be tuned on the test set. Curves and
average precision are threshold-free summaries; the best point on a curve is an
oracle bound and is labelled as one wherever it appears.

Every sweep replays stored candidates exactly (see
:meth:`owlcaribou.peaks.Candidates.select`), and the notebook checks that the
headline setting reproduces the full run's detections before trusting any
curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import GroundTruth
from .failures import EDGE_MARGIN, full_and_interior_statistics, interior_statistics
from .peaks import Candidates, PeakConfig
from .uncertainty import METRICS, bootstrap_weights, group_totals, metrics_from_totals, patch_statistics

PROTOCOL = PeakConfig()   # adapt_ts 0.3, neg_ts 0.1, kernel 3


class SweepError(ValueError):
    """A sweep was asked for something the stored candidates cannot answer."""


@dataclass
class CandidateTable:
    """All candidates for one model over one split, in patch order.

    Row and column stay on the heatmap grid; ``scale`` maps them to patch
    pixels on replay, so the table never depends on a coordinate convention
    it could get wrong.
    """

    names: list[str]
    patch: np.ndarray       # int32, index into names, non-decreasing
    row: np.ndarray         # int32
    col: np.ndarray         # int32
    score: np.ndarray       # float32
    peaks: np.ndarray       # float32, one per patch
    floor: float
    kernel_size: int
    scale: int

    @classmethod
    def from_stream(
        cls,
        stream: Iterable[tuple[str, Candidates]],
        names: Sequence[str],
        scale: int,
    ) -> "CandidateTable":
        position = {name: i for i, name in enumerate(names)}
        per_patch: list[Candidates | None] = [None] * len(names)
        for name, candidates in stream:
            per_patch[position[name]] = candidates
        missing = [names[i] for i, c in enumerate(per_patch) if c is None]
        if missing:
            raise SweepError(f"{len(missing)} patch(es) produced no candidate record, e.g. {missing[:3]}")

        floors = {c.floor for c in per_patch}
        kernels = {c.kernel_size for c in per_patch}
        if len(floors) != 1 or len(kernels) != 1:
            raise SweepError("candidates were extracted with mixed settings")

        counts = np.array([len(c) for c in per_patch], dtype=np.int64)
        return cls(
            names=list(names),
            patch=np.repeat(np.arange(len(names), dtype=np.int32), counts),
            row=np.concatenate([c.row for c in per_patch]).astype(np.int32),
            col=np.concatenate([c.col for c in per_patch]).astype(np.int32),
            score=np.concatenate([c.score for c in per_patch]).astype(np.float32),
            peaks=np.array([c.peak for c in per_patch], dtype=np.float32),
            floor=floors.pop(),
            kernel_size=kernels.pop(),
            scale=int(scale),
        )

    def __len__(self) -> int:
        return int(self.score.size)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary, names=np.array(self.names), patch=self.patch, row=self.row,
            col=self.col, score=self.score, peaks=self.peaks,
            meta=np.array([self.floor, self.kernel_size, self.scale], dtype=np.float64),
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CandidateTable":
        with np.load(path, allow_pickle=False) as data:
            floor, kernel, scale = data["meta"].tolist()
            return cls(
                names=[str(n) for n in data["names"]], patch=data["patch"], row=data["row"],
                col=data["col"], score=data["score"], peaks=data["peaks"],
                floor=float(floor), kernel_size=int(kernel), scale=int(scale),
            )

    # ---------------------------------------------------------------- replay

    def _group(self, keep: np.ndarray) -> dict[str, np.ndarray]:
        """Kept candidates as ``{patch name: (N, 2) x, y in patch pixels}``."""
        patch = self.patch[keep]
        xy = np.column_stack([self.col[keep], self.row[keep]]).astype(np.float64) * self.scale
        if patch.size == 0:
            return {}
        starts = np.flatnonzero(np.r_[True, patch[1:] != patch[:-1]])
        ends = np.r_[starts[1:], patch.size]
        return {self.names[int(patch[s])]: xy[s:e] for s, e in zip(starts, ends)}

    def relative(self, adapt_ts: float, neg_ts: float = PROTOCOL.neg_ts) -> dict[str, np.ndarray]:
        """Detections under the upstream rule at ``adapt_ts`` / ``neg_ts``.

        Vectorised form of :meth:`Candidates.select`: the cut is computed in
        float64 and compared in float32, exactly as the single-patch path does.
        """
        if adapt_ts < self.floor:
            raise SweepError(f"adapt_ts {adapt_ts} is below the stored floor {self.floor}")
        peaks64 = self.peaks.astype(np.float64)
        cut = (adapt_ts * peaks64).astype(np.float32)
        alive = peaks64 >= neg_ts
        keep = (self.score >= cut[self.patch]) & alive[self.patch]
        return self._group(keep)

    def absolute(self, threshold: float) -> dict[str, np.ndarray]:
        """Every stored local maximum scoring at least ``threshold``."""
        lowest = self.minimum_absolute_threshold
        if threshold < lowest:
            raise SweepError(f"threshold {threshold} is below the stored floor {lowest:.4g}")
        return self._group(self.score >= np.float32(threshold))

    @property
    def minimum_absolute_threshold(self) -> float:
        """Lowest absolute threshold every patch's candidates fully cover."""
        return float(self.floor * self.peaks.max()) if self.peaks.size else 0.0


def reproduces(
    table: CandidateTable,
    reference: Mapping[str, np.ndarray],
    config: PeakConfig = PROTOCOL,
    tolerance: float = 1e-6,
) -> dict:
    """Does replaying ``config`` give back the detections a full run wrote?

    ``reference`` is a detections table loaded with
    :func:`owlcaribou.metrics.load_predictions`. Point sets are compared per
    patch, ignoring order. Both sides are heatmap indices times the same
    integer scale, so they agree exactly; ``tolerance`` is an absolute bound
    that only absorbs float round-off from the CSV round trip.
    """
    if config.kernel_size != table.kernel_size:
        raise SweepError("kernel size differs from the stored candidates")
    def ordered(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return points[np.lexsort((points[:, 1], points[:, 0]))]

    ours = table.relative(config.adapt_ts, config.neg_ts)
    differing = []
    for name in table.names:
        a = ordered(ours.get(name, np.empty((0, 2))))
        b = ordered(reference.get(name, np.empty((0, 2))))
        if len(a) != len(b) or (len(a) and not np.allclose(a, b, rtol=0.0, atol=tolerance)):
            differing.append((name, len(a), len(b)))
    return {
        "patches": len(table.names),
        "identical": len(table.names) - len(differing),
        "differing": len(differing),
        "examples": differing[:5],
        "our_detections": int(sum(len(v) for v in ours.values())),
        "reference_detections": int(sum(len(v) for v in reference.values())),
    }


# -------------------------------------------------------------------- curves

@dataclass
class Curve:
    """Per-patch statistics at every value of one swept parameter."""

    model: str
    mode: str                       # "relative" or "absolute"
    values: np.ndarray              # swept parameter, ascending
    statistics: np.ndarray          # (values, patches, stat columns)
    interior: bool = False
    # Derived from ``statistics`` once; not a constructor argument.
    pooled: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.pooled = metrics_from_totals(self.statistics.sum(axis=1))

    def rows(self) -> list[dict]:
        return [
            {"model": self.model, "mode": self.mode, "interior": self.interior,
             "value": float(v), **{m: float(self.pooled[m][i]) for m in METRICS}}
            for i, v in enumerate(self.values)
        ]

    def at(self, value: float) -> dict[str, float]:
        index = int(np.argmin(np.abs(self.values - value)))
        if not np.isclose(self.values[index], value):
            raise SweepError(f"{value} is not on the swept grid")
        return {m: float(self.pooled[m][index]) for m in METRICS}


def trace(
    table: CandidateTable,
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
    mode: str,
    values: Sequence[float],
    model: str = "",
    interior: bool = False,
    margin: float = EDGE_MARGIN,
) -> Curve:
    """Score every value of the swept parameter on every patch.

    ``interior`` scores only the interior of each patch, as in the failure
    analysis and with the same band width by default (``EDGE_MARGIN``):
    matching is done on the whole patch, then recall is counted over interior
    animals and precision over interior detections (see
    :func:`owlcaribou.failures.interior_statistics`).
    """
    if mode not in ("relative", "absolute"):
        raise SweepError(f"unknown mode {mode!r}")
    values = np.asarray(sorted(values), dtype=np.float64)

    stacks = []
    for value in values:
        predictions = table.relative(value) if mode == "relative" else table.absolute(value)
        if interior:
            stacks.append(interior_statistics(predictions, truth, patch_names, radius, margin))
        else:
            stacks.append(patch_statistics(predictions, truth, patch_names, radius))
    return Curve(model=model, mode=mode, values=values, statistics=np.stack(stacks), interior=interior)


def trace_both(
    table: CandidateTable,
    truth: GroundTruth,
    patch_names: Sequence[str],
    radius: float,
    mode: str,
    values: Sequence[float],
    model: str = "",
    margin: float = EDGE_MARGIN,
) -> tuple[Curve, Curve]:
    """``(whole-patch, interior)`` curves from one matching per patch per value.

    Equal, bit for bit, to calling :func:`trace` with ``interior=False`` and
    then ``interior=True``, at half the matching work.
    """
    if mode not in ("relative", "absolute"):
        raise SweepError(f"unknown mode {mode!r}")
    values = np.asarray(sorted(values), dtype=np.float64)

    full_stacks, interior_stacks = [], []
    for value in values:
        predictions = table.relative(value) if mode == "relative" else table.absolute(value)
        full, interior = full_and_interior_statistics(predictions, truth, patch_names, radius, margin)
        full_stacks.append(full)
        interior_stacks.append(interior)
    return (
        Curve(model=model, mode=mode, values=values, statistics=np.stack(full_stacks), interior=False),
        Curve(model=model, mode=mode, values=values, statistics=np.stack(interior_stacks), interior=True),
    )


def average_precision(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Area under the precision-recall points, all-points interpolation.

    Works on one curve (1-D) or a stack of replicate curves (2-D, one per row).
    Precision is replaced by its running maximum from high recall to low, as in
    the standard all-points definition, and integrated over recall from 0 to the
    highest recall reached. A threshold with no counted detections has
    undefined precision; on full patches its recall is then zero, so it
    contributes nothing and is set to 0.

    The curve is sampled at the swept thresholds rather than at every
    candidate. Each grid point is itself an exact operating point, so the
    result is a lower bound on the exact all-points area. How far below
    depends on how a model's scores are spread, which differs between models;
    a shared grid does not make that equal, so a small difference between two
    models can sit within the approximation.
    """
    p = np.atleast_2d(np.nan_to_num(np.asarray(precision, dtype=np.float64), nan=0.0))
    r = np.atleast_2d(np.nan_to_num(np.asarray(recall, dtype=np.float64), nan=0.0))
    order = np.argsort(r, axis=1, kind="stable")
    r = np.take_along_axis(r, order, axis=1)
    p = np.take_along_axis(p, order, axis=1)
    envelope = np.maximum.accumulate(p[:, ::-1], axis=1)[:, ::-1]
    steps = np.diff(np.concatenate([np.zeros((r.shape[0], 1)), r], axis=1), axis=1)
    ap = (steps * envelope).sum(axis=1)
    return ap if np.ndim(precision) > 1 else ap[:1]


def bootstrap_average_precision(
    curves: Mapping[str, Curve],
    groups: Sequence[str],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Average precision per model with mosaic-bootstrap intervals, paired.

    Every model is scored on the same resampled mosaics, so the difference in
    average precision between two models is measured on identical draws of the
    survey. Requires all curves to share one grid.
    """
    grids = {tuple(c.values) for c in curves.values()}
    if len(grids) != 1:
        raise SweepError("curves must share one grid to be compared")

    first = next(iter(curves.values()))
    labels, _ = group_totals(first.statistics[0], groups)
    weights = bootstrap_weights(len(labels), n_boot, seed)
    alpha = (1.0 - confidence) / 2.0

    point, replicate = {}, {}
    for model, curve in curves.items():
        per_value = np.stack([group_totals(s, groups)[1] for s in curve.statistics])  # V,G,K
        pooled = metrics_from_totals(per_value.sum(axis=1))
        point[model] = float(average_precision(pooled["precision"], pooled["recall"])[0])
        totals = np.einsum("bg,vgk->bvk", weights, per_value)                         # B,V,K
        flat = metrics_from_totals(totals.reshape(-1, totals.shape[-1]))
        precision = flat["precision"].reshape(n_boot, -1)
        recall = flat["recall"].reshape(n_boot, -1)
        replicate[model] = average_precision(precision, recall)

    summary = {
        model: {"ap": point[model],
                "ci_low": float(np.quantile(replicate[model], alpha)),
                "ci_high": float(np.quantile(replicate[model], 1 - alpha))}
        for model in curves
    }

    def difference(a: str, b: str) -> dict:
        d = replicate[a] - replicate[b]
        return {"a": a, "b": b, "estimate": point[a] - point[b],
                "ci_low": float(np.quantile(d, alpha)), "ci_high": float(np.quantile(d, 1 - alpha)),
                "share_a_better": float((d > 0).mean())}

    return {"models": summary, "difference": difference, "mosaics": len(labels),
            "n_boot": n_boot, "confidence": confidence, "seed": seed}
