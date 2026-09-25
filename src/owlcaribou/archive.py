"""Comparing this pipeline against an earlier run made with the upstream evaluator.

That run produced a detections table per model with the upstream code on a
subset of the same patches. Reproducing it with our own code, on the same
patches and the same checkpoint, is the strongest evidence that this pipeline
is faithful: it exercises checkpoint loading, the forward pass and peak
extraction at once. The reference table is not part of this repository.

The reference format writes one row per detection, plus a row with blank
coordinates for a patch that had none. That blank row is how an evaluated but
empty patch is recorded, so it is read as "present, zero detections" rather
than skipped.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArchivedDetections:
    """Detections from the reference run, per patch, in its own coordinates."""

    points: dict[str, np.ndarray]

    @property
    def images(self) -> list[str]:
        return sorted(self.points)

    @property
    def total(self) -> int:
        return int(sum(len(v) for v in self.points.values()))

    def for_patch(self, name: str) -> np.ndarray:
        found = self.points.get(name)
        return found if found is not None else np.empty((0, 2), dtype=np.float64)


def load_archived_detections(csv_path: str | Path) -> ArchivedDetections:
    points: dict[str, list[tuple[float, float]]] = {}

    with open(Path(csv_path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = Path(str(row["images"]).strip().replace("\\", "/")).name
            points.setdefault(name, [])
            raw_x, raw_y = str(row.get("x", "")).strip(), str(row.get("y", "")).strip()
            if not raw_x or not raw_y:
                continue  # evaluated, no detections
            points[name].append((float(raw_x), float(raw_y)))

    return ArchivedDetections(
        points={k: np.asarray(v, dtype=np.float64).reshape(-1, 2) for k, v in points.items()},
    )


def compare(
    results,
    archived: ArchivedDetections,
    *,
    tolerance: float = 0.5,
    archived_scale: float = 1.0,
) -> dict:
    """Compare our detections against the reference table, patch by patch.

    Points are matched as sets, not by row order: the reference table's
    ordering comes from a different traversal and carries no meaning. A patch
    agrees when both sides hold the same number of points and every point has
    a partner within ``tolerance`` pixels.

    ``archived_scale`` converts the reference coordinates into the same space
    as ours. The reference tables store heatmap indices, because the upstream
    evaluator applied the ``down_ratio`` factor when it scored rather than
    when it wrote the table; we emit patch pixels. Pass the model's prediction
    scale and the two become comparable. The report carries both coordinate
    ranges so a unit mismatch is visible rather than appearing as a mass
    displacement.
    """
    ours = {r.image: np.column_stack([r.x, r.y]) for r in results}
    shared = sorted(set(ours) & set(archived.points))

    agreeing = 0
    count_diff: list[tuple[str, int, int]] = []
    moved: list[tuple[str, float]] = []

    for name in shared:
        mine = ours[name]
        theirs = archived.for_patch(name) * archived_scale
        if len(mine) != len(theirs):
            count_diff.append((name, len(mine), len(theirs)))
            continue
        if len(mine) == 0:
            agreeing += 1
            continue
        distance = np.linalg.norm(mine[:, None, :] - theirs[None, :, :], axis=2)
        worst = float(max(distance.min(axis=1).max(), distance.min(axis=0).max()))
        if worst <= tolerance:
            agreeing += 1
        else:
            moved.append((name, worst))

    def _span(arrays) -> tuple[float, float] | None:
        stacked = [a for a in arrays if len(a)]
        if not stacked:
            return None
        joined = np.concatenate(stacked)
        return round(float(joined.min()), 1), round(float(joined.max()), 1)

    only_ours = sorted(set(ours) - set(archived.points))
    only_archived = sorted(set(archived.points) - set(ours))
    return {
        "tolerance": tolerance,
        "our_range": _span(ours.values()),
        "archived_range": _span([v * archived_scale for v in archived.points.values()]),
        "archived_scale": archived_scale,
        "patches_compared": len(shared),
        "only_ours": only_ours,
        "only_archived": only_archived,
        "agreeing": agreeing,
        "disagreeing": len(shared) - agreeing,
        "count_differences": count_diff[:10],
        "displaced": moved[:10],
        "our_detections": int(sum(len(v) for v in ours.values())),
        "archived_detections": archived.total,
        # A patch present on one side only was never compared, so it fails.
        "identical": (len(shared) > 0 and agreeing == len(shared)
                      and not only_ours and not only_archived),
    }
