"""The locked CAH test split: patches, ground truth, and the mosaic they came from.

Three things here are easy to get wrong and each would quietly corrupt the
metrics rather than raise:

1. ``gt.csv`` holds rows only for patches that contain animals. 755 of the
   2,607 patches have none. Driving evaluation from the CSV would silently
   drop every background patch and make the empty-patch false-positive rate
   meaningless, so the patch list always comes from the directory.

2. Ground-truth coordinates are in 512 px patch space, while the models
   predict on a grid reduced by ``down_ratio``. This module does not rescale
   anything; it reports truth in patch pixels and leaves the conversion to the
   caller, which reads the ratio from the model config rather than assuming it.

3. ``base_images`` names the source mosaic, but only for annotated patches.
   The split draws on 43 mosaics, and 17 of them contributed nothing but
   background, so they never appear in ``gt.csv`` at all. Grouping by
   ``base_images`` alone would silently drop 17 mosaics and leave all 755
   empty patches unassigned. The mosaic is therefore read from the patch
   filename, and that rule is checked against ``base_images`` on every
   annotated patch before it is trusted (see :func:`verify_mosaic_rule`).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH_SIZE = 512

# Channel statistics the checkpoints were trained against.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_REQUIRED_COLUMNS = {"images", "x", "y"}

# A patch is named after its mosaic plus a tile index; background tiles carry
# "neg_" before the index. The mosaic group is non-greedy so the suffix binds to
# the final "_<n>.png" or "_neg_<n>.png" only.
_PATCH_NAME = re.compile(r"^(?P<mosaic>.+?)_(?P<neg>neg_)?(?P<index>\d+)\.png$")


class DataError(RuntimeError):
    """The split on disk does not agree with the ground truth."""


def mosaic_from_patch_name(name: str) -> str:
    """Source mosaic of a patch, read from its filename.

    Trust this only after :func:`verify_mosaic_rule` has passed on the split
    in hand; the rule describes this dataset's naming, not a general law.
    """
    match = _PATCH_NAME.match(Path(str(name)).name)
    if match is None:
        raise DataError(f"patch name does not follow the mosaic naming rule: {name!r}")
    return match.group("mosaic")


def is_background_name(name: str) -> bool:
    match = _PATCH_NAME.match(Path(str(name)).name)
    return bool(match and match.group("neg"))


@dataclass(frozen=True)
class GroundTruth:
    """Annotated points per patch, in 512 px patch coordinates.

    ``points`` only contains patches that have at least one annotation.
    ``mosaic`` maps those same patches to the image they were cut from.
    Absence from ``points`` means a patch has no animals, not that it is
    unlabelled: every patch in the split was reviewed.
    """

    points: Mapping[str, np.ndarray]
    mosaic: Mapping[str, str]

    @property
    def total_points(self) -> int:
        return int(sum(len(v) for v in self.points.values()))

    def for_patch(self, name: str) -> np.ndarray:
        """Points for one patch; an empty (0, 2) array when it has none."""
        found = self.points.get(name)
        return found if found is not None else np.empty((0, 2), dtype=np.float64)

    def mosaic_for(self, name: str) -> str | None:
        return self.mosaic.get(name)


def load_ground_truth(csv_path: str | Path) -> GroundTruth:
    """Read ``gt.csv`` into per-patch point arrays."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DataError(f"missing ground truth: {csv_path}")

    collected: dict[str, list[tuple[float, float]]] = {}
    mosaic: dict[str, str] = {}

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise DataError(
                f"{csv_path.name} is missing column(s) {sorted(missing)}; "
                f"found {sorted(columns)}"
            )
        has_mosaic = "base_images" in columns

        for line, row in enumerate(reader, start=2):
            # Defensive: the file stores bare names, but a path would join wrong.
            name = Path(str(row["images"]).strip().replace("\\", "/")).name
            try:
                point = (float(row["x"]), float(row["y"]))
            except (TypeError, ValueError):
                raise DataError(
                    f"{csv_path.name} line {line}: non-numeric coordinate "
                    f"x={row['x']!r} y={row['y']!r}"
                ) from None
            collected.setdefault(name, []).append(point)
            if has_mosaic and name not in mosaic:
                mosaic[name] = str(row["base_images"]).strip()

    points = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in collected.items()
    }
    return GroundTruth(points=points, mosaic=mosaic)


class CahPatches(Dataset):
    """Every patch in the split, annotated or not, in a stable order.

    Yields ``(image, index)``. The image is normalised CHW float32; the index
    maps back to :attr:`names`, which is what results are keyed on. Ground
    truth is deliberately not returned here — evaluation happens after
    inference, over whole-run outputs.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expect_images: int | None = None,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise DataError(f"missing patch directory: {self.root}")

        self.patch_size = patch_size
        self.names: list[str] = sorted(p.name for p in self.root.glob("*.png"))
        if not self.names:
            raise DataError(f"no .png patches under {self.root}")
        if expect_images is not None and len(self.names) != expect_images:
            raise DataError(
                f"{self.root} holds {len(self.names)} patches, expected "
                f"{expect_images}; refusing to evaluate an incomplete split"
            )

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.names)

    def path_for(self, index: int) -> Path:
        return self.root / self.names[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.path_for(index)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            if image.size != (self.patch_size, self.patch_size):
                raise DataError(
                    f"{path.name} is {image.size}, expected "
                    f"({self.patch_size}, {self.patch_size})"
                )
            # np.array (not asarray) so the buffer is writable and torch can
            # adopt it without warning; div_ below writes in place.
            array = np.array(image, dtype=np.uint8)

        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self._mean) / self._std
        return tensor, index


def cross_check(patches: CahPatches, truth: GroundTruth) -> dict[str, int]:
    """Confirm the ground truth and the directory describe the same split.

    A patch named in ``gt.csv`` but absent from disk means the split is
    incomplete, which would inflate recall by removing hard positives. That is
    an error. The reverse — patches on disk with no rows — is normal and
    counted, since those are the backgrounds.
    """
    on_disk = set(patches.names)
    annotated = set(truth.points)

    orphaned = sorted(annotated - on_disk)
    if orphaned:
        raise DataError(
            f"{len(orphaned)} annotated patch(es) missing from {patches.root}, "
            f"e.g. {orphaned[:3]}"
        )

    return {
        "patches": len(on_disk),
        "with_points": len(annotated),
        "background": len(on_disk - annotated),
        "points": truth.total_points,
        # Only mosaics named in gt.csv, i.e. those with animals; the full
        # count including background-only mosaics comes from verify_mosaic_rule.
        "mosaics_with_animals": len(set(truth.mosaic.values())),
    }


def verify_mosaic_rule(patch_names: Sequence[str], truth: GroundTruth) -> dict[str, int]:
    """Check the filename rule against every fact the ground truth records.

    Three things must hold before patches are grouped by mosaic:

    * every patch name parses;
    * for every annotated patch, the parsed mosaic equals ``base_images``;
    * a ``neg_`` name means no annotations, and no annotations means ``neg_``.

    Any single failure raises, because a wrong grouping would not error later —
    it would just produce confidence intervals that describe the wrong thing.
    """
    from pathlib import PurePosixPath

    mismatched: list[tuple[str, str, str]] = []
    annotated = set(truth.points)

    for name in patch_names:
        parsed = mosaic_from_patch_name(name)  # raises on an unparseable name
        recorded = truth.mosaic_for(name)
        if recorded is not None:
            expected = PurePosixPath(recorded).stem
            if parsed != expected:
                mismatched.append((name, parsed, expected))

    if mismatched:
        raise DataError(
            f"mosaic rule disagrees with base_images on {len(mismatched)} patch(es), "
            f"e.g. {mismatched[:2]}"
        )

    background_named = {n for n in patch_names if is_background_name(n)}
    unannotated = {n for n in patch_names if n not in annotated}
    if background_named != unannotated:
        raise DataError(
            f"'neg_' naming and missing annotations disagree: "
            f"{len(background_named - unannotated)} 'neg_' patch(es) have animals, "
            f"{len(unannotated - background_named)} unannotated patch(es) lack 'neg_'"
        )

    mosaics = {mosaic_from_patch_name(n) for n in patch_names}
    with_animals = {mosaic_from_patch_name(n) for n in patch_names if n in annotated}
    return {
        "patches": len(patch_names),
        "annotated_checked": len(annotated & set(patch_names)),
        "mosaics": len(mosaics),
        "mosaics_with_animals": len(with_animals),
        "background_only_mosaics": len(mosaics - with_animals),
    }


def collate(batch: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, list[int]]:
    images = torch.stack([item[0] for item in batch])
    return images, [int(item[1]) for item in batch]
