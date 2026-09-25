"""Turning a heatmap into points.

This is our implementation of the local-maximum detection strategy used
upstream to evaluate the checkpoints. It is written to be numerically
equivalent to the upstream routine, because the evaluation setting fixes the
operating point: change the peak rule and the numbers stop being comparable.

The behaviour worth understanding, and the reason a threshold sweep has to go
back to the heatmaps:

``adapt_ts`` is **relative, not absolute**. The cut is ``adapt_ts * max`` where
the max is that patch's own peak. A quiet patch keeps weak peaks; a patch with
one very strong animal discards moderate ones. Consequently, filtering a
finished detections table by score afterwards is *not* a threshold sweep —
every surviving point already passed a per-patch bar, and those bars differ
between patches. Any genuine sweep has to re-extract from the heatmaps with a
permissive setting.

``neg_ts`` is absolute: if a patch's peak never reaches it, the patch is
declared empty and contributes no detections at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PeakConfig:
    """Operating point for peak extraction.

    Defaults are the setting used upstream to evaluate the checkpoints.
    """

    kernel_size: int = 3
    adapt_ts: float = 0.3
    neg_ts: float = 0.1

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")


@dataclass(frozen=True)
class Peaks:
    """Detected points for one patch, on the heatmap grid.

    ``row``/``col`` are heatmap indices, deliberately not yet scaled to patch
    pixels — the caller applies the model's ``down_ratio``. Naming them row and
    col rather than y and x keeps the axis order unambiguous at every hand-off.
    """

    row: np.ndarray
    col: np.ndarray
    score: np.ndarray
    peak: float

    def __len__(self) -> int:
        return int(self.row.size)


def _suppress_non_maxima(heatmap: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Zero every pixel that is not equal to the maximum of its neighbourhood.

    Padding is implicitly -inf for max pooling, so border pixels compete only
    against real neighbours. Plateaus keep every tied pixel, matching upstream.
    """
    pad = kernel_size // 2
    pooled = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    return heatmap * (pooled == heatmap)


def extract_peaks(
    heatmap: torch.Tensor,
    config: PeakConfig | None = None,
) -> list[Peaks]:
    """Extract points from a ``[B, 1, H, W]`` heatmap, one result per image.

    The threshold is computed from the raw map before suppression, which is
    equivalent to using the suppressed map: a global maximum is always a local
    maximum and so always survives.
    """
    config = config or PeakConfig()

    if heatmap.dim() != 4:
        raise ValueError(f"expected [B, 1, H, W] heatmap, got shape {tuple(heatmap.shape)}")
    if heatmap.shape[1] != 1:
        raise ValueError(
            f"expected a single-channel detection heatmap, got {heatmap.shape[1]} channels"
        )

    heatmap = heatmap.detach().float()
    suppressed = _suppress_non_maxima(heatmap, config.kernel_size)

    results: list[Peaks] = []
    for index in range(heatmap.shape[0]):
        raw = heatmap[index, 0]
        peak = float(raw.max().item())

        if peak < config.neg_ts:
            results.append(_empty(peak))
            continue

        kept = suppressed[index, 0]
        mask = kept >= config.adapt_ts * peak
        # A relative cut against a non-positive peak would admit everything;
        # neg_ts normally catches this, but guard rather than rely on it.
        mask &= kept > 0

        rows, cols = torch.nonzero(mask, as_tuple=True)
        if rows.numel() == 0:
            results.append(_empty(peak))
            continue

        results.append(
            Peaks(
                row=rows.cpu().numpy().astype(np.int64),
                col=cols.cpu().numpy().astype(np.int64),
                score=kept[rows, cols].cpu().numpy().astype(np.float64),
                peak=peak,
            )
        )

    return results


def _empty(peak: float) -> Peaks:
    return Peaks(
        row=np.empty(0, dtype=np.int64),
        col=np.empty(0, dtype=np.int64),
        score=np.empty(0, dtype=np.float64),
        peak=peak,
    )


CANDIDATE_FLOOR = 0.01


class CandidateError(ValueError):
    """A replay asked for more than the stored candidates can answer."""


@dataclass(frozen=True)
class Candidates:
    """Every local maximum on one heatmap down to a floor, for later replay.

    Stored instead of a finished detection list so that any operating point at
    or above the floor can be reproduced without running the model again. The
    floor is relative to this heatmap's own peak, which is what makes both a
    relative sweep (``adapt_ts`` down to the floor) and an absolute one (any
    threshold at or above ``floor * peak``) exact.

    Scores are kept as float32, the dtype the model produced, because the
    upstream rule compares in float32 and a borderline point could otherwise
    flip on replay.
    """

    row: np.ndarray
    col: np.ndarray
    score: np.ndarray
    peak: float
    floor: float
    kernel_size: int

    def __len__(self) -> int:
        return int(self.row.size)

    def select(self, config: PeakConfig | None = None) -> Peaks:
        """Reproduce :func:`extract_peaks` at ``config`` from the stored candidates."""
        config = config or PeakConfig()
        if config.kernel_size != self.kernel_size:
            raise CandidateError(
                f"candidates were suppressed with a {self.kernel_size}x{self.kernel_size} "
                f"kernel; cannot replay kernel {config.kernel_size}"
            )
        if config.adapt_ts < self.floor:
            raise CandidateError(
                f"adapt_ts {config.adapt_ts} is below the stored floor {self.floor}"
            )
        if self.peak < config.neg_ts:
            return _empty(self.peak)
        cut = np.float32(config.adapt_ts * self.peak)
        keep = self.score >= cut
        if not keep.any():
            return _empty(self.peak)
        return Peaks(
            row=self.row[keep].astype(np.int64),
            col=self.col[keep].astype(np.int64),
            score=self.score[keep].astype(np.float64),
            peak=self.peak,
        )


def extract_candidates(
    heatmap: torch.Tensor,
    floor: float = CANDIDATE_FLOOR,
    kernel_size: int = 3,
) -> list[Candidates]:
    """Every local maximum at or above ``floor`` times its heatmap's peak.

    Uses the same suppression and the same float32 comparison as
    :func:`extract_peaks`, so :meth:`Candidates.select` reproduces it exactly
    for any ``adapt_ts`` at or above ``floor``. No ``neg_ts`` is applied here;
    that is a replay-time choice.
    """
    if not 0.0 < floor < 1.0:
        raise ValueError(f"floor must lie in (0, 1), got {floor}")
    if heatmap.dim() != 4 or heatmap.shape[1] != 1:
        raise ValueError(f"expected [B, 1, H, W] heatmap, got shape {tuple(heatmap.shape)}")

    heatmap = heatmap.detach().float()
    suppressed = _suppress_non_maxima(heatmap, kernel_size)
    out: list[Candidates] = []
    for index in range(heatmap.shape[0]):
        peak = float(heatmap[index, 0].max().item())
        kept = suppressed[index, 0]
        mask = (kept >= floor * peak) & (kept > 0)
        rows, cols = torch.nonzero(mask, as_tuple=True)
        out.append(
            Candidates(
                row=rows.cpu().numpy().astype(np.int32),
                col=cols.cpu().numpy().astype(np.int32),
                score=kept[rows, cols].cpu().numpy().astype(np.float32),
                peak=peak,
                floor=floor,
                kernel_size=kernel_size,
            )
        )
    return out


def to_patch_coordinates(peaks: Peaks, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Map heatmap indices to patch pixels, returning ``(x, y)``.

    Column is the horizontal axis and becomes x; row is vertical and becomes y.
    ``scale`` is the model's ``down_ratio`` when the stitcher does not upsample,
    and 1 when it does — read from the resolved config, never assumed.
    """
    return peaks.col.astype(np.float64) * scale, peaks.row.astype(np.float64) * scale
