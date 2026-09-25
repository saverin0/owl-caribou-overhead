"""Build a model, load a checkpoint, and turn patches into detections.

Model construction goes through the vendored definitions, which is the only
thing taken from upstream. Everything else here is ours.

Two deliberate choices:

* ``pretrained`` is forced off for every architecture and asserted. The
  vendored code would otherwise try to fetch backbone weights over the network
  from a hardcoded path belonging to someone else's machine. Every weight we
  use comes from the verified public checkpoint, which is loaded whole.

* No tiling. The locked split is already 512 px patches and the models take
  512 px input, so the upstream stitcher would emit exactly one tile per
  image. We assert the sizes agree rather than carry a stitching path that
  never runs. A different split would need that path written.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PATCH_SIZE, CahPatches, collate
from .io import ARTIFACT_MD5, verify_file
from .peaks import (
    CANDIDATE_FLOOR,
    Candidates,
    PeakConfig,
    extract_candidates,
    extract_peaks,
    to_patch_coordinates,
)


class InferenceError(RuntimeError):
    """The model could not be built, or the checkpoint does not fit it."""


@dataclass(frozen=True)
class ModelSpec:
    """How to instantiate one evaluated model.

    Mirrors the configuration the checkpoints were released and evaluated
    with, so results stay comparable to the evaluation setting.
    """

    architecture: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    down_ratio: int = 2
    stitcher_upsamples: bool = False
    needs_dinov3: bool = False

    @property
    def prediction_scale(self) -> int:
        """Factor mapping heatmap indices to patch pixels.

        With no upsampling the prediction grid is reduced by ``down_ratio``,
        so indices scale by it. With upsampling the grid is already at patch
        resolution and the factor is 1.
        """
        return 1 if self.stitcher_upsamples else self.down_ratio


_SWIN = {
    "swin_num_layers_per_scale": [0, 1, 2, 2, 2, 3],
    "swin_window_sizes": [8, 8, 8, 4, 4, 4],
    "swin_num_heads": [1, 1, 2, 4, 8, 16],
    "drop_path_rate": 0.0,
}

MODEL_SPECS: Mapping[str, ModelSpec] = {
    "owl-c": ModelSpec(
        architecture="OWLC",
        kwargs={"num_layers": 34, "pretrained": False, "down_ratio": 2, "head_conv": 64},
    ),
    "caribou-owl-c": ModelSpec(
        architecture="OWLC",
        kwargs={"num_layers": 34, "pretrained": False, "down_ratio": 2, "head_conv": 64},
    ),
    "owl-t": ModelSpec(
        architecture="OWLT",
        kwargs={
            "num_layers": 34,
            "pretrained_cnn": False,
            "head_conv": 64,
            "down_ratio": 2,
            **_SWIN,
        },
    ),
    "owl-d": ModelSpec(
        architecture="OWLD_H",
        kwargs={
            "pretrained": False,
            "head_conv": 64,
            "readout_type": "ignore",
            "freeze_backbone": True,
            "down_ratio": 2,
        },
        needs_dinov3=True,
    ),
}

_PRETRAIN_FLAGS = ("pretrained", "pretrained_cnn")

# Shared by every forward-pass entry point, so the timing probe rounds its
# warm-up to the same batch boundary the run uses.
DEFAULT_BATCH_SIZE = 8


def ensure_importable(third_party: str | Path) -> Path:
    """Put the vendored packages on the import path. Idempotent."""
    root = Path(third_party).resolve()
    if not (root / "animaloc").is_dir():
        raise InferenceError(f"vendored model code not found under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def build_model(
    model: str,
    third_party: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Instantiate an architecture with no pretrained weights fetched."""
    try:
        spec = MODEL_SPECS[model]
    except KeyError:
        raise InferenceError(
            f"unknown model {model!r}; known: {', '.join(sorted(MODEL_SPECS))}"
        ) from None

    root = ensure_importable(third_party)
    kwargs = dict(spec.kwargs)

    for flag in _PRETRAIN_FLAGS:
        if kwargs.get(flag):
            raise InferenceError(
                f"{model}: {flag} must be False; weights come from the checkpoint"
            )
    if spec.needs_dinov3:
        # Passed explicitly so the upstream fallback path is never consulted.
        kwargs["dinov3_root"] = str(root / "dinov3")

    import animaloc.models as registry  # noqa: PLC0415 - needs sys.path set first

    try:
        architecture = registry.MODELS[spec.architecture]
    except KeyError:
        raise InferenceError(
            f"architecture {spec.architecture!r} is not registered"
        ) from None
    # Built outside the lookup's try, so an error inside the constructor
    # surfaces as itself rather than as "not registered".
    net = architecture(**kwargs)

    return net.to(device)


def load_checkpoint(
    net: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify: bool = True,
) -> dict[str, Any]:
    """Load weights, insisting every parameter is accounted for.

    The released files wrap the weights in a training checkpoint under
    ``model_state_dict``. A strict load is the point: a silently partial load
    would leave randomly initialised layers and produce plausible-looking but
    meaningless detections.

    A released checkpoint is checked against its published digest before it is
    unpickled, whatever the caller verified earlier; the sidecar marker makes
    that cheap after the first read, and a mismatch raises
    :class:`owlcaribou.io.AssetError`. ``verify=False`` is for files that are
    not part of the release.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise InferenceError(f"missing checkpoint: {checkpoint_path}")
    if verify and checkpoint_path.name in ARTIFACT_MD5:
        verify_file(checkpoint_path, ARTIFACT_MD5[checkpoint_path.name])

    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
    elif isinstance(blob, dict) and all(torch.is_tensor(v) for v in blob.values()):
        state = blob
    else:
        keys = list(blob)[:6] if isinstance(blob, dict) else type(blob).__name__
        raise InferenceError(
            f"{checkpoint_path.name}: cannot find weights in checkpoint (saw {keys})"
        )

    state, stripped = _unwrap_state_dict(state)

    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise InferenceError(
            f"{checkpoint_path.name} does not fit this architecture: {error}"
        ) from None

    return {
        "checkpoint": checkpoint_path.name,
        "tensors": len(state),
        "unwrapped": "".join(stripped) or None,
        "epoch": blob.get("epoch") if isinstance(blob, dict) else None,
    }


_WRAPPER_PREFIXES = ("module.", "model.")


def _unwrap_state_dict(state: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Strip outer-module prefixes from checkpoint keys.

    The released checkpoints save a wrapper that holds the network as
    ``model``, so every key arrives as ``model.backbone...`` while the
    architecture expects ``backbone...``. ``module.`` appears the same way when
    a model was saved through DataParallel.

    A prefix is only stripped when *every* key carries it, so this cannot
    silently mangle a checkpoint that genuinely has a submodule of that name.
    Stripping is repeated in case a checkpoint carries more than one wrapper.
    """
    unwrapped = dict(state)
    stripped: list[str] = []
    while unwrapped:
        for prefix in _WRAPPER_PREFIXES:
            if all(key.startswith(prefix) for key in unwrapped):
                unwrapped = {key[len(prefix):]: value for key, value in unwrapped.items()}
                stripped.append(prefix)
                break
        else:
            break
    return unwrapped, stripped


def _detection_heatmap(output: Any) -> torch.Tensor:
    """Pull the single-channel detection map out of a model's output.

    OWL-D returns a pair whose second element is ``None`` (its classification
    branch is unused); the others return the map directly.
    """
    if isinstance(output, (list, tuple)):
        if not output or not torch.is_tensor(output[0]):
            raise InferenceError("model output has no detection heatmap")
        output = output[0]
    if not torch.is_tensor(output):
        raise InferenceError(f"unexpected model output of type {type(output).__name__}")
    return output


@dataclass
class PatchResult:
    """What one patch produced."""

    image: str
    x: np.ndarray
    y: np.ndarray
    score: np.ndarray
    peak: float

    @property
    def count(self) -> int:
        return int(self.x.size)


def run_inference(
    net: torch.nn.Module,
    patches: CahPatches,
    spec: ModelSpec,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 2,
    peak_config: PeakConfig | None = None,
    autocast_dtype: torch.dtype | None = None,
    limit: int | None = None,
    indices: Sequence[int] | None = None,
) -> Iterator[PatchResult]:
    """Run every patch and yield its detections, in dataset order.

    ``autocast_dtype`` is off by default. Reduced precision is faster on an
    A100 but changes the heatmap, and the peak rule thresholds relative to each
    patch's own maximum, so it can move detections. Measure the difference
    before trusting it for a headline run.

    ``limit`` restricts the run to the first N patches, for timing probes.
    ``indices`` runs an explicit subset instead, which is how a named set of
    patches is reproduced.
    """
    peak_config = peak_config or PeakConfig()
    scale = spec.prediction_scale
    for batch_indices, heatmap in _heatmaps(
        net, patches, device=device, batch_size=batch_size, num_workers=num_workers,
        autocast_dtype=autocast_dtype, limit=limit, indices=indices,
    ):
        for offset, peaks in enumerate(extract_peaks(heatmap, peak_config)):
            # Subset passes through the underlying dataset's index, so this
            # still names the right patch when a subset is running.
            index = batch_indices[offset]
            x, y = to_patch_coordinates(peaks, scale)
            yield PatchResult(
                image=patches.names[index],
                x=x,
                y=y,
                score=peaks.score,
                peak=peaks.peak,
            )


def run_candidates(
    net: torch.nn.Module,
    patches: CahPatches,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 2,
    floor: float = CANDIDATE_FLOOR,
    limit: int | None = None,
    indices: Sequence[int] | None = None,
) -> Iterator[tuple[str, Candidates]]:
    """Run every patch and yield all its local maxima down to ``floor``.

    Candidates stay on the heatmap grid; scaling to patch pixels happens when
    they are replayed. Always full precision: a sweep exists to describe the
    evaluated operating point faithfully, and reduced precision would move it.
    """
    for batch_indices, heatmap in _heatmaps(
        net, patches, device=device, batch_size=batch_size, num_workers=num_workers,
        autocast_dtype=None, limit=limit, indices=indices,
    ):
        for offset, candidates in enumerate(extract_candidates(heatmap, floor)):
            yield patches.names[batch_indices[offset]], candidates


def _heatmaps(
    net: torch.nn.Module,
    patches: CahPatches,
    *,
    device: str | torch.device,
    batch_size: int,
    num_workers: int,
    autocast_dtype: torch.dtype | None,
    limit: int | None,
    indices: Sequence[int] | None,
) -> Iterator[tuple[list[int], torch.Tensor]]:
    """The forward pass shared by detection and candidate extraction.

    Yields each batch's dataset indices with its float32 detection heatmap.
    One code path, so a sweep replays exactly the heatmaps a run scored.
    """
    if patches.patch_size != PATCH_SIZE:
        raise InferenceError(
            f"patch size {patches.patch_size} differs from the evaluated {PATCH_SIZE}"
        )
    if indices is not None and limit is not None:
        raise InferenceError("pass indices or limit, not both")

    device = torch.device(device)
    subset: Any = patches
    if indices is not None:
        chosen = list(indices)
        if not chosen:
            raise InferenceError("indices is empty; nothing to run")
        subset = torch.utils.data.Subset(patches, chosen)
    elif limit is not None:
        subset = torch.utils.data.Subset(patches, range(min(limit, len(patches))))

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    net.eval()
    with torch.inference_mode():
        for images, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            if autocast_dtype is not None and device.type == "cuda":
                with torch.autocast("cuda", dtype=autocast_dtype):
                    output = net(images)
                yield batch_indices, _detection_heatmap(output).float()
            else:
                yield batch_indices, _detection_heatmap(net(images)).float()


def time_inference(
    net: torch.nn.Module,
    patches: CahPatches,
    spec: ModelSpec,
    *,
    device: str | torch.device = "cpu",
    warmup: int = 4,
    measure: int = 20,
    **kwargs: Any,
) -> dict[str, float]:
    """Measure seconds per patch, discarding whole warm-up batches.

    The first batches include kernel autotuning and weight paging, so timing
    them would overstate the cost of a long run.

    Work happens per batch, and a batch is computed when its first patch is
    yielded. ``warmup`` is therefore rounded up to a batch boundary and the
    clock starts once the last warm-up patch has arrived, so every timed
    second belongs to the measured patches and nothing is timed part-way
    through a batch.
    """
    device = torch.device(device)
    batch_size = int(kwargs.get("batch_size", DEFAULT_BATCH_SIZE))
    if warmup % batch_size:
        warmup = -(-warmup // batch_size) * batch_size
    results: list[PatchResult] = []

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    stream = run_inference(net, patches, spec, device=device, limit=warmup + measure, **kwargs)
    started: float | None = None
    if warmup == 0:
        _sync()
        started = time.perf_counter()
    for position, result in enumerate(stream):
        results.append(result)
        if position + 1 == warmup:
            _sync()
            started = time.perf_counter()
    _sync()

    measured = len(results) - warmup
    if started is None or measured <= 0:
        raise InferenceError("not enough patches to measure after warm-up")

    elapsed = time.perf_counter() - started
    per_patch = elapsed / measured
    return {
        "warmup_patches": float(warmup),
        "measured_patches": float(measured),
        "seconds_per_patch": per_patch,
        "detections": float(sum(r.count for r in results[warmup:])),
        "projected_full_split_minutes": per_patch * len(patches) / 60.0,
    }


def to_rows(results: Iterable[PatchResult]) -> tuple[list[dict], list[dict]]:
    """Split results into a detection table and a per-image table.

    Every patch appears in the per-image table, including ones with no
    detections. Upstream recorded empty patches as a detection row with blank
    coordinates; keeping two tables says the same thing without a row that is
    not a detection.
    """
    detections: list[dict] = []
    per_image: list[dict] = []
    for result in results:
        for x, y, score in zip(result.x, result.y, result.score):
            detections.append(
                {"images": result.image, "x": float(x), "y": float(y), "score": float(score)}
            )
        per_image.append(
            {"images": result.image, "count": result.count, "peak": float(result.peak)}
        )
    return detections, per_image
