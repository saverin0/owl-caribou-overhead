"""Paths, integrity checks and run markers.

Checkpoints and the CAH split are large and live on Drive, which is slow to
read. Re-hashing 4.3 GB on every run is wasteful, so a verified file gets a
sidecar marker recording its size and digest; later runs trust the marker
unless asked to re-hash. The marker is only ever written after a full read.

Nothing here reaches the network. Assets are expected to exist already; if one
is missing we say so and stop rather than fetching it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Published digests for the release artifacts (Zenodo record 20802844).
# These describe public files; they are not secrets.
ARTIFACT_MD5: Mapping[str, str] = {
    "Caribou-OWL-C.pth": "4242ff6e031544718d38b4bf3cfcf8af",
    "OWL-C.pth": "4d1cb40b4ee16d531ded09c16333d73b",
    "OWL-D.pth": "30ff57bba505de4be743c2e0ef0c23ed",
    "OWL-T.pth": "6ec0cadc861c260a1acaca3657921be5",
    "test.zip": "086a0430c769b164cbcc9d2d48db6fac",
    "train.zip": "d55575fecc0e5e6d7180c96b74e75097",
}

CHECKPOINT_FOR_MODEL: Mapping[str, str] = {
    "caribou-owl-c": "Caribou-OWL-C.pth",
    "owl-c": "OWL-C.pth",
    "owl-t": "OWL-T.pth",
    "owl-d": "OWL-D.pth",
}

# The locked test domain, as recorded when the split was extracted.
CAH_IMAGES = 2607
CAH_POINTS = 12456

_READ_CHUNK = 8 * 1024 * 1024


class AssetError(RuntimeError):
    """An expected asset is missing, or its contents do not match."""


@dataclass(frozen=True)
class Paths:
    """Where assets are read from and where results are written.

    ``assets`` holds the verified checkpoints and the extracted CAH split, and
    nothing in it is ever changed. The only thing written there is a small
    ``.verified.json`` marker beside each checkpoint, and only when the folder
    is writable (see :func:`verify_file`). ``results`` is ours to write.
    """

    assets: Path
    results: Path

    @property
    def weights(self) -> Path:
        return self.assets / "cache" / "weights"

    @property
    def test_data(self) -> Path:
        return self.assets / "data" / "test"

    @property
    def ground_truth(self) -> Path:
        return self.test_data / "gt.csv"

    @property
    def test_archive(self) -> Path:
        """The released archive the CAH split was extracted from."""
        return self.assets / "cache" / "archives" / "test.zip"

    def checkpoint(self, model: str) -> Path:
        try:
            filename = CHECKPOINT_FOR_MODEL[model]
        except KeyError:
            known = ", ".join(sorted(CHECKPOINT_FOR_MODEL))
            raise AssetError(f"unknown model {model!r}; known: {known}") from None
        return self.weights / filename

    def run_dir(self, run: str, model: str | None = None) -> Path:
        return self.results / run / model if model else self.results / run


def on_colab() -> bool:
    """True on a Colab runtime, mounted or not.

    Tested by the presence of the ``google.colab`` package rather than of
    ``/content/drive``, which only exists once Drive has been mounted; the
    session needs to know it is on Colab *before* mounting.
    """
    try:
        import google.colab  # noqa: F401, PLC0415 - only exists on Colab
    except ImportError:
        return False
    return True


def default_paths(
    assets: str | os.PathLike[str] | None = None,
    results: str | os.PathLike[str] | None = None,
) -> Paths:
    """Resolve asset and result roots, on Colab or locally.

    Both can be overridden, which is what the notebooks do. The defaults only
    describe where things already live; no directory is created for assets.
    """
    if assets is None:
        base = Path("/content/drive/MyDrive") if on_colab() else Path.cwd()
        assets = base / "OWL_Caribou_Project"
    if results is None:
        results = Path(assets).parent / "owl_caribou_overhead" / "results"
    resolved = Paths(assets=Path(assets), results=Path(results))
    resolved.results.mkdir(parents=True, exist_ok=True)
    return resolved


def md5sum(path: str | os.PathLike[str], chunk_size: int = _READ_CHUNK) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Write JSON so a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_json(path: str | os.PathLike[str]) -> Any | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_csv(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write rows to CSV atomically, so a partial file is never left behind.

    An empty table still writes a header when field names are given, because
    "ran and found nothing" and "never ran" must not look the same on disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames) if fieldnames else (list(rows[0]) if rows else [])
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def read_csv(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise AssetError(f"missing table: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _marker_for(path: Path) -> Path:
    return path.with_name(path.name + ".verified.json")


def verify_file(
    path: str | os.PathLike[str],
    expected_md5: str,
    *,
    rehash: bool = False,
) -> str:
    """Return the digest of ``path``, raising if it is not ``expected_md5``.

    A sidecar marker lets a later run skip the read. The marker is trusted
    only when it carries the expected digest *and* the size still matches,
    so a truncated file is still caught; ``rehash=True`` also catches a
    replacement of the same size. Writing the marker is best effort: on a
    read-only assets folder the file is verified all the same and simply
    hashed again on the next run.
    """
    path = Path(path)
    if not path.is_file():
        raise AssetError(f"missing asset: {path}")

    size = path.stat().st_size
    marker = _marker_for(path)

    if not rehash:
        recorded = read_json(marker)
        if (
            isinstance(recorded, dict)
            and recorded.get("md5") == expected_md5
            and recorded.get("size") == size
        ):
            return expected_md5

    actual = md5sum(path)
    if actual != expected_md5:
        raise AssetError(
            f"checksum mismatch for {path.name}: "
            f"expected {expected_md5}, read {actual}. "
            "Remove that file and its marker, then restore it."
        )

    try:
        write_json(
            marker,
            {
                "md5": actual,
                "size": size,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except OSError:
        pass  # read-only assets: verified anyway, re-hashed next time
    return actual


def verify_checkpoints(
    paths: Paths,
    models: Iterable[str],
    *,
    rehash: bool = False,
) -> dict[str, str]:
    """Verify one checkpoint per model. Raises on the first bad or missing one."""
    digests: dict[str, str] = {}
    for model in models:
        checkpoint = paths.checkpoint(model)
        expected = ARTIFACT_MD5[checkpoint.name]
        digests[model] = verify_file(checkpoint, expected, rehash=rehash)
    return digests


def verify_test_split(paths: Paths) -> dict[str, int]:
    """Check the CAH split is present and the expected size.

    Both counts are taken for real: the patches by listing the directory, the
    points by reading ``gt.csv``, one row per annotated animal. Neither is
    trusted from a marker, so a truncated split or ground truth is caught
    before anything is scored.
    """
    root = paths.test_data
    if not root.is_dir():
        raise AssetError(f"missing CAH test split: {root}")
    if not paths.ground_truth.is_file():
        raise AssetError(f"missing ground truth: {paths.ground_truth}")

    images = sum(1 for _ in root.glob("*.png"))
    if images != CAH_IMAGES:
        raise AssetError(
            f"CAH split has {images} patches, expected {CAH_IMAGES}. "
            "The split is incomplete; do not evaluate against it."
        )
    with open(paths.ground_truth, newline="", encoding="utf-8") as handle:
        points = sum(1 for _ in csv.DictReader(handle))
    if points != CAH_POINTS:
        raise AssetError(
            f"ground truth lists {points} points, expected {CAH_POINTS}. "
            "The annotations are incomplete; do not evaluate against them."
        )
    return {"images": images, "points": points}


def stage_split(
    paths: Paths,
    destination: str | os.PathLike[str],
) -> Path:
    """Copy the CAH archive to fast local disk, verify it, and unpack it there.

    Reading 2,607 small files over the Drive mount costs a round trip per file
    (measured: 19 minutes for the first pass of the full run, 22 seconds once
    cached). One 1.2 GB archive crosses in a single sequential read. The copy
    is checked against the published digest before it is trusted, and the
    unpacked split is counted before it is used. A later call in the same
    runtime reuses the staged copy if its marker still agrees.

    Returns the directory holding ``gt.csv`` and the patches.
    """
    import shutil  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    destination = Path(destination)
    marker = destination / "_STAGED.json"
    recorded = read_json(marker)
    if isinstance(recorded, dict) and recorded.get("md5") == ARTIFACT_MD5["test.zip"]:
        split = Path(recorded["split"])
        if (split / "gt.csv").is_file() and sum(1 for _ in split.glob("*.png")) == CAH_IMAGES:
            return split

    source = paths.test_archive
    if not source.is_file():
        raise AssetError(f"missing archive: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    local = destination / "test.zip"
    shutil.copyfile(source, local)
    digest = md5sum(local)
    if digest != ARTIFACT_MD5["test.zip"]:
        local.unlink(missing_ok=True)
        raise AssetError(f"staged archive digest {digest} does not match the published one")

    extracted = destination / "extracted"
    if extracted.exists():
        shutil.rmtree(extracted)
    with zipfile.ZipFile(local) as archive:
        archive.extractall(extracted)
    local.unlink()

    roots = [p.parent for p in extracted.rglob("gt.csv")]
    if len(roots) != 1:
        raise AssetError(f"expected one gt.csv in the archive, found {len(roots)}")
    split = roots[0]
    images = sum(1 for _ in split.glob("*.png"))
    if images != CAH_IMAGES:
        raise AssetError(f"staged split has {images} patches, expected {CAH_IMAGES}")

    write_json(marker, {"md5": digest, "split": str(split), "images": images,
                        "staged_at": datetime.now(timezone.utc).isoformat()})
    return split


def mark_complete(directory: str | os.PathLike[str], **fields: Any) -> Path:
    """Record that a unit of work finished. Written last, on purpose."""
    return write_json(
        Path(directory) / "_SUCCESS.json",
        {"completed_at": datetime.now(timezone.utc).isoformat(), **fields},
    )


def is_complete(directory: str | os.PathLike[str], **must_match: Any) -> bool:
    """True if the directory carries a success marker agreeing with ``must_match``.

    Used to skip work that is already done. A marker from a different code
    revision or model set does not count, so changing either forces a rerun.
    A ``None`` in ``must_match`` never matches: a run whose code hash is
    unknown cannot prove it is the same work, so it is redone rather than
    silently reused.
    """
    if any(value is None for value in must_match.values()):
        return False
    recorded = read_json(Path(directory) / "_SUCCESS.json")
    if not isinstance(recorded, dict):
        return False
    return all(recorded.get(key) == value for key, value in must_match.items())


def require_one_run(
    run: str | os.PathLike[str],
    models: Iterable[str],
    producer: str = "01_full_cah",
) -> str:
    """Check that every model's results are complete and come from one code version.

    Reads each ``run/<model>/_SUCCESS.json``. Raises if a marker is missing,
    names a different model, carries no code hash, or if the models were
    produced by different code, so a notebook can never score one model's
    new results next to another's old ones. ``producer`` names the notebook
    that writes these results, for the error message. Returns the shared
    code hash.
    """
    hashes: dict[str, Any] = {}
    for model in models:
        recorded = read_json(Path(run) / model / "_SUCCESS.json")
        if not isinstance(recorded, dict) or recorded.get("model") != model:
            raise AssetError(f"{model} has no complete results in {run}; run {producer} first")
        hashes[model] = recorded.get("code_hash")
    if not hashes:
        raise AssetError("no models given")
    if any(value is None for value in hashes.values()):
        raise AssetError(f"results without a code hash cannot be tied to one version: {hashes}")
    if len(set(hashes.values())) != 1:
        raise AssetError(
            f"results come from different code versions {hashes}; run {producer} again"
        )
    return next(iter(hashes.values()))
