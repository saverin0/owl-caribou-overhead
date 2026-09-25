"""Starting a run: mount storage, find the GPU, check the assets are sound.

One call at the top of every notebook, so the checks cannot be skipped by
accident and every run records what it ran on.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .io import Paths, default_paths, on_colab, verify_checkpoints, verify_test_split, write_json
from .sync import RUNTIME_DIR


class SessionError(RuntimeError):
    """The environment is not fit to run on."""


@dataclass
class Session:
    paths: Paths
    device: str
    third_party: Path
    gpu: dict[str, Any] | None = None
    assets: dict[str, Any] = field(default_factory=dict)
    code_hash: str | None = None

    def provenance(self, **extra: Any) -> dict[str, Any]:
        """A record of what produced a result."""
        import torch

        return {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "code_hash": self.code_hash,
            "device": self.device,
            "gpu": self.gpu,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "assets": self.assets,
            **extra,
        }

    def record(self, directory: str | Path, **extra: Any) -> Path:
        return write_json(Path(directory) / "provenance.json", self.provenance(**extra))


def _mount_drive() -> None:
    if Path("/content/drive/MyDrive").is_dir():
        return
    from google.colab import drive  # noqa: PLC0415 - only exists on Colab

    drive.mount("/content/drive")


def _find_third_party(explicit: str | Path | None) -> Path:
    """Locate the vendored model code, unpacked by the sync cell or local."""
    if explicit is not None:
        return Path(explicit)

    candidates = [
        Path(RUNTIME_DIR) / "third_party",  # where the sync cell unpacks
        Path.cwd() / "third_party",
        Path.cwd().parent / "third_party",
    ]
    # Also try relative to this file, which works from an installed checkout.
    candidates.append(Path(__file__).resolve().parents[2] / "third_party")
    for candidate in candidates:
        if (candidate / "animaloc").is_dir():
            return candidate
    raise SessionError(
        "cannot find third_party/. On Colab, run the sync cell first; "
        "locally, start from the project root."
    )


def start_session(
    *,
    assets: str | Path | None = None,
    results: str | Path | None = None,
    third_party: str | Path | None = None,
    require_gpu: bool = True,
    models: Sequence[str] = (),
    rehash: bool = False,
    verify_assets: bool = True,
) -> Session:
    """Prepare a run and fail early if anything is missing.

    ``models`` names the checkpoints to verify. Verification uses the sidecar
    markers, so it is quick; pass ``rehash=True`` to read all 4.3 GB again.
    """
    import torch

    if on_colab():
        _mount_drive()

    device = "cpu"
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 7:
            gpu = {
                "name": properties.name,
                "compute_capability": list(capability),
                "vram_gb": round(properties.total_memory / 2**30, 2),
            }
            device = "cuda"
        elif require_gpu:
            raise SessionError(
                f"{properties.name} is too old for these models; use a T4, L4, A100 or newer"
            )
        # An older GPU is simply not used when no GPU is required.
    elif require_gpu:
        raise SessionError(
            "no CUDA GPU. In Colab choose Runtime -> Change runtime type -> GPU."
        )

    resolved = default_paths(assets, results)
    vendored = _find_third_party(third_party)
    if str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))

    summary: dict[str, Any] = {}
    if verify_assets:
        summary["test_split"] = verify_test_split(resolved)
        if models:
            summary["checkpoints"] = verify_checkpoints(resolved, models, rehash=rehash)

    session = Session(
        paths=resolved,
        device=device,
        third_party=vendored,
        gpu=gpu,
        assets=summary,
        code_hash=os.environ.get("OWLCARIBOU_CODE_HASH"),
    )

    where = "Colab" if on_colab() else "local"
    print(f"session: {where} | device {session.device}" + (f" ({gpu['name']}, {gpu['vram_gb']} GB)" if gpu else ""))
    print(f"  torch   {torch.__version__} | CUDA {torch.version.cuda or 'none'} | "
          f"Python {platform.python_version()}")
    print(f"  assets  {resolved.assets}")
    print(f"  results {resolved.results}")
    print(f"  code    {session.code_hash or 'unsynced (running from disk)'}")
    if summary.get("test_split"):
        split = summary["test_split"]
        print(f"  split   {split['images']} patches, {split['points']} points")
    if summary.get("checkpoints"):
        print(f"  weights verified: {', '.join(sorted(summary['checkpoints']))}")
    return session
