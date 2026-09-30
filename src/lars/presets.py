"""Named serving configurations: `lars serve --preset 30b`.

A preset names the backend, the model, any LoRA adapters (several are served as an
ensemble, `lars.engine.EnsembleEngine`), and the calibrator for each (or for the model).
Models, adapters and calibrators live on the Hugging Face Hub and are downloaded on first
use; a local directory or file with the same argument is used as is.
The published adapters are trained on the commercially licensed corpus (DESIGN.md 8.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

M30 = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
M4 = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
CALIBRATION_FILE = "lars-calibration.json"


@dataclass(frozen=True)
class Preset:
    backend: str
    model: str
    adapters: tuple[str, ...]
    calibrations: tuple[str, ...]
    description: str


# Everything published lives in one Hub repository, one folder per tier, each with its pooled
# calibrator. The 4B ships with its adapter fused in (`scripts/fuse_mixed.py`: the adapter's
# layers at 8 bits, the rest at the base's 4 bits), which is as accurate as the adapter and
# faster. The 30B ships as an adapter: fusing its attention-only LoRA gains nothing.
REPO = "moedex/lars"

# Two tiers. The 30B-A3B runs about 3B parameters per token, so it is nearly as fast as the 4B;
# what it costs is memory (about 18 GB resident against about 5 GB). Pick the smallest tier whose
# evaluation on your decision clears your bar; `serve --escalate` covers questions neither knows.
PRESETS: dict[str, Preset] = {
    "4b": Preset("mlx", f"{REPO}/4b", (), (f"{REPO}/4b/{CALIBRATION_FILE}",),
                 "Qwen3-4B 4-bit with its LoRA fused in and its pooled calibrator: about 5 GB, for small machines"),
    "30b": Preset("mlx", M30, (f"{REPO}/30b-a3b",), (f"{REPO}/30b-a3b/{CALIBRATION_FILE}",),
                  "Qwen3-30B-A3B 4-bit with the attention LoRA and its pooled calibrator"),
}


def split_hub_ref(value: str | None) -> tuple[str, str] | None:
    """`org/repo`, or `org/repo/<path>` inside it, as (repo, path); None for a local path."""
    if not value or Path(value).exists() or value.startswith((".", "/", "~")):
        return None
    parts = value.split("/")
    if len(parts) < 2 or not all(parts):
        return None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _is_hub_id(value: str) -> bool:
    return split_hub_ref(value) is not None


def resolve_hub(value: str | None, cached_only: bool = False) -> str | None:
    """A local path for a Hub reference, downloaded to the cache: a repository, a folder in one
    (`org/repo/4b`), or a file (`org/repo/4b/lars-calibration.json`). Anything else is returned as is."""
    ref = split_hub_ref(value)
    if ref is None:
        return value
    repo, sub = ref
    from huggingface_hub import hf_hub_download, snapshot_download

    if sub and Path(sub).suffix:
        return hf_hub_download(repo_id=repo, filename=sub, local_files_only=cached_only)
    if not sub:
        return snapshot_download(repo_id=repo, local_files_only=cached_only)
    root = snapshot_download(repo_id=repo, allow_patterns=[f"{sub}/*"], local_files_only=cached_only)
    return str(Path(root) / sub)


def resolve_adapter(adapter: str | None) -> str | None:
    """A local adapter directory, or a Hub repository (or a folder in one) downloaded to the local cache."""
    return resolve_hub(adapter)


def resolve_calibration(calibration: str | None) -> str | None:
    """A local calibrator file, or `<org>/<repo>/<path>` downloaded from a Hub repository."""
    return resolve_hub(calibration)


def resolve_model(model: str | None) -> str | None:
    """A model inside a Hub repository folder is downloaded here; a plain `org/repo` is left to the backend."""
    ref = split_hub_ref(model)
    return resolve_hub(model) if ref and ref[1] else model
