"""Model artifact registry.

Artifacts are serialised with joblib (pickle). Pickle files can execute code when
loaded, so CollectIQ only loads files it produced itself: every artifact's SHA-256
is recorded in ``registry.json`` at training time and verified before loading.
Files whose hash does not match - or that are not in the registry - are refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.config import path
from src.utils.helpers import file_sha256, read_json, write_json

REGISTRY_FILE = "registry.json"


class UntrustedArtifactError(RuntimeError):
    """Raised when an artifact is missing from the registry or fails hash verification."""


def _registry_path() -> Path:
    return path("artifacts_dir") / REGISTRY_FILE


def load_registry() -> dict[str, Any]:
    """Return the registry dict (empty if none)."""
    p = _registry_path()
    return read_json(p) if p.exists() else {}


def save_artifact(name: str, obj: Any, metadata: dict[str, Any]) -> str:
    """Serialise ``obj`` and register its checksum. Returns the relative path."""
    art_dir = path("artifacts_dir")
    art_dir.mkdir(parents=True, exist_ok=True)
    file = art_dir / f"{name}.joblib"
    joblib.dump(obj, file, compress=3)
    reg = load_registry()
    reg[name] = {**metadata, "file": file.name, "sha256": file_sha256(file)}
    write_json(reg, _registry_path())
    return f"models/artifacts/{file.name}"


def load_artifact(name: str) -> Any:
    """Load a registered artifact after verifying its SHA-256 checksum."""
    reg = load_registry()
    if name not in reg:
        raise UntrustedArtifactError(f"Artifact '{name}' is not registered; refusing to load.")
    file = path("artifacts_dir") / reg[name]["file"]
    if not file.exists():
        raise FileNotFoundError(f"Artifact file missing: {file}")
    if file_sha256(file) != reg[name]["sha256"]:
        raise UntrustedArtifactError(f"Checksum mismatch for '{name}'; refusing to load.")
    return joblib.load(file)


def artifact_metadata(name: str) -> dict[str, Any] | None:
    """Registry entry for an artifact, if present."""
    return load_registry().get(name)
