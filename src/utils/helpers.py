"""Small shared utilities: logging, hashing, JSON IO and timestamps."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent format."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    return logging.getLogger(name)


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(file_path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic content hash of a DataFrame (used as a dataset version)."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401 - JSON hook
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def to_json(obj: Any, **kwargs: Any) -> str:
    """JSON-serialise objects that may contain numpy / pandas scalars."""
    return json.dumps(obj, cls=_NumpyEncoder, **kwargs)


def write_json(obj: Any, file_path: str | Path) -> None:
    """Write ``obj`` as pretty JSON."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    Path(file_path).write_text(to_json(obj, indent=2), encoding="utf-8")


def read_json(file_path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(file_path).read_text(encoding="utf-8"))
