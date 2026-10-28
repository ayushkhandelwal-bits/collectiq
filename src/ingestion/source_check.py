"""Primary-source availability probe.

CollectIQ's preferred public source is LendingClub loan data hosted on Kaggle.
Kaggle requires an authenticated account (API token) to download files. This
module checks - without scraping and without asking for credentials - whether an
already-authorised route exists. The result is recorded in the source manifest.
"""

from __future__ import annotations

import importlib.util
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import load_settings
from src.utils.helpers import get_logger, utc_now

log = get_logger(__name__)


@dataclass
class SourceStatus:
    """Outcome of probing the primary public data source."""

    source_name: str
    source_url: str
    checked_at: str
    network_reachable: bool
    network_detail: str
    credentials_available: bool
    client_installed: bool
    usable: bool
    decision: str

    def to_dict(self) -> dict:
        return asdict(self)


def _credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def probe_primary_source(timeout: float | None = None) -> SourceStatus:
    """Check whether the Kaggle LendingClub source can be lawfully downloaded here."""
    cfg = load_settings()["source"]
    timeout = timeout or cfg["probe_timeout_seconds"]
    reachable, detail = False, ""
    try:
        req = urllib.request.Request(cfg["primary_url"], method="HEAD",
                                     headers={"User-Agent": "CollectIQ-source-probe"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed URL
            reachable = resp.status < 400
            detail = f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code}"
    except Exception as exc:  # network policy, DNS, proxy denial, timeout
        detail = f"{type(exc).__name__}: {exc}"[:200]

    creds = _credentials_present()
    client = importlib.util.find_spec("kaggle") is not None
    usable = reachable and creds and client
    if usable:
        decision = "PRIMARY_AVAILABLE"
    else:
        reasons = []
        if not reachable:
            reasons.append(f"source not reachable ({detail})")
        if not creds:
            reasons.append("no authorised Kaggle API credentials in environment")
        if not client:
            reasons.append("Kaggle API client not installed")
        decision = "SYNTHETIC_FALLBACK: " + "; ".join(reasons)
    status = SourceStatus(cfg["primary_name"], cfg["primary_url"], utc_now(), reachable, detail,
                          creds, client, usable, decision)
    log.info("Primary source probe: %s", decision)
    return status
