"""Typed install report for proof data packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataPackageInstallReport:
    package_id: str
    package_version: str
    files_total: int
    files_downloaded: int
    files_reused_from_cache: int
    files_installed_from_existing: int
    bytes_downloaded: int
    bytes_reused: int
    verification_passed: bool
    install_location: Path
    elapsed_seconds: float
