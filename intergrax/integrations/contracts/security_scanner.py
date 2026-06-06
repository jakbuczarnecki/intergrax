# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Security scanner integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class ScanFinding(BaseModel):
    """Normalized security finding row."""

    id: str
    severity: str = ""
    title: str = ""
    resource: str = ""
    detail: str = ""


class ScanReport(BaseModel):
    """Aggregated scan result for image or repository targets."""

    target: str
    status: str = "completed"
    findings: Sequence[ScanFinding] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class SecurityScannerBackend(Protocol):
    """Image/SBOM and repository security scanning facade."""

    def scan_image(self, image_ref: str) -> ScanReport:
        """Scan a container image reference (registry tag or digest)."""

    def scan_repo(self, repo_path: str) -> ScanReport:
        """Scan a repository path or VCS URL for policy violations."""
