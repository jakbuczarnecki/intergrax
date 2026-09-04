"""Transport contracts for proof data package distribution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TransportDownloadResult:
    bytes_written: int
    resumed: bool
    supports_range: bool
    final_uri: str


class DataPackageTransportPort(Protocol):
    """Obtain package file bytes without scenario semantics."""

    def download_file(
        self,
        source_uri: str,
        destination_partial: Path,
        *,
        resume_from_byte: int = 0,
    ) -> TransportDownloadResult:
        """Stream bytes into ``destination_partial`` and return transfer metadata."""
