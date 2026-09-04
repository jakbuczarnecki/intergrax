# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Provider-neutral embedding execution diagnostics contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class EmbeddingProviderExecutionSnapshotStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class EmbeddingProviderExecutionSnapshot:
    configured_device: str | None
    resolved_device: str | None
    provider_batch_size: int | None
    max_length: int | None
    evidence_source: str


@dataclass(frozen=True, slots=True)
class EmbeddingProviderExecutionSnapshotResult:
    status: EmbeddingProviderExecutionSnapshotStatus
    snapshot: EmbeddingProviderExecutionSnapshot | None
    reason: str | None


@runtime_checkable
class EmbeddingProviderExecutionDiagnostics(Protocol):
    def execution_snapshot(self) -> EmbeddingProviderExecutionSnapshot: ...
