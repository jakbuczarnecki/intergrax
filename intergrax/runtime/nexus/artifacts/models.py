# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestMismatchError,
    ArtifactLineage,
    ArtifactProvenance,
    ArtifactRetentionClass,
    ArtifactScope,
    DurableArtifactRef,
    compute_artifact_digest,
)
from intergrax.contracts.tracing import TraceArtifactRef


@dataclass(frozen=True)
class Artifact:
    """
    Persistent artifact produced during run execution.

    Artifacts are part of execution substrate (debug / replay / eval),
    not user/profile memory.

    The content digest is never empty: it is derived from ``data`` when omitted and
    must match ``data`` when supplied (``ArtifactDigestMismatchError`` otherwise).
    ``version`` is assigned by the store on ``put`` (append-only history).
    """

    tenant_id: str
    artifact_id: str
    run_id: str
    step_id: str | None

    kind: str
    mime_type: str

    created_at_utc: datetime

    data: bytes
    size_bytes: int

    content_digest: str = ""
    version: int = 1
    artifact_class: ArtifactClass = ArtifactClass.EXECUTION_EVIDENCE
    retention_class: ArtifactRetentionClass = ArtifactRetentionClass.DEFAULT
    expires_at_utc: datetime | None = None
    workspace_id: str | None = None
    producer: ArtifactProvenance | None = None
    lineage: ArtifactLineage | None = None

    def __post_init__(self) -> None:
        computed = compute_artifact_digest(self.data)
        if not self.content_digest:
            object.__setattr__(self, "content_digest", computed)
        elif self.content_digest != computed:
            raise ArtifactDigestMismatchError(
                f"artifact {self.artifact_id!r}: supplied digest does not match the content"
            )

    def to_durable_ref(self, *, uri: str | None = None) -> DurableArtifactRef:
        """Project this artifact onto the shared durable reference contract."""
        if self.producer is None or self.lineage is None:
            raise ValueError(
                f"artifact {self.artifact_id!r} has no producer identity or lineage; a durable reference requires both"
            )
        return DurableArtifactRef(
            artifact_id=self.artifact_id,
            artifact_class=self.artifact_class,
            type=self.kind,
            uri=uri or f"artifact://{self.tenant_id}/{self.artifact_id}@{self.version}",
            mime_type=self.mime_type,
            content_digest=self.content_digest,
            size_bytes=len(self.data),
            version=self.version,
            scope=ArtifactScope(tenant_id=self.tenant_id, workspace_id=self.workspace_id),
            producer=self.producer,
            lineage=self.lineage,
            retention_class=self.retention_class,
            created_at=self.created_at_utc,
        )


__all__ = ["Artifact", "TraceArtifactRef"]


def __getattr__(name: str) -> Any:
    """COMPAT-X shim: ``ArtifactRef`` was an alias of ``TraceArtifactRef`` (PDC-D8-R018).

    The alias is no longer bound at module level, so the only ``ArtifactRef`` is the artifact
    reference contract in ``intergrax.contracts.artifact_ref``. Existing imports keep resolving.
    """
    if name == "ArtifactRef":
        return TraceArtifactRef
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
