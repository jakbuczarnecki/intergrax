# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Storage-only read sources registered behind the governed access port.

A source turns one owner's store into the neutral ``ArtifactReadSource`` shape. Sources hold
no security logic: authorization, tenant check and audit are done by the governed port.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.collaborative_work.content_storage import (
    ArtifactContentStore,
    GetArtifactContentRequest,
    artifact_content_object_key,
)
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactLineage,
    ArtifactProvenance,
    ArtifactScope,
    DurableArtifactRef,
)
from intergrax.contracts.collaborative_work import ArtifactContentRef
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactReadRequest,
    GovernedArtifactContent,
)
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore

#: Producer recorded for artifacts stored before producer identity became part of the contract.
LEGACY_UNKNOWN_PRODUCER = "legacy_unknown"


class ExecutionArtifactSource:
    """Read source over the execution ``ArtifactStore`` (evidence and tool-output classes)."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def _fetch(self, request: ArtifactReadRequest):
        try:
            if request.version is None:
                artifact = self._store.get(request.tenant_id, request.artifact_id)
            else:
                artifact = self._store.get_version(request.tenant_id, request.artifact_id, request.version)
        except KeyError:
            return None
        if artifact.artifact_class is not request.artifact_class:
            return None
        return artifact

    def read(self, request: ArtifactReadRequest) -> GovernedArtifactContent | None:
        artifact = self._fetch(request)
        if artifact is None:
            return None
        if artifact.producer is None or artifact.lineage is None:
            from dataclasses import replace

            artifact = replace(
                artifact,
                producer=artifact.producer or ArtifactProvenance(created_by_agent_id=LEGACY_UNKNOWN_PRODUCER),
                lineage=artifact.lineage or ArtifactLineage(run_id=artifact.run_id),
            )
        return GovernedArtifactContent(ref=artifact.to_durable_ref(), content=artifact.data)

    # ---- lifecycle (legal hold / deletion), delegated to the store's own invariants ----

    def place_legal_hold(self, request: ArtifactReadRequest, *, placed_by: str, reason: str) -> None:
        self._store.place_legal_hold(request.tenant_id, request.artifact_id, placed_by=placed_by, reason=reason)

    def release_legal_hold(self, request: ArtifactReadRequest) -> None:
        self._store.release_legal_hold(request.tenant_id, request.artifact_id)

    def delete(self, request: ArtifactReadRequest) -> None:
        self._store.delete_artifact(request.tenant_id, request.artifact_id)


@dataclass(frozen=True)
class BusinessArtifactBinding:
    """What the composition root resolves for one Collaborative Work artifact version."""

    content_ref: ArtifactContentRef
    version: int
    producer: ArtifactProvenance
    lineage: ArtifactLineage
    artifact_type: str = "work_artifact"


class CollaborativeWorkContentSource:
    """Read source over Collaborative Work content (business class). Read and download only.

    Collaborative Work defers hard delete, so this source exposes no lifecycle operations; the
    governed port reports ``operation_unsupported`` for hold and delete on business artifacts.
    """

    def __init__(
        self,
        content_store: ArtifactContentStore,
        resolve_binding: Callable[[str, str, str, int | None], BusinessArtifactBinding | None],
    ) -> None:
        self._content_store = content_store
        self._resolve = resolve_binding

    def _binding(self, request: ArtifactReadRequest) -> BusinessArtifactBinding | None:
        if not request.workspace_id:
            return None
        return self._resolve(request.tenant_id, request.workspace_id, request.artifact_id, request.version)

    def read(self, request: ArtifactReadRequest) -> GovernedArtifactContent | None:
        binding = self._binding(request)
        if binding is None or request.workspace_id is None:
            return None
        stored = self._content_store.get(
            GetArtifactContentRequest(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                content_ref=binding.content_ref,
            )
        )
        if stored is None:
            return None
        ref = DurableArtifactRef(
            artifact_id=request.artifact_id,
            artifact_class=ArtifactClass.BUSINESS,
            type=binding.artifact_type,
            uri=binding.content_ref.content_ref,
            mime_type=binding.content_ref.media_type,
            content_digest=binding.content_ref.integrity_digest,
            size_bytes=len(stored.body),
            version=binding.version,
            scope=ArtifactScope(tenant_id=request.tenant_id, workspace_id=request.workspace_id),
            producer=binding.producer,
            lineage=binding.lineage,
        )
        return GovernedArtifactContent(ref=ref, content=stored.body)

    def download_key(self, request: ArtifactReadRequest) -> str | None:
        binding = self._binding(request)
        if binding is None or request.workspace_id is None:
            return None
        return artifact_content_object_key(request.tenant_id, request.workspace_id, binding.content_ref)
