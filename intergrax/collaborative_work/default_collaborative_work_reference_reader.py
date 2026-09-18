# © Artur Czarnecki. All rights reserved.

"""Default Collaborative Work reference reader — scoped catalog projection without hydration."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkArtifactCanonicalRef,
    CollaborativeWorkArtifactVersionCanonicalRef,
    CollaborativeWorkCanonicalRef,
    CollaborativeWorkItemCanonicalRef,
    CollaborativeWorkReferenceEntityKind,
    CollaborativeWorkReferenceReadOutcome,
    CollaborativeWorkReferenceReadRequest,
    CollaborativeWorkReferenceReadResult,
    CollaborativeWorkVersionSelection,
    validate_collaborative_work_reference_read_request,
)
from intergrax.collaborative_work.repository import (
    CollaborativeWorkScopedReferenceCatalog,
    CollaborativeWorkScopedReferenceListing,
    CollaborativeWorkScopedReferenceQuery,
)
from intergrax.contracts.agent_run import RequestIdentity

__all__ = [
    "DefaultCollaborativeWorkReferenceReader",
    "CollaborativeWorkReferenceReadCapabilityBinding",
    "CollaborativeWorkReferenceReadConfigurationError",
]

_KIND_BY_ENTITY = {
    CollaborativeWorkReferenceEntityKind.WORK_ITEM: "work_item",
    CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT: "work_artifact",
    CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION: "work_artifact_version",
}


class CollaborativeWorkReferenceReadConfigurationError(ValueError):
    """Default reader wiring violates mandatory capability authority invariants."""


@dataclass(frozen=True, slots=True)
class CollaborativeWorkReferenceReadCapabilityBinding:
    """Authoritative tenant/workspace binding for a configured catalog surface."""

    tenant_id: str
    workspace_id: str

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        if not tenant:
            raise CollaborativeWorkReferenceReadConfigurationError(
                "tenant_id must be non-empty"
            )
        if not workspace:
            raise CollaborativeWorkReferenceReadConfigurationError(
                "workspace_id must be non-empty"
            )
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)


def _binding_rejects_request(
    binding: CollaborativeWorkReferenceReadCapabilityBinding,
    identity: RequestIdentity,
    request: CollaborativeWorkReferenceReadRequest,
) -> bool:
    if binding.tenant_id != identity.tenant_id:
        return True
    if binding.tenant_id != request.scope.tenant_id:
        return True
    return binding.workspace_id != request.scope.workspace_id


def _catalog_query(request: CollaborativeWorkReferenceReadRequest) -> CollaborativeWorkScopedReferenceQuery:
    include_historical = (
        request.query.version_selection
        is CollaborativeWorkVersionSelection.INCLUDE_HISTORICAL
    )
    kinds = frozenset(_KIND_BY_ENTITY[kind] for kind in request.query.entity_kinds)
    return CollaborativeWorkScopedReferenceQuery(
        tenant_id=request.scope.tenant_id,
        workspace_id=request.scope.workspace_id,
        limit=request.query.limit,
        entity_kinds=kinds,
        include_historical=include_historical,
        work_item_id=request.scope.work_item_id,
        work_artifact_id=request.scope.work_artifact_id,
        work_artifact_version_id=request.scope.work_artifact_version_id,
    )


def _listing_within_query_scope(
    listing: CollaborativeWorkScopedReferenceListing,
    query: CollaborativeWorkScopedReferenceQuery,
) -> bool:
    if listing.tenant_id != query.tenant_id:
        return False
    if listing.workspace_id != query.workspace_id:
        return False
    if query.work_item_id is not None and listing.work_item_id != query.work_item_id:
        return False
    if query.work_artifact_id is not None and listing.work_artifact_id is not None:
        if listing.work_artifact_id != query.work_artifact_id:
            return False
    if query.work_artifact_version_id is not None and listing.work_artifact_version_id is not None:
        if listing.work_artifact_version_id != query.work_artifact_version_id:
            return False
    return True


def _to_canonical_ref(
    listing: CollaborativeWorkScopedReferenceListing,
) -> CollaborativeWorkCanonicalRef:
    if listing.entity_kind == "work_item":
        if listing.work_item_state is None:
            raise CollaborativeWorkReferenceReadConfigurationError(
                "work_item listing requires work_item_state"
            )
        return CollaborativeWorkItemCanonicalRef(
            tenant_id=listing.tenant_id,
            workspace_id=listing.workspace_id,
            work_item_id=listing.work_item_id,
            state=listing.work_item_state,
        )
    if listing.entity_kind == "work_artifact":
        if listing.work_artifact_id is None or listing.current_version_id is None:
            raise CollaborativeWorkReferenceReadConfigurationError(
                "work_artifact listing requires artifact and current_version ids"
            )
        return CollaborativeWorkArtifactCanonicalRef(
            tenant_id=listing.tenant_id,
            workspace_id=listing.workspace_id,
            work_item_id=listing.work_item_id,
            work_artifact_id=listing.work_artifact_id,
            current_version_id=listing.current_version_id,
        )
    if listing.work_artifact_id is None or listing.work_artifact_version_id is None:
        raise CollaborativeWorkReferenceReadConfigurationError(
            "work_artifact_version listing requires artifact and version ids"
        )
    return CollaborativeWorkArtifactVersionCanonicalRef(
        tenant_id=listing.tenant_id,
        workspace_id=listing.workspace_id,
        work_item_id=listing.work_item_id,
        work_artifact_id=listing.work_artifact_id,
        work_artifact_version_id=listing.work_artifact_version_id,
    )


def _reference_sort_key(ref: CollaborativeWorkCanonicalRef) -> tuple[str, str, str, str]:
    if isinstance(ref, CollaborativeWorkItemCanonicalRef):
        return ("work_item", ref.work_item_id, "", "")
    if isinstance(ref, CollaborativeWorkArtifactCanonicalRef):
        return ("work_artifact", ref.work_item_id, ref.work_artifact_id, "")
    return (
        "work_artifact_version",
        ref.work_item_id,
        ref.work_artifact_id,
        ref.work_artifact_version_id,
    )


@dataclass
class DefaultCollaborativeWorkReferenceReader:
    """Enumerate canonical Collaborative Work references under CW catalog semantics."""

    catalog: CollaborativeWorkScopedReferenceCatalog | None = None
    capability_binding: CollaborativeWorkReferenceReadCapabilityBinding | None = None

    def __post_init__(self) -> None:
        if self.catalog is not None and self.capability_binding is None:
            raise CollaborativeWorkReferenceReadConfigurationError(
                "capability_binding is required when catalog is configured"
            )

    def read_references(
        self,
        identity: RequestIdentity,
        request: CollaborativeWorkReferenceReadRequest,
    ) -> CollaborativeWorkReferenceReadResult:
        invalid = validate_collaborative_work_reference_read_request(identity, request)
        if invalid is not None:
            return CollaborativeWorkReferenceReadResult(outcome=invalid, reason="identity_scope")

        if self.catalog is None:
            return CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.UNAVAILABLE,
                reason="catalog_not_configured",
            )

        binding = self.capability_binding
        if binding is None:
            return CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.UNAVAILABLE,
                reason="capability_binding_missing",
            )

        if _binding_rejects_request(binding, identity, request):
            return CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.SCOPE_REJECTED,
                reason="capability_workspace_binding",
            )

        scoped_query = _catalog_query(request)
        try:
            listings = self.catalog.list_scoped_references(scoped_query)
        except Exception:
            return CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.UNAVAILABLE,
                reason="backend_error",
            )

        refs: list[CollaborativeWorkCanonicalRef] = []
        for listing in listings:
            if not _listing_within_query_scope(listing, scoped_query):
                return CollaborativeWorkReferenceReadResult(
                    outcome=CollaborativeWorkReferenceReadOutcome.UNAVAILABLE,
                    reason="catalog_contract_violation",
                )
            try:
                refs.append(_to_canonical_ref(listing))
            except CollaborativeWorkReferenceReadConfigurationError:
                return CollaborativeWorkReferenceReadResult(
                    outcome=CollaborativeWorkReferenceReadOutcome.UNAVAILABLE,
                    reason="catalog_contract_violation",
                )

        refs.sort(key=_reference_sort_key)
        return CollaborativeWorkReferenceReadResult(
            outcome=CollaborativeWorkReferenceReadOutcome.OK,
            references=tuple(refs),
            reason="ok",
        )
