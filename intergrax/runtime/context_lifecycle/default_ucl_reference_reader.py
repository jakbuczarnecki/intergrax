# © Artur Czarnecki. All rights reserved.

"""Default UCL reference reader — scoped catalog projection without payload hydration."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.context_lifecycle.repository import (
    OptimizationArtifactScopedReferenceCatalog,
    OptimizationArtifactScopedReferenceQuery,
    ScopedOptimizationArtifactListing,
)
from intergrax.ucl.contracts.ucl_reference_read import (
    UclOptimizationArtifactCanonicalRef,
    UclReferenceLifecycleSelection,
    UclReferenceReadOutcome,
    UclReferenceReadRequest,
    UclReferenceReadResult,
    validate_ucl_reference_read_request,
)

__all__ = [
    "DefaultUclReferenceReader",
    "UclReferenceReadCapabilityBinding",
    "UclReferenceReadConfigurationError",
]


class UclReferenceReadConfigurationError(ValueError):
    """Default reader wiring violates mandatory capability authority invariants."""


@dataclass(frozen=True, slots=True)
class UclReferenceReadCapabilityBinding:
    """Authoritative tenant/workspace/context-scope binding for a configured catalog."""

    tenant_id: str
    workspace_id: str
    context_scope_id: str

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        scope = (self.context_scope_id or "").strip()
        if not tenant:
            raise UclReferenceReadConfigurationError("tenant_id must be non-empty")
        if not workspace:
            raise UclReferenceReadConfigurationError("workspace_id must be non-empty")
        if not scope:
            raise UclReferenceReadConfigurationError("context_scope_id must be non-empty")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "context_scope_id", scope)


def _binding_rejects_request(
    binding: UclReferenceReadCapabilityBinding,
    identity: RequestIdentity,
    request: UclReferenceReadRequest,
) -> bool:
    if binding.tenant_id != identity.tenant_id:
        return True
    if binding.tenant_id != request.scope.tenant_id:
        return True
    if binding.workspace_id != request.scope.workspace_id:
        return True
    return binding.context_scope_id != request.scope.context_scope_id


def _to_canonical_ref(listing: ScopedOptimizationArtifactListing) -> UclOptimizationArtifactCanonicalRef:
    reference = listing.reference
    workspace_id = reference.workspace_id
    if workspace_id is None:
        raise UclReferenceReadConfigurationError(
            "scoped listing reference requires canonical workspace_id"
        )
    return UclOptimizationArtifactCanonicalRef(
        tenant_id=reference.tenant_id,
        workspace_id=workspace_id,
        context_scope_id=listing.context_scope_id,
        artifact_id=reference.artifact_id,
        artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
        artifact_content_hash=reference.artifact_content_hash,
        artifact_type=reference.artifact_type.value,
        lifecycle_status=listing.lifecycle_status.value,
    )


@dataclass
class DefaultUclReferenceReader:
    """Enumerate canonical optimization artifact references under UCL catalog semantics."""

    catalog: OptimizationArtifactScopedReferenceCatalog | None = None
    capability_binding: UclReferenceReadCapabilityBinding | None = None

    def __post_init__(self) -> None:
        if self.catalog is not None and self.capability_binding is None:
            raise UclReferenceReadConfigurationError(
                "capability_binding is required when catalog is configured"
            )

    async def read_references(
        self,
        identity: RequestIdentity,
        request: UclReferenceReadRequest,
    ) -> UclReferenceReadResult:
        invalid = validate_ucl_reference_read_request(identity, request)
        if invalid is not None:
            return UclReferenceReadResult(outcome=invalid, reason="identity_scope")

        if self.catalog is None:
            return UclReferenceReadResult(
                outcome=UclReferenceReadOutcome.UNAVAILABLE,
                reason="catalog_not_configured",
            )

        binding = self.capability_binding
        if binding is None:
            return UclReferenceReadResult(
                outcome=UclReferenceReadOutcome.UNAVAILABLE,
                reason="capability_binding_missing",
            )

        if _binding_rejects_request(binding, identity, request):
            return UclReferenceReadResult(
                outcome=UclReferenceReadOutcome.SCOPE_REJECTED,
                reason="capability_context_scope_binding",
            )

        resource = request.scope.resource
        if resource is not None and resource.resource_kind != "source_ref":
            return UclReferenceReadResult(
                outcome=UclReferenceReadOutcome.SCOPE_REJECTED,
                reason="resource_scope_unsupported",
            )

        include_historical = (
            request.query.lifecycle_selection
            is UclReferenceLifecycleSelection.INCLUDE_HISTORICAL
        )
        try:
            listings = self.catalog.list_scoped_artifact_references(
                OptimizationArtifactScopedReferenceQuery(
                    tenant_id=request.scope.tenant_id,
                    workspace_id=request.scope.workspace_id,
                    context_scope_id=request.scope.context_scope_id,
                    limit=request.query.limit,
                    include_historical=include_historical,
                )
            )
        except Exception:
            return UclReferenceReadResult(
                outcome=UclReferenceReadOutcome.UNAVAILABLE,
                reason="backend_error",
            )

        refs: list[UclOptimizationArtifactCanonicalRef] = []
        for listing in listings:
            if resource is not None and resource.resource_id not in listing.source_refs:
                continue
            refs.append(_to_canonical_ref(listing))

        refs.sort(
            key=lambda item: (item.artifact_id, item.artifact_lookup_key_hash),
        )
        return UclReferenceReadResult(
            outcome=UclReferenceReadOutcome.OK,
            references=tuple(refs[: request.query.limit]),
            reason="ok",
        )
