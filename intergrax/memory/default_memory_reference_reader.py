# © Artur Czarnecki. All rights reserved.

"""Default Memory reference reader — scoped enumeration without payload hydration."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlGovernanceDenied,
    MemoryControlPlaneScope,
    MemoryControlScopeRef,
    UserProfileMemoryCapability,
)
from intergrax.memory.contracts.memory_reference_read import (
    MemoryRecordCanonicalRef,
    MemoryReferenceReadOutcome,
    MemoryReferenceReadRequest,
    MemoryReferenceReadResult,
    validate_memory_reference_read_request,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemorySecurityContext,
)
from intergrax.memory.memory_scope_authority import (
    assert_memory_scope_authorized,
    governance_service_or_default,
)
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.recall.retrieval import candidates_from_profile_scan

__all__ = [
    "DefaultMemoryReferenceReader",
    "MemoryReferenceReadCapabilityBinding",
    "MemoryReferenceReadConfigurationError",
]


class MemoryReferenceReadConfigurationError(ValueError):
    """Default reader wiring violates mandatory capability authority invariants."""


@dataclass(frozen=True, slots=True)
class MemoryReferenceReadCapabilityBinding:
    """Authoritative tenant/workspace binding for a configured user-profile surface."""

    tenant_id: str
    workspace_id: str

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        if not tenant:
            raise MemoryReferenceReadConfigurationError(
                "tenant_id must be non-empty"
            )
        if not workspace:
            raise MemoryReferenceReadConfigurationError(
                "workspace_id must be non-empty"
            )
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)


def _control_scope_for_user_read(
    request: MemoryReferenceReadRequest,
) -> MemoryControlScopeRef:
    user_id = request.scope.user_id
    if user_id is None:
        raise MemoryControlAccessDenied(
            "user-scoped reference read requires scope.user_id"
        )
    return MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=request.scope.tenant_id,
        user_id=user_id,
    )


def _security_context(
    identity: RequestIdentity,
    scope: MemoryControlScopeRef,
    operation: MemoryGovernanceOperation,
) -> MemorySecurityContext:
    return MemorySecurityContext(
        identity=identity,
        scope=scope,
        operation=operation,
        reference_time=None,
    )


def _capability_binding_rejects_request(
    binding: MemoryReferenceReadCapabilityBinding,
    identity: RequestIdentity,
    request: MemoryReferenceReadRequest,
) -> bool:
    if binding.tenant_id != identity.tenant_id:
        return True
    if binding.tenant_id != request.scope.tenant_id:
        return True
    return binding.workspace_id != request.scope.workspace_id


@dataclass
class DefaultMemoryReferenceReader:
    """Enumerate canonical user-profile memory references under Memory governance."""

    user_profile: UserProfileMemoryCapability | None = None
    capability_binding: MemoryReferenceReadCapabilityBinding | None = None
    security_governance: MemorySecurityGovernanceService | None = field(
        default_factory=build_default_memory_security_governance_service
    )

    def __post_init__(self) -> None:
        if self.user_profile is not None and self.capability_binding is None:
            raise MemoryReferenceReadConfigurationError(
                "capability_binding is required when user_profile is configured"
            )

    async def read_references(
        self,
        identity: RequestIdentity,
        request: MemoryReferenceReadRequest,
    ) -> MemoryReferenceReadResult:
        invalid = validate_memory_reference_read_request(identity, request)
        if invalid is not None:
            return MemoryReferenceReadResult(outcome=invalid, reason="identity_scope")

        if self.user_profile is None:
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.UNAVAILABLE,
                reason="user_profile_capability_not_configured",
            )

        binding = self.capability_binding
        assert binding is not None

        if _capability_binding_rejects_request(binding, identity, request):
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.SCOPE_REJECTED,
                reason="capability_workspace_binding",
            )

        if request.scope.resource is not None:
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.SCOPE_REJECTED,
                reason="resource_scope_unsupported_for_surface",
            )

        if request.scope.user_id is None:
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.SCOPE_REJECTED,
                reason="user_id_required",
            )

        try:
            control_scope = _control_scope_for_user_read(request)
            assert_memory_scope_authorized(identity, control_scope)
        except MemoryControlAccessDenied:
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.ACCESS_DENIED,
                reason="identity_user_scope",
            )

        governance_service = governance_service_or_default(self.security_governance)
        recall_governance_request = MemoryGovernanceEvaluationRequest(
            context=_security_context(
                identity, control_scope, MemoryGovernanceOperation.RECALL
            ),
        )
        disclosure = governance_service.evaluate(recall_governance_request)
        if not disclosure.permits_disclosure():
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.ACCESS_DENIED,
                reason=disclosure.reason_code.value,
            )

        user_id = request.scope.user_id or ""
        try:
            entries = await self.user_profile.list_active_memory_entries(user_id)
        except MemoryControlGovernanceDenied as exc:
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.ACCESS_DENIED,
                reason=exc.decision.reason_code.value,
            )
        except Exception:
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.UNAVAILABLE,
                reason="backend_error",
            )

        candidates = candidates_from_profile_scan(
            entries,
            query="",
            candidate_limit=request.query.limit,
        )
        candidates = governance_service.filter_recall_candidates(
            recall_governance_request,
            candidates,
        )
        refs: list[MemoryRecordCanonicalRef] = []
        for candidate in candidates[: request.query.limit]:
            entry = candidate.record
            refs.append(
                MemoryRecordCanonicalRef(
                    tenant_id=request.scope.tenant_id,
                    memory_id=entry.entry_id,
                    revision=entry.revision,
                )
            )
        refs.sort(key=lambda item: (item.memory_id, item.revision))
        return MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=tuple(refs),
            evaluated_scope=request.scope,
            reason="ok",
        )
