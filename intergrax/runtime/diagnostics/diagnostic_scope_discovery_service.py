# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only diagnostic execution scope discovery service (DG-002)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticExecutionScopeCandidate,
    DiagnosticScopeDiscoveryIntegrityError,
    DiagnosticScopeDiscoveryRequest,
    DiagnosticScopeDiscoveryResult,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReference,
    build_diagnostic_scope_discovery_result,
    provider_unavailable_result,
    unsupported_reference_result,
    validate_scope_discovery_request,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeDiscoveryProvider,
    DiagnosticScopeDiscoveryProviderRegistry,
    DiagnosticScopeProviderIntegrityError,
    DiagnosticScopeProviderResult,
    DiagnosticScopeProviderUnavailableError,
    validate_scope_provider_result,
)
from intergrax.runtime.diagnostics.diagnostic_subject import ExecutionDiagnosticSubjectRef


class DiagnosticScopeDiscoveryService:
    """Read-only derived resolver for tenant-scoped diagnostic execution scope."""

    def __init__(
        self,
        providers: tuple[DiagnosticScopeDiscoveryProvider, ...],
    ) -> None:
        self._registry = DiagnosticScopeDiscoveryProviderRegistry(providers)

    def discover_scope(
        self,
        request: DiagnosticScopeDiscoveryRequest,
    ) -> DiagnosticScopeDiscoveryResult:
        validated_request = validate_scope_discovery_request(request)
        provider = self._registry.resolve_for_reference(validated_request.reference)
        if provider is None:
            return unsupported_reference_result()

        try:
            provider_result = provider.discover(
                tenant_id=validated_request.tenant_id,
                reference=validated_request.reference,
                candidate_limit=validated_request.candidate_limit,
            )
        except DiagnosticScopeProviderIntegrityError as exc:
            raise DiagnosticScopeDiscoveryIntegrityError(str(exc)) from exc
        except DiagnosticScopeProviderUnavailableError as exc:
            return provider_unavailable_result(
                limitations=(f"provider unavailable: {type(exc).__name__}",),
            )

        return _project_provider_result(
            validate_scope_provider_result(provider_result),
            candidate_limit=validated_request.candidate_limit,
            tenant_id=validated_request.tenant_id,
        )


def _project_provider_result(
    provider_result: DiagnosticScopeProviderResult,
    *,
    candidate_limit: int,
    tenant_id: str,
) -> DiagnosticScopeDiscoveryResult:
    deduplicated_candidates = _deduplicate_execution_candidates(
        provider_result.candidates,
        tenant_id=tenant_id,
    )
    projected_candidates = deduplicated_candidates[:candidate_limit]
    resolved_scope = provider_result.resolved_scope
    if resolved_scope is not None:
        _assert_execution_scope_tenant(resolved_scope, tenant_id=tenant_id)

    return build_diagnostic_scope_discovery_result(
        status=provider_result.status,
        resolved_scope=resolved_scope,
        candidates=projected_candidates,
        candidate_count=provider_result.candidate_count,
        candidate_count_exact=provider_result.candidate_count_exact,
        provenance=provider_result.provenance,
        limitations=provider_result.limitations,
    )


def _deduplicate_execution_candidates(
    candidates: tuple[DiagnosticExecutionScopeCandidate, ...],
    *,
    tenant_id: str,
) -> tuple[DiagnosticExecutionScopeCandidate, ...]:
    seen: set[tuple[str, str]] = set()
    ordered: list[DiagnosticExecutionScopeCandidate] = []
    for candidate in sorted(
        candidates,
        key=_execution_candidate_sort_key,
    ):
        subject_ref = candidate.subject_ref
        _assert_execution_scope_tenant(subject_ref, tenant_id=tenant_id)
        identity = (str(subject_ref.task_id), str(subject_ref.run_id))
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(candidate)
    return tuple(ordered)


def _execution_candidate_sort_key(
    candidate: DiagnosticExecutionScopeCandidate,
) -> tuple[str, str]:
    subject_ref = candidate.subject_ref
    return (str(subject_ref.task_id), str(subject_ref.run_id))


def _assert_execution_scope_tenant(
    subject_ref: ExecutionDiagnosticSubjectRef,
    *,
    tenant_id: str,
) -> None:
    if subject_ref.tenant_id != tenant_id:
        raise DiagnosticScopeDiscoveryIntegrityError(
            "execution subject tenant_id does not match discovery request tenant",
        )
