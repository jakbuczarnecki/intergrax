# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure scoped integration adaptation core (AW-7C-P2/P3)."""

from __future__ import annotations

from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationError,
    ScopedIntegrationAdaptationFailureReason,
    ScopedIntegrationAdaptationOperationId,
    ScopedIntegrationAdaptationRequest,
    ScopedIntegrationAdaptationStrategy,
    ScopedIntegrationAdaptationTarget,
    ScopedIntegrationAdaptationTargetResolver,
    derive_scoped_integration_adaptation_scope_fingerprint,
)


class ScopedIntegrationAdaptationService:
    """Internal pure core — resolver, strategy selection, scope containment only."""

    def __init__(
        self,
        *,
        target_resolver: ScopedIntegrationAdaptationTargetResolver,
        strategies: tuple[ScopedIntegrationAdaptationStrategy, ...],
    ) -> None:
        seen: set[str] = set()
        for strategy in strategies:
            strategy_id = strategy.strategy_id
            if strategy_id in seen:
                raise ScopedIntegrationAdaptationError(
                    ScopedIntegrationAdaptationFailureReason.ARTIFACT_INVALID,
                    detail=f"duplicate strategy_id: {strategy_id}",
                )
            seen.add(strategy_id)
        self._target_resolver = target_resolver
        self._strategies = strategies

    def adapt(
        self,
        request: ScopedIntegrationAdaptationRequest,
    ) -> ScopedIntegrationAdaptationArtifact:
        _validate_request_identity_continuity(request)
        target = _resolve_target(self._target_resolver, request)
        _verify_target_continuity(request, target)
        artifact = _select_and_adapt(request, target, self._strategies)
        _verify_artifact_containment(request, target, artifact)
        return artifact


class ScopedIntegrationAdaptationPortAdapter:
    """Explicit composition adapter — not a global registry or default strategy."""

    def __init__(
        self,
        *,
        target_resolver: ScopedIntegrationAdaptationTargetResolver,
        strategies: tuple[ScopedIntegrationAdaptationStrategy, ...],
    ) -> None:
        self._core = ScopedIntegrationAdaptationService(
            target_resolver=target_resolver,
            strategies=strategies,
        )

    def adapt(
        self,
        request: ScopedIntegrationAdaptationRequest,
    ) -> ScopedIntegrationAdaptationArtifact:
        return self._core.adapt(request)


def _validate_request_identity_continuity(
    request: ScopedIntegrationAdaptationRequest,
) -> None:
    scope = request.scope
    if scope.tenant_id != request.tenant_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH,
            detail="scope tenant mismatch",
        )
    if scope.integration_category != request.integration_category:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="scope integration_category mismatch",
        )
    if scope.provider_id != request.provider_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="scope provider_id mismatch",
        )
    if scope.resource_scope != request.resource_scope:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="scope resource_scope mismatch",
        )


def _resolve_target(
    resolver: ScopedIntegrationAdaptationTargetResolver,
    request: ScopedIntegrationAdaptationRequest,
) -> ScopedIntegrationAdaptationTarget:
    try:
        return resolver.resolve_target(request)
    except ScopedIntegrationAdaptationError:
        raise
    except TimeoutError as exc:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.ADAPTATION_FAILED,
            detail=str(exc),
        ) from exc
    except OSError as exc:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.ADAPTATION_FAILED,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.ADAPTATION_FAILED,
            detail=str(exc),
        ) from exc


def _verify_target_continuity(
    request: ScopedIntegrationAdaptationRequest,
    target: ScopedIntegrationAdaptationTarget,
) -> None:
    scope = request.scope
    if target.tenant_id != request.tenant_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH,
            detail="target tenant mismatch",
        )
    if target.tenant_id != scope.tenant_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH,
            detail="target scope tenant mismatch",
        )
    if target.provider_id != request.provider_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="target provider mismatch",
        )
    if target.provider_id != scope.provider_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="target scope provider mismatch",
        )
    if target.integration_category != request.integration_category:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="target category mismatch",
        )
    if target.integration_category != scope.integration_category:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="target scope category mismatch",
        )
    if target.resource_scope != request.resource_scope:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="target resource_scope mismatch",
        )
    if target.resource_scope != scope.resource_scope:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="target scope resource_scope mismatch",
        )
    if target.current_revision != scope.candidate_revision:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.REVISION_MISMATCH,
            detail="target revision mismatch",
        )


def _select_and_adapt(
    request: ScopedIntegrationAdaptationRequest,
    target: ScopedIntegrationAdaptationTarget,
    strategies: tuple[ScopedIntegrationAdaptationStrategy, ...],
) -> ScopedIntegrationAdaptationArtifact:
    matching = [
        strategy
        for strategy in strategies
        if strategy.supports(request, target)
    ]
    if not matching:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.STRATEGY_UNAVAILABLE,
        )
    if len(matching) > 1:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.STRATEGY_AMBIGUOUS,
        )
    try:
        return matching[0].adapt(request, target)
    except ScopedIntegrationAdaptationError:
        raise
    except TimeoutError as exc:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.ADAPTATION_FAILED,
            detail=str(exc),
        ) from exc
    except OSError as exc:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.ADAPTATION_FAILED,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.ADAPTATION_FAILED,
            detail=str(exc),
        ) from exc


def _verify_artifact_containment(
    request: ScopedIntegrationAdaptationRequest,
    target: ScopedIntegrationAdaptationTarget,
    artifact: ScopedIntegrationAdaptationArtifact,
) -> None:
    del target
    req_scope = request.scope
    art_scope = artifact.scope
    if artifact.tenant_id != request.tenant_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH,
            detail="artifact tenant mismatch",
        )
    if art_scope.tenant_id != request.tenant_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH,
            detail="artifact scope tenant mismatch",
        )
    if artifact.integration_category != request.integration_category:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="artifact category mismatch",
        )
    if art_scope.integration_category != request.integration_category:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="artifact scope category mismatch",
        )
    if artifact.provider_id != request.provider_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="artifact provider mismatch",
        )
    if art_scope.provider_id != request.provider_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="artifact scope provider mismatch",
        )
    if artifact.resource_scope != request.resource_scope:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.SCOPE_WIDENING,
            detail="artifact resource_scope mismatch",
        )
    if art_scope.resource_scope != request.resource_scope:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.SCOPE_WIDENING,
            detail="artifact scope resource_scope mismatch",
        )
    if artifact.candidate_id != req_scope.candidate_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="candidate_id mismatch",
        )
    if art_scope.candidate_id != req_scope.candidate_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="artifact scope candidate_id mismatch",
        )
    if artifact.candidate_id != art_scope.candidate_id:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
            detail="artifact candidate_id scope mismatch",
        )
    if artifact.candidate_revision != req_scope.candidate_revision:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.REVISION_MISMATCH,
            detail="artifact candidate_revision mismatch",
        )
    if art_scope.candidate_revision != req_scope.candidate_revision:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.REVISION_MISMATCH,
            detail="artifact scope candidate_revision mismatch",
        )
    if not _operations_subset(art_scope.permitted_operations, req_scope.permitted_operations):
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.OPERATION_WIDENING,
        )
    if not _network_subset(art_scope.network_allowlist, req_scope.network_allowlist):
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.NETWORK_WIDENING,
        )
    if art_scope.credential_grant_ref != req_scope.credential_grant_ref:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.SCOPE_WIDENING,
            detail="credential grant ref mismatch",
        )
    if art_scope.expires_at > req_scope.expires_at:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.EXPIRY_WIDENING,
        )
    expected_scope_fp = derive_scoped_integration_adaptation_scope_fingerprint(art_scope)
    if artifact.scope_fingerprint != expected_scope_fp:
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.FINGERPRINT_MISMATCH,
            detail="artifact scope_fingerprint mismatch",
        )
    if artifact.artifact_fingerprint != artifact.derived_artifact_fingerprint():
        raise ScopedIntegrationAdaptationError(
            ScopedIntegrationAdaptationFailureReason.FINGERPRINT_MISMATCH,
            detail="artifact fingerprint mismatch",
        )


def _operations_subset(
    artifact_ops: tuple[ScopedIntegrationAdaptationOperationId, ...],
    request_ops: tuple[ScopedIntegrationAdaptationOperationId, ...],
) -> bool:
    request_set = frozenset(request_ops)
    return all(op in request_set for op in artifact_ops)


def _network_subset(
    artifact_allowlist: NetworkEgressAllowlist,
    request_allowlist: NetworkEgressAllowlist,
) -> bool:
    request_hosts = {host.canonical_form() for host in request_allowlist.hosts}
    return all(host.canonical_form() in request_hosts for host in artifact_allowlist.hosts)


__all__ = [
    "ScopedIntegrationAdaptationPortAdapter",
    "ScopedIntegrationAdaptationService",
]
