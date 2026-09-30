# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference scoped integration adaptation strategy — qualification proof only (AW-7C-P3)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta

from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationOperationId,
    ScopedIntegrationAdaptationRequest,
    ScopedIntegrationAdaptationScope,
    ScopedIntegrationAdaptationTarget,
    build_scoped_integration_adaptation_artifact,
    scoped_integration_adaptation_operation_id,
)

REFERENCE_SCOPED_INTEGRATION_ADAPTATION_STRATEGY_ID = "aw-7c-reference-adaptation.v1"
REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID = "aw-7c-reference-provider"
REFERENCE_SPECIFICATION_TYPE = "aw-7c.reference_scoped_adaptation_spec.v1"
REFERENCE_SPECIFICATION_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ReferenceScopedIntegrationAdaptationSpecification:
    adaptation_intent: str
    normalized_resource_scope: str
    operation_subset: tuple[str, ...]

    @property
    def specification_type(self) -> str:
        return REFERENCE_SPECIFICATION_TYPE

    @property
    def specification_version(self) -> str:
        return REFERENCE_SPECIFICATION_VERSION

    @property
    def specification_fingerprint(self) -> str:
        parts = [
            f"type={self.specification_type}",
            f"version={self.specification_version}",
            f"intent={self.adaptation_intent}",
            f"resource_scope={self.normalized_resource_scope}",
            f"operations={','.join(self.operation_subset)}",
        ]
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


class ReferenceScopedIntegrationAdaptationStrategy:
    """Deterministic local reference strategy — active only when explicitly injected."""

    @property
    def strategy_id(self) -> str:
        return REFERENCE_SCOPED_INTEGRATION_ADAPTATION_STRATEGY_ID

    def supports(
        self,
        request: ScopedIntegrationAdaptationRequest,
        target: ScopedIntegrationAdaptationTarget,
    ) -> bool:
        return request.provider_id == REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID

    def adapt(
        self,
        request: ScopedIntegrationAdaptationRequest,
        target: ScopedIntegrationAdaptationTarget,
    ) -> ScopedIntegrationAdaptationArtifact:
        del target
        req_scope = request.scope
        narrowed_ops = _narrow_operations(req_scope.permitted_operations)
        narrowed_network = _narrow_network(req_scope.network_allowlist)
        narrowed_expiry = req_scope.expires_at - timedelta(minutes=5)
        narrowed_scope = ScopedIntegrationAdaptationScope(
            tenant_id=req_scope.tenant_id,
            integration_category=req_scope.integration_category,
            provider_id=req_scope.provider_id,
            resource_scope=req_scope.resource_scope,
            permitted_operations=narrowed_ops,
            network_allowlist=narrowed_network,
            credential_grant_ref=req_scope.credential_grant_ref,
            expires_at=narrowed_expiry,
            candidate_id=req_scope.candidate_id,
            candidate_revision=req_scope.candidate_revision,
        )
        specification = ReferenceScopedIntegrationAdaptationSpecification(
            adaptation_intent="reference-narrowing-proof",
            normalized_resource_scope=req_scope.resource_scope.strip().lower(),
            operation_subset=tuple(op.value for op in narrowed_ops),
        )
        artifact_id = _deterministic_artifact_id(
            request_id=request.request_id,
            strategy_id=self.strategy_id,
            scope=narrowed_scope,
            specification=specification,
        )
        return build_scoped_integration_adaptation_artifact(
            artifact_id=artifact_id,
            tenant_id=request.tenant_id,
            integration_category=request.integration_category,
            provider_id=request.provider_id,
            resource_scope=request.resource_scope,
            strategy_id=self.strategy_id,
            candidate_id=req_scope.candidate_id,
            candidate_revision=req_scope.candidate_revision,
            scope=narrowed_scope,
            specification=specification,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


def _narrow_operations(
    permitted: tuple[ScopedIntegrationAdaptationOperationId, ...],
) -> tuple[ScopedIntegrationAdaptationOperationId, ...]:
    if len(permitted) <= 1:
        return permitted
    return (permitted[0],)


def _narrow_network(allowlist: NetworkEgressAllowlist) -> NetworkEgressAllowlist:
    if not allowlist.hosts:
        return allowlist
    return NetworkEgressAllowlist(hosts=(allowlist.hosts[0],))


def _deterministic_artifact_id(
    *,
    request_id: str,
    strategy_id: str,
    scope: ScopedIntegrationAdaptationScope,
    specification: ReferenceScopedIntegrationAdaptationSpecification,
) -> str:
    parts = [
        f"request_id={request_id}",
        f"strategy_id={strategy_id}",
        f"candidate_id={scope.candidate_id}",
        f"candidate_revision={scope.candidate_revision}",
        f"spec_fp={specification.specification_fingerprint}",
    ]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"ref-art:{digest}"


def reference_read_operation() -> ScopedIntegrationAdaptationOperationId:
    return scoped_integration_adaptation_operation_id("READ_CONFIGURATION")


def reference_write_operation() -> ScopedIntegrationAdaptationOperationId:
    return scoped_integration_adaptation_operation_id("WRITE_CONFIGURATION")


__all__ = [
    "REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID",
    "REFERENCE_SCOPED_INTEGRATION_ADAPTATION_STRATEGY_ID",
    "ReferenceScopedIntegrationAdaptationSpecification",
    "ReferenceScopedIntegrationAdaptationStrategy",
    "reference_read_operation",
    "reference_write_operation",
]
