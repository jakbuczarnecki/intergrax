# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference execution-bound scoped adaptive integration (AW-7C-P4/CERT/CLOSURE)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionOutcome,
    ScopedAdaptiveIntegrationExecutionOutput,
    ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
    validate_execution_bound_qualification_proof,
    validate_handoff_credential_grant_identity,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialScopeAdmissionDeniedError,
    CredentialScopeMismatchError,
    CredentialUseGrant,
    CredentialUseGrantExpiredError,
    CredentialUseScope,
    ExecutionBoundCredentialGrantProvider,
)
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationOperationEvidence,
    ScopedAdaptedIntegrationOperationPort,
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationOperationId,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.scoped_adaptive_integration_sandbox_validation import (
    ScopedAdaptiveIntegrationSandboxSecurityError,
    validate_qualified_allowlist_attestation,
)
from intergrax.runtime.sandbox.contracts import (
    SandboxSecurityCapable,
    SandboxSecurityCapabilities,
)


class ReferenceScopedAdaptedIntegrationOperation:
    """Deterministic adapted operation — injected only for qualification."""

    def __init__(self) -> None:
        self.last_operation: ScopedIntegrationAdaptationOperationId | None = None

    def execute(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        requested_operation: ScopedIntegrationAdaptationOperationId,
        execution_id: str,
        tenant_id: str,
    ) -> ScopedAdaptedIntegrationOperationEvidence:
        self.last_operation = requested_operation
        return ScopedAdaptedIntegrationOperationEvidence(
            evidence_ref=(
                f"ref-op:{artifact.artifact_id}:{execution_id}:{requested_operation.value}"
            ),
            tenant_id=tenant_id,
            execution_id=execution_id,
            artifact_id=artifact.artifact_id,
            executed_operation=requested_operation,
        )


@dataclass(frozen=True, slots=True)
class ReferenceScopedAdaptiveIntegrationSandboxSession:
    """Reference sandbox session — attests substrate truth via SandboxSecurityCapable."""

    capabilities: SandboxSecurityCapabilities

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return self.capabilities


@dataclass(frozen=True, slots=True)
class ReferenceExecutionBoundCredentialGrantProvider:
    """Reference grant resolver — binds grant_id and operation to active execution."""

    grant_id: str
    credential_ref: CredentialRef
    tenant_id: str
    provider_id: str
    integration_id: str
    target_scope: NetworkEgressAllowlist
    expires_at: datetime

    def resolve_grant(
        self,
        *,
        execution_id: ExecutionId,
        credential_grant_ref: str,
        tenant_id: str,
        provider_id: str,
        integration_id: str,
        requested_operation: ScopedIntegrationAdaptationOperationId,
    ) -> CredentialUseGrant:
        return CredentialUseGrant(
            grant_id=credential_grant_ref,
            credential_ref=self.credential_ref,
            tenant_id=tenant_id,
            provider_id=provider_id,
            integration_id=integration_id,
            operation=requested_operation.value,
            execution_id=str(execution_id),
            target_scope=self.target_scope,
            expires_at=self.expires_at,
        )


def execute_reference_scoped_adaptive_integration(
    *,
    handoff: ScopedAdaptiveIntegrationExecutionHandoff,
    execution_id: ExecutionId,
    tenant_id: str,
    sandbox_security_source: SandboxSecurityCapable,
    credential_broker: ScopedCredentialBroker,
    credential_grant_provider: ExecutionBoundCredentialGrantProvider,
    operation_port: ScopedAdaptedIntegrationOperationPort,
) -> ScopedAdaptiveIntegrationExecutionRuntimeEnvelope:
    proof_error = validate_execution_bound_qualification_proof(handoff)
    if proof_error is not None:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED,
            error_detail=proof_error,
        )
    if handoff.tenant_id != tenant_id:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail="handoff tenant mismatch",
        )
    if handoff.artifact.tenant_id != tenant_id:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail="artifact tenant mismatch",
        )
    if not isinstance(sandbox_security_source, SandboxSecurityCapable):
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.SANDBOX_SECURITY_UNSATISFIED,
            error_detail="sandbox security source missing",
        )
    sandbox_capabilities = sandbox_security_source.security_capabilities()
    try:
        validate_qualified_allowlist_attestation(
            qualified_allowlist=handoff.network_allowlist,
            capabilities=sandbox_capabilities,
        )
    except ScopedAdaptiveIntegrationSandboxSecurityError as exc:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.SANDBOX_SECURITY_UNSATISFIED,
            error_detail=str(exc),
        )

    credential_grant = credential_grant_provider.resolve_grant(
        execution_id=execution_id,
        credential_grant_ref=handoff.artifact.scope.credential_grant_ref,
        tenant_id=tenant_id,
        provider_id=handoff.provider_id,
        integration_id=handoff.resource_scope,
        requested_operation=handoff.requested_operation,
    )
    grant_binding = validate_handoff_credential_grant_identity(
        handoff=handoff,
        grant_grant_id=credential_grant.grant_id,
    )
    if grant_binding is not None:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED,
            error_detail=grant_binding,
        )
    if credential_grant.operation != handoff.requested_operation.value:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED,
            error_detail="credential grant operation mismatch",
        )

    scope = CredentialUseScope(
        tenant_id=tenant_id,
        provider_id=handoff.provider_id,
        integration_id=handoff.resource_scope,
        operation=handoff.requested_operation.value,
        execution_id=str(execution_id),
        target_scope=handoff.network_allowlist,
    )
    try:
        resolved = credential_broker.resolve_scoped(credential_grant, scope)
    except (
        CredentialScopeAdmissionDeniedError,
        CredentialScopeMismatchError,
        CredentialUseGrantExpiredError,
    ) as exc:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED,
            error_detail=str(exc),
        )
    del resolved
    evidence = operation_port.execute(
        artifact=handoff.artifact,
        requested_operation=handoff.requested_operation,
        execution_id=str(execution_id),
        tenant_id=tenant_id,
    )
    if evidence.tenant_id != tenant_id:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail="operation evidence tenant mismatch",
        )
    if evidence.executed_operation != handoff.requested_operation:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail="operation evidence operation mismatch",
        )
    if evidence.execution_id != str(execution_id):
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail="operation evidence execution_id mismatch",
        )
    return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
        outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED,
        output=ScopedAdaptiveIntegrationExecutionOutput(
            evidence_ref=evidence.evidence_ref,
            tenant_id=evidence.tenant_id,
        ),
    )


__all__ = [
    "ReferenceExecutionBoundCredentialGrantProvider",
    "ReferenceScopedAdaptedIntegrationOperation",
    "ReferenceScopedAdaptiveIntegrationSandboxSession",
    "execute_reference_scoped_adaptive_integration",
]
