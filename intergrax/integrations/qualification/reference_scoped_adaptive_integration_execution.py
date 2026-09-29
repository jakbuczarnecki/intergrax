# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference execution-bound scoped adaptive integration (AW-7C-P4/CERT)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionOutcome,
    ScopedAdaptiveIntegrationExecutionOutput,
    ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
    validate_execution_bound_qualification_proof,
    validate_handoff_credential_grant_identity,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.contracts.credential import (
    CredentialScopeAdmissionDeniedError,
    CredentialScopeMismatchError,
    CredentialUseGrant,
    CredentialUseGrantExpiredError,
    CredentialUseScope,
)
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationOperationEvidence,
    ScopedAdaptedIntegrationOperationPort,
    ScopedIntegrationAdaptationArtifact,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.scoped_adaptive_integration_sandbox_validation import (
    ScopedAdaptiveIntegrationSandboxSecurityError,
    validate_qualified_allowlist_attestation,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities


class ReferenceScopedAdaptedIntegrationOperation:
    """Deterministic adapted operation — injected only for qualification."""

    def __init__(self) -> None:
        self.expected_operation: str = ""
        self.last_operation: str | None = None

    def execute(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        execution_id: str,
        tenant_id: str,
    ) -> ScopedAdaptedIntegrationOperationEvidence:
        operation_id = self.expected_operation
        self.last_operation = operation_id
        return ScopedAdaptedIntegrationOperationEvidence(
            evidence_ref=f"ref-op:{artifact.artifact_id}:{execution_id}:{operation_id}",
            tenant_id=tenant_id,
            execution_id=execution_id,
            artifact_id=artifact.artifact_id,
        )


def execute_reference_scoped_adaptive_integration(
    *,
    handoff: ScopedAdaptiveIntegrationExecutionHandoff,
    execution_id: ExecutionId,
    tenant_id: str,
    sandbox_capabilities: SandboxSecurityCapabilities,
    credential_broker: ScopedCredentialBroker,
    credential_grant: CredentialUseGrant,
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
    grant_binding = validate_handoff_credential_grant_identity(
        handoff=handoff,
        grant_grant_id=credential_grant.grant_id,
    )
    if grant_binding is not None:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED,
            error_detail=grant_binding,
        )
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
        execution_id=str(execution_id),
        tenant_id=tenant_id,
    )
    if evidence.tenant_id != tenant_id:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail="operation evidence tenant mismatch",
        )
    return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
        outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED,
        output=ScopedAdaptiveIntegrationExecutionOutput(
            evidence_ref=evidence.evidence_ref,
            tenant_id=evidence.tenant_id,
        ),
    )


__all__ = [
    "ReferenceScopedAdaptedIntegrationOperation",
    "execute_reference_scoped_adaptive_integration",
]
