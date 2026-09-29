# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference execution-bound scoped adaptive integration (AW-7C-P4 qualification)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionOutput,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.contracts.credential import (
    CredentialUseGrant,
    CredentialUseScope,
)
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationOperationEvidence,
    ScopedAdaptedIntegrationOperationPort,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.scoped_adaptive_integration_sandbox_validation import (
    ScopedAdaptiveIntegrationSandboxSecurityError,
    validate_qualified_allowlist_attestation,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities


class ReferenceScopedAdaptedIntegrationOperation:
    """Deterministic adapted operation — injected only for qualification."""

    def execute(
        self,
        *,
        artifact: object,
        execution_id: str,
        tenant_id: str,
    ) -> ScopedAdaptedIntegrationOperationEvidence:
        from intergrax.integrations.contracts.scoped_integration_adaptation import (
            ScopedIntegrationAdaptationArtifact,
        )

        if type(artifact) is not ScopedIntegrationAdaptationArtifact:
            raise TypeError("artifact must be ScopedIntegrationAdaptationArtifact")
        return ScopedAdaptedIntegrationOperationEvidence(
            evidence_ref=f"ref-op:{artifact.artifact_id}:{execution_id}",
            tenant_id=tenant_id,
            execution_id=execution_id,
            artifact_id=artifact.artifact_id,
        )


class ScopedAdaptiveIntegrationCredentialDenied(Exception):
    """Credential resolution rejected before adapted operation."""


def execute_reference_scoped_adaptive_integration(
    *,
    handoff: ScopedAdaptiveIntegrationExecutionHandoff,
    execution_id: ExecutionId,
    tenant_id: str,
    sandbox_capabilities: SandboxSecurityCapabilities,
    credential_broker: ScopedCredentialBroker,
    credential_grant: CredentialUseGrant,
    operation_port: ScopedAdaptedIntegrationOperationPort,
) -> ScopedAdaptiveIntegrationExecutionOutput:
    if handoff.tenant_id != tenant_id:
        raise ValueError("handoff tenant mismatch")
    if handoff.artifact.tenant_id != tenant_id:
        raise ValueError("artifact tenant mismatch")
    try:
        validate_qualified_allowlist_attestation(
            qualified_allowlist=handoff.network_allowlist,
            capabilities=sandbox_capabilities,
        )
    except ScopedAdaptiveIntegrationSandboxSecurityError as exc:
        raise ScopedAdaptiveIntegrationCredentialDenied(str(exc)) from exc

    scope = CredentialUseScope(
        tenant_id=tenant_id,
        provider_id=handoff.provider_id,
        integration_id=handoff.resource_scope,
        operation=handoff.permitted_operations[0].value,
        execution_id=str(execution_id),
        target_scope=handoff.network_allowlist,
    )
    resolved = credential_broker.resolve_scoped(credential_grant, scope)
    del resolved
    evidence = operation_port.execute(
        artifact=handoff.artifact,
        execution_id=str(execution_id),
        tenant_id=tenant_id,
    )
    return ScopedAdaptiveIntegrationExecutionOutput(
        evidence_ref=evidence.evidence_ref,
        tenant_id=evidence.tenant_id,
    )


__all__ = [
    "ReferenceScopedAdaptedIntegrationOperation",
    "ScopedAdaptiveIntegrationCredentialDenied",
    "execute_reference_scoped_adaptive_integration",
]
