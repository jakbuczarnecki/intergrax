# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference execution-bound scoped adaptive integration (AW-7C-P4/CERT/CLOSURE-R1)."""

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
    ScopedCredentialResolutionResult,
)
from intergrax.integrations.contracts.scoped_adapted_integration_effect_execution import (
    ScopedAdaptedIntegrationEffectExecutionIngress,
    ScopedAdaptedIntegrationEffectExecutor,
)
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationEffectRequest,
    ScopedAdaptedIntegrationEffectRequestPort,
    ScopedAdaptedIntegrationOperationEvidence,
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationOperationId,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.scoped_adaptive_integration_effect_request_validation import (
    validate_admitted_scoped_adaptive_integration_effect_request,
)
from intergrax.integrations.scoped_adaptive_integration_sandbox_validation import (
    ScopedAdaptiveIntegrationSandboxSecurityError,
    validate_qualified_allowlist_attestation,
)
from intergrax.runtime.sandbox.contracts import (
    SandboxSecurityCapable,
    SandboxSecurityCapabilities,
)


class ReferenceScopedAdaptedIntegrationEffectRequestPreparer:
    """Deterministic effect preparer — no physical I/O, no credential material."""

    def __init__(self) -> None:
        self.last_requested_operation: ScopedIntegrationAdaptationOperationId | None = None
        self.last_effect_request: ScopedAdaptedIntegrationEffectRequest | None = None

    def prepare(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        requested_operation: ScopedIntegrationAdaptationOperationId,
        execution_id: str,
        tenant_id: str,
        admitted_network_allowlist: NetworkEgressAllowlist,
    ) -> ScopedAdaptedIntegrationEffectRequest:
        self.last_requested_operation = requested_operation
        effect_request = ScopedAdaptedIntegrationEffectRequest(
            artifact_id=artifact.artifact_id,
            artifact_fingerprint=artifact.artifact_fingerprint,
            tenant_id=tenant_id,
            integration_category=artifact.integration_category,
            provider_id=artifact.provider_id,
            resource_scope=artifact.resource_scope,
            requested_operation=requested_operation,
            network_allowlist=admitted_network_allowlist,
            target_scope=admitted_network_allowlist,
            specification=artifact.specification,
            execution_id=execution_id,
        )
        self.last_effect_request = effect_request
        return effect_request


@dataclass(frozen=True, slots=True)
class ReferenceScopedAdaptedIntegrationEffectExecutionContext:
    """Concrete executor ingress — binds attested sandbox and broker resolution."""

    effect_request: ScopedAdaptedIntegrationEffectRequest
    bound_execution_id: ExecutionId
    sandbox_security_source: SandboxSecurityCapable
    credential_resolution: ScopedCredentialResolutionResult

    @property
    def execution_id(self) -> str:
        return str(self.bound_execution_id)

    @property
    def sandbox_resource(self) -> SandboxSecurityCapable:
        return self.sandbox_security_source

    @property
    def sandbox_session_id(self) -> int:
        return id(self.sandbox_security_source)

    @property
    def credential_use_evidence_grant_id(self) -> str:
        return self.credential_resolution.use_evidence.grant_id

    @property
    def credential_use_evidence_fingerprint(self) -> str:
        return self.credential_resolution.use_evidence.credential_fingerprint


class ReferenceScopedAdaptedIntegrationEffectExecutor:
    """Canonical reference physical effect — consumes typed executor ingress only."""

    def __init__(self) -> None:
        self.last_ingress: ScopedAdaptedIntegrationEffectExecutionIngress | None = None
        self.call_count = 0

    def execute(
        self,
        ingress: ScopedAdaptedIntegrationEffectExecutionIngress,
    ) -> ScopedAdaptedIntegrationOperationEvidence:
        self.call_count += 1
        self.last_ingress = ingress
        effect_request = ingress.effect_request
        if ingress.execution_id != effect_request.execution_id:
            raise ValueError("execution identity mismatch")
        if not isinstance(ingress.sandbox_resource, SandboxSecurityCapable):
            raise ValueError("sandbox security source missing")
        ingress.sandbox_resource.security_capabilities()
        evidence = ingress.credential_resolution.use_evidence
        if evidence.tenant_id != effect_request.tenant_id:
            raise ValueError("credential evidence tenant mismatch")
        if evidence.execution_id != effect_request.execution_id:
            raise ValueError("credential evidence execution_id mismatch")
        if evidence.operation != effect_request.requested_operation.value:
            raise ValueError("credential evidence operation mismatch")
        if evidence.integration_id != effect_request.resource_scope:
            raise ValueError("credential evidence integration mismatch")
        if evidence.provider_id != effect_request.provider_id:
            raise ValueError("credential evidence provider mismatch")
        _ = ingress.credential_resolution.resolved_credential.value
        return ScopedAdaptedIntegrationOperationEvidence(
            evidence_ref=(
                f"ref-effect:{effect_request.artifact_id}:"
                f"{effect_request.execution_id}:"
                f"{effect_request.requested_operation.value}:"
                f"{ingress.credential_use_evidence_fingerprint}:"
                f"sbx={ingress.sandbox_session_id}"
            ),
            tenant_id=effect_request.tenant_id,
            execution_id=effect_request.execution_id,
            artifact_id=effect_request.artifact_id,
            executed_operation=effect_request.requested_operation,
        )


@dataclass(frozen=True, slots=True)
class ReferenceScopedAdaptiveIntegrationSandboxSession:
    """Reference sandbox session — attests substrate truth via SandboxSecurityCapable."""

    capabilities: SandboxSecurityCapabilities

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return self.capabilities


@dataclass(frozen=True, slots=True)
class ReferenceExecutionBoundCredentialGrantProvider:
    """Reference grant resolver — provider-owned authoritative facts only."""

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
        if credential_grant_ref != self.grant_id:
            raise CredentialScopeMismatchError("credential grant ref mismatch")
        if tenant_id != self.tenant_id:
            raise CredentialScopeMismatchError("credential grant tenant mismatch")
        if provider_id != self.provider_id:
            raise CredentialScopeMismatchError("credential grant provider mismatch")
        if integration_id != self.integration_id:
            raise CredentialScopeMismatchError("credential grant integration mismatch")
        return CredentialUseGrant(
            grant_id=self.grant_id,
            credential_ref=self.credential_ref,
            tenant_id=self.tenant_id,
            provider_id=self.provider_id,
            integration_id=self.integration_id,
            operation=requested_operation.value,
            execution_id=str(execution_id),
            target_scope=self.target_scope,
            expires_at=self.expires_at,
        )


def _validate_grant_against_effect_request(
    *,
    grant: CredentialUseGrant,
    effect_request: ScopedAdaptedIntegrationEffectRequest,
    execution_id: ExecutionId,
) -> str | None:
    if grant.tenant_id != effect_request.tenant_id:
        return "grant tenant mismatch"
    if grant.provider_id != effect_request.provider_id:
        return "grant provider mismatch"
    if grant.integration_id != effect_request.resource_scope:
        return "grant integration mismatch"
    if grant.operation != effect_request.requested_operation.value:
        return "credential grant operation mismatch"
    if grant.execution_id != str(execution_id):
        return "grant execution_id mismatch"
    if grant.execution_id != effect_request.execution_id:
        return "grant effect execution_id mismatch"
    return None


def execute_reference_scoped_adaptive_integration(
    *,
    handoff: ScopedAdaptiveIntegrationExecutionHandoff,
    execution_id: ExecutionId,
    tenant_id: str,
    sandbox_security_source: SandboxSecurityCapable,
    credential_broker: ScopedCredentialBroker,
    credential_grant_provider: ExecutionBoundCredentialGrantProvider,
    effect_preparer: ScopedAdaptedIntegrationEffectRequestPort,
    effect_executor: ScopedAdaptedIntegrationEffectExecutor,
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

    try:
        effect_request = effect_preparer.prepare(
            artifact=handoff.artifact,
            requested_operation=handoff.requested_operation,
            execution_id=str(execution_id),
            tenant_id=tenant_id,
            admitted_network_allowlist=handoff.network_allowlist,
        )
    except Exception as exc:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail=f"effect preparer failed: {exc}",
        )

    effect_validation = validate_admitted_scoped_adaptive_integration_effect_request(
        effect_request=effect_request,
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=tenant_id,
        admitted_network_allowlist=handoff.network_allowlist,
    )
    if effect_validation is not None:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail=effect_validation,
        )

    try:
        credential_grant = credential_grant_provider.resolve_grant(
            execution_id=execution_id,
            credential_grant_ref=handoff.artifact.scope.credential_grant_ref,
            tenant_id=tenant_id,
            provider_id=handoff.provider_id,
            integration_id=handoff.resource_scope,
            requested_operation=effect_request.requested_operation,
        )
    except CredentialScopeMismatchError as exc:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED,
            error_detail=str(exc),
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
    grant_validation = _validate_grant_against_effect_request(
        grant=credential_grant,
        effect_request=effect_request,
        execution_id=execution_id,
    )
    if grant_validation is not None:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED,
            error_detail=grant_validation,
        )

    scope = CredentialUseScope(
        tenant_id=effect_request.tenant_id,
        provider_id=effect_request.provider_id,
        integration_id=effect_request.resource_scope,
        operation=effect_request.requested_operation.value,
        execution_id=effect_request.execution_id,
        target_scope=effect_request.target_scope,
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

    ingress = ReferenceScopedAdaptedIntegrationEffectExecutionContext(
        effect_request=effect_request,
        bound_execution_id=execution_id,
        sandbox_security_source=sandbox_security_source,
        credential_resolution=resolved,
    )
    try:
        evidence = effect_executor.execute(ingress)
    except Exception as exc:
        return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
            outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
            error_detail=f"effect executor failed: {exc}",
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
    "ReferenceScopedAdaptedIntegrationEffectExecutionContext",
    "ReferenceScopedAdaptedIntegrationEffectExecutor",
    "ReferenceScopedAdaptedIntegrationEffectRequestPreparer",
    "ReferenceScopedAdaptiveIntegrationSandboxSession",
    "execute_reference_scoped_adaptive_integration",
]
