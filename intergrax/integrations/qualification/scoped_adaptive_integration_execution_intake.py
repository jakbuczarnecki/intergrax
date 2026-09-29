# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution intake for scoped adaptive integration reference path (AW-7C-CERT)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.contracts.execution_request import ExecutionRequest
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.integrations.contracts.credential import CredentialUseGrant
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationOperationPort,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.qualification.reference_scoped_adaptive_integration_execution import (
    ReferenceScopedAdaptedIntegrationOperation,
    execute_reference_scoped_adaptive_integration,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities


class ScopedAdaptiveIntegrationReferenceExecutionIntake(
    CanonicalExecutionIntakePort[
        ScopedAdaptiveIntegrationExecutionHandoff,
        ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
    ],
):
    """Execution-bound A2 intake — validates CQ proof before credential/sandbox/operation."""

    def __init__(
        self,
        *,
        credential_broker: ScopedCredentialBroker,
        credential_grant_for_execution: Callable[[ExecutionId], CredentialUseGrant],
        sandbox_capabilities: SandboxSecurityCapabilities,
        operation_port: ScopedAdaptedIntegrationOperationPort | None = None,
    ) -> None:
        self._credential_broker = credential_broker
        self._credential_grant_for_execution = credential_grant_for_execution
        self._sandbox_capabilities = sandbox_capabilities
        self._operation_port = operation_port or ReferenceScopedAdaptedIntegrationOperation()

    @property
    def operation_port(self) -> ScopedAdaptedIntegrationOperationPort:
        return self._operation_port

    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[ScopedAdaptiveIntegrationExecutionHandoff],
    ) -> CanonicalExecutionIntakeResult[ScopedAdaptiveIntegrationExecutionRuntimeEnvelope]:
        payload = request.payload
        if type(payload) is ExecutionRequest:
            handoff = payload.input
        else:
            handoff = payload
        if type(handoff) is not ScopedAdaptiveIntegrationExecutionHandoff:
            raise TypeError("payload must carry ScopedAdaptiveIntegrationExecutionHandoff")
        run_id = request.run_id if request.run_id is not None else mint_run_id()
        attempt_id = request.attempt_id if request.attempt_id is not None else mint_attempt_id()
        execution_id = request.execution_id if request.execution_id is not None else mint_execution_id()
        validate_run_id(run_id)
        validate_attempt_id(attempt_id)
        validate_execution_id(execution_id)
        if type(self._operation_port) is ReferenceScopedAdaptedIntegrationOperation:
            self._operation_port.expected_operation = handoff.requested_operation.value
        credential_grant = self._credential_grant_for_execution(execution_id)
        envelope = execute_reference_scoped_adaptive_integration(
            handoff=handoff,
            execution_id=execution_id,
            tenant_id=request.tenant_id,
            sandbox_capabilities=self._sandbox_capabilities,
            credential_broker=self._credential_broker,
            credential_grant=credential_grant,
            operation_port=self._operation_port,
        )
        return CanonicalExecutionIntakeResult(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            result=envelope,
        )


__all__ = ["ScopedAdaptiveIntegrationReferenceExecutionIntake"]
