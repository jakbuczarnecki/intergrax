# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ExecutionRuntime delegate for AW-7C scoped adaptive integration (CLOSURE)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
)
from intergrax.contracts.execution_identity import peek_active_execution_id
from intergrax.contracts.execution_request import ExecutionRequest
from intergrax.integrations.contracts.credential import ExecutionBoundCredentialGrantProvider
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationOperationPort,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.qualification.reference_scoped_adaptive_integration_execution import (
    execute_reference_scoped_adaptive_integration,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapable


class ScopedAdaptiveIntegrationExecutionRuntimeDelegate:
    """Runs AW-7C execution under canonical active ExecutionId — no identity minting."""

    def __init__(
        self,
        *,
        credential_broker: ScopedCredentialBroker,
        credential_grant_provider: ExecutionBoundCredentialGrantProvider,
        sandbox_security_source: SandboxSecurityCapable,
        operation_port: ScopedAdaptedIntegrationOperationPort,
    ) -> None:
        self._credential_broker = credential_broker
        self._credential_grant_provider = credential_grant_provider
        self._sandbox_security_source = sandbox_security_source
        self._operation_port = operation_port
        self.execute_calls = 0

    @property
    def operation_port(self) -> ScopedAdaptedIntegrationOperationPort:
        return self._operation_port

    async def execute(
        self,
        request: ExecutionRequest[
            ScopedAdaptiveIntegrationExecutionHandoff,
            ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
        ],
    ) -> ScopedAdaptiveIntegrationExecutionRuntimeEnvelope:
        self.execute_calls += 1
        handoff = request.input
        if type(handoff) is not ScopedAdaptiveIntegrationExecutionHandoff:
            raise TypeError("payload must carry ScopedAdaptiveIntegrationExecutionHandoff")
        execution_id = peek_active_execution_id()
        if execution_id is None:
            from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
                ScopedAdaptiveIntegrationExecutionOutcome,
            )

            return ScopedAdaptiveIntegrationExecutionRuntimeEnvelope(
                outcome=ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED,
                error_detail="active_execution_id_missing",
            )
        return execute_reference_scoped_adaptive_integration(
            handoff=handoff,
            execution_id=execution_id,
            tenant_id=handoff.tenant_id,
            sandbox_security_source=self._sandbox_security_source,
            credential_broker=self._credential_broker,
            credential_grant_provider=self._credential_grant_provider,
            operation_port=self._operation_port,
        )


__all__ = ["ScopedAdaptiveIntegrationExecutionRuntimeDelegate"]
