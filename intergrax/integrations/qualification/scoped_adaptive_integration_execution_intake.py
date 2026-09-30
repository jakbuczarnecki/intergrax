# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution composition for AW-7C reference / qualification path (CLOSURE)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
)
from intergrax.contracts.execution_intake import CanonicalExecutionIntakePort
from intergrax.contracts.execution_request import ExecutionRequest
from intergrax.integrations.contracts.credential import ExecutionBoundCredentialGrantProvider
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationOperationPort,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.qualification.reference_scoped_adaptive_integration_execution import (
    ReferenceScopedAdaptedIntegrationOperation,
)
from intergrax.integrations.qualification.scoped_adaptive_integration_execution_runtime_delegate import (
    ScopedAdaptiveIntegrationExecutionRuntimeDelegate,
)
from intergrax.runtime.execution.canonical_intake_adapter import (
    CanonicalExecutionRuntimeAdapter,
)
from intergrax.runtime.execution.runtime import ExecutionRuntime
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapable


def build_scoped_adaptive_integration_canonical_execution_intake(
    *,
    credential_broker: ScopedCredentialBroker,
    credential_grant_provider: ExecutionBoundCredentialGrantProvider,
    sandbox_security_source: SandboxSecurityCapable,
    operation_port: ScopedAdaptedIntegrationOperationPort | None = None,
) -> tuple[
    CanonicalExecutionRuntimeAdapter[
        ExecutionRequest[
            ScopedAdaptiveIntegrationExecutionHandoff,
            ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
        ],
        ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
    ],
    ScopedAdaptiveIntegrationExecutionRuntimeDelegate,
]:
    """Assemble ExecutionDelegate → ExecutionRuntime → CanonicalExecutionRuntimeAdapter."""
    delegate = ScopedAdaptiveIntegrationExecutionRuntimeDelegate(
        credential_broker=credential_broker,
        credential_grant_provider=credential_grant_provider,
        sandbox_security_source=sandbox_security_source,
        operation_port=operation_port or ReferenceScopedAdaptedIntegrationOperation(),
    )
    runtime = ExecutionRuntime(delegate)
    intake: CanonicalExecutionIntakePort[
        ExecutionRequest[
            ScopedAdaptiveIntegrationExecutionHandoff,
            ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
        ],
        ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
    ] = CanonicalExecutionRuntimeAdapter(runtime)
    return intake, delegate


__all__ = [
    "build_scoped_adaptive_integration_canonical_execution_intake",
]
