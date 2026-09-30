# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution composition for AW-7C reference / qualification path (CLOSURE-R1)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
)
from intergrax.contracts.execution_intake import CanonicalExecutionIntakePort
from intergrax.contracts.execution_request import ExecutionRequest
from intergrax.integrations.contracts.credential import ExecutionBoundCredentialGrantProvider
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationEffectRequestPort,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.qualification.reference_scoped_adaptive_integration_execution import (
    ReferenceScopedAdaptedIntegrationEffectExecutor,
    ReferenceScopedAdaptedIntegrationEffectRequestPreparer,
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
    effect_preparer: ScopedAdaptedIntegrationEffectRequestPort | None = None,
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
    """Assemble EffectPreparer + canonical executor → delegate → ExecutionRuntime → intake."""
    canonical_executor = ReferenceScopedAdaptedIntegrationEffectExecutor()
    delegate = ScopedAdaptiveIntegrationExecutionRuntimeDelegate(
        credential_broker=credential_broker,
        credential_grant_provider=credential_grant_provider,
        sandbox_security_source=sandbox_security_source,
        effect_preparer=effect_preparer or ReferenceScopedAdaptedIntegrationEffectRequestPreparer(),
        effect_executor=canonical_executor,
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
