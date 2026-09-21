# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Typed runtime lifecycle binding for framework LLM adapter implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationCancellationPort,
        ExternalOperationStatusPort,
    )
    from intergrax.contracts.external_operation_termination import (
        ExternalOperationCapabilities,
        ExternalOperationTerminationPort,
    )
    from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
        ProviderStreamTransportRegistry,
    )
    from intergrax.runtime.external_operations.admission.execution_gate import (
        ExternalOperationExecutionGate,
    )
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateStore,
    )
    from intergrax.runtime.external_operations.external_operation_ownership import (
        ProcessLocalExternalOperationOwner,
    )
    from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
        DependencyAttemptExecutionBoundary,
    )


class LLMRuntimeLifecycleBinding(Protocol):
    """Composition-root binding surface for provider external-operation integration."""

    def bind_external_operation_admission_gate(
        self,
        gate: ExternalOperationExecutionGate | None,
    ) -> None:
        ...

    def bind_external_operation_ports(
        self,
        *,
        store: ExternalOperationStateStore | None,
        owner: ProcessLocalExternalOperationOwner | None = None,
        cancellation_port: ExternalOperationCancellationPort | None = None,
        status_port: ExternalOperationStatusPort | None = None,
        termination_port: ExternalOperationTerminationPort | None = None,
        stream_registry: ProviderStreamTransportRegistry | None = None,
        capabilities: ExternalOperationCapabilities | None = None,
    ) -> None:
        ...

    def bind_provider_dependency_boundary(
        self,
        boundary: DependencyAttemptExecutionBoundary | None,
    ) -> None:
        ...
