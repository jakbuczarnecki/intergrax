# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLM provider external operation lifecycle (W4-C)."""

from __future__ import annotations

from intergrax.contracts.dependency_concurrency_admission import DependencyConcurrencyKind
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationCancellationPort,
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
    ExternalOperationStatusPort,
)
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    ExternalOperationTerminationPort,
    TerminationResult,
)
from intergrax.contracts.external_operation_identity import (
    ExternalOperationIdentity,
    mint_stable_operation_id,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
    assert_may_begin_physical_attempt,
    finalize_operation_intent,
    mark_physical_running,
    mark_physical_terminal,
    request_operation_cancellation,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.external_operations.operation_termination import (
    dispatch_provider_termination_sync,
    mark_observed_cancellation_terminal,
)


def llm_external_operation_identity(
    *,
    provider_slug: str,
    model: str,
    call_scope: str,
    physical_attempt_sequence: int = 1,
) -> ExternalOperationIdentity:
    execution_id = require_active_execution_id()
    _run_id, attempt_id = require_active_execution_identity()
    operation_id = mint_stable_operation_id(
        execution_id=execution_id,
        dependency_kind=DependencyConcurrencyKind.LLM_PROVIDER,
        dependency_identity=provider_slug,
        logical_scope=f"{model}:{call_scope}",
    )
    return ExternalOperationIdentity(
        execution_id=execution_id,
        attempt_id=attempt_id,
        dependency_kind=DependencyConcurrencyKind.LLM_PROVIDER,
        dependency_identity=provider_slug,
        operation_id=operation_id,
        physical_attempt_sequence=physical_attempt_sequence,
    )


class LlmExternalOperationAttempt:
    __slots__ = (
        "_store",
        "_owner",
        "_identity",
        "_cancellation_port",
        "_status_port",
        "_termination_port",
        "_capabilities",
        "_revision",
    )

    def __init__(
        self,
        *,
        store: ExternalOperationStateStore | None,
        owner: ProcessLocalExternalOperationOwner | None,
        identity: ExternalOperationIdentity | None,
        cancellation_port: ExternalOperationCancellationPort | None = None,
        status_port: ExternalOperationStatusPort | None = None,
        termination_port: ExternalOperationTerminationPort | None = None,
        capabilities: ExternalOperationCapabilities | None = None,
    ) -> None:
        self._store = store
        self._owner = owner
        self._identity = identity
        self._cancellation_port = cancellation_port
        self._status_port = status_port
        self._termination_port = termination_port
        self._capabilities = capabilities
        self._revision: int | None = None

    @property
    def enabled(self) -> bool:
        return (
            self._store is not None
            and self._identity is not None
            and self._owner is not None
        )

    @property
    def operation_id(self) -> str | None:
        if self._identity is None:
            return None
        return self._identity.operation_id

    def _binding(
        self,
    ) -> tuple[
        ExternalOperationStateStore,
        ExternalOperationIdentity,
        ProcessLocalExternalOperationOwner,
    ] | None:
        if self._store is None or self._identity is None or self._owner is None:
            return None
        return self._store, self._identity, self._owner

    def before_physical_call(self) -> None:
        binding = self._binding()
        if binding is None:
            return
        store, identity, _owner = binding
        record = assert_may_begin_physical_attempt(store, identity)
        self._revision = record.revision

    def mark_running(self) -> None:
        binding = self._binding()
        if binding is None or self._revision is None:
            return
        store, identity, owner = binding
        record = mark_physical_running(
            store,
            operation_id=identity.operation_id,
            expected_revision=self._revision,
            owner=owner,
        )
        self._revision = record.revision

    def request_cancel(self) -> None:
        binding = self._binding()
        if binding is None:
            return
        store, identity, _owner = binding
        updated = request_operation_cancellation(
            store,
            operation_id=identity.operation_id,
            cancellation_port=self._cancellation_port,
        )
        self._revision = updated.revision

    def _terminal(
        self,
        physical: ExternalOperationPhysicalState,
        *,
        finalize_intent: bool,
        intent_state: ExternalOperationIntentState | None = None,
    ) -> None:
        binding = self._binding()
        if binding is None or self._revision is None:
            return
        store, identity, _owner = binding
        record = mark_physical_terminal(
            store,
            operation_id=identity.operation_id,
            expected_revision=self._revision,
            physical_state=physical,
            intent_state=intent_state,
        )
        self._revision = record.revision
        if finalize_intent:
            finalized = finalize_operation_intent(
                store,
                operation_id=identity.operation_id,
                expected_revision=self._revision,
            )
            self._revision = finalized.revision

    def mark_succeeded(self) -> None:
        self._terminal(ExternalOperationPhysicalState.SUCCEEDED, finalize_intent=True)

    def mark_failed(self) -> None:
        self._terminal(ExternalOperationPhysicalState.FAILED, finalize_intent=False)

    def mark_cancelled(self) -> None:
        self._terminal(ExternalOperationPhysicalState.CANCELLED, finalize_intent=True)

    def complete_cancellation_after_termination(
        self,
        termination_result: TerminationResult | None = None,
    ) -> None:
        """CAS physical terminal from observed provider termination (W4-D)."""
        binding = self._binding()
        if binding is None or self._revision is None or self._identity is None:
            return
        store, identity, _owner = binding
        if termination_result is None:
            caps = self._capabilities
            if caps is None:
                caps = ExternalOperationCapabilities(
                    supports_native_cancel=False,
                    supports_stream_abort=False,
                    supports_remote_termination=False,
                )
            termination_result = dispatch_provider_termination_sync(
                identity,
                termination_port=self._termination_port,
                capabilities=caps,
            )
        record = mark_observed_cancellation_terminal(
            store,
            operation_id=identity.operation_id,
            expected_revision=self._revision,
            termination_result=termination_result,
        )
        self._revision = record.revision

    def mark_unknown(self) -> None:
        self._terminal(
            ExternalOperationPhysicalState.UNKNOWN,
            finalize_intent=False,
            intent_state=ExternalOperationIntentState.TERMINATING,
        )
