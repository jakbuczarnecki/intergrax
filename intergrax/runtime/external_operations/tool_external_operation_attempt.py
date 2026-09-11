# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool boundary external operation lifecycle (W4-C)."""

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
)
from intergrax.contracts.external_operation_identity import (
    ExternalOperationIdentity,
    mint_stable_operation_id,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationStartSuppressedError,
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


def tool_external_operation_identity(
    *,
    tool_id: str,
    step_id: str,
    physical_attempt_sequence: int,
) -> ExternalOperationIdentity:
    execution_id = require_active_execution_id()
    _run_id, attempt_id = require_active_execution_identity()
    operation_id = mint_stable_operation_id(
        execution_id=execution_id,
        dependency_kind=DependencyConcurrencyKind.TOOL,
        dependency_identity=tool_id,
        logical_scope=step_id,
    )
    return ExternalOperationIdentity(
        execution_id=execution_id,
        attempt_id=attempt_id,
        dependency_kind=DependencyConcurrencyKind.TOOL,
        dependency_identity=tool_id,
        operation_id=operation_id,
        physical_attempt_sequence=physical_attempt_sequence,
    )


class ToolExternalOperationAttempt:
    """Records terminal durable state before dependency permit release."""

    __slots__ = (
        "_store",
        "_owner",
        "_identity",
        "_cancellation_port",
        "_revision",
        "_terminal_recorded",
    )

    def __init__(
        self,
        *,
        store: ExternalOperationStateStore | None,
        owner: ProcessLocalExternalOperationOwner | None,
        identity: ExternalOperationIdentity | None,
        cancellation_port: ExternalOperationCancellationPort | None = None,
    ) -> None:
        self._store = store
        self._owner = owner
        self._identity = identity
        self._cancellation_port = cancellation_port
        self._revision: int | None = None
        self._terminal_recorded = False

    @property
    def enabled(self) -> bool:
        return self._store is not None and self._identity is not None and self._owner is not None

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

    def before_physical_submit(self) -> None:
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

    def mark_succeeded(self) -> None:
        self._mark_terminal(
            ExternalOperationPhysicalState.SUCCEEDED,
            finalize_intent=True,
        )

    def mark_failed(self) -> None:
        self._mark_terminal(
            ExternalOperationPhysicalState.FAILED,
            finalize_intent=False,
        )

    def mark_cancelled(self) -> None:
        self._mark_terminal(
            ExternalOperationPhysicalState.CANCELLED,
            finalize_intent=True,
        )

    def mark_unknown(self) -> None:
        self._mark_terminal(
            ExternalOperationPhysicalState.UNKNOWN,
            intent_state=ExternalOperationIntentState.TERMINATING,
            finalize_intent=False,
        )

    def request_cancel(self) -> None:
        binding = self._binding()
        if binding is None:
            return
        store, identity, _owner = binding
        request_operation_cancellation(
            store,
            operation_id=identity.operation_id,
            cancellation_port=self._cancellation_port,
        )

    def ensure_terminal_recorded(self) -> None:
        binding = self._binding()
        if binding is None or self._terminal_recorded or self._revision is None:
            return
        store, identity, _owner = binding
        record = mark_physical_terminal(
            store,
            operation_id=identity.operation_id,
            expected_revision=self._revision,
            physical_state=ExternalOperationPhysicalState.UNKNOWN,
        )
        self._revision = record.revision
        finalized = finalize_operation_intent(
            store,
            operation_id=identity.operation_id,
            expected_revision=self._revision,
        )
        self._revision = finalized.revision
        self._terminal_recorded = True

    def _mark_terminal(
        self,
        physical: ExternalOperationPhysicalState,
        *,
        finalize_intent: bool = False,
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
        self._terminal_recorded = True


__all__ = [
    "ExternalOperationStartSuppressedError",
    "ToolExternalOperationAttempt",
    "tool_external_operation_identity",
]
