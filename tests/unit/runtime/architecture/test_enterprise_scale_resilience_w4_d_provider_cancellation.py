# © Artur Czarnecki. All rights reserved.

"""W4-D — provider native cancellation qualification matrix."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from intergrax.contracts.dependency_concurrency_admission import DependencyConcurrencyKind
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.external_operation_identity import (
    ExternalOperationIdentity,
    mint_stable_operation_id,
)
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    TerminationResult,
)
from intergrax.llm_adapters._shared.provider_external_operation_capabilities import (
    external_operation_capabilities_for_provider,
)
from intergrax.llm_adapters._shared.provider_stream_admission import (
    stream_with_external_operation_lifecycle,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters.providers.openai.cancellation import openai_external_operation_ports
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationStartSuppressedError,
    ProcessLocalExternalOperationOwner,
    assert_may_begin_physical_attempt,
    mark_physical_running,
    request_operation_cancellation,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateUpdate,
    InMemoryExternalOperationStateStore,
    StaleExternalOperationStateError,
)
from intergrax.runtime.external_operations.llm_external_operation_attempt import (
    LlmExternalOperationAttempt,
)
from intergrax.runtime.external_operations.operation_termination import (
    cancellation_physical_state_from_termination,
    dispatch_provider_termination_sync,
    mark_observed_cancellation_terminal,
    request_cancel_dispatch_terminate_and_observe,
    timeout_physical_state,
)
from intergrax.runtime.nexus.tools.tool_operation_termination import (
    TOOL_EXTERNAL_OPERATION_CAPABILITIES,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _identity() -> ExternalOperationIdentity:
    execution_id = mint_execution_id()
    operation_id = mint_stable_operation_id(
        execution_id=execution_id,
        dependency_kind=DependencyConcurrencyKind.LLM_PROVIDER,
        dependency_identity="openai",
        logical_scope="gpt-test:sync",
    )
    return ExternalOperationIdentity(
        execution_id=execution_id,
        attempt_id=mint_attempt_id(),
        dependency_kind=DependencyConcurrencyKind.LLM_PROVIDER,
        dependency_identity="openai",
        operation_id=operation_id,
        physical_attempt_sequence=1,
    )


class _ConfirmedTerminationPort:
    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        return TerminationResult.physical_stop_confirmed()


def test_native_cancel_success_cas_cancelled() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    cancel, termination, _registry = openai_external_operation_ports()
    ext_op = LlmExternalOperationAttempt(
        store=store,
        owner=owner,
        identity=identity,
        cancellation_port=cancel,
        termination_port=_ConfirmedTerminationPort(),
        capabilities=external_operation_capabilities_for_provider("openai"),
    )
    store.create_if_absent(identity.operation_id, owner_token=owner.token)
    ext_op.before_physical_call()
    ext_op.mark_running()
    ext_op.request_cancel()
    ext_op.complete_cancellation_after_termination(
        TerminationResult.physical_stop_confirmed(),
    )
    loaded = store.load(identity.operation_id)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.CANCELLED
    assert loaded.intent_state is ExternalOperationIntentState.TERMINATED


def test_provider_without_cancel_capability_marks_unknown() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    caps = external_operation_capabilities_for_provider("ollama")
    assert caps.supports_native_cancel is False
    ext_op = LlmExternalOperationAttempt(
        store=store,
        owner=owner,
        identity=identity,
        capabilities=caps,
    )
    store.create_if_absent(identity.operation_id, owner_token=owner.token)
    ext_op.before_physical_call()
    ext_op.mark_running()
    ext_op.request_cancel()
    ext_op.complete_cancellation_after_termination()
    loaded = store.load(identity.operation_id)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.UNKNOWN


def test_cancel_race_twenty_workers_one_terminal_winner() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token=owner.token)
    running = mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner,
    )
    barrier = threading.Barrier(20)
    cas_success_count = 0
    stale_count = 0
    lock = threading.Lock()

    def writer(physical: ExternalOperationPhysicalState) -> None:
        nonlocal stale_count, cas_success_count
        barrier.wait()
        try:
            store.compare_and_set(
                identity.operation_id,
                expected_revision=running.revision,
                update=ExternalOperationStateUpdate(physical_state=physical),
            )
        except StaleExternalOperationStateError:
            with lock:
                stale_count += 1
            return
        with lock:
            cas_success_count += 1

    threads = [
        threading.Thread(
            target=writer,
            args=(
                ExternalOperationPhysicalState.CANCELLED
                if i % 2 == 0
                else ExternalOperationPhysicalState.SUCCEEDED,
            ),
        )
        for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert cas_success_count == 1
    assert stale_count == 19


class _CloseableStream:
    def __init__(self) -> None:
        self.closed = False
        self.tokens = iter(("a", "b", "c"))

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        return next(self.tokens)

    def close(self) -> None:
        self.closed = True


def test_stream_cancellation_closes_transport() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    cancel, termination, registry = openai_external_operation_ports()
    ext_op = LlmExternalOperationAttempt(
        store=store,
        owner=owner,
        identity=identity,
        cancellation_port=cancel,
        termination_port=termination,
        capabilities=external_operation_capabilities_for_provider("openai"),
    )
    stream = _CloseableStream()

    def factory() -> Iterator[str]:
        return stream

    wrapped = stream_with_external_operation_lifecycle(
        ext_op=ext_op,
        factory=factory,
        stream_registry=registry,
    )
    ext_op.before_physical_call()
    ext_op.mark_running()
    consumed = []
    for index, token in enumerate(wrapped):
        consumed.append(token)
        if index == 0:
            dispatch_provider_termination_sync(
                identity,
                termination_port=termination,
                capabilities=external_operation_capabilities_for_provider("openai"),
            )
            break
    assert stream.closed is True


def test_retry_blocked_after_cancelled_operation() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token=owner.token)
    running = mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner,
    )
    cancelled = request_operation_cancellation(store, operation_id=identity.operation_id)
    mark_observed_cancellation_terminal(
        store,
        operation_id=identity.operation_id,
        expected_revision=cancelled.revision,
        termination_result=TerminationResult.physical_stop_confirmed(),
    )
    retry_identity = ExternalOperationIdentity(
        execution_id=identity.execution_id,
        attempt_id=mint_attempt_id(),
        dependency_kind=identity.dependency_kind,
        dependency_identity=identity.dependency_identity,
        operation_id=identity.operation_id,
        physical_attempt_sequence=2,
    )
    with pytest.raises(ExternalOperationStartSuppressedError):
        assert_may_begin_physical_attempt(store, retry_identity)


def test_timeout_is_not_cancellation_physical_state() -> None:
    assert timeout_physical_state(exceeded_deadline=True) is (
        ExternalOperationPhysicalState.UNKNOWN
    )
    assert timeout_physical_state(exceeded_deadline=False) is (
        ExternalOperationPhysicalState.FAILED
    )
    assert (
        cancellation_physical_state_from_termination(TerminationResult.not_supported())
        is ExternalOperationPhysicalState.UNKNOWN
    )
    assert (
        cancellation_physical_state_from_termination(
            TerminationResult.physical_stop_confirmed(),
        )
        is ExternalOperationPhysicalState.CANCELLED
    )


def test_tool_capabilities_declare_no_native_cancel() -> None:
    assert TOOL_EXTERNAL_OPERATION_CAPABILITIES.supports_native_cancel is False


def test_dispatch_observe_pipeline() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token=owner.token)
    mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner,
    )
    terminal = request_cancel_dispatch_terminate_and_observe(
        store,
        identity,
        cancellation_port=None,
        termination_port=_ConfirmedTerminationPort(),
        capabilities=ExternalOperationCapabilities(
            supports_native_cancel=True,
            supports_stream_abort=True,
            supports_remote_termination=True,
        ),
    )
    assert terminal.physical_state is ExternalOperationPhysicalState.CANCELLED
