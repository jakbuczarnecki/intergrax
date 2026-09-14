# © Artur Czarnecki. All rights reserved.

"""P2.1-S2B — capability-gated delegated provider cancel/interrupt control."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionCancelProvider,
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcome,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    DelegatedExecutionInterruptProvider,
    delegated_control_outcome,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionCapabilities,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    DelegatedExecutionTransportError,
    delegated_success_outcome,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.execution.delegated_execution.control_service import (
    DelegatedExecutionControlService,
)
from intergrax.runtime.execution.delegated_execution.local_provider import (
    LocalDelegatedExecutionProvider,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTROL_SERVICE_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "delegated_execution"
    / "control_service.py"
)
_T0 = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class EchoPayload:
    value: str


@dataclass(frozen=True)
class EchoResult:
    value: str


def _invocation(
    *,
    provider_id: str,
    run_id: str,
    invocation_id: str = "inv-control-1",
) -> ProviderInvocation:
    return ProviderInvocation.model_validate(
        {
            "invocation_id": invocation_id,
            "provider_id": provider_id,
            "operation": "execute_delegate",
            "task_id": "task-1",
            "run_id": run_id,
            "request_digest": "sha256:" + ("cd" * 32),
            "started_at": _T0.isoformat(),
            "provider_request_id": "preq-control-1",
            "provider_operation_id": "pop-control-1",
        }
    )


def _control_request(
    *,
    provider_id: str,
    operation: DelegatedExecutionControlOperation,
    execution_id: str | None = None,
    run_id: str | None = None,
    invocation_id: str = "inv-control-1",
) -> DelegatedExecutionControlRequest:
    run = run_id or str(mint_run_id())
    child = execution_id or str(mint_execution_id())
    return DelegatedExecutionControlRequest(
        execution_id=child,
        run_id=run,
        provider_invocation=_invocation(
            provider_id=provider_id,
            run_id=run,
            invocation_id=invocation_id,
        ),
        operation=operation,
    )


class FakeControlProvider(
    DelegatedExecutionProvider[EchoPayload, EchoResult],
    DelegatedExecutionCancelProvider,
    DelegatedExecutionInterruptProvider,
):
    def __init__(
        self,
        *,
        provider_id: str = "fake_control",
        cancel_calls: list[DelegatedExecutionControlRequest] | None = None,
        interrupt_calls: list[DelegatedExecutionControlRequest] | None = None,
        cancel_behavior: str = "accepted",
        interrupt_behavior: str = "accepted",
        raise_transport_on_cancel: bool = False,
    ) -> None:
        self._provider_id = provider_id
        self._cancel_calls = cancel_calls if cancel_calls is not None else []
        self._interrupt_calls = interrupt_calls if interrupt_calls is not None else []
        self._cancel_behavior = cancel_behavior
        self._interrupt_behavior = interrupt_behavior
        self._raise_transport_on_cancel = raise_transport_on_cancel
        self._cancel_count = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(
            provider_id=self._provider_id,
            supports_cancel=True,
            supports_interrupt=True,
        )

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        invocation = _invocation(
            provider_id=self._provider_id,
            run_id=str(request.context.run_id),
        )
        outcome = ProviderInvocationOutcome.model_validate(
            {
                "invocation_id": invocation.invocation_id,
                "status": ProviderInvocationStatus.SUCCEEDED,
                "completed_at": "2026-09-07T08:00:01+00:00",
                "provider_request_id": invocation.provider_request_id,
                "provider_operation_id": invocation.provider_operation_id,
            }
        )
        return delegated_success_outcome(
            result=EchoResult(value=request.payload.value),
            provider_invocation=invocation,
            provider_outcome=outcome,
        )

    async def cancel_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> DelegatedExecutionControlOutcome:
        self._cancel_calls.append(request)
        if self._raise_transport_on_cancel:
            raise DelegatedExecutionTransportError("transport down")
        self._cancel_count += 1
        if self._cancel_behavior == "provider_failure":
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.PROVIDER_FAILURE,
                request=request,
                provider_id=self._provider_id,
                failure_code="PROVIDER_CONTROL_FAILED",
                failure_message="provider control failed",
            )
        if self._cancel_count > 1 or self._cancel_behavior == "already_terminal":
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.ALREADY_TERMINAL,
                request=request,
                provider_id=self._provider_id,
            )
        if self._cancel_behavior == "not_found":
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.NOT_FOUND,
                request=request,
                provider_id=self._provider_id,
            )
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.ACCEPTED,
            request=request,
            provider_id=self._provider_id,
        )

    async def interrupt_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> DelegatedExecutionControlOutcome:
        self._interrupt_calls.append(request)
        if self._interrupt_behavior == "completed":
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.COMPLETED,
                request=request,
                provider_id=self._provider_id,
            )
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.ACCEPTED,
            request=request,
            provider_id=self._provider_id,
        )


@pytest.mark.asyncio
async def test_t1_unsupported_cancel_fail_closed() -> None:
    async def delegate(
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> EchoResult:
        return EchoResult(value=request.payload.value)

    provider = LocalDelegatedExecutionProvider(delegate)
    service = DelegatedExecutionControlService(provider)
    assert not isinstance(provider, DelegatedExecutionCancelProvider)

    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await service.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.UNSUPPORTED
    assert outcome.failure_code == "CONTROL_UNSUPPORTED"


@pytest.mark.asyncio
async def test_t2_unsupported_interrupt_fail_closed() -> None:
    async def delegate(
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> EchoResult:
        return EchoResult(value=request.payload.value)

    provider = LocalDelegatedExecutionProvider(delegate)
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.INTERRUPT,
    )
    outcome = await service.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.UNSUPPORTED


@pytest.mark.asyncio
async def test_t3_cancel_capable_provider_dispatched_once() -> None:
    cancel_calls: list[DelegatedExecutionControlRequest] = []
    provider = FakeControlProvider(cancel_calls=cancel_calls)
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await service.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED
    assert len(cancel_calls) == 1


@pytest.mark.asyncio
async def test_t4_interrupt_capable_provider() -> None:
    interrupt_calls: list[DelegatedExecutionControlRequest] = []
    provider = FakeControlProvider(
        interrupt_calls=interrupt_calls,
        interrupt_behavior="completed",
    )
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.INTERRUPT,
    )
    outcome = await service.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.COMPLETED
    assert len(interrupt_calls) == 1


def test_t5_control_service_has_no_reflection_dispatch() -> None:
    tree = ast.parse(_CONTROL_SERVICE_MODULE.read_text(encoding="utf-8"))
    forbidden = {"hasattr", "getattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def test_t6_control_service_does_not_import_local_provider() -> None:
    source = _CONTROL_SERVICE_MODULE.read_text(encoding="utf-8")
    assert "local_provider" not in source
    assert "LocalDelegatedExecutionProvider" not in source


@pytest.mark.asyncio
async def test_t7_execution_identity_preserved_in_outcome() -> None:
    execution_id = str(mint_execution_id())
    run_id = str(mint_run_id())
    provider = FakeControlProvider()
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
        execution_id=execution_id,
        run_id=run_id,
    )
    outcome = await service.apply_control(request)
    assert str(outcome.execution_id) == execution_id
    assert str(outcome.run_id) == run_id


@pytest.mark.asyncio
async def test_t8_provider_native_identity_not_execution_alias() -> None:
    execution_id = str(mint_execution_id())
    run_id = str(mint_run_id())
    provider = FakeControlProvider()
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
        execution_id=execution_id,
        run_id=run_id,
    )
    outcome = await service.apply_control(request)
    assert outcome.invocation_id == "inv-control-1"
    assert outcome.invocation_id != execution_id
    assert outcome.provider_request_id == "preq-control-1"


@pytest.mark.asyncio
async def test_t9_wrong_provider_binding_fail_closed() -> None:
    provider = FakeControlProvider(provider_id="provider-a")
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id="provider-b",
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await service.apply_control(request)
    assert (
        outcome.category
        is DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH
    )


@pytest.mark.asyncio
async def test_t10_transport_failure_typed() -> None:
    provider = FakeControlProvider(raise_transport_on_cancel=True)
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await service.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.TRANSPORT_FAILURE
    assert outcome.failure_code == "TRANSPORT_FAILURE"
    assert "TransportError" not in (outcome.failure_message or "")


@pytest.mark.asyncio
async def test_t11_provider_failure_neutral() -> None:
    provider = FakeControlProvider(cancel_behavior="provider_failure")
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await service.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.PROVIDER_FAILURE
    assert outcome.failure_code == "PROVIDER_CONTROL_FAILED"


@pytest.mark.asyncio
async def test_t12_repeated_cancel_idempotent_outcome() -> None:
    provider = FakeControlProvider()
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    first = await service.apply_control(request)
    second = await service.apply_control(request)
    assert first.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED
    assert second.category is DelegatedExecutionControlOutcomeCategory.ALREADY_TERMINAL


@pytest.mark.asyncio
async def test_t13_completed_work_not_false_cancel() -> None:
    provider = FakeControlProvider(cancel_behavior="not_found")
    service = DelegatedExecutionControlService(provider)
    request = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await service.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.NOT_FOUND
    assert outcome.category is not DelegatedExecutionControlOutcomeCategory.ACCEPTED


def test_t14_control_service_no_nexus_imports() -> None:
    tree = ast.parse(_CONTROL_SERVICE_MODULE.read_text(encoding="utf-8"))
    forbidden = (
        "intergrax.runtime.nexus",
        "NexusLoop",
        "GraphExecutor",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in forbidden:
                assert prefix not in module
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden:
                    assert prefix not in alias.name


def test_architecture_gate_no_execution_identity_mint() -> None:
    source = _CONTROL_SERVICE_MODULE.read_text(encoding="utf-8")
    assert "mint_execution_id" not in source
    assert "mint_run_id" not in source
    assert "execution_terminal" not in source


@pytest.mark.asyncio
async def test_t15_external_fake_provider_through_service() -> None:
    provider = FakeControlProvider(provider_id="external_conformance")
    service = DelegatedExecutionControlService(provider)
    outcome = await service.apply_control(
        _control_request(
            provider_id="external_conformance",
            operation=DelegatedExecutionControlOperation.CANCEL,
        ),
    )
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED


class _CancelCapableWithoutProtocol(DelegatedExecutionProvider[EchoPayload, EchoResult]):
    """Advertises cancel but does not implement cancel contract."""

    @property
    def provider_id(self) -> str:
        return "lying_cap"

    @property
    def provider_version(self) -> str:
        return "0.0.1"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(
            provider_id=self.provider_id,
            supports_cancel=True,
        )

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_capability_without_contract_fail_closed() -> None:
    service = DelegatedExecutionControlService(_CancelCapableWithoutProtocol())
    outcome = await service.apply_control(
        _control_request(
            provider_id="lying_cap",
            operation=DelegatedExecutionControlOperation.CANCEL,
        ),
    )
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.UNSUPPORTED
    assert outcome.failure_code == "CONTROL_CONTRACT_MISSING"
