# © Artur Czarnecki. All rights reserved.

"""P2.1-S2B / P2.1-S2B-C1 — delegated provider control correlation hardening."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionCancelProvider,
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcome,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    DelegatedExecutionDurableControlOutcomeCategory,
    DelegatedExecutionInterruptProvider,
    delegated_control_outcome,
)
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE,
    DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationNotFoundError,
    DelegatedInvocationCorrelationPersistenceError,
)
from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
    mint_delegated_execution_invocation_binding,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionBudgetMode,
    DelegatedExecutionBudgetProjection,
    DelegatedExecutionCapabilities,
    DelegatedExecutionContext,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    DelegatedExecutionTransportError,
    delegated_success_outcome,
    digest_delegated_execution_request,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
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
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    InMemoryDelegatedInvocationCorrelationStore,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationService,
)
from intergrax.runtime.execution.delegated_execution.durable_control_service import (
    DelegatedExecutionDurableControlService,
)
from intergrax.runtime.execution.delegated_execution.local_provider import (
    LocalDelegatedExecutionProvider,
)
from intergrax.runtime.execution.delegated_execution.provider_resolver import (
    MappingDelegatedExecutionProviderResolver,
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
_DURABLE_CONTROL_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "delegated_execution"
    / "durable_control_service.py"
)
_RESOLVER_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "delegated_execution_provider_resolver.py"
)
_BINDING_MODULE = (
    _REPO_ROOT / "intergrax" / "contracts" / "delegated_execution_invocation_binding.py"
)
_T0 = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
_PAYLOAD_DIGEST = "sha256:" + ("ef" * 32)


@dataclass(frozen=True)
class EchoPayload:
    value: str


@dataclass(frozen=True)
class EchoResult:
    value: str


def _operation() -> DelegatedExecutionOperationMetadata:
    return DelegatedExecutionOperationMetadata.model_validate(
        {
            "operation": "execute_delegate",
            "task_id": "task-1",
        }
    )


def _context(
    *,
    execution_id: object | None = None,
    run_id: object | None = None,
    parent_execution_id: object | None = None,
) -> DelegatedExecutionContext:
    parent = parent_execution_id or mint_execution_id()
    child = execution_id or mint_execution_id()
    run = run_id or mint_run_id()
    return DelegatedExecutionContext(
        execution_id=child,
        parent_execution_id=parent,
        run_id=run,
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("tools.read",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )


def _invocation_for_context(
    *,
    provider_id: str,
    context: DelegatedExecutionContext,
    operation: DelegatedExecutionOperationMetadata | None = None,
    invocation_id: str = "inv-control-1",
    run_id_override: str | None = None,
) -> ProviderInvocation:
    op = operation or _operation()
    digest = digest_delegated_execution_request(
        context=context,
        operation=op,
        payload_digest=_PAYLOAD_DIGEST,
    )
    return ProviderInvocation.model_validate(
        {
            "invocation_id": invocation_id,
            "provider_id": provider_id,
            "operation": op.operation,
            "task_id": op.task_id,
            "run_id": run_id_override or str(context.run_id),
            "request_digest": digest,
            "started_at": _T0.isoformat(),
            "provider_request_id": "preq-control-1",
            "provider_operation_id": "pop-control-1",
        }
    )


def _binding(
    *,
    provider_id: str,
    context: DelegatedExecutionContext | None = None,
    operation: DelegatedExecutionOperationMetadata | None = None,
    invocation: ProviderInvocation | None = None,
) -> DelegatedExecutionInvocationBinding:
    ctx = context or _context()
    op = operation or _operation()
    inv = invocation or _invocation_for_context(
        provider_id=provider_id,
        context=ctx,
        operation=op,
    )
    return mint_delegated_execution_invocation_binding(
        context=ctx,
        operation=op,
        payload_digest=_PAYLOAD_DIGEST,
        provider_invocation=inv,
    )


def _control_request(
    *,
    provider_id: str,
    operation: DelegatedExecutionControlOperation,
    execution_id: object | None = None,
    run_id: object | None = None,
    invocation_id: str = "inv-control-1",
    binding: DelegatedExecutionInvocationBinding | None = None,
) -> DelegatedExecutionControlRequest:
    if binding is not None:
        return DelegatedExecutionControlRequest(
            invocation_binding=binding,
            operation=operation,
        )
    ctx = _context(execution_id=execution_id, run_id=run_id)
    resolved_binding = _binding(
        provider_id=provider_id,
        context=ctx,
        invocation=_invocation_for_context(
            provider_id=provider_id,
            context=ctx,
            invocation_id=invocation_id,
        ),
    )
    return DelegatedExecutionControlRequest(
        invocation_binding=resolved_binding,
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
        cancel_spoof_field: str | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._cancel_calls = cancel_calls if cancel_calls is not None else []
        self._interrupt_calls = interrupt_calls if interrupt_calls is not None else []
        self._cancel_behavior = cancel_behavior
        self._interrupt_behavior = interrupt_behavior
        self._raise_transport_on_cancel = raise_transport_on_cancel
        self._cancel_spoof_field = cancel_spoof_field
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
        ctx = request.context
        invocation = _invocation_for_context(
            provider_id=self._provider_id,
            context=ctx,
            operation=request.operation,
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
        outcome = delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.ACCEPTED,
            request=request,
            provider_id=self._provider_id,
        )
        return _maybe_spoof_control_outcome(
            outcome,
            request=request,
            provider_id=self._provider_id,
            spoof_field=self._cancel_spoof_field,
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


def _maybe_spoof_control_outcome(
    outcome: DelegatedExecutionControlOutcome,
    *,
    request: DelegatedExecutionControlRequest,
    provider_id: str,
    spoof_field: str | None,
) -> DelegatedExecutionControlOutcome:
    if spoof_field is None:
        return outcome
    if spoof_field == "execution_id":
        return DelegatedExecutionControlOutcome(
            category=outcome.category,
            operation=outcome.operation,
            execution_id=mint_execution_id(),
            run_id=outcome.run_id,
            provider_id=outcome.provider_id,
            invocation_id=outcome.invocation_id,
            provider_request_id=outcome.provider_request_id,
            provider_operation_id=outcome.provider_operation_id,
        )
    if spoof_field == "run_id":
        return DelegatedExecutionControlOutcome(
            category=outcome.category,
            operation=outcome.operation,
            execution_id=outcome.execution_id,
            run_id=mint_run_id(),
            provider_id=outcome.provider_id,
            invocation_id=outcome.invocation_id,
            provider_request_id=outcome.provider_request_id,
            provider_operation_id=outcome.provider_operation_id,
        )
    if spoof_field == "provider_id":
        return DelegatedExecutionControlOutcome(
            category=outcome.category,
            operation=outcome.operation,
            execution_id=outcome.execution_id,
            run_id=outcome.run_id,
            provider_id="spoof-provider",
            invocation_id=outcome.invocation_id,
            provider_request_id=outcome.provider_request_id,
            provider_operation_id=outcome.provider_operation_id,
        )
    if spoof_field == "invocation_id":
        return DelegatedExecutionControlOutcome(
            category=outcome.category,
            operation=outcome.operation,
            execution_id=outcome.execution_id,
            run_id=outcome.run_id,
            provider_id=outcome.provider_id,
            invocation_id="spoof-invocation",
            provider_request_id=outcome.provider_request_id,
            provider_operation_id=outcome.provider_operation_id,
        )
    if spoof_field == "provider_request_id":
        return DelegatedExecutionControlOutcome(
            category=outcome.category,
            operation=outcome.operation,
            execution_id=outcome.execution_id,
            run_id=outcome.run_id,
            provider_id=outcome.provider_id,
            invocation_id=outcome.invocation_id,
            provider_request_id="spoof-preq",
            provider_operation_id=outcome.provider_operation_id,
        )
    if spoof_field == "provider_operation_id":
        return DelegatedExecutionControlOutcome(
            category=outcome.category,
            operation=outcome.operation,
            execution_id=outcome.execution_id,
            run_id=outcome.run_id,
            provider_id=outcome.provider_id,
            invocation_id=outcome.invocation_id,
            provider_request_id=outcome.provider_request_id,
            provider_operation_id="spoof-pop",
        )
    if spoof_field == "operation":
        return DelegatedExecutionControlOutcome(
            category=outcome.category,
            operation=DelegatedExecutionControlOperation.INTERRUPT,
            execution_id=outcome.execution_id,
            run_id=outcome.run_id,
            provider_id=outcome.provider_id,
            invocation_id=outcome.invocation_id,
            provider_request_id=outcome.provider_request_id,
            provider_operation_id=outcome.provider_operation_id,
        )
    raise AssertionError(f"unknown spoof_field: {spoof_field}")


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


@pytest.mark.asyncio
async def test_c1_t1_cross_child_invocation_rejected_before_provider() -> None:
    run = mint_run_id()
    ctx_a = _context(run_id=run)
    ctx_b = _context(run_id=run)
    inv_a = _invocation_for_context(provider_id="fake_control", context=ctx_a)
    cancel_calls: list[DelegatedExecutionControlRequest] = []
    provider = FakeControlProvider(cancel_calls=cancel_calls)
    with pytest.raises(ValidationError):
        mint_delegated_execution_invocation_binding(
            context=ctx_b,
            operation=_operation(),
            payload_digest=_PAYLOAD_DIGEST,
            provider_invocation=inv_a,
        )
    assert len(cancel_calls) == 0


@pytest.mark.asyncio
async def test_c1_t2_wrong_run_rejected_at_binding() -> None:
    ctx = _context()
    with pytest.raises(ValidationError):
        _binding(
            provider_id="fake_control",
            context=ctx,
            invocation=_invocation_for_context(
                provider_id="fake_control",
                context=ctx,
                run_id_override=str(mint_run_id()),
            ),
        )


@pytest.mark.asyncio
async def test_c1_t3_wrong_provider_binding_fail_closed() -> None:
    provider = FakeControlProvider(provider_id="provider-a")
    service = DelegatedExecutionControlService(provider)
    outcome = await service.apply_control(
        _control_request(
            provider_id="provider-b",
            operation=DelegatedExecutionControlOperation.CANCEL,
        ),
    )
    assert (
        outcome.category
        is DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spoof_field",),
    [
        ("execution_id",),
        ("run_id",),
        ("provider_id",),
        ("invocation_id",),
        ("provider_request_id",),
        ("provider_operation_id",),
        ("operation",),
    ],
)
async def test_c1_t4_t10_malicious_provider_outcome_fail_closed(
    spoof_field: str,
) -> None:
    cancel_calls: list[DelegatedExecutionControlRequest] = []
    provider = FakeControlProvider(
        cancel_calls=cancel_calls,
        cancel_spoof_field=spoof_field,
    )
    service = DelegatedExecutionControlService(provider)
    outcome = await service.apply_control(
        _control_request(
            provider_id=provider.provider_id,
            operation=DelegatedExecutionControlOperation.CANCEL,
        ),
    )
    assert (
        outcome.category
        is DelegatedExecutionControlOutcomeCategory.CONTROL_OUTCOME_CONTRACT_MISMATCH
    )
    assert outcome.failure_code == "CONTROL_OUTCOME_CONTRACT_MISMATCH"
    assert len(cancel_calls) == 1


@pytest.mark.asyncio
async def test_c1_t11_valid_outcome_unchanged() -> None:
    provider = FakeControlProvider()
    service = DelegatedExecutionControlService(provider)
    outcome = await service.apply_control(
        _control_request(
            provider_id=provider.provider_id,
            operation=DelegatedExecutionControlOperation.CANCEL,
        ),
    )
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED


def test_c1_architecture_gate_binding_has_no_global_registry() -> None:
    source = _BINDING_MODULE.read_text(encoding="utf-8")
    assert "global " not in source
    assert "Registry" not in source


def test_c1_architecture_gate_binding_no_reflection() -> None:
    tree = ast.parse(_BINDING_MODULE.read_text(encoding="utf-8"))
    forbidden = {"hasattr", "getattr", "setattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


class _TrackingResolver(DelegatedExecutionProviderResolver):
    def __init__(self, provider: FakeControlProvider) -> None:
        self._provider = provider
        self.resolve_calls = 0

    def resolve(self, provider_id: str):
        self.resolve_calls += 1
        if provider_id == self._provider.provider_id:
            return self._provider
        return None


def _durable_service(
    correlation: DelegatedInvocationCorrelationService,
    provider: FakeControlProvider,
) -> DelegatedExecutionDurableControlService:
    return DelegatedExecutionDurableControlService(
        correlation,
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
    )


def _persist_control_binding(
    provider: FakeControlProvider,
) -> tuple[DelegatedInvocationCorrelationService, DelegatedExecutionInvocationBinding]:
    binding = _control_request(
        provider_id=provider.provider_id,
        operation=DelegatedExecutionControlOperation.CANCEL,
    ).invocation_binding
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    correlation.persist_binding(binding, persisted_at=_T0)
    return correlation, binding


@pytest.mark.asyncio
async def test_s2c2_c1_t6_durable_control_not_found() -> None:
    provider = FakeControlProvider()
    correlation = DelegatedInvocationCorrelationService(
        InMemoryDelegatedInvocationCorrelationStore(),
    )
    durable = _durable_service(correlation, provider)
    outcome = await durable.apply_control_by_execution_id(
        mint_execution_id(),
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert (
        outcome.category
        is DelegatedExecutionDurableControlOutcomeCategory.CORRELATION_NOT_FOUND
    )
    assert outcome.failure_message == DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE
    assert provider._cancel_calls == []


@pytest.mark.asyncio
async def test_s2c2_c1_t7_durable_control_integrity_failure() -> None:
    provider = FakeControlProvider()

    class BrokenLookup:
        def load_binding_by_execution_id(self, execution_id: object):
            raise DelegatedInvocationCorrelationIntegrityError("corrupt")

    durable = DelegatedExecutionDurableControlService(
        BrokenLookup(),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
    )
    outcome = await durable.apply_control_by_execution_id(
        mint_execution_id(),
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert (
        outcome.category
        is DelegatedExecutionDurableControlOutcomeCategory.CORRELATION_INTEGRITY_FAILURE
    )
    assert (
        outcome.failure_message
        == DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE
    )
    assert provider._cancel_calls == []


@pytest.mark.asyncio
async def test_s2c2_c1_t8_durable_control_persistence_failure() -> None:
    provider = FakeControlProvider()

    class BrokenLookup:
        def load_binding_by_execution_id(self, execution_id: object):
            raise DelegatedInvocationCorrelationPersistenceError("down")

    durable = DelegatedExecutionDurableControlService(
        BrokenLookup(),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
    )
    outcome = await durable.apply_control_by_execution_id(
        mint_execution_id(),
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert (
        outcome.category
        is DelegatedExecutionDurableControlOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE
    )
    assert provider._cancel_calls == []


@pytest.mark.asyncio
async def test_s2c2_c1_t11_resolver_not_called_on_lookup_failure() -> None:
    provider = FakeControlProvider()
    resolver = _TrackingResolver(provider)

    class BrokenLookup:
        def load_binding_by_execution_id(self, execution_id: object):
            raise DelegatedInvocationCorrelationNotFoundError(
                DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
            )

    durable = DelegatedExecutionDurableControlService(BrokenLookup(), resolver)
    await durable.apply_control_by_execution_id(
        mint_execution_id(),
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert resolver.resolve_calls == 0
    assert provider._cancel_calls == []


@pytest.mark.asyncio
async def test_s2c2_c1_t13_real_durable_control_success() -> None:
    provider = FakeControlProvider()
    correlation, binding = _persist_control_binding(provider)
    durable = _durable_service(correlation, provider)
    outcome = await durable.apply_control_by_execution_id(
        binding.execution_id,
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert (
        outcome.category
        is DelegatedExecutionDurableControlOutcomeCategory.RESOLVED_CONTROL
    )
    assert outcome.control_outcome is not None
    assert outcome.control_outcome.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED
    assert len(provider._cancel_calls) == 1


@pytest.mark.asyncio
async def test_s2c2_c1_t14_provider_unavailable_with_real_binding() -> None:
    provider = FakeControlProvider(provider_id="bound-provider")
    correlation, binding = _persist_control_binding(provider)
    durable = DelegatedExecutionDurableControlService(
        correlation,
        MappingDelegatedExecutionProviderResolver({}),
    )
    outcome = await durable.apply_control_by_execution_id(
        binding.execution_id,
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert (
        outcome.category
        is DelegatedExecutionDurableControlOutcomeCategory.RESOLVED_CONTROL
    )
    assert outcome.control_outcome is not None
    assert (
        outcome.control_outcome.category
        is DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH
    )


@pytest.mark.asyncio
async def test_s2c2_c1_t15_provider_id_mismatch_fail_closed() -> None:
    bound = FakeControlProvider(provider_id="bound-provider")
    resolved = FakeControlProvider(provider_id="other-provider")
    correlation, binding = _persist_control_binding(bound)
    durable = DelegatedExecutionDurableControlService(
        correlation,
        MappingDelegatedExecutionProviderResolver(
            {bound.provider_id: resolved},
        ),
    )
    outcome = await durable.apply_control_by_execution_id(
        binding.execution_id,
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert (
        outcome.control_outcome is not None
        and outcome.control_outcome.category
        is DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH
    )
    assert bound._cancel_calls == []
    assert resolved._cancel_calls == []


def test_s2c2_c1_t18_custom_resolver_contract() -> None:
    provider = FakeControlProvider()
    resolver = _TrackingResolver(provider)
    assert resolver.resolve(provider.provider_id) is provider


def test_s2c2_c1_t20_durable_control_no_reflection() -> None:
    tree = ast.parse(_DURABLE_CONTROL_MODULE.read_text(encoding="utf-8"))
    forbidden = {"hasattr", "getattr", "setattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def test_s2c2_c1_t21_durable_control_no_global_registry() -> None:
    source = _DURABLE_CONTROL_MODULE.read_text(encoding="utf-8")
    assert "global " not in source
    assert "Registry" not in source


def test_s2c2_c1_t17_resolver_abi_no_any() -> None:
    assert "Any" not in _RESOLVER_CONTRACT.read_text(encoding="utf-8")
