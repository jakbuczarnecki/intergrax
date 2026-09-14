# © Artur Czarnecki. All rights reserved.

"""P2.1-S2A — delegated provider production adoption through child execution boundary."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionCapabilities,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    assert_provider_native_ids_distinct_from_execution,
    delegated_failure_outcome,
    delegated_success_outcome,
    digest_delegated_execution_payload,
    digest_delegated_execution_request,
)
from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    require_active_execution_id,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.execution.active_execution_budget import peek_active_execution_budget
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.delegated_execution.service import (
    DelegatedExecutionService,
    DelegatedExecutionWorkUnit,
    delegated_execution_service,
)
from intergrax.runtime.execution.delegated_execution.local_provider import (
    LocalDelegatedExecutionProvider,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SERVICE_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "delegated_execution" / "service.py"
)
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


@dataclass(frozen=True)
class EchoPayload:
    value: str


@dataclass(frozen=True)
class EchoResult:
    value: str
    child_execution_id: str
    parent_execution_id: str


def _operation(**overrides: object) -> DelegatedExecutionOperationMetadata:
    base = {
        "operation": "execute_delegate",
        "task_id": "task_adoption",
    }
    base.update(overrides)
    return DelegatedExecutionOperationMetadata.model_validate(base)


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _root_authority(*scopes: str) -> ParentExecutionAuthority:
    return ParentExecutionAuthority.scoped(scopes)


class _RecordingProvider:
    """Spy provider capturing dispatch context and admission ordering."""

    def __init__(self) -> None:
        self.calls: list[DelegatedExecutionRequest[EchoPayload]] = []
        self.admission_observed_before_execute = False
        self.active_execution_id_at_dispatch: str | None = None

    @property
    def provider_id(self) -> str:
        return "recording_fake"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(provider_id=self.provider_id)

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        self.active_execution_id_at_dispatch = str(require_active_execution_id())
        self.calls.append(request)
        assert_provider_native_ids_distinct_from_execution(
            execution_id=request.context.execution_id,
            provider_request_id="preq-fake-1",
            provider_operation_id="pop-fake-1",
            invocation_id="inv-fake-1",
        )
        payload_digest = digest_delegated_execution_payload(request.payload)
        request_digest = digest_delegated_execution_request(
            context=request.context,
            operation=request.operation,
            payload_digest=payload_digest,
        )
        return delegated_success_outcome(
            result=EchoResult(
                value=request.payload.value,
                child_execution_id=str(request.context.execution_id),
                parent_execution_id=str(request.context.parent_execution_id),
            ),
            provider_invocation=ProviderInvocation.model_validate(
                {
                    "invocation_id": "inv-fake-1",
                    "provider_id": self.provider_id,
                    "operation": request.operation.operation,
                    "task_id": request.operation.task_id,
                    "run_id": str(request.context.run_id),
                    "request_digest": request_digest,
                    "started_at": "2026-09-07T08:00:00+00:00",
                    "provider_request_id": "preq-fake-1",
                    "provider_operation_id": "pop-fake-1",
                }
            ),
            provider_outcome=ProviderInvocationOutcome.model_validate(
                {
                    "invocation_id": "inv-fake-1",
                    "status": ProviderInvocationStatus.SUCCEEDED,
                    "completed_at": "2026-09-07T08:00:01+00:00",
                    "provider_request_id": "preq-fake-1",
                    "provider_operation_id": "pop-fake-1",
                }
            ),
        )


class _UnsupportedOperationProvider:
    @property
    def provider_id(self) -> str:
        return "unsupported_fake"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(
            provider_id=self.provider_id,
            supports_cancel=False,
        )

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        if request.operation.operation != "execute_delegate":
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.UNSUPPORTED,
                failure_code="UNSUPPORTED_OPERATION",
                failure_message="operation not supported",
            )
        return delegated_failure_outcome(
            category=DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
            failure_code="PROVIDER_EXECUTION_FAILED",
            failure_message="delegated execution provider failed",
        )


class _EchoDelegate:
    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> EchoResult:
        return EchoResult(
            value=request.payload.value,
            child_execution_id=str(request.context.execution_id),
            parent_execution_id=str(request.context.parent_execution_id),
        )


async def _run_under_root(
    service: DelegatedExecutionService[EchoPayload, EchoResult],
    *,
    payload: EchoPayload | None = None,
    requested_permission_scopes: tuple[str, ...] | None = None,
    requested_budget: RunBudget | None = None,
    admission_hooks: tuple[
        ExecutionAdmissionHook[DelegatedExecutionWorkUnit[EchoPayload]],
        ...,
    ] = (),
    root_authority: ParentExecutionAuthority | None = None,
) -> DelegatedExecutionOutcome[EchoResult]:
    root = _root_identity()
    authority = root_authority or _root_authority("read", "write", "delete")
    captured: list[DelegatedExecutionOutcome[EchoResult]] = []

    class RootDelegate:
        async def execute(self, request: EchoPayload) -> EchoResult:
            outcome = await service.execute_delegated(
                payload=request,
                operation=_operation(),
                requested_permission_scopes=requested_permission_scopes,
                requested_budget=requested_budget,
                admission_hooks=admission_hooks,
            )
            captured.append(outcome)
            if outcome.category is not DelegatedExecutionOutcomeCategory.SUCCESS:
                return EchoResult(
                    value="failed",
                    child_execution_id="",
                    parent_execution_id="",
                )
            assert outcome.result is not None
            return outcome.result

    await ExecutionBoundary[EchoPayload, EchoResult](
        RootDelegate(),
        identity=root,
        authority=authority,
    ).execute(payload or EchoPayload(value="ping"))

    assert len(captured) == 1
    return captured[0]


@pytest.mark.asyncio
async def test_t1_child_identity_distinct_from_parent() -> None:
    provider = _RecordingProvider()
    service = delegated_execution_service(provider, ledger=_UNLIMITED_LEDGER)
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    assert outcome.result is not None
    assert outcome.result.child_execution_id != outcome.result.parent_execution_id
    ctx = provider.calls[0].context
    assert str(ctx.execution_id) == outcome.result.child_execution_id
    assert str(ctx.parent_execution_id) == outcome.result.parent_execution_id


@pytest.mark.asyncio
async def test_t3_admission_before_provider_execute() -> None:
    provider = _RecordingProvider()
    admitted: list[bool] = []

    class _AdmissionGate:
        async def admit(
            self,
            request: DelegatedExecutionWorkUnit[EchoPayload],
        ) -> None:
            _ = request
            admitted.append(True)

    service = delegated_execution_service(provider, ledger=_UNLIMITED_LEDGER)
    await _run_under_root(service, admission_hooks=(_AdmissionGate(),))
    assert admitted == [True]
    assert provider.active_execution_id_at_dispatch is not None
    assert provider.active_execution_id_at_dispatch == str(
        provider.calls[0].context.execution_id,
    )


@pytest.mark.asyncio
async def test_t4_authority_widening_fail_closed() -> None:
    provider = _RecordingProvider()
    service = delegated_execution_service(provider, ledger=_UNLIMITED_LEDGER)
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: EchoPayload) -> EchoResult:
            await service.execute_delegated(
                payload=request,
                operation=_operation(),
                requested_permission_scopes=("admin",),
            )
            return EchoResult(value="x", child_execution_id="", parent_execution_id="")

    with pytest.raises(DelegationAuthorityError):
        await ExecutionBoundary[EchoPayload, EchoResult](
            RootDelegate(),
            identity=root,
            authority=_root_authority("read"),
        ).execute(EchoPayload(value="ping"))


@pytest.mark.asyncio
async def test_t5_budget_narrowing_reserved_child() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=50))
    provider = _RecordingProvider()
    service = delegated_execution_service(provider, ledger=ledger)
    child_budget = RunBudget(max_tool_calls=10)
    await _run_under_root(
        service,
        requested_budget=child_budget,
        root_authority=_root_authority("read"),
    )
    projection = provider.calls[0].context.budget
    assert projection.reservation_allowance is not None
    assert projection.reservation_allowance.max_tool_calls == 10


@pytest.mark.asyncio
async def test_t6_parent_lineage_in_provider_context() -> None:
    provider = _RecordingProvider()
    service = delegated_execution_service(provider, ledger=_UNLIMITED_LEDGER)
    root = _root_identity()
    captured: list[DelegatedExecutionOutcome[EchoResult]] = []

    class RootDelegate:
        async def execute(self, request: EchoPayload) -> EchoResult:
            outcome = await service.execute_delegated(
                payload=request,
                operation=_operation(),
            )
            captured.append(outcome)
            assert outcome.result is not None
            return outcome.result

    await ExecutionBoundary[EchoPayload, EchoResult](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read"),
    ).execute(EchoPayload(value="ping"))

    assert str(provider.calls[0].context.parent_execution_id) == str(root.execution_id)


@pytest.mark.asyncio
async def test_t8_local_provider_through_adoption_path() -> None:
    local = LocalDelegatedExecutionProvider(_EchoDelegate())
    service = delegated_execution_service(local, ledger=_UNLIMITED_LEDGER)
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    assert outcome.result is not None
    assert outcome.result.value == "ping"


@pytest.mark.asyncio
async def test_t9_transport_failure_stable_category() -> None:
    class _TimeoutDelegate:
        async def execute(
            self,
            request: DelegatedExecutionRequest[EchoPayload],
        ) -> EchoResult:
            raise TimeoutError("connect timed out")

    local = LocalDelegatedExecutionProvider(_TimeoutDelegate())
    service = delegated_execution_service(local, ledger=_UNLIMITED_LEDGER)
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE
    assert outcome.failure_code == "TRANSPORT_TIMEOUT"
    assert outcome.failure_message == "delegated execution transport timed out"


@pytest.mark.asyncio
async def test_t10_provider_failure_typed_neutral() -> None:
    service = delegated_execution_service(
        _UnsupportedOperationProvider(),
        ledger=_UNLIMITED_LEDGER,
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert outcome.failure_code == "PROVIDER_EXECUTION_FAILED"


@pytest.mark.asyncio
async def test_t11_unsupported_capability_explicit() -> None:
    service = delegated_execution_service(
        _UnsupportedOperationProvider(),
        ledger=_UNLIMITED_LEDGER,
    )
    root = _root_identity()
    captured: list[DelegatedExecutionOutcome[EchoResult]] = []

    class RootDelegateCapture:
        async def execute(self, request: EchoPayload) -> EchoResult:
            outcome = await service.execute_delegated(
                payload=request,
                operation=_operation(operation="unsupported_op"),
            )
            captured.append(outcome)
            return EchoResult(value="x", child_execution_id="", parent_execution_id="")

    await ExecutionBoundary[EchoPayload, EchoResult](
        RootDelegateCapture(),
        identity=root,
        authority=_root_authority("read"),
    ).execute(EchoPayload(value="ping"))

    assert captured[0].category is DelegatedExecutionOutcomeCategory.UNSUPPORTED


@pytest.mark.asyncio
async def test_t15_budget_cleanup_after_provider_failure() -> None:
    service = delegated_execution_service(
        _UnsupportedOperationProvider(),
        ledger=_UNLIMITED_LEDGER,
    )
    await _run_under_root(service)
    assert peek_active_execution_budget() is None
    assert peek_active_execution_id() is None


def test_t12_service_module_has_no_nexus_orchestration_imports() -> None:
    tree = ast.parse(_SERVICE_MODULE.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "intergrax.runtime.nexus.execution",
        "intergrax.runtime.nexus.responses",
        "intergrax.runtime.nexus.loop",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in forbidden_prefixes:
                assert not module.startswith(prefix)


def test_t13_service_module_does_not_import_local_provider() -> None:
    source = _SERVICE_MODULE.read_text(encoding="utf-8")
    assert "LocalDelegatedExecutionProvider" not in source
    assert "local_provider" not in source


@pytest.mark.asyncio
async def test_conformance_fake_provider_composition() -> None:
    class FakeDelegatedProvider(DelegatedExecutionProvider[EchoPayload, EchoResult]):
        @property
        def provider_id(self) -> str:
            return "fake_conformance"

        @property
        def provider_version(self) -> str:
            return "0.1.0"

        @property
        def capabilities(self) -> DelegatedExecutionCapabilities:
            return DelegatedExecutionCapabilities(provider_id=self.provider_id)

        async def execute(
            self,
            request: DelegatedExecutionRequest[EchoPayload],
        ) -> DelegatedExecutionOutcome[EchoResult]:
            payload_digest = digest_delegated_execution_payload(request.payload)
            request_digest = digest_delegated_execution_request(
                context=request.context,
                operation=request.operation,
                payload_digest=payload_digest,
            )
            invocation = ProviderInvocation.model_validate(
                {
                    "invocation_id": "inv-conformance-1",
                    "provider_id": self.provider_id,
                    "operation": request.operation.operation,
                    "task_id": request.operation.task_id,
                    "run_id": str(request.context.run_id),
                    "request_digest": request_digest,
                    "started_at": "2026-09-07T08:00:00+00:00",
                    "provider_request_id": "preq-conformance-1",
                    "provider_operation_id": "pop-conformance-1",
                }
            )
            provider_outcome = ProviderInvocationOutcome.model_validate(
                {
                    "invocation_id": "inv-conformance-1",
                    "status": ProviderInvocationStatus.SUCCEEDED,
                    "completed_at": "2026-09-07T08:00:01+00:00",
                    "provider_request_id": "preq-conformance-1",
                    "provider_operation_id": "pop-conformance-1",
                }
            )
            return delegated_success_outcome(
                result=EchoResult(
                    value=request.payload.value,
                    child_execution_id=str(request.context.execution_id),
                    parent_execution_id=str(request.context.parent_execution_id),
                ),
                provider_invocation=invocation,
                provider_outcome=provider_outcome,
            )

    service = delegated_execution_service(
        FakeDelegatedProvider(),
        ledger=_UNLIMITED_LEDGER,
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
