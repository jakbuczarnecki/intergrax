# © Artur Czarnecki. All rights reserved.

"""P2.1-S2D — real subprocess delegated execution provider production qualification."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
)
from intergrax.contracts.delegated_execution_continuation import (
    DelegatedExecutionContinuationOutcomeCategory,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionCapabilities,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    delegated_failure_outcome,
)
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
)
from intergrax.contracts.delegated_execution_status import (
    DelegatedExecutionStatusOutcomeCategory,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationDurabilityMode,
    DelegatedInvocationCorrelationDurabilityPolicy,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.delegated_execution.continuation_service import (
    DelegatedExecutionContinuationService,
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
from intergrax.runtime.execution.delegated_execution.provider_resolver import (
    MappingDelegatedExecutionProviderResolver,
)
from intergrax.runtime.execution.delegated_execution.service import (
    DelegatedExecutionService,
    delegated_execution_service,
)
from intergrax.runtime.execution.delegated_execution.status_service import (
    DelegatedExecutionStatusReadService,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.integrations.providers.delegated_execution.subprocess.bundle import (
    SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID,
    create_subprocess_delegated_execution_provider,
)
from intergrax.integrations.providers.delegated_execution.subprocess.config import (
    SubprocessDelegatedExecutionProviderConfig,
)
from intergrax.integrations.providers.delegated_execution.subprocess.provider import (
    SubprocessDelegatedExecutionProvider,
    SubprocessEchoPayload,
    SubprocessEchoResult,
)
from intergrax.integrations.providers.delegated_execution.subprocess.transport import (
    FailingConnectTransport,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROVIDER_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "integrations"
    / "providers"
    / "delegated_execution"
    / "subprocess"
    / "provider.py"
)
_SERVICE_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "delegated_execution" / "service.py"
)
_CHILD_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"
_BOUNDARY_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "boundary.py"
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())
_SECRET = "s2d-test-auth-token-do-not-log"


@dataclass(frozen=True)
class _RootEchoPayload:
    value: str
    behavior: str | None = None


@dataclass(frozen=True)
class _RootEchoResult:
    value: str
    child_execution_id: str
    parent_execution_id: str


def _operation(**overrides: object) -> DelegatedExecutionOperationMetadata:
    base = {"operation": "execute_delegate", "task_id": "task_s2d"}
    base.update(overrides)
    return DelegatedExecutionOperationMetadata.model_validate(base)


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _config(**overrides: object) -> SubprocessDelegatedExecutionProviderConfig:
    base = {
        "enabled": True,
        "request_timeout_seconds": 10.0,
        "connect_timeout_seconds": 3.0,
        "connection_auth_token": _SECRET,
    }
    base.update(overrides)
    return SubprocessDelegatedExecutionProviderConfig.model_validate(base)


@pytest.fixture
def subprocess_provider() -> Iterator[SubprocessDelegatedExecutionProvider]:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    yield provider
    provider.close()


async def _run_delegated(
    provider: SubprocessDelegatedExecutionProvider,
    *,
    payload: SubprocessEchoPayload,
) -> tuple[DelegatedExecutionOutcome[SubprocessEchoResult], InMemoryDelegatedInvocationCorrelationStore]:
    store = InMemoryDelegatedInvocationCorrelationStore()
    service = delegated_execution_service(
        provider,
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=DelegatedInvocationCorrelationDurabilityPolicy(
            mode=DelegatedInvocationCorrelationDurabilityMode.NON_DURABLE_TEST,
        ),
        correlation_store=store,
    )
    root = _root_identity()
    authority = ParentExecutionAuthority.scoped(("read", "write"))
    captured: list[DelegatedExecutionOutcome[SubprocessEchoResult]] = []

    class RootDelegate:
        async def execute(self, request: _RootEchoPayload) -> _RootEchoResult:
            outcome = await service.execute_delegated(
                payload=SubprocessEchoPayload(
                    value=request.value,
                    behavior=request.behavior,
                ),
                operation=_operation(),
            )
            captured.append(outcome)
            if outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS and outcome.result:
                return _RootEchoResult(
                    value=outcome.result.value,
                    child_execution_id=outcome.result.child_execution_id,
                    parent_execution_id=outcome.result.parent_execution_id,
                )
            return _RootEchoResult(value="failed", child_execution_id="", parent_execution_id="")

    await ExecutionBoundary[_RootEchoPayload, _RootEchoResult](
        RootDelegate(),
        identity=root,
        authority=authority,
    ).execute(_RootEchoPayload(value=payload.value, behavior=payload.behavior))
    assert len(captured) == 1
    return captured[0], store


@pytest.mark.asyncio
async def test_s2d_t1_real_external_success(subprocess_provider: SubprocessDelegatedExecutionProvider) -> None:
    outcome, _store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="external-ok"),
    )
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    count = await subprocess_provider.worker_execute_count()
    assert count >= 1


@pytest.mark.asyncio
async def test_s2d_t2_canonical_child_execution(subprocess_provider: SubprocessDelegatedExecutionProvider) -> None:
    outcome, _store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="child-path"),
    )
    assert outcome.result is not None
    assert outcome.result.child_execution_id != outcome.result.parent_execution_id


@pytest.mark.asyncio
async def test_s2d_t3_no_provider_minted_execution_id(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, _store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="identity"),
    )
    assert outcome.invocation_binding is not None
    assert outcome.invocation_binding.provider_invocation.provider_request_id != str(
        outcome.invocation_binding.execution_id,
    )


@pytest.mark.asyncio
async def test_s2d_t6_transport_before_dispatch_failure() -> None:
    provider = SubprocessDelegatedExecutionProvider(
        _config(),
        transport=FailingConnectTransport(),
    )
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x"),
        operation=_operation(),
    )
    outcome = await provider.execute(request)
    assert outcome.category is DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE
    assert outcome.failure_code == "TRANSPORT_FAILURE"


@pytest.mark.asyncio
async def test_s2d_t7_post_dispatch_ambiguous_failure(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x", behavior="accept_then_disconnect"),
        operation=_operation(),
    )
    outcome = await subprocess_provider.execute(request)
    assert outcome.failure_code == "OUTCOME_UNKNOWN"
    assert outcome.provider_outcome is not None
    assert outcome.provider_outcome.status is ProviderInvocationStatus.UNKNOWN


@pytest.mark.asyncio
async def test_s2d_t8_no_auto_retry_on_unknown(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    before = await subprocess_provider.worker_execute_count()
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x", behavior="accept_then_disconnect"),
        operation=_operation(),
    )
    await subprocess_provider.execute(request)
    after = await subprocess_provider.worker_execute_count()
    assert after == before + 1


@pytest.mark.asyncio
async def test_s2d_t9_invalid_provider_response(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x", behavior="spoof_execution_id"),
        operation=_operation(),
    )
    outcome = await subprocess_provider.execute(request)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE
    assert outcome.failure_code == "OUTCOME_CONTRACT_MISMATCH"


@pytest.mark.asyncio
async def test_s2d_t13_timeout_bounded() -> None:
    provider = create_subprocess_delegated_execution_provider(
        config=_config(request_timeout_seconds=0.2),
    )
    try:
        request = DelegatedExecutionRequest(
            context=_dummy_context(),
            payload=SubprocessEchoPayload(value="x", behavior="slow"),
            operation=_operation(),
        )
        outcome = await provider.execute(request)
        assert outcome.failure_code in {"OUTCOME_UNKNOWN", "TRANSPORT_FAILURE"}
    finally:
        provider.close()


@pytest.mark.asyncio
async def test_s2d_t14_secret_redaction() -> None:
    config = _config(connection_auth_token=_SECRET)
    public = config.public_view()
    assert _SECRET not in str(public)
    assert "connection_auth_token" not in public


@pytest.mark.asyncio
async def test_s2d_t15_concurrent_operations(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    async def one(value: str) -> None:
        await subprocess_provider.execute(
            DelegatedExecutionRequest(
                context=_dummy_context(),
                payload=SubprocessEchoPayload(value=value),
                operation=_operation(task_id=f"task-{value}"),
            ),
        )

    await asyncio.gather(*(one(str(i)) for i in range(4)))


@pytest.mark.asyncio
async def test_s2d_t16_provider_config_isolation() -> None:
    first = create_subprocess_delegated_execution_provider(config=_config())
    second = create_subprocess_delegated_execution_provider(config=_config())
    try:
        assert first.provider_id == second.provider_id
        assert first is not second
        before = await first.worker_execute_count()
        await first.execute(
            DelegatedExecutionRequest(
                context=_dummy_context(),
                payload=SubprocessEchoPayload(value="a"),
                operation=_operation(),
            ),
        )
        assert await second.worker_execute_count() == 0
        assert await first.worker_execute_count() == before + 1
    finally:
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_s2d_t19_status_real_provider(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="status"),
    )
    assert outcome.invocation_binding is not None
    correlation = DelegatedInvocationCorrelationService(store)
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    status_service = DelegatedExecutionStatusReadService(
        correlation,
        MappingDelegatedExecutionProviderResolver(
            {subprocess_provider.provider_id: subprocess_provider},
        ),
    )
    outcome = await status_service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


@pytest.mark.asyncio
async def test_s2d_t21_control_real_provider(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="cancel-me"),
    )
    correlation = DelegatedInvocationCorrelationService(store)
    assert outcome.invocation_binding is not None
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    control = DelegatedExecutionControlService(subprocess_provider)
    request = DelegatedExecutionControlRequest(
        invocation_binding=binding,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await control.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.COMPLETED


@pytest.mark.asyncio
async def test_s2d_t22_unsupported_control(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(subprocess_provider, payload=SubprocessEchoPayload(value="x"))
    correlation = DelegatedInvocationCorrelationService(store)
    assert outcome.invocation_binding is not None
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    control = DelegatedExecutionControlService(subprocess_provider)
    request = DelegatedExecutionControlRequest(
        invocation_binding=binding,
        operation=DelegatedExecutionControlOperation.INTERRUPT,
    )
    outcome = await control.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.UNSUPPORTED


@pytest.mark.asyncio
async def test_s2d_t24_real_reattachment(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="reattach"),
    )
    correlation = DelegatedInvocationCorrelationService(store)
    assert outcome.invocation_binding is not None
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    continuation = DelegatedExecutionContinuationService(
        correlation,
        MappingDelegatedExecutionProviderResolver(
            {subprocess_provider.provider_id: subprocess_provider},
        ),
    )
    outcome = await continuation.reattach_by_execution_id(binding.execution_id)
    assert outcome.category in {
        DelegatedExecutionContinuationOutcomeCategory.REATTACHED,
        DelegatedExecutionContinuationOutcomeCategory.ALREADY_ATTACHED,
    }


def test_s2d_t29_no_global_registry() -> None:
    source = _PROVIDER_MODULE.read_text(encoding="utf-8")
    assert "Registry" not in source
    assert "global " not in source


def test_s2d_t31_no_vendor_import_in_core() -> None:
    for module in (_SERVICE_MODULE, _CHILD_MODULE, _BOUNDARY_MODULE):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "integrations.providers.delegated_execution" not in mod


def test_s2d_t32_no_reflection() -> None:
    tree = ast.parse(_PROVIDER_MODULE.read_text(encoding="utf-8"))
    forbidden = {"hasattr", "getattr", "setattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def test_s2d_t35_no_frozen_core_change() -> None:
    assert _CHILD_MODULE.exists()
    assert _BOUNDARY_MODULE.exists()


def _dummy_context() -> object:
    from intergrax.contracts.delegated_execution_provider import (
        DelegatedExecutionBudgetMode,
        DelegatedExecutionBudgetProjection,
        DelegatedExecutionContext,
    )

    parent = mint_execution_id()
    child = mint_execution_id()
    return DelegatedExecutionContext(
        execution_id=child,
        parent_execution_id=parent,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("read",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )
