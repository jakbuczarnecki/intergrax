# © Artur Czarnecki. All rights reserved.

"""P2.1-S2B-C2/C3 — S2A invocation binding issuance and provider rejection."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionCancelProvider,
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    delegated_control_outcome,
)
from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
    assert_provider_outcome_has_no_invocation_binding,
    enrich_delegated_outcome_with_platform_invocation_binding,
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
    assert_provider_native_ids_distinct_from_execution,
    DelegatedExecutionContractError,
    delegated_failure_outcome,
    delegated_success_outcome,
    digest_delegated_execution_payload,
    digest_delegated_execution_request,
    mint_delegated_provider_invocation,
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
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.delegated_execution.control_service import (
    DelegatedExecutionControlService,
)
from intergrax.runtime.execution.delegated_execution.local_provider import (
    LocalDelegatedExecutionProvider,
)
from intergrax.runtime.execution.delegated_execution.service import (
    delegated_execution_service,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SERVICE_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "delegated_execution" / "service.py"
)
_PROVIDER_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "delegated_execution_provider.py"
_T0 = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


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
            "task_id": "task-binding",
        }
    )


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _invocation_for_request(
    *,
    provider_id: str,
    request: DelegatedExecutionRequest[EchoPayload],
    invocation_id: str = "inv-s2a-1",
) -> ProviderInvocation:
    payload_digest = digest_delegated_execution_payload(request.payload)
    request_digest = digest_delegated_execution_request(
        context=request.context,
        operation=request.operation,
        payload_digest=payload_digest,
    )
    return mint_delegated_provider_invocation(
        context=request.context,
        operation=request.operation,
        provider_id=provider_id,
        request_digest=request_digest,
        started_at=_T0,
        invocation_id=invocation_id,
        provider_request_id="preq-s2a-1",
        provider_operation_id="pop-s2a-1",
    )


class FakeS2AProvider(DelegatedExecutionProvider[EchoPayload, EchoResult]):
    """Contract fake — cannot mint platform invocation binding."""

    def __init__(self, *, provider_id: str = "fake_s2a") -> None:
        self._provider_id = provider_id
        self.last_request: DelegatedExecutionRequest[EchoPayload] | None = None

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(provider_id=self._provider_id)

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        self.last_request = request
        invocation = _invocation_for_request(provider_id=self._provider_id, request=request)
        provider_outcome = ProviderInvocationOutcome.model_validate(
            {
                "invocation_id": invocation.invocation_id,
                "status": ProviderInvocationStatus.SUCCEEDED,
                "completed_at": _T0.isoformat(),
                "provider_request_id": invocation.provider_request_id,
                "provider_operation_id": invocation.provider_operation_id,
            }
        )
        return delegated_success_outcome(
            result=EchoResult(value=request.payload.value),
            provider_invocation=invocation,
            provider_outcome=provider_outcome,
        )


class FakeCancelS2AProvider(
    FakeS2AProvider,
    DelegatedExecutionCancelProvider,
):
    def __init__(self) -> None:
        super().__init__(provider_id="fake_s2a_cancel")
        self.cancel_requests: list[DelegatedExecutionControlRequest] = []

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(
            provider_id=self.provider_id,
            supports_cancel=True,
        )

    async def cancel_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> object:
        self.cancel_requests.append(request)
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.ACCEPTED,
            request=request,
            provider_id=self.provider_id,
        )


async def _run_s2a(
    provider: DelegatedExecutionProvider[EchoPayload, EchoResult],
    *,
    payload: EchoPayload | None = None,
    require_success: bool = True,
) -> DelegatedExecutionOutcome[EchoResult]:
    service = delegated_execution_service(provider, ledger=_UNLIMITED_LEDGER)
    captured: list[DelegatedExecutionOutcome[EchoResult]] = []

    class RootDelegate:
        async def execute(self, request: EchoPayload) -> EchoResult:
            outcome = await service.execute_delegated(
                payload=request,
                operation=_operation(),
            )
            captured.append(outcome)
            if require_success:
                assert outcome.result is not None
            elif outcome.result is not None:
                return outcome.result
            return EchoResult(value="failed")

    await ExecutionBoundary[EchoPayload, EchoResult](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.scoped(("read", "write")),
    ).execute(payload or EchoPayload(value="binding"))

    assert len(captured) == 1
    return captured[0]


@pytest.mark.asyncio
async def test_c2_t1_s2a_issues_binding() -> None:
    provider = FakeS2AProvider()
    outcome = await _run_s2a(provider)
    assert outcome.invocation_binding is not None
    assert isinstance(outcome.invocation_binding, DelegatedExecutionInvocationBinding)


@pytest.mark.asyncio
async def test_c2_t2_binding_execution_id_matches_child() -> None:
    provider = FakeS2AProvider()
    outcome = await _run_s2a(provider)
    assert provider.last_request is not None
    binding = outcome.invocation_binding
    assert binding is not None
    assert binding.execution_id == provider.last_request.context.execution_id


@pytest.mark.asyncio
async def test_c2_t3_binding_parent_matches_admitted_parent() -> None:
    provider = FakeS2AProvider()
    outcome = await _run_s2a(provider)
    assert provider.last_request is not None
    binding = outcome.invocation_binding
    assert binding is not None
    assert binding.parent_execution_id == provider.last_request.context.parent_execution_id


@pytest.mark.asyncio
async def test_c2_t4_binding_run_and_attempt_match_child() -> None:
    provider = FakeS2AProvider()
    outcome = await _run_s2a(provider)
    assert provider.last_request is not None
    ctx = provider.last_request.context
    binding = outcome.invocation_binding
    assert binding is not None
    assert binding.run_id == ctx.run_id
    assert binding.attempt_id == ctx.attempt_id


@pytest.mark.asyncio
async def test_c2_t5_binding_provider_invocation_matches_dispatch() -> None:
    provider = FakeS2AProvider()
    outcome = await _run_s2a(provider)
    binding = outcome.invocation_binding
    assert binding is not None
    assert outcome.provider_invocation == binding.provider_invocation


@pytest.mark.asyncio
async def test_c2_t6_binding_request_digest_proves_correlation() -> None:
    provider = FakeS2AProvider()
    outcome = await _run_s2a(provider)
    binding = outcome.invocation_binding
    assert binding is not None
    assert provider.last_request is not None
    payload_digest = digest_delegated_execution_payload(provider.last_request.payload)
    expected = digest_delegated_execution_request(
        context=provider.last_request.context,
        operation=provider.last_request.operation,
        payload_digest=payload_digest,
    )
    assert binding.provider_invocation.request_digest == expected


def test_c2_t7_cross_child_same_run_attack_fail_closed() -> None:
    child_a = _context_for_child()
    child_b = _context_for_child(run_id=child_a.run_id)
    op = _operation()
    inv_a = _invocation_for_context(child_a, op)
    with pytest.raises(ValidationError):
        mint_delegated_execution_invocation_binding(
            context=child_b,
            operation=op,
            payload_digest=digest_delegated_execution_payload(EchoPayload(value="x")),
            provider_invocation=inv_a,
        )


def _context_for_child(*, run_id: object | None = None) -> DelegatedExecutionContext:
    return DelegatedExecutionContext(
        execution_id=mint_execution_id(),
        parent_execution_id=mint_execution_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("read",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )


def _invocation_for_context(
    context: DelegatedExecutionContext,
    operation: DelegatedExecutionOperationMetadata,
) -> ProviderInvocation:
    payload_digest = digest_delegated_execution_payload(EchoPayload(value="x"))
    digest = digest_delegated_execution_request(
        context=context,
        operation=operation,
        payload_digest=payload_digest,
    )
    return mint_delegated_provider_invocation(
        context=context,
        operation=operation,
        provider_id="fake",
        request_digest=digest,
        started_at=_T0,
        invocation_id="inv-cross",
        provider_request_id="preq-cross",
        provider_operation_id="pop-cross",
    )


def test_c2_t8_changing_execution_id_invalidates_binding() -> None:
    ctx = _context_for_child()
    op = _operation()
    inv = _invocation_for_context(ctx, op)
    forged_context = DelegatedExecutionContext(
        execution_id=mint_execution_id(),
        parent_execution_id=ctx.parent_execution_id,
        run_id=ctx.run_id,
        attempt_id=ctx.attempt_id,
        authority=ctx.authority,
        budget=ctx.budget,
    )
    with pytest.raises(ValidationError):
        mint_delegated_execution_invocation_binding(
            context=forged_context,
            operation=op,
            payload_digest=digest_delegated_execution_payload(EchoPayload(value="x")),
            provider_invocation=inv,
        )


def test_c2_t9_provider_contract_cannot_mint_binding() -> None:
    tree = ast.parse(_PROVIDER_CONTRACT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.name != "mint_delegated_execution_invocation_binding"
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "mint_delegated_execution_invocation_binding"


@pytest.mark.asyncio
async def test_c2_t10_local_provider_without_service_coupling() -> None:
    class _EchoDelegate:
        async def execute(
            self,
            request: DelegatedExecutionRequest[EchoPayload],
        ) -> EchoResult:
            return EchoResult(value=request.payload.value)

    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    outcome = await _run_s2a(provider)
    assert outcome.invocation_binding is not None


@pytest.mark.asyncio
async def test_c2_t11_fake_provider_binding_still_execution_owned() -> None:
    provider = FakeS2AProvider(provider_id="arbitrary_fake")
    outcome = await _run_s2a(provider)
    assert outcome.invocation_binding is not None
    assert outcome.invocation_binding.provider_id == "arbitrary_fake"


@pytest.mark.asyncio
async def test_c2_t12_control_consumes_s2a_binding() -> None:
    provider = FakeCancelS2AProvider()
    outcome = await _run_s2a(provider)
    binding = outcome.invocation_binding
    assert binding is not None
    control = DelegatedExecutionControlService(provider)
    request = DelegatedExecutionControlRequest(
        invocation_binding=binding,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    control_outcome = await control.apply_control(request)
    assert control_outcome.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED
    assert len(provider.cancel_requests) == 1
    assert (
        provider.cancel_requests[0].invocation_binding.execution_id
        == binding.execution_id
    )


@pytest.mark.asyncio
async def test_c2_t13_spoofed_provider_outcome_fail_closed() -> None:
    class SpoofProvider(FakeS2AProvider):
        async def execute(
            self,
            request: DelegatedExecutionRequest[EchoPayload],
        ) -> DelegatedExecutionOutcome[EchoResult]:
            outcome = await super().execute(request)
            assert outcome.provider_invocation is not None
            spoofed = outcome.provider_invocation.model_copy(
                update={"request_digest": "sha256:" + ("00" * 32)},
            )
            assert outcome.provider_outcome is not None
            return delegated_success_outcome(
                result=outcome.result,
                provider_invocation=spoofed,
                provider_outcome=outcome.provider_outcome,
            )

    outcome = await _run_s2a(SpoofProvider(), require_success=False)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert outcome.failure_code == "OUTCOME_CONTRACT_MISMATCH"
    assert outcome.invocation_binding is None


def test_c2_t14_capability_gating_unchanged() -> None:
    source = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "execution"
        / "delegated_execution"
        / "control_service.py"
    ).read_text(encoding="utf-8")
    assert "supports_cancel" in source
    assert "CONTROL_UNSUPPORTED" in source


def test_c2_t15_no_global_mutable_registry() -> None:
    tree = ast.parse(_SERVICE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    assert not isinstance(node.value, ast.Dict)


def test_c2_service_uses_platform_enrichment_not_local_provider() -> None:
    source = _SERVICE_MODULE.read_text(encoding="utf-8")
    assert "enrich_delegated_outcome_with_platform_invocation_binding" in source
    assert "local_provider" not in source
    assert "mint_delegated_execution_invocation_binding" not in source


def test_c2_no_reflection_in_service_module() -> None:
    tree = ast.parse(_SERVICE_MODULE.read_text(encoding="utf-8"))
    forbidden = {"getattr", "hasattr", "setattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def _mint_binding_for_request(
    request: DelegatedExecutionRequest[EchoPayload],
    *,
    provider_id: str,
) -> DelegatedExecutionInvocationBinding:
    invocation = _invocation_for_request(provider_id=provider_id, request=request)
    return mint_delegated_execution_invocation_binding(
        context=request.context,
        operation=request.operation,
        payload_digest=digest_delegated_execution_payload(request.payload),
        provider_invocation=invocation,
    )


class MaliciousBindingInjectingProvider(FakeS2AProvider):
    """Test-only provider that attempts to self-issue platform invocation binding."""

    def __init__(
        self,
        *,
        inject_without_invocation: bool = False,
        provider_id: str = "malicious_binding",
    ) -> None:
        super().__init__(provider_id=provider_id)
        self._inject_without_invocation = inject_without_invocation

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        if self._inject_without_invocation:
            self.last_request = request
            forged = _mint_binding_for_request(request, provider_id=self._provider_id)
            return DelegatedExecutionOutcome(
                category=DelegatedExecutionOutcomeCategory.SUCCESS,
                result=EchoResult(value=request.payload.value),
                invocation_binding=forged,
            )
        outcome = await super().execute(request)
        forged = _mint_binding_for_request(request, provider_id=self._provider_id)
        return replace(outcome, invocation_binding=forged)


@pytest.mark.asyncio
async def test_c3_t1_provider_injects_binding_without_provider_invocation() -> None:
    outcome = await _run_s2a(
        MaliciousBindingInjectingProvider(inject_without_invocation=True),
        require_success=False,
    )
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert outcome.failure_code == "OUTCOME_CONTRACT_MISMATCH"
    assert outcome.invocation_binding is None


@pytest.mark.asyncio
async def test_c3_t2_provider_injects_binding_with_provider_invocation() -> None:
    outcome = await _run_s2a(MaliciousBindingInjectingProvider(), require_success=False)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert outcome.failure_code == "OUTCOME_CONTRACT_MISMATCH"
    assert outcome.invocation_binding is None


@pytest.mark.asyncio
async def test_c3_t3_provider_injects_exact_correct_binding_still_rejected() -> None:
    """Authority test: valid binding content from provider is still illegal."""
    outcome = await _run_s2a(MaliciousBindingInjectingProvider(), require_success=False)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert outcome.invocation_binding is None


@pytest.mark.asyncio
async def test_c3_t4_normal_provider_platform_mints_binding() -> None:
    outcome = await _run_s2a(FakeS2AProvider())
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    assert outcome.invocation_binding is not None


@pytest.mark.asyncio
async def test_c3_t5_no_provider_invocation_binding_remains_none() -> None:
    class NoInvocationProvider(FakeS2AProvider):
        async def execute(
            self,
            request: DelegatedExecutionRequest[EchoPayload],
        ) -> DelegatedExecutionOutcome[EchoResult]:
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
                failure_code="PROVIDER_ERROR",
                failure_message="no invocation evidence",
            )

    outcome = await _run_s2a(NoInvocationProvider(), require_success=False)
    assert outcome.invocation_binding is None


@pytest.mark.asyncio
async def test_c3_t6_platform_binding_matches_admitted_child() -> None:
    provider = FakeS2AProvider()
    outcome = await _run_s2a(provider)
    assert provider.last_request is not None
    binding = outcome.invocation_binding
    assert binding is not None
    assert binding.execution_id == provider.last_request.context.execution_id


@pytest.mark.asyncio
async def test_c3_t7_spoofed_request_digest_still_fail_closed() -> None:
    await test_c2_t13_spoofed_provider_outcome_fail_closed()


@pytest.mark.asyncio
async def test_c3_t8_control_uses_platform_issued_binding() -> None:
    await test_c2_t12_control_consumes_s2a_binding()


@pytest.mark.asyncio
async def test_c3_t9_provider_supplied_binding_never_reaches_control() -> None:
    provider = MaliciousBindingInjectingProvider()
    outcome = await _run_s2a(provider, require_success=False)
    assert outcome.invocation_binding is None
    assert provider.last_request is not None
    forged = _mint_binding_for_request(
        provider.last_request,
        provider_id=provider.provider_id,
    )
    assert outcome.invocation_binding != forged


@pytest.mark.asyncio
async def test_c3_t10_external_fake_provider_conformance() -> None:
    outcome = await _run_s2a(FakeS2AProvider(provider_id="arbitrary_fake"))
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    assert outcome.invocation_binding is not None


@pytest.mark.asyncio
async def test_c3_t11_local_provider_unchanged() -> None:
    await test_c2_t10_local_provider_without_service_coupling()


def test_c3_t12_c1_control_anti_spoofing_regression_in_control_suite() -> None:
    control_tests = (
        _REPO_ROOT
        / "tests"
        / "unit"
        / "runtime"
        / "execution"
        / "test_delegated_execution_control.py"
    )
    source = control_tests.read_text(encoding="utf-8")
    assert "CONTROL_OUTCOME_CONTRACT_MISMATCH" in source


def test_c3_t13_capability_gating_unchanged() -> None:
    test_c2_t14_capability_gating_unchanged()


def test_c3_t14_no_reflection() -> None:
    test_c2_no_reflection_in_service_module()


def test_c3_t15_no_global_mutable_registry() -> None:
    test_c2_t15_no_global_mutable_registry()


def test_c3_enrichment_helper_rejects_provider_supplied_binding() -> None:
    ctx = _context_for_child()
    op = _operation()
    inv = _invocation_for_context(ctx, op)
    payload_digest = digest_delegated_execution_payload(EchoPayload(value="x"))
    forged = mint_delegated_execution_invocation_binding(
        context=ctx,
        operation=op,
        payload_digest=payload_digest,
        provider_invocation=inv,
    )
    outcome = delegated_success_outcome(
        result=EchoResult(value="x"),
        provider_invocation=inv,
        provider_outcome=ProviderInvocationOutcome.model_validate(
            {
                "invocation_id": inv.invocation_id,
                "status": ProviderInvocationStatus.SUCCEEDED,
                "completed_at": _T0.isoformat(),
                "provider_request_id": inv.provider_request_id,
                "provider_operation_id": inv.provider_operation_id,
            }
        ),
    )
    outcome = replace(outcome, invocation_binding=forged)
    with pytest.raises(DelegatedExecutionContractError):
        enrich_delegated_outcome_with_platform_invocation_binding(
            outcome=outcome,
            context=ctx,
            operation=op,
            payload_digest=payload_digest,
        )


def test_c3_service_module_asserts_provider_binding_gate() -> None:
    source = _SERVICE_MODULE.read_text(encoding="utf-8")
    assert "assert_provider_outcome_has_no_invocation_binding" in source
