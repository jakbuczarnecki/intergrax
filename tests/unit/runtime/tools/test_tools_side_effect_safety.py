# © Artur Czarnecki. All rights reserved.

"""TOOLS-SIDE-EFFECT-SAFETY — idempotency identity, retry safety, outcome states."""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.idempotency_store import (
    ActiveInvocationClaimError,
    ClaimOutcome,
    IdempotencyOperationConflictError,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
    DeclarativePolicyViolationError,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlScopeAssignmentState,
    UniqueDeclarativeHitlCandidate,
    maybe_assign_declarative_hitl_scope,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker, ToolInvocationAdmission
from intergrax.runtime.tools.idempotent_invoker import IdempotentToolInvoker
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.core.contracts import (
    SideEffectRetrySafety,
    ToolContract,
    ToolRetryPolicy,
)
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_HITL_TASK_ID = mint_task_id()
_HITL_RUN_ID = mint_run_id()
_HITL_TOOL_ID = "governance.hitl.tool"
_HITL_RULE_ID = "governance.hitl.rule"


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        self.calls += 1
        return ValueOutput(result=request.input.value * 2)


class DummyHandler:
    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        return ValueOutput(result=request.input.value * 2)


class DummyState:
    def __init__(self, tenant_id: str = "tenant_test") -> None:
        self._tenant_id = tenant_id
        self.run_id = "run1"

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def context(self):
        return type("Ctx", (), {"config": type("Cfg", (), {"policy_bundle": None})()})()

    def trace_event(self, *args, **kwargs) -> None:
        del args, kwargs


class GovernanceDummyState(DummyState):
    def __init__(
        self,
        *,
        tenant_id: str = "tenant_test",
        run_id: str = _HITL_RUN_ID,
        task_id: str = _HITL_TASK_ID,
    ) -> None:
        DummyState.__init__(self, tenant_id=tenant_id)
        self.run_id = run_id
        self.request = type("Req", (), {"task_id": task_id})()
        self.declarative_hitl_grant: DeclarativeHitlApprovalGrant | None = None
        self._governance_context = type(
            "Ctx",
            (),
            {"config": type("Cfg", (), {"policy_bundle": None})()},
        )()

    @property
    def context(self):
        return self._governance_context

    @property
    def task_id(self) -> str:
        return self.request.task_id


class AllowAllScopePolicy:
    def is_allowed(self, *args, **kwargs) -> bool:
        return True


def _register_tool(
    registry: ToolRegistry,
    *,
    tool_id: str,
    side_effects: bool,
    retry_policy: ToolRetryPolicy = ToolRetryPolicy(),
    side_effect_retry_safety: SideEffectRetrySafety = SideEffectRetrySafety.NOT_RETRY_SAFE,
    timeout_ms: int = 30_000,
) -> None:
    registry.register(
        contract=ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=tool_id,
            input_schema=ValueInput,
            output_schema=ValueOutput,
            error_mapping={},
            side_effects=side_effects,
            retry_policy=retry_policy,
            side_effect_retry_safety=side_effect_retry_safety,
            timeout_ms=timeout_ms,
        ),
        handler=DummyHandler(),
    )


def _idempotent_invoker(
    executor: CountingExecutor,
    *,
    tools: tuple[str, ...] = ("tool_a",),
    side_effects: bool = True,
    store: InMemoryIdempotencyStore | None = None,
) -> tuple[IdempotentToolInvoker, InMemoryIdempotencyStore]:
    ledger = store or InMemoryIdempotencyStore()
    registry = ToolRegistry()
    for tool_id in tools:
        _register_tool(registry, tool_id=tool_id, side_effects=side_effects)
    base = RuntimeToolInvoker(registry=registry, executor=executor)
    invoker = IdempotentToolInvoker(
        base_invoker=base,
        idempotency_store=ledger,
    )
    return invoker, ledger


def _request(
    *,
    tool_id: str = "tool_a",
    value: int = 5,
    key: str = "key-x",
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id=tool_id,
        input=ValueInput(value=value),
        idempotency_key=key,
    )


def test_cross_tool_same_key_fails_closed() -> None:
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor, tools=("tool_a", "tool_b"))
    state = DummyState()

    invoker.invoke(state=state, agent_id="agent", request=_request(tool_id="tool_a"))
    with pytest.raises(IdempotencyOperationConflictError):
        invoker.invoke(state=state, agent_id="agent", request=_request(tool_id="tool_b"))
    assert executor.calls == 1


def test_cross_input_same_tool_key_fails_closed() -> None:
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor)
    state = DummyState()

    invoker.invoke(state=state, agent_id="agent", request=_request(value=5))
    with pytest.raises(IdempotencyOperationConflictError):
        invoker.invoke(state=state, agent_id="agent", request=_request(value=9))
    assert executor.calls == 1


def test_same_canonical_input_replays() -> None:
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor)
    state = DummyState()
    request = _request(value=5)

    r1 = invoker.invoke(state=state, agent_id="agent", request=request)
    r2 = invoker.invoke(state=state, agent_id="agent", request=request)
    assert r1.success and r2.success
    assert r1.output == r2.output
    assert executor.calls == 1


def test_read_only_tool_retries_per_policy() -> None:
    contract = ToolContract(
        tool_id="read_tool",
        name="read_tool",
        description="read",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=42)

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    state = DummyState()
    request = ToolExecutionRequest(
        run_id="run_retry",
        tool_id="read_tool",
        step_id="1",
        input=ValueInput(value=1),
    )
    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success
    assert calls["n"] == 3


def test_side_effect_tool_without_retry_proof_executes_once() -> None:
    contract = ToolContract(
        tool_id="mutate_tool",
        name="mutate_tool",
        description="mutate",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            calls["n"] += 1
            raise RuntimeError("transient")

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    state = DummyState()
    request = ToolExecutionRequest(
        run_id="run_mutate",
        tool_id="mutate_tool",
        step_id="1",
        input=ValueInput(value=1),
    )
    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert calls["n"] == 1


def test_side_effect_tool_explicit_retry_safe_retries() -> None:
    contract = ToolContract(
        tool_id="retry_safe_tool",
        name="retry_safe_tool",
        description="retry safe",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=7)

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    state = DummyState()
    request = ToolExecutionRequest(
        run_id="run_safe_retry",
        tool_id="retry_safe_tool",
        step_id="1",
        input=ValueInput(value=1),
    )
    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success
    assert calls["n"] == 3


def test_side_effect_timeout_marks_uncertain_and_blocks_replay() -> None:
    contract = ToolContract(
        tool_id="slow_tool",
        name="slow_tool",
        description="slow",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        timeout_ms=50,
    )

    class SlowExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            time.sleep(0.2)
            return ValueOutput(result=1)

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="slow_tool", side_effects=True, timeout_ms=50)
    base = RuntimeToolInvoker(registry=registry, executor=SlowExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(tool_id="slow_tool", key="timeout-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT
    assert store.get_status("tenant_test", "timeout-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)


def test_validation_failure_before_claim_has_no_ledger_state() -> None:
    executor = CountingExecutor()
    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(registry=registry, executor=executor)
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    bad_request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="tool_a",
        input=ValueOutput(result=1),
        idempotency_key="validation-key",
    )
    r1 = invoker.invoke(state=state, agent_id="agent", request=bad_request)
    r2 = invoker.invoke(state=state, agent_id="agent", request=bad_request)
    assert r1.success is False
    assert r2.success is False
    assert executor.calls == 0
    assert store.get_status("tenant_test", "validation-key") is None


def test_scope_denial_before_claim_leaves_ledger_untouched() -> None:
    executor = CountingExecutor()
    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        scope_policy=StaticToolScopePolicy(allowed_tools=set()),
    )
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="scope-deny-key")

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(state=state, agent_id="agent", request=request)

    assert executor.calls == 0
    assert store.get_status("tenant_test", "scope-deny-key") is None

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert executor.calls == 0


def test_output_validation_failure_after_executor_marks_uncertain() -> None:
    external_effects = {"n": 0}

    class BadOutputExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> BaseModel:
            external_effects["n"] += 1
            return ValueInput(value=999)

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(registry=registry, executor=BadOutputExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="bad-output-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert external_effects["n"] == 1
    assert store.get_status("tenant_test", "bad-output-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert external_effects["n"] == 1


def test_mapped_validation_error_after_executor_marks_uncertain() -> None:
    external_effects = {"n": 0}

    class MutatingFailExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            external_effects["n"] += 1
            raise ValueError("mutation already applied")

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    registry.register(
        contract=ToolContract(
            tool_id="tool_a",
            name="tool_a",
            description="tool_a",
            input_schema=ValueInput,
            output_schema=ValueOutput,
            error_mapping={ValueError: RuntimeErrorCode.VALIDATION_ERROR},
            side_effects=True,
        ),
        handler=DummyHandler(),
    )
    base = RuntimeToolInvoker(registry=registry, executor=MutatingFailExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="mapped-validation-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.VALIDATION_ERROR
    assert external_effects["n"] == 1
    assert store.get_status("tenant_test", "mapped-validation-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert external_effects["n"] == 1


def test_unknown_executor_failure_marks_uncertain() -> None:
    calls = {"n": 0}

    class FailExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            calls["n"] += 1
            raise RuntimeError("boom")

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(registry=registry, executor=FailExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="fail-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert store.get_status("tenant_test", "fail-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert calls["n"] == 1


def test_stale_owner_cannot_mark_uncertain() -> None:
    store = InMemoryIdempotencyStore()
    acquired = store.claim(
        "tenant-a",
        "uncertain-key",
        "owner-a",
        lease_seconds=30,
        operation_identity=None,
    )
    assert acquired.claim is not None
    stale_claim = InvocationClaim(
        tenant_id="tenant-a",
        key="uncertain-key",
        owner_id="owner-a",
        lease_expires_at=acquired.claim.lease_expires_at,
        fence=1,
    )
    entry = store._store[("tenant-a", "uncertain-key")]  # noqa: SLF001
    entry.claim = acquired.claim.model_copy(update={"fence": 2, "owner_id": "owner-b"})
    with pytest.raises(StaleClaimError):
        store.mark_uncertain_with_claim("tenant-a", "uncertain-key", stale_claim)


def test_active_claim_blocks_concurrent_owner() -> None:
    store = InMemoryIdempotencyStore()
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor, store=store)
    state = DummyState()
    request = _request()

    store.claim("tenant_test", "key-x", "owner-a", lease_seconds=30)
    with pytest.raises(ActiveInvocationClaimError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert executor.calls == 0


def test_completed_replays_with_matching_operation_identity() -> None:
    store = InMemoryIdempotencyStore()
    executor = CountingExecutor()
    invoker, ledger = _idempotent_invoker(executor, store=store)
    state = DummyState()
    request = _request()
    operation_identity = compute_invocation_operation_identity(
        request.tool_id,
        request.input,
    )

    invoker.invoke(state=state, agent_id="agent", request=request)
    outcome = ledger.claim(
        "tenant_test",
        "key-x",
        "owner-b",
        lease_seconds=30,
        operation_identity=operation_identity,
    )
    assert outcome.outcome == ClaimOutcome.REPLAY_COMPLETED


def _policy_bundle(*, action: str, tool_id: str, rule_id: str) -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=f"governance.{action}")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": rule_id,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": tool_id,
                "action": action,
            }
        ],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _hitl_grant(*, bundle: object, key: str) -> DeclarativeHitlApprovalGrant:
    provenance = bundle.declarative_policy_runtime.provenance.rules_digest_sha256
    return DeclarativeHitlApprovalGrant(
        grant_id="grant-governance-first",
        invocation_scope_id="dhr_scope",
        task_id=_HITL_TASK_ID,
        run_id=_HITL_RUN_ID,
        step_id="step1",
        tool_id=_HITL_TOOL_ID,
        idempotency_key=key,
        matched_rule_ids=(_HITL_RULE_ID,),
        human_request_id="hr-governance-first",
        policy_provenance_digest=provenance,
        pause_id="pause-governance-first",
        approved_at="2026-08-30T00:00:00+00:00",
    )


def _governance_idempotent_invoker(
    executor: CountingExecutor,
    *,
    store: InMemoryIdempotencyStore | None = None,
) -> tuple[IdempotentToolInvoker, InMemoryIdempotencyStore, RuntimeToolInvoker]:
    ledger = store or InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id=_HITL_TOOL_ID, side_effects=True)
    base = RuntimeToolInvoker(registry=registry, executor=executor)
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=ledger)
    return invoker, ledger, base


def test_concurrent_governance_allow_executes_once() -> None:
    executor = CountingExecutor()
    invoker, _, _ = _governance_idempotent_invoker(executor)
    state = DummyState()
    request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id=_HITL_TOOL_ID,
        input=ValueInput(value=3),
        idempotency_key="concurrency-key",
    )
    barrier = threading.Barrier(2)
    errors: list[Exception] = []
    results: list[ToolExecutionResult[ValueOutput]] = []

    def worker() -> None:
        try:
            barrier.wait()
            results.append(
                invoker.invoke(state=state, agent_id="agent", request=request)
            )
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert executor.calls == 1
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ActiveInvocationClaimError)


def test_deny_then_allow_executes_once_without_cached_denial() -> None:
    executor = CountingExecutor()
    invoker, store, _ = _governance_idempotent_invoker(executor)
    state = GovernanceDummyState()
    deny_bundle = _policy_bundle(
        action="deny",
        tool_id=_HITL_TOOL_ID,
        rule_id=_HITL_RULE_ID,
    )
    state.context.config.policy_bundle = deny_bundle
    request = ToolExecutionRequest(
        run_id=_HITL_RUN_ID,
        step_id="step1",
        tool_id=_HITL_TOOL_ID,
        input=ValueInput(value=4),
        idempotency_key="deny-then-allow-key",
    )

    with pytest.raises(DeclarativePolicyViolationError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert executor.calls == 0
    assert store.get_status(state.tenant_id, "deny-then-allow-key") is None

    allow_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="governance.allow")
    allow_env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    state.context.config.policy_bundle = wire_policy_bundle(allow_env)

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success
    assert executor.calls == 1
    assert store.get_status(state.tenant_id, "deny-then-allow-key") == InvocationStatus.COMPLETED


def test_hitl_resume_claims_and_executes_once() -> None:
    executor = CountingExecutor()
    invoker, store, _ = _governance_idempotent_invoker(executor)
    state = GovernanceDummyState()
    bundle = _policy_bundle(
        action="require_hitl",
        tool_id=_HITL_TOOL_ID,
        rule_id=_HITL_RULE_ID,
    )
    state.context.config.policy_bundle = bundle
    base_request = ToolExecutionRequest(
        run_id=_HITL_RUN_ID,
        step_id="step1",
        tool_id=_HITL_TOOL_ID,
        input=ValueInput(value=7),
        idempotency_key="hitl-resume-key",
    )

    with pytest.raises(DeclarativePolicyHitlRequiredError):
        invoker.invoke(state=state, agent_id="agent", request=base_request)
    assert executor.calls == 0
    assert store.get_status(state.tenant_id, "hitl-resume-key") is None

    state.declarative_hitl_grant = _hitl_grant(
        bundle=bundle,
        key="hitl-resume-key",
    )
    request = maybe_assign_declarative_hitl_scope(
        base_request,
        state=state,
        assignment_state=DeclarativeHitlScopeAssignmentState(),
        unique_candidate=UniqueDeclarativeHitlCandidate(candidate_index=0),
        request_index=0,
    )
    first = invoker.invoke(state=state, agent_id="agent", request=request)
    assert first.success
    assert executor.calls == 1
    assert store.get_status(state.tenant_id, "hitl-resume-key") == InvocationStatus.COMPLETED

    replay = invoker.invoke(state=state, agent_id="agent", request=request)
    assert replay.success
    assert executor.calls == 1


def _deny_governance_bundle() -> object:
    return _policy_bundle(
        action="deny",
        tool_id=_HITL_TOOL_ID,
        rule_id=_HITL_RULE_ID,
    )


def _governance_request(
    *,
    value: int = 5,
    key: str = "admission-key",
    run_id: str = _HITL_RUN_ID,
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id=run_id,
        step_id="step1",
        tool_id=_HITL_TOOL_ID,
        input=ValueInput(value=value),
        idempotency_key=key,
    )


def test_manual_admission_forgery_cannot_execute() -> None:
    executor = CountingExecutor()
    _, _, base = _governance_idempotent_invoker(executor)
    state = GovernanceDummyState()
    state.context.config.policy_bundle = _deny_governance_bundle()
    request = _governance_request()

    with pytest.raises(TypeError, match="cannot be constructed directly"):
        ToolInvocationAdmission(
            agent_id="agent",
            tool_id=_HITL_TOOL_ID,
            operation_identity=compute_invocation_operation_identity(
                request.tool_id,
                request.input,
            ),
        )

    forged = ToolInvocationAdmission._mint(
        agent_id="agent",
        tool_id=request.tool_id,
        operation_identity=compute_invocation_operation_identity(
            request.tool_id,
            request.input,
        ),
        tenant_id=state.tenant_id,
        run_id=request.run_id,
        task_id=state.task_id,
        mint=object(),
    )
    with pytest.raises(RuntimeError, match="Invalid tool invocation admission token"):
        base._execute_after_admission(
            state=state,
            agent_id="agent",
            request=request,
            admission=forged,
        )
    assert executor.calls == 0


def test_admission_cannot_cross_operation() -> None:
    executor = CountingExecutor()
    _, _, base = _governance_idempotent_invoker(executor)
    state = GovernanceDummyState()
    allow_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="governance.allow")
    allow_env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    state.context.config.policy_bundle = wire_policy_bundle(allow_env)
    request_x = _governance_request(value=3, key="cross-op-key")
    admission = base.admit(state=state, agent_id="agent", request=request_x)
    assert not isinstance(admission, ToolExecutionResult)

    request_y = _governance_request(value=9, key="cross-op-key")
    with pytest.raises(RuntimeError, match="operation identity mismatch"):
        base._execute_after_admission(
            state=state,
            agent_id="agent",
            request=request_y,
            admission=admission,
        )
    assert executor.calls == 0


def test_admission_cannot_cross_agent() -> None:
    executor = CountingExecutor()
    _, _, base = _governance_idempotent_invoker(executor)
    state = GovernanceDummyState()
    allow_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="governance.allow")
    allow_env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    state.context.config.policy_bundle = wire_policy_bundle(allow_env)
    request = _governance_request()
    admission = base.admit(state=state, agent_id="agent-a", request=request)
    assert not isinstance(admission, ToolExecutionResult)

    with pytest.raises(RuntimeError, match="agent mismatch"):
        base._execute_after_admission(
            state=state,
            agent_id="agent-b",
            request=request,
            admission=admission,
        )
    assert executor.calls == 0


def test_admission_cannot_cross_runtime_state() -> None:
    executor = CountingExecutor()
    _, _, base = _governance_idempotent_invoker(executor)
    allow_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="governance.allow")
    allow_env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(allow_env)
    state_a = GovernanceDummyState(tenant_id="tenant-a")
    state_a.context.config.policy_bundle = bundle
    state_b = GovernanceDummyState(tenant_id="tenant-b")
    state_b.context.config.policy_bundle = bundle
    request = _governance_request()
    admission = base.admit(state=state_a, agent_id="agent", request=request)
    assert not isinstance(admission, ToolExecutionResult)

    with pytest.raises(RuntimeError, match="tenant mismatch"):
        base._execute_after_admission(
            state=state_b,
            agent_id="agent",
            request=request,
            admission=admission,
        )
    assert executor.calls == 0


def test_admission_cannot_cross_invoker() -> None:
    executor_a = CountingExecutor()
    executor_b = CountingExecutor()
    _, _, base_a = _governance_idempotent_invoker(executor_a)
    _, _, base_b = _governance_idempotent_invoker(executor_b)
    state = GovernanceDummyState()
    allow_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="governance.allow")
    allow_env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    state.context.config.policy_bundle = wire_policy_bundle(allow_env)
    request = _governance_request()
    admission = base_a.admit(state=state, agent_id="agent", request=request)
    assert not isinstance(admission, ToolExecutionResult)

    with pytest.raises(RuntimeError, match="Invalid tool invocation admission token"):
        base_b._execute_after_admission(
            state=state,
            agent_id="agent",
            request=request,
            admission=admission,
        )
    assert executor_a.calls == 0
    assert executor_b.calls == 0
