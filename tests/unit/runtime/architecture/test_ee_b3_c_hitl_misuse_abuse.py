# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-10 HITL approval misuse abuse."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlScopeAssignmentState,
    UniqueDeclarativeHitlCandidate,
    maybe_assign_declarative_hitl_scope,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_TOOL_ID = "b3c.hitl.tool"
_RULE_ID = "b3c.hitl.rule"


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        self.calls += 1
        return ValueOutput(result=request.input.value)


class DummyHandler:
    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        return ValueOutput(result=request.input.value)


class _HitlConfig:
    policy_bundle: object | None = None
    production_mode = False


class _HitlContext:
    config = _HitlConfig()


class HitlState:
    run_id = _RUN_ID
    task_id = _TASK_ID
    tenant_id = "tenant-b3c"
    declarative_hitl_grant: DeclarativeHitlApprovalGrant | None = None
    context = _HitlContext()

    def trace_event(self, **kwargs: object) -> None:
        del kwargs


def _hitl_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="b3c.hitl")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": _RULE_ID,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
    )
    return wire_policy_bundle(env)


def _hitl_grant(*, bundle: object, key: str) -> DeclarativeHitlApprovalGrant:
    runtime = bundle.declarative_policy_runtime  # type: ignore[attr-defined]
    provenance = runtime.provenance.rules_digest_sha256
    return DeclarativeHitlApprovalGrant(
        grant_id="grant-b3c",
        invocation_scope_id="dhr_scope",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=_TOOL_ID,
        agent_id="agent-a",
        idempotency_key=key,
        matched_rule_ids=(_RULE_ID,),
        human_request_id="hr-b3c",
        policy_provenance_digest=provenance,
        pause_id="pause-b3c",
        approved_at="2026-09-14T00:00:00+00:00",
    )


def test_ee_b3_c_hitl_grant_for_other_idempotency_key_does_not_authorize() -> None:
    executor = CountingExecutor()
    registry = ToolRegistry()
    registry.register(
        contract=ToolContract(
            tool_id=_TOOL_ID,
            name=_TOOL_ID,
            description=_TOOL_ID,
            input_schema=ValueInput,
            output_schema=ValueOutput,
            error_mapping={},
            side_effects=True,
        ),
        handler=DummyHandler(),  # type: ignore[arg-type]
    )
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,  # type: ignore[arg-type]
        pre_effect_coordinator=IdempotencyPreEffectCoordinator(
            idempotency_store=InMemoryIdempotencyStore(),
        ),
    )
    bundle = _hitl_bundle()
    state = HitlState()
    state.context.config.policy_bundle = bundle
    state.declarative_hitl_grant = _hitl_grant(bundle=bundle, key="approved-key")
    base_request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=_TOOL_ID,
        input=ValueInput(value=1),
        idempotency_key="different-key",
    )
    scoped = maybe_assign_declarative_hitl_scope(
        base_request,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        assignment_state=DeclarativeHitlScopeAssignmentState(),
        unique_candidate=UniqueDeclarativeHitlCandidate(candidate_index=0),
        request_index=0,
    )
    with pytest.raises(DeclarativePolicyHitlRequiredError):
        invoker.invoke(state=state, agent_id="agent-a", request=scoped)  # type: ignore[arg-type]
    assert executor.calls == 0
