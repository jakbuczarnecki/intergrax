# © Artur Czarnecki. All rights reserved.
"""TOOL-LIFE-X Batch B — real ``require_meaningful_side_effect_authorization`` evidence."""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from testing_support.builder import (
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.policy.declarative_tool_authorization_gate import (
    require_meaningful_side_effect_authorization,
)
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
    SideEffectAuthorizationFailureReason,
)
from intergrax.tools.core.contracts import (
    SideEffectRetrySafety,
    ToolContract,
    ToolRetryPolicy,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RUN_ID = canonical_run_id_for_tests("tool-life-x-batch-b-governance")
_AGENT_ID = "batch-b-governance-agent"


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class AllowAllScopePolicy:
    def is_allowed(self, *args: object, **kwargs: object) -> bool:
        return True


class RevocableScopePolicy:
    def __init__(self) -> None:
        self._revoked = False

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        del agent_id, tool_id
        return not self._revoked

    def revoke(self) -> None:
        self._revoked = True


@contextmanager
def _governed_scope():
    with canonical_governed_execution_scope(
        _RUN_ID,
        governance_tenant_id="tenant_test",
        governance_principal_id="principal-batch-b-governance",
    ):
        yield


@pytest.fixture(autouse=True)
def _active_governance_identity() -> None:
    with _governed_scope():
        yield


def _enforce_allow_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="batch-b.gov.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _audit_only_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="batch-b.gov.audit")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="audit_only",
    )
    return wire_policy_bundle(env)


class DummyState:
    def __init__(self, *, policy_bundle: object | None = None) -> None:
        self.run_id = _RUN_ID
        self.tenant_id = "tenant_test"
        self.declarative_hitl_grant = None
        self.request = type("Req", (), {"metadata": {}})()
        self._context = type(
            "Ctx",
            (),
            {
                "config": type(
                    "Cfg",
                    (),
                    {"policy_bundle": policy_bundle, "production_mode": False},
                )()
            },
        )()

    @property
    def context(self) -> object:
        return self._context

    @property
    def task_id(self) -> str | None:
        return None

    def trace_event(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _request(
    *,
    tool_id: str,
    key: str | None = None,
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=tool_id,
        input=ValueInput(value=1),
        idempotency_key=key,
    )


def _side_effect_contract(tool_id: str = "batch_b.gov.mutate") -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name="mutate",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )


def test_batch_b_governance_unauthorized_side_effect_zero_executor_calls() -> None:
    calls = {"n": 0}

    class CountingExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            return ValueOutput(result=1)

    contract = _side_effect_contract()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=CountingExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError) as exc:
        invoker.invoke(
            state=DummyState(policy_bundle=None),
            agent_id=_AGENT_ID,
            request=_request(tool_id=contract.tool_id, key="gov-no-bundle"),
        )
    assert exc.value.reason is SideEffectAuthorizationFailureReason.NOT_CONFIGURED
    assert calls["n"] == 0


def test_batch_b_governance_non_enforcing_mode_blocks_before_executor() -> None:
    calls = {"n": 0}

    class CountingExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            return ValueOutput(result=1)

    contract = _side_effect_contract("batch_b.gov.audit_only")
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=CountingExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError) as exc:
        invoker.invoke(
            state=DummyState(policy_bundle=_audit_only_bundle()),
            agent_id=_AGENT_ID,
            request=_request(tool_id=contract.tool_id),
        )
    assert exc.value.reason is SideEffectAuthorizationFailureReason.NON_ENFORCING_MODE
    assert calls["n"] == 0


def test_batch_b_governance_real_gate_invoked_on_each_retry_attempt() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=7)

    contract = _side_effect_contract("batch_b.gov.retry_safe")
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    gate_calls = {"n": 0}
    real_gate = require_meaningful_side_effect_authorization

    def _counting_gate(**kwargs: object) -> None:
        gate_calls["n"] += 1
        real_gate(**kwargs)  # type: ignore[arg-type]

    with patch(
        "intergrax.runtime.nexus.tools.invoker.require_meaningful_side_effect_authorization",
        side_effect=_counting_gate,
    ):
        result = invoker.invoke(
            state=DummyState(policy_bundle=_enforce_allow_bundle()),
            agent_id=_AGENT_ID,
            request=_request(tool_id=contract.tool_id),
        )
    assert result.success
    assert calls["n"] == 3
    assert gate_calls["n"] == 3


def test_batch_b_governance_scope_revoked_during_backoff_blocks_retry() -> None:
    calls = {"n": 0}
    scope = RevocableScopePolicy()

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="batch_b.gov.backoff_revoke",
        name="mutate",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=50),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=scope,
    )

    def _sleep_then_revoke(
        _seconds: float,
        *,
        should_abort: object = None,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        del should_abort, poll_interval_seconds
        scope.revoke()
        time.sleep(0)

    with patch(
        "intergrax.runtime.nexus.tools.invoker.cooperative_delay_seconds",
        side_effect=_sleep_then_revoke,
    ):
        with pytest.raises(ToolScopeViolationError):
            invoker.invoke(
                state=DummyState(policy_bundle=_enforce_allow_bundle()),
                agent_id=_AGENT_ID,
                request=_request(tool_id=contract.tool_id),
            )
    assert calls["n"] == 1
