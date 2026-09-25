# © Artur Czarnecki. All rights reserved.

"""REL / U5 — reliability idempotency store must reach governed declarative tool invoker."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CANONICAL_DECLARATIVE_INVOKER_COMPOSITION = (
    _REPO_ROOT / "intergrax/applications/_shared/scenario_runtime_baseline.py",
    _REPO_ROOT / "intergrax/applications/_shared/harness_host_runtime.py",
    _REPO_ROOT / "intergrax/applications/_shared/acp_session_host_wiring.py",
    _REPO_ROOT / "applications/local_workspace_application/host/factory.py",
)

_RUN_SEED = "rel-decl-invoker"
_RUN_ID = canonical_run_id_for_tests(_RUN_SEED)


def _calls_missing_idempotency_keyword(source_path: Path) -> list[int]:
    tree = ast.parse(
        source_path.read_text(encoding="utf-8-sig"), filename=str(source_path)
    )
    missing_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name: str | None = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name != "build_declarative_invoker_for_application_host":
            continue
        keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        if "idempotency_store" not in keyword_names:
            missing_lines.append(node.lineno)
    return missing_lines


def test_canonical_application_compositions_forward_idempotency_store_to_declarative_invoker() -> (
    None
):
    violations: list[str] = []
    for path in _CANONICAL_DECLARATIVE_INVOKER_COMPOSITION:
        assert path.is_file(), f"missing composition root: {path}"
        missing = _calls_missing_idempotency_keyword(path)
        if missing:
            rel = path.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}: lines {missing}")
    assert violations == [], (
        "build_declarative_invoker_for_application_host must receive idempotency_store "
        "from application reliability wiring:\n" + "\n".join(violations)
    )


class _DoubleIn(BaseModel):
    value: int


class _DoubleOut(BaseModel):
    result: int


class _CountingHandler:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest) -> _DoubleOut:
        self.calls += 1
        return _DoubleOut(result=request.input.value * 2)


def _state_with_enforce_allow() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.decl.idem")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    state = build_runtime_state_for_tests(run_id=_RUN_SEED)
    state.context.config.policy_bundle = wire_policy_bundle(env)
    return state


def test_declarative_invoker_uses_wired_idempotency_store_for_side_effect_dedupe() -> (
    None
):
    """Observable: duplicate invocation with same idempotency key executes effect once."""
    registry = ToolRegistry()
    handler = _CountingHandler()
    registry.register(
        contract=ToolContract(
            tool_id="double",
            name="double",
            description="double value",
            input_schema=_DoubleIn,
            output_schema=_DoubleOut,
            error_mapping={},
            side_effects=True,
        ),
        handler=handler,
    )
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["double"]),
        wiring_context=ToolWiringContext(),
        registry=registry,
    )
    store = InMemoryIdempotencyStore()
    catalog = build_declarative_invoker_from_tool_wiring(
        wiring,
        idempotency_store=store,
        production_mode=False,
    )
    assert catalog is not None
    invoker = catalog.tool_invoker
    state = _state_with_enforce_allow()
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id="double",
        input=_DoubleIn(value=5),
        idempotency_key="platform-wiring-dedupe-key",
    )
    with canonical_execution_identity_scope(_RUN_SEED):
        first = invoker.invoke(state=state, agent_id="agent-test", request=request)
        second = invoker.invoke(state=state, agent_id="agent-test", request=request)
    assert first.success and second.success
    assert first.output == second.output
    assert handler.calls == 1


def test_declarative_invoker_without_store_fails_closed_when_idempotency_key_present() -> (
    None
):
    registry = ToolRegistry()
    handler = _CountingHandler()
    registry.register(
        contract=ToolContract(
            tool_id="double",
            name="double",
            description="double value",
            input_schema=_DoubleIn,
            output_schema=_DoubleOut,
            error_mapping={},
            side_effects=True,
        ),
        handler=handler,
    )
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["double"]),
        wiring_context=ToolWiringContext(),
        registry=registry,
    )
    catalog = build_declarative_invoker_from_tool_wiring(wiring, production_mode=False)
    assert catalog is not None
    invoker = catalog.tool_invoker
    state = _state_with_enforce_allow()
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id="double",
        input=_DoubleIn(value=3),
        idempotency_key="no-store-key",
    )
    with canonical_execution_identity_scope(_RUN_SEED):
        with pytest.raises(RuntimeError, match="pre-effect coordinator"):
            invoker.invoke(state=state, agent_id="agent-test", request=request)
    assert handler.calls == 0
