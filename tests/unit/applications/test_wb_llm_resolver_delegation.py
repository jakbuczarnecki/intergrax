# © Artur Czarnecki. All rights reserved.

"""W-B Part A — the Tier-3 resolver module only delegates (NT-WB-01, NT-WB-03)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.applications._shared import llm_resolver as facade
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from testing_support.builder import (
    build_in_memory_session_manager,
    build_runtime_request_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    canonical_task_id_for_tests,
)

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FACADE = _REPO_ROOT / "intergrax/applications/_shared/llm_resolver.py"
_FORBIDDEN_CALLS = frozenset(
    {
        "create_adapter",
        "create_adapter_with_failover",
        "ModelRouter",
        "wrap_routing_evaluating_adapter",
    },
)


def test_nt_wb_01_facade_creates_no_adapters() -> None:
    tree = ast.parse(_FACADE.read_text(encoding="utf-8"))

    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert called.isdisjoint(_FORBIDDEN_CALLS), sorted(called & _FORBIDDEN_CALLS)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert imported.isdisjoint(_FORBIDDEN_CALLS), sorted(imported & _FORBIDDEN_CALLS)

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assert not any(isinstance(t, ast.Name) and t.id == "_last_routing_evaluation" for t in targets)

    public_functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    assert public_functions
    for function in public_functions:
        names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
        # A function either calls the Tier-1 module or calls a sibling public function of this module
        # (the environment helpers go through the module-level resolver so that patches of it still apply).
        delegates = "_canonical" in names or bool(names & {"resolve_llm_adapter", "resolve_optional_llm_adapter"})
        assert delegates, f"{function.name} does not delegate to the Tier-1 module"


class _StubAdapter(BaseLLMAdapter):
    provider = LLMProvider.OPENAI
    model: str = "stub"

    def __init__(self, *, model: str) -> None:
        super().__init__()
        self.model = model

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(self, messages: list[ChatMessage], **kwargs: object) -> LLMAdapterResponse:
        del messages, kwargs
        return LLMAdapterResponse(
            content="ok",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider=str(self.provider),
        )


def _routing_env_and_stubs(monkeypatch: pytest.MonkeyPatch) -> ApplicationEnvironmentProfile:
    def _fake(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        return _StubAdapter(model=profile.model or "solo")

    monkeypatch.setattr("intergrax.runtime.wiring.llm_resolver.create_adapter", _fake)
    monkeypatch.setattr("intergrax.runtime.wiring.llm_resolver.create_adapter_with_failover", _fake)

    primary = LLMProfile(provider=LLMProvider.OPENAI, model="primary")
    economy = LLMProfile(provider=LLMProvider.OPENAI, model="economy")
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="wb.nt03")
    return env.model_copy(
        update={
            "llm_profile": primary,
            "llm_routing_profile": LLMRoutingProfile(
                default_profile=primary,
                allowed_profiles=(primary, economy),
                rules=(BudgetBelowRule(threshold=0.25, profile=economy),),
            ),
        },
    )


def _tracker_steps(adapter: LLMAdapter, *, request_tenant: str) -> list[str]:
    seed = "wb-nt03"
    request = build_runtime_request_for_tests(
        seed=seed,
        tenant_id=request_tenant,
        agent_id="agent-1",
        session_id="sess-1",
        user_id="user-1",
        message="hello",
        metadata={"task_id": str(canonical_task_id_for_tests(seed))},
    )
    context = RuntimeContext.build(
        config=RuntimeConfig(
            llm_adapter=adapter,
            enable_rag=False,
            enable_websearch=False,
            tools_mode="off",
            production_mode=False,
        ),
        session_manager=build_in_memory_session_manager(),
    )
    state = RuntimeState(
        context=context,
        request=request,
        run_id=str(canonical_run_id_for_tests(seed)),
    )
    with canonical_execution_identity_scope(seed):
        state.configure_llm_tracker()
    return [event.step for event in state.trace_events]


def test_nt_wb_03_tier3_routing_record_is_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _routing_env_and_stubs(monkeypatch)
    adapter = facade.resolve_llm_adapter(env, tenant_id="tenant-1")
    assert _tracker_steps(adapter, request_tenant="tenant-1").count("llm_routing_rule") == 1


def test_unscoped_routing_record_is_consumed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: an adapter resolved without a tenant still gets its routing diag event exactly once."""
    env = _routing_env_and_stubs(monkeypatch)
    adapter = facade.resolve_llm_adapter(env)
    assert _tracker_steps(adapter, request_tenant="tenant-1").count("llm_routing_rule") == 1


def test_foreign_tenant_run_does_not_receive_the_routing_record(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _routing_env_and_stubs(monkeypatch)
    adapter = facade.resolve_llm_adapter(env, tenant_id="tenant-1")
    assert "llm_routing_rule" not in _tracker_steps(adapter, request_tenant="tenant-2")


@pytest.mark.parametrize(
    ("helper", "resolver"),
    [
        ("resolve_environment_llm_adapter", "resolve_llm_adapter"),
        ("resolve_optional_environment_llm_adapter", "resolve_optional_llm_adapter"),
    ],
)
def test_tier3_environment_resolvers_go_through_the_module_level_resolver(helper: str, resolver: str) -> None:
    """Patches of the Tier-3 module-level resolvers keep applying to the environment helpers."""
    tree = ast.parse(_FACADE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == helper)
    called = {n.func.id for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    assert resolver in called
    assert "_canonical" not in names
