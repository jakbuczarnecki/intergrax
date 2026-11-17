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
        assert "_canonical" in names, f"{function.name} does not call the Tier-1 module"


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


def test_nt_wb_03_tier3_routing_record_is_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        return _StubAdapter(model=profile.model or "solo")

    monkeypatch.setattr("intergrax.runtime.wiring.llm_resolver.create_adapter", _fake)
    monkeypatch.setattr("intergrax.runtime.wiring.llm_resolver.create_adapter_with_failover", _fake)

    primary = LLMProfile(provider=LLMProvider.OPENAI, model="primary")
    economy = LLMProfile(provider=LLMProvider.OPENAI, model="economy")
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="wb.nt03")
    env = env.model_copy(
        update={
            "llm_profile": primary,
            "llm_routing_profile": LLMRoutingProfile(
                default_profile=primary,
                allowed_profiles=(primary, economy),
                rules=(BudgetBelowRule(threshold=0.25, profile=economy),),
            ),
        },
    )
    adapter = facade.resolve_llm_adapter(env, tenant_id="tenant-1")

    seed = "wb-nt03"
    request = build_runtime_request_for_tests(
        seed=seed,
        tenant_id="tenant-1",
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
    steps = [event.step for event in state.trace_events]
    assert "llm_routing_rule" in steps
