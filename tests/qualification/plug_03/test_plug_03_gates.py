# © Artur Czarnecki. All rights reserved.

"""PLUG-03 — external replacement qualification gates and targeted proofs."""

from __future__ import annotations

import ast
import importlib.metadata
from collections.abc import Iterator
from pathlib import Path

import pytest

from intergrax.context.budget.compaction import (
    ContextCompactionInput,
    ContextCompactionStrategy,
    NoOpContextCompactionStrategy,
)
from intergrax.context.registry import ContextPluginRegistry
from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.catalog_snapshot import snapshot_catalogs
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.examples.custom_memory_kv.plugin import CustomMemoryKvPlugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.tools.tool_invocation_pattern import resolve_invocation_pattern
from intergrax.tools.invocation_pattern.errors import ToolInvocationPatternResolutionError
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.usefixtures("catalog_fixture_installed")]

# Catalog fixture EP tool id (avoid importing fixture package before session install).
_FIXTURE_ECHO_TOOL_ID = "fixture_ep.echo"

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PUBLIC_PLUGIN_ROOTS = (
    _REPO_ROOT / "examples" / "platform_plugins",
    _REPO_ROOT / "tests" / "fixtures" / "plugin_packages",
)

_NEXUS_ALLOWLIST_RELATIVE: frozenset[str] = frozenset()


def _module_imports_nexus(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime.nexus"):
                hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime.nexus"):
                    hits.append(alias.name)
    return hits


def _iter_public_plugin_python_files() -> Iterator[Path]:
    for root in _PUBLIC_PLUGIN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in _NEXUS_ALLOWLIST_RELATIVE:
                continue
            yield path


@pytest.fixture(autouse=True)
def _reset_catalog_state() -> Iterator[None]:
    clear_catalog()
    clear_skill_catalog()
    clear_tool_catalog()
    reset_default_integrations_state()
    reset_default_skills_for_tests()
    reset_default_tools_bootstrap()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    yield
    clear_catalog()
    clear_skill_catalog()
    clear_tool_catalog()
    reset_default_integrations_state()
    reset_default_skills_for_tests()
    reset_default_tools_bootstrap()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()


def test_public_external_plugin_packages_do_not_import_nexus() -> None:
    violations: list[str] = []
    for path in _iter_public_plugin_python_files():
        hits = _module_imports_nexus(path)
        if hits:
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {hits}")
    assert violations == [], "public plugin packages must not import Nexus:\n" + "\n".join(violations)


def test_memory_session_storage_fixture_does_not_import_nexus() -> None:
    plugin_root = (
        _REPO_ROOT
        / "tests"
        / "fixtures"
        / "plugin_packages"
        / "memory_store_plugin"
        / "memory_store_plugin"
    )
    violations: list[str] = []
    for path in plugin_root.glob("*.py"):
        hits = _module_imports_nexus(path)
        if hits:
            violations.append(f"{path.name}: {hits}")
    assert violations == []


def test_tools_discovered_but_unselected_not_in_execution_registry() -> None:
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    snap = snapshot_catalogs()
    assert "fixture_ep" in snap.tool_bundle_ids

    registry = build_registry_from_profile(ToolProfile.lab(), ctx=ToolWiringContext())
    assert not registry.has(_FIXTURE_ECHO_TOOL_ID)


def test_tools_profile_selection_executes_custom_not_catalog_default() -> None:
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    registry = build_registry_from_profile(
        ToolProfile(enabled_bundles=["fixture_ep"]),
        ctx=ToolWiringContext(),
    )
    assert registry.has(_FIXTURE_ECHO_TOOL_ID)


def test_integration_discovered_but_unselected_keeps_default_binding() -> None:
    bootstrap_catalogs(
        register_shipped=False,
        discover_entry_points=True,
        integration_plugins=(CustomMemoryKvPlugin,),
    )
    snap = snapshot_catalogs()
    assert "fixture_ep_kv" in snap.integration_slugs

    profile = IntegrationProfile(key_value_cache="custom_memory_kv")
    assert profile.key_value_cache is not None
    assert profile.key_value_cache.resolved_slug() == "custom_memory_kv"
    assert profile.key_value_cache.resolved_slug() != "fixture_ep_kv"
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    cache.set("tenant-a", "selected", b"custom_memory_kv")
    assert cache.get("tenant-a", "selected") == b"custom_memory_kv"


def test_integration_explicit_slug_activates_fixture_provider() -> None:
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    profile = IntegrationProfile(key_value_cache="fixture_ep_kv")
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    cache.set("tenant-a", "proof-key", b"custom")
    assert cache.get("tenant-a", "proof-key") == b"custom"


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _CustomMarkerPattern:
    marker = "plug-03-custom-pattern"

    @property
    def pattern_id(self) -> str:
        return "plug_03_custom_pattern"

    def execute(self, **_kwargs: object) -> object:
        raise AssertionError("execute should not run during resolver qualification")


def test_canonical_resolver_selects_custom_pattern_without_shipped_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "plug_03_custom_pattern",
                f"{__name__}:_CustomMarkerPattern",
                "intergrax.tool_invocation_patterns",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    def _forbidden_shipped_default(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shipped default pattern must not be instantiated")

    monkeypatch.setattr(
        "intergrax.runtime.nexus.tools.tool_invocation_pattern.pattern_for_mode",
        _forbidden_shipped_default,
    )

    resolved = resolve_invocation_pattern(
        mode=ToolInvocationMode.SINGLE_PASS,
        max_iterations=1,
        entry_point_pattern_id="plug_03_custom_pattern",
    )
    assert resolved.pattern_id == "plug_03_custom_pattern"


class _CustomMarkerCompaction(ContextCompactionStrategy):
    @property
    def strategy_id(self) -> str:
        return "plug_03_custom_compaction"

    def compact(self, item: ContextCompactionInput) -> None:
        _ = item
        return None


def test_context_custom_compaction_default_strategy_not_invoked() -> None:
    registry = ContextPluginRegistry()
    registry.set_compaction_strategy(_CustomMarkerCompaction())
    strategy = registry.compaction_strategy
    assert strategy.strategy_id == "plug_03_custom_compaction"
    assert not isinstance(strategy, NoOpContextCompactionStrategy)


def test_explicit_missing_tool_invocation_pattern_id_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    def _forbidden_shipped_default(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shipped default pattern must not be instantiated")

    monkeypatch.setattr(
        "intergrax.runtime.nexus.tools.tool_invocation_pattern.pattern_for_mode",
        _forbidden_shipped_default,
    )

    with pytest.raises(ToolInvocationPatternResolutionError, match="missing.custom.pattern"):
        resolve_invocation_pattern(
            mode=ToolInvocationMode.SINGLE_PASS,
            max_iterations=1,
            entry_point_pattern_id="missing.custom.pattern",
        )


def test_plug03_session_storage_canonical_session_manager_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.applications._shared.memory_wiring import (
        build_session_manager_from_environment,
        resolve_memory_platform_wiring,
    )
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
        MemoryProfile,
    )
    from intergrax.integrations.registry.profile import IntegrationProfile
    from intergrax.llm.messages import ChatMessage
    from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
    from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.fixture_session_storage import (
        FIXTURE_SESSION_STORAGE_MARKER,
        FixtureExternalSessionStorage,
    )
    from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin import (
        ExternalInMemorySessionStoragePlugin,
    )

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="plug03.session.q4")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        session_storage_plugin_id="external.in_memory_session_storage",
    )
    manager = build_session_manager_from_environment(
        env,
        memory_wiring=resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(ExternalInMemorySessionStoragePlugin,),
        ),
    )
    assert isinstance(manager._storage, FixtureExternalSessionStorage)
    assert not isinstance(manager._storage, InMemorySessionStorage)

    async def _exercise() -> None:
        session = await manager.create_session(tenant_id="tenant-a", user_id="user-a")
        await manager.append_message(
            tenant_id="tenant-a",
            session_id=session.id,
            message=ChatMessage(role="user", content="plug03 session proof"),
        )
        history = await manager.get_history(tenant_id="tenant-a", session_id=session.id)
        assert len(history) == 1
        assert history[0].content == "plug03 session proof"

    import asyncio

    asyncio.run(_exercise())
    assert getattr(manager._storage, "fixture_marker", "") == FIXTURE_SESSION_STORAGE_MARKER


from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from intergrax.skills.examples.custom_pack import CustomPackSkillPlugin
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.schema import PolicyRuleAction
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _Plug03PackAgent(Agent):
    def __init__(self, *, include_skill: bool) -> None:
        self._include_skill = include_skill

    def get_contract(self) -> AgentContract:
        manifests = CustomPackSkillPlugin.skill_manifests() if self._include_skill else ()
        return AgentContract(
            id="plug03_pack_stub",
            name="Plug03 Pack Stub",
            description="stub",
            capabilities=["stub.cap"],
            skills=list(manifests),
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        return CapabilityMatchResult(matched=True, agent_id="plug03_pack_stub", score=1.0)


class _Plug03ExternalPolicyHandler:
    rule_id = "plug03_external_policy_handler"
    evaluate_calls = 0

    def evaluate(self, rule: object, *, context: PolicyEvaluationContext) -> PolicyRuleAction:
        type(self).evaluate_calls += 1
        return PolicyRuleAction.ALLOW


def _plug03_skill_gateway_fixture(
    *,
    include_skill: bool,
) -> tuple[AgentContract, object, object]:
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from intergrax.skills.registry.factory import build_registry_from_profile as build_skill_registry
    from intergrax.skills.registry.plugin_register import register_skill_plugin
    from intergrax.skills.registry.profile import SkillProfile
    from intergrax.tools.examples.custom_echo import CustomEchoToolPlugin
    from intergrax.tools.registry.factory import build_registry_from_profile as build_tool_registry
    from intergrax.tools.registry.plugin_register import register_tool_plugin
    from intergrax.tools.registry.profile import ToolProfile
    from intergrax.tools.registry.wiring import ToolWiringContext
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    register_skill_plugin(CustomPackSkillPlugin)
    register_tool_plugin(CustomEchoToolPlugin)
    skill_bundles = ["custom_pack"] if include_skill else []
    skill_registry = build_skill_registry(SkillProfile(enabled_bundles=skill_bundles))
    tool_registry = build_tool_registry(
        ToolProfile(enabled_bundles=["custom_echo"]),
        ctx=ToolWiringContext(),
    )
    agent_registry = AgentRegistry()
    agent_registry.register(
        _Plug03PackAgent(include_skill=include_skill),
        skill_registry=skill_registry,
        tool_registry=tool_registry,
    )
    contract = agent_registry.get_contract("plug03_pack_stub")

    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tool_registry=tool_registry,
    )
    ctx = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    state = RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="plug03_pack_stub",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message="plug03 skill proof",
            task_id="task_00000000000000000000000000000001",
            run_id="run_00000000000000000000000000000001",
        ),
        run_id="run_00000000000000000000000000000001",
        tool_traces=[],
    )
    return contract, tool_registry, state


def _plug03_gateway_invoke_scope(run_id: str):
    from contextlib import contextmanager

    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        reset_active_execution_identity,
    )
    from intergrax.dev_support.execution_identity_scope import canonical_run_id_for_tests
    from intergrax.runtime.execution.active_execution_budget import (
        ActiveExecutionBudgetState,
        bind_active_execution_budget,
        reset_active_execution_budget,
    )
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
    from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode

    @contextmanager
    def _scope():
        canonical_run_id = canonical_run_id_for_tests(run_id)
        execution_id = mint_execution_id()
        ledger = create_execution_budget_ledger(None)
        budget_token = bind_active_execution_budget(
            ActiveExecutionBudgetState(
                execution_id=execution_id,
                mode=ExecutionBudgetAllocationMode.SHARED,
                ledger=ledger,
            ),
        )
        identity_token = bind_active_execution_identity(
            run_id=canonical_run_id,
            attempt_id=mint_attempt_id(),
            execution_id=execution_id,
        )
        try:
            yield
        finally:
            reset_active_execution_identity(identity_token)
            reset_active_execution_budget(budget_token)

    return _scope()


@pytest.mark.asyncio
async def test_plug03_custom_skill_enables_canonical_tool_execution() -> None:
    from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
    from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway
    from intergrax.tools.examples.custom_echo.plugin import CUSTOM_ECHO_TOOL_ID

    contract, tool_registry, state = _plug03_skill_gateway_fixture(include_skill=True)
    assert CUSTOM_ECHO_TOOL_ID in contract.allowed_tools
    assert tool_registry.has(CUSTOM_ECHO_TOOL_ID)

    gateway = RuntimeToolGateway.for_state(state, allowed_tools=contract.allowed_tools)
    with _plug03_gateway_invoke_scope(state.run_id):
        response = await gateway.invoke(
            ToolRequest(
                request_id="plug03-skill-positive",
                tool_name=CUSTOM_ECHO_TOOL_ID,
                agent_id="plug03_pack_stub",
                step_id="1",
                input={"message": "plug03-skill-proof"},
            )
        )
    assert response.status == ToolResponseStatus.SUCCESS
    assert response.output is not None
    assert response.output["message"] == "plug03-skill-proof"
    assert state.used_tools is True


@pytest.mark.asyncio
async def test_plug03_without_custom_skill_tool_not_allowed() -> None:
    from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
    from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
    from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway
    from intergrax.tools.examples.custom_echo.plugin import CUSTOM_ECHO_TOOL_ID

    contract, tool_registry, state = _plug03_skill_gateway_fixture(include_skill=False)
    assert CUSTOM_ECHO_TOOL_ID not in contract.allowed_tools
    assert tool_registry.has(CUSTOM_ECHO_TOOL_ID)
    assert not ToolAccessPolicy.is_tool_allowed(CUSTOM_ECHO_TOOL_ID, contract.allowed_tools)

    gateway = RuntimeToolGateway.for_state(state, allowed_tools=contract.allowed_tools)
    with _plug03_gateway_invoke_scope(state.run_id):
        response = await gateway.invoke(
            ToolRequest(
                request_id="plug03-skill-negative",
                tool_name=CUSTOM_ECHO_TOOL_ID,
                agent_id="plug03_pack_stub",
                step_id="1",
                input={"message": "plug03-skill-proof"},
            )
        )
    assert response.status == ToolResponseStatus.DENIED
    assert response.error == f"tool_not_allowed:{CUSTOM_ECHO_TOOL_ID}"
    assert state.used_tools is False
    assert state.tool_traces == []


from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import SecurityFailMode, SecurityInspectionResult


class _Plug03SentinelDefense:
    plugin_id = "plug03.sentinel.defense"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 58
    fail_mode = SecurityFailMode.FAIL_CLOSED
    inspect_calls = 0

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        type(self).inspect_calls += 1
        return SecurityInspectionResult(
            allowed=False,
            reasons=["plug03-security-sentinel"],
            plugin_id=self.plugin_id,
            hook_point=point.value,
        )


@pytest.mark.asyncio
async def test_plug03_security_defense_canonical_hook_invokes_custom_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.core.catalog_bootstrap import bootstrap_catalogs
    from intergrax.core.plugin_env import INTERGRAX_DISCOVER_PLUGINS_ENV
    from intergrax.core.security_bootstrap import bootstrap_security_providers
    from intergrax.runtime.hooks.hook_context import HookContext
    from intergrax.runtime.hooks.hook_point import HookPoint
    from intergrax.runtime.security.defense_plugin import PluginSecurityDefenseMiddleware
    from intergrax.runtime.security.defense_registry import get_security_defense_plugin

    _Plug03SentinelDefense.inspect_calls = 0
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    entries = _EntryPoints(
        [
            _EntryPoint(
                "plug03_sentinel",
                f"{__name__}:_Plug03SentinelDefense",
                "intergrax.security_defenses",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    bootstrap_security_providers(discover_entry_points=True)
    plugin = get_security_defense_plugin("plug03.sentinel.defense")
    assert plugin is not None

    middleware = PluginSecurityDefenseMiddleware(plugin)
    ctx = HookContext(
        run_id="plug03-sec",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={"tool_id": "demo.tool", "arguments": {}},
    )
    result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action.value == "block"
    assert _Plug03SentinelDefense.inspect_calls == 1


def test_plug03_policy_pipeline_custom_handler_changes_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.applications._shared.policy_wiring import wire_policy_bundle
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
        PolicyRulesProfile,
    )
    from intergrax.runtime.policy.declarative_enforcer import DeclarativePolicyEnforcer
    from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
    from intergrax.runtime.policy.rules.schema import PolicyRuleAction

    _Plug03ExternalPolicyHandler.evaluate_calls = 0
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(
            [
                _EntryPoint(
                    "plug03_handler",
                    f"{__name__}:_Plug03ExternalPolicyHandler",
                    "intergrax.policy_rules",
                ),
            ]
        ),
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="plug03.policy.q4")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "plug03.sentinel.rule",
                "handler_id": "plug03_external_policy_handler",
                "resource_kind": "tool",
                "resource_id": "plug03.sentinel.tool",
                "action": "deny",
            }
        ],
        policy_enforcement_mode="enforce",
        allowed_handler_ids=["plug03_external_policy_handler"],
    )
    bundle = wire_policy_bundle(env)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    enforcer = DeclarativePolicyEnforcer(runtime=runtime)
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="plug03.sentinel.tool"),
    )
    assert _Plug03ExternalPolicyHandler.evaluate_calls == 1
    assert decision.action is PolicyRuleAction.ALLOW

    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "plug03.sentinel.rule",
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": "plug03.sentinel.tool",
                "action": "deny",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    shipped_bundle = wire_policy_bundle(env)
    shipped_enforcer = DeclarativePolicyEnforcer(runtime=shipped_bundle.declarative_policy_runtime)
    shipped_decision = shipped_enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="plug03.sentinel.tool"),
    )
    assert shipped_decision.action is PolicyRuleAction.DENY
