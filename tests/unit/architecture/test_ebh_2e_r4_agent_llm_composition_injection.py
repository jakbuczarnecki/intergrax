# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R4 — agent LLM self-wiring elimination and composition injection."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import pytest

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult, TStructured
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from model_routing_qualifier.model_routing_qualifier_agent import ModelRoutingQualifierAgent
from tool_selection_qualifier.tool_selection_qualifier_agent import ToolSelectionQualifierAgent
from web_search_qualifier.web_search_qualifier_agent import WebSearchQualifierAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _minimal_runtime_request() -> RuntimeRequest:
    return RuntimeRequest(
        agent_id="qualifier",
        user_id="user",
        session_id="session",
        message="",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )

_QUALIFIER_AGENT_MODULES = (
    _REPO_ROOT / "agents/web_search_qualifier/web_search_qualifier_agent.py",
    _REPO_ROOT / "agents/tool_selection_qualifier/tool_selection_qualifier_agent.py",
    _REPO_ROOT / "agents/model_routing_qualifier/model_routing_qualifier_agent.py",
)

_FORBIDDEN_IMPORT = "intergrax.llm_adapters.registry.profile"
_FORBIDDEN_CALLS = ("create_adapter(", "create_adapter_with_failover(", "llm_profile_from_env(")
_OPTIONAL_ADAPTER_PATTERN = re.compile(r"llm_adapter:\s*LLMAdapter\s*\|\s*None")


class _ExternalStructuralAdapter:
    """Structural LLMAdapter — no framework base class."""

    @property
    def slug(self) -> str:
        return "external-ebh-2e-r4"

    def invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        raise NotImplementedError

    def invoke_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        raise NotImplementedError

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        raise NotImplementedError

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        raise NotImplementedError

    def supports_structured_output(self) -> bool:
        return False

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type[TStructured],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[TStructured]:
        raise NotImplementedError

    def supports_vision(self) -> bool:
        return False

    def supports_audio_input(self) -> bool:
        return False

    def supports_audio_output(self) -> bool:
        return False


def _module_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(_module_source(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            hits.append(node.module)
    return hits


@pytest.mark.parametrize("path", _QUALIFIER_AGENT_MODULES, ids=lambda p: p.name)
def test_ebh_2e_r4_qualifier_agent_forbids_registry_profile_import(path: Path) -> None:
    offenders = [imp for imp in _module_imports(path) if _FORBIDDEN_IMPORT in imp]
    assert not offenders, f"{path.name}: {offenders}"


@pytest.mark.parametrize("path", _QUALIFIER_AGENT_MODULES, ids=lambda p: p.name)
def test_ebh_2e_r4_qualifier_agent_forbids_factory_and_env_calls(path: Path) -> None:
    source = _module_source(path)
    hits = [token for token in _FORBIDDEN_CALLS if token in source]
    assert not hits, f"{path.name}: {hits}"


@pytest.mark.parametrize("path", _QUALIFIER_AGENT_MODULES, ids=lambda p: p.name)
def test_ebh_2e_r4_qualifier_agent_requires_mandatory_llm_adapter(path: Path) -> None:
    source = _module_source(path)
    assert "LLMAdapter" in source
    assert not _OPTIONAL_ADAPTER_PATTERN.search(source), f"{path.name}: optional llm_adapter"


@pytest.mark.parametrize(
    ("agent_cls", "factory_name"),
    [
        (WebSearchQualifierAgent, "build_local_workspace_web_search_qualifier_from_context"),
        (ToolSelectionQualifierAgent, "build_local_workspace_tool_selection_qualifier_from_context"),
        (ModelRoutingQualifierAgent, "build_local_workspace_model_routing_qualifier_from_context"),
    ],
)
def test_ebh_2e_r4_production_factory_injects_llm_adapter(
    agent_cls: type,
    factory_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.host import agent_factories
    from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

    injected = _ExternalStructuralAdapter()
    factory = getattr(agent_factories, factory_name)
    ctx = ApplicationBuildContext.for_manifest(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        environment=LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment,
    )
    binding = AgentBinding.mount(agent_cls, contract_id="qualifier", factory=factory)
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        lambda *_args, **_kwargs: injected,
    )
    agent = factory(ctx, binding)
    runtime = agent.build_context(_minimal_runtime_request())
    assert runtime.config.llm_adapter is injected


@pytest.mark.parametrize(
    "agent_cls",
    [WebSearchQualifierAgent, ToolSelectionQualifierAgent, ModelRoutingQualifierAgent],
)
def test_ebh_2e_r4_injected_structural_adapter_reaches_runtime_context(
    agent_cls: type,
) -> None:
    injected = _ExternalStructuralAdapter()
    agent = agent_cls(llm_adapter=injected)
    runtime = agent.build_context(_minimal_runtime_request())
    assert runtime.config.llm_adapter is injected


def test_ebh_2e_r4_agents_tree_has_no_consumer_create_adapter() -> None:
    agents_root = _REPO_ROOT / "agents"
    offenders: list[str] = []
    for path in agents_root.rglob("*.py"):
        if not path.is_file():
            continue
        source = _module_source(path)
        if any(token in source for token in _FORBIDDEN_CALLS):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, "\n".join(offenders)
