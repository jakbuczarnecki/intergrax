# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R1 — canonical LLM execution contract vs runtime base separation."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import pytest

from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.contracts.structured_result import TStructured
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.registration_contract import LLMAdapterFactory

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_PATH = _REPO_ROOT / "intergrax/llm_adapters/contracts/llm_adapter.py"
_BASE_PATH = _REPO_ROOT / "intergrax/llm_adapters/base/base_llm_adapter.py"

_FORBIDDEN_CONTRACT_PREFIXES = (
    "intergrax.runtime.",
    "intergrax.llm_adapters._shared.",
    "intergrax.llm_adapters.providers.",
    "intergrax.llm_adapters.registry.",
    "intergrax.llm_adapters.base.",
    "tiktoken",
)
_FORBIDDEN_CONTRACT_SUBSTRINGS = (
    "intergrax.llm_adapters.tracking",
    "intergrax.llm_adapters.governance",
)

_PLATFORM_CONSUMER_ROOTS = (
    _REPO_ROOT / "intergrax/runtime",
    _REPO_ROOT / "intergrax/applications",
    _REPO_ROOT / "intergrax/agents",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "intergrax/rag",
    _REPO_ROOT / "intergrax/supervisor",
    _REPO_ROOT / "intergrax/contracts",
)

_BASE_IMPORT_ALLOWLIST = (
    _REPO_ROOT / "intergrax/llm_adapters/providers",
    _REPO_ROOT / "intergrax/llm_adapters/registry",
    _REPO_ROOT / "intergrax/llm_adapters/routing",
    _REPO_ROOT / "intergrax/llm_adapters/_shared",
    _REPO_ROOT / "intergrax/llm_adapters/base",
    _REPO_ROOT / "intergrax/codecraft",
    _REPO_ROOT / "intergrax/runtime/external_operations",
    _REPO_ROOT / "intergrax/runtime/nexus/agents",
    _REPO_ROOT / "intergrax/runtime/token_optimization/proofs",
    _REPO_ROOT / "intergrax/agents/authoring",
    _REPO_ROOT / "testing_support",
    _REPO_ROOT / "tests",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "platform_proofs",
    _REPO_ROOT / "proof_infrastructure",
)

_SHIPPED_PROVIDER_MODULES = (
    "intergrax.llm_adapters.providers.claude_adapter",
    "intergrax.llm_adapters.providers.gemini_adapter",
    "intergrax.llm_adapters.providers.mistral_adapter",
    "intergrax.llm_adapters.providers.ollama_adapter",
    "intergrax.llm_adapters.providers.native_ollama_adapter",
    "intergrax.llm_adapters.providers.openai_chat_completions_adapter",
    "intergrax.llm_adapters.providers.openai_responses_adapter",
    "intergrax.llm_adapters.providers.aws_bedrock_adapter",
    "intergrax.llm_adapters.providers.cohere_native_adapter",
)


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            hits.append(node.module)
    return hits


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def test_ebh_2e_r1_contract_module_is_pure() -> None:
    problems: list[str] = []
    for imported in _module_imports(_CONTRACT_PATH):
        if any(imported.startswith(prefix) for prefix in _FORBIDDEN_CONTRACT_PREFIXES):
            problems.append(imported)
        if any(token in imported for token in _FORBIDDEN_CONTRACT_SUBSTRINGS):
            problems.append(imported)
    assert not problems, "\n".join(problems)


def test_ebh_2e_r1_framework_base_not_in_contracts_package() -> None:
    contracts_dir = _REPO_ROOT / "intergrax/llm_adapters/contracts"
    offenders: list[str] = []
    for path in contracts_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in {
                "BaseLLMAdapter",
                "LLMAdapterUsageLog",
            }:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.name}")
    assert not offenders


def test_ebh_2e_r1_single_execution_contract_authority() -> None:
    llm_mod = importlib.import_module("intergrax.llm_adapters.contracts.llm_adapter")
    assert getattr(llm_mod, "LLMAdapter", None) is not None
    tree = ast.parse(_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == "Protocol"
            for base in node.bases
        )
        and "LLM" in node.name
        and "Adapter" in node.name
    ]
    assert protocol_names == ["LLMAdapter"]


class _ExternalStructuralAdapter:
    """Structural provider plugin — no framework base inheritance."""

    provider = "external-structural"
    model = "ext-model"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="external")

    def supports_streaming(self) -> bool:
        return False

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        raise NotImplementedError

    def supports_tools(self) -> bool:
        return False

    def supports_strict_tool_argument_conformance(self) -> bool:
        return False

    def generate_with_tools(
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


def test_ebh_2e_r1_external_structural_adapter_satisfies_contract() -> None:
    adapter = _ExternalStructuralAdapter()
    assert isinstance(adapter, LLMAdapter)


def test_ebh_2e_r1_registry_accepts_structural_adapter() -> None:
    LLMAdapterRegistry.reset_for_testing()
    key = "ebh2e-r1-external"

    def _factory(**_kwargs: object) -> LLMAdapter:
        return _ExternalStructuralAdapter()

    LLMAdapterRegistry.register(key, _factory)
    created = LLMAdapterRegistry.create(key)
    assert isinstance(created, LLMAdapter)
    assert not isinstance(created, BaseLLMAdapter)


def test_ebh_2e_r1_registry_factory_annotation_returns_contract() -> None:
    hints = LLMAdapterFactory.__call__.__annotations__
    assert hints.get("return") == "LLMAdapter"


def test_ebh_2e_r1_shipped_provider_adapters_satisfy_contract() -> None:
    for module_name in _SHIPPED_PROVIDER_MODULES:
        path = _REPO_ROOT / Path(*module_name.split("."))
        path = path.with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        adapter_types = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id == "BaseLLMAdapter"
                for base in node.bases
            )
        ]
        assert adapter_types, f"no BaseLLMAdapter subclass in {module_name}"


def test_ebh_2e_r1_platform_consumers_do_not_import_framework_base() -> None:
    offenders: list[str] = []
    for root in _PLATFORM_CONSUMER_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            if _is_under(path, _REPO_ROOT / "intergrax/llm_adapters"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if (
                "from intergrax.llm_adapters.base" in text
                or "import BaseLLMAdapter" in text
            ):
                if any(_is_under(path, allow) for allow in _BASE_IMPORT_ALLOWLIST):
                    continue
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r1_application_resolver_return_type_is_contract() -> None:
    hints = resolve_llm_adapter.__annotations__
    assert hints.get("return") in {LLMAdapter, "LLMAdapter"}


def test_ebh_2e_r1_runtime_config_llm_field_is_contract() -> None:
    hints = RuntimeConfig.__annotations__
    assert hints.get("llm_adapter") is LLMAdapter
