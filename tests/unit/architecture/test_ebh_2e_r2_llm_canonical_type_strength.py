# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R2 — canonical LLM configuration/execution type strength gate."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.native_tool_choice import NativeForcedFunctionChoice
from intergrax.llm_adapters.contracts.serialized_value import JsonValue
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.registration_contract import LLMAdapterRegistrationSpec

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax/llm_adapters/contracts"

# (relative path under contracts/, qualified symbol fragment, reason)
_ANY_OBJECT_ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset(
    {
        (
            "tool_call.py",
            "tool_calls_from_openai_message",
            "opaque provider SDK message interop at conversion boundary",
        ),
        (
            "tool_call.py",
            "tool_calls_from_langchain_message",
            "opaque provider SDK message interop at conversion boundary",
        ),
        (
            "tool_call.py",
            "tool_calls_from_openai_dicts",
            "opaque provider wire item interop at conversion boundary",
        ),
        (
            "llm_profile.py",
            "_coerce_and_validate_options",
            "pydantic before-validator accepts untyped wire payload",
        ),
        (
            "llm_profile.py",
            "_coerce_fallback_profiles",
            "pydantic before-validator accepts untyped wire payload",
        ),
        (
            "llm_profile.py",
            "_validate_options_map",
            "options map coercion before JSON validation",
        ),
        (
            "strict_tool_call_validation.py",
            "_validate_type",
            "JSON Schema instance validation against runtime payloads",
        ),
        (
            "strict_tool_call_validation.py",
            "validate_json_against_canonical_schema",
            "JSON Schema instance validation against runtime payloads",
        ),
        (
            "strict_tool_call_validation.py",
            "_parameters_from_definition",
            "JSON Schema document extracted from canonical tool wire schema",
        ),
    }
)

_FORBIDDEN_PUBLIC_NAMES = frozenset({"Any"})


def _contract_modules() -> list[Path]:
    return sorted(_CONTRACTS_ROOT.rglob("*.py"))


def _annotation_mentions_forbidden(node: ast.AST) -> list[str]:
    hits: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in _FORBIDDEN_PUBLIC_NAMES:
            hits.append(child.id)
        if isinstance(child, ast.Attribute) and child.attr in _FORBIDDEN_PUBLIC_NAMES:
            hits.append(child.attr)
    return hits


def _symbol_key(path: Path, node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.ClassDef):
        return node.name
    return ""


def test_ebh_2e_r2_contracts_dynamic_discovery_no_unexplained_any() -> None:
    offenders: list[str] = []
    for path in _contract_modules():
        rel = path.relative_to(_CONTRACTS_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.AnnAssign)
            ):
                continue
            symbol = _symbol_key(path, node) if isinstance(node, ast.ClassDef) else getattr(node, "name", "")
            if isinstance(node, ast.ClassDef):
                target_nodes = [node] + list(node.body)
            else:
                target_nodes = [node]
            for target in target_nodes:
                for forbidden in _annotation_mentions_forbidden(target):
                    key = (rel, symbol, "")
                    allowed = any(
                        rel == entry[0] and (entry[1] == symbol or entry[1] in symbol)
                        for entry in _ANY_OBJECT_ALLOWLIST
                    )
                    if not allowed:
                        offenders.append(f"{rel}:{symbol} uses {forbidden}")
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r2_llm_profile_options_bounded_json_value() -> None:
    hints = get_type_hints(LLMProfile, include_extras=True)
    options_ann = hints["options"]
    assert "JsonValue" in str(options_ann)


def test_ebh_2e_r2_llm_adapter_tools_canonical_only() -> None:
    sig = inspect.signature(LLMAdapter.generate_with_tools)
    tools = sig.parameters["tools"]
    assert "CanonicalFunctionToolDefinition" in str(tools.annotation)
    assert "Mapping" not in str(tools.annotation)
    assert "Any" not in str(tools.annotation)


def test_ebh_2e_r2_llm_adapter_tool_choice_typed() -> None:
    sig = inspect.signature(LLMAdapter.generate_with_tools)
    tool_choice = sig.parameters["tool_choice"]
    assert "NativeToolChoice" in str(tool_choice.annotation)
    assert "dict" not in str(tool_choice.annotation)


def test_ebh_2e_r2_structured_output_generic() -> None:
    sig = inspect.signature(LLMAdapter.generate_structured)
    assert "TStructured" in str(sig.return_annotation)
    assert "output_model" in sig.parameters
    assert "TStructured" in str(sig.parameters["output_model"].annotation)


def test_ebh_2e_r2_external_provider_custom_serialized_options() -> None:
    from collections.abc import Iterable, Sequence

    from intergrax.llm.messages import ChatMessage
    from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter as AdapterPort
    from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
    from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
    from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult, TStructured

    class _ExternalStub:
        provider = "external_r2_proof"
        model = "proof-model"

        @property
        def context_window_tokens(self) -> int:
            return 4096

        def generate_messages(
            self,
            messages: Sequence[ChatMessage],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
            run_id: str | None = None,
        ) -> LLMAdapterResponse:
            return build_adapter_response(content="ok")

        def supports_streaming(self) -> bool:
            return False

        def stream_messages(
            self,
            messages: Sequence[ChatMessage],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
            run_id: str | None = None,
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
            temperature: float | None = None,
            max_tokens: int | None = None,
            tool_choice: NativeToolChoice | None = None,
            run_id: str | None = None,
        ) -> LLMAdapterResponse:
            raise NotImplementedError

        def stream_with_tools(
            self,
            messages: Sequence[ChatMessage],
            tools: Sequence[CanonicalFunctionToolDefinition],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
            tool_choice: NativeToolChoice | None = None,
            run_id: str | None = None,
        ) -> Iterable[LLMStreamEvent]:
            raise NotImplementedError

        def supports_structured_output(self) -> bool:
            return False

        def generate_structured(
            self,
            messages: Sequence[ChatMessage],
            output_model: type[TStructured],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
            run_id: str | None = None,
        ) -> LLMStructuredResult[TStructured]:
            raise NotImplementedError

        def supports_vision(self) -> bool:
            return False

        def supports_audio_input(self) -> bool:
            return False

        def supports_audio_output(self) -> bool:
            return False

    LLMAdapterRegistry.reset_for_testing()
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id="external_r2_proof",
            factory=lambda **kwargs: _ExternalStub(),
        ),
        override=True,
    )
    profile = LLMProfile(
        provider="external_r2_proof",
        model="proof-model",
        options={"custom_flag": True, "nested": {"k": 1}},
    )
    assert profile.options["custom_flag"] is True
    adapter = LLMAdapterRegistry.create("external_r2_proof", model="proof-model")
    assert isinstance(adapter, AdapterPort)


def test_ebh_2e_r2_contracts_do_not_import_providers() -> None:
    forbidden_prefixes = (
        "intergrax.llm_adapters.providers.",
        "intergrax.applications.",
        "intergrax.runtime.",
    )
    offenders: list[str] = []
    for path in _contract_modules():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(node.module.startswith(prefix) for prefix in forbidden_prefixes):
                    offenders.append(f"{path.name}: from {node.module}")
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r2_native_tool_choice_projects_for_external_provider() -> None:
    from intergrax.llm_adapters.contracts.native_tool_choice import (
        project_native_tool_choice_for_provider,
    )

    choice = NativeForcedFunctionChoice(function_name="demo.tool")
    projected = project_native_tool_choice_for_provider(choice, provider="external_r2_proof")
    assert projected == {"type": "function", "name": "demo.tool"}
