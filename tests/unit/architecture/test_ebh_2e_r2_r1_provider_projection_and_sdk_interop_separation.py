# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R2-R1 — provider projection and SDK interop separation gate."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters._shared.openai_tool_choice_projection import (
    project_openai_compatible_tool_choice,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter as AdapterPort
from intergrax.llm_adapters.contracts.native_tool_choice import (
    NativeForcedFunctionChoice,
    NativeToolChoice,
)
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult, TStructured
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.registration_contract import LLMAdapterRegistrationSpec

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax/llm_adapters/contracts"

_PROVIDER_LEAK_TERMS = frozenset(
    {
        "openai",
        "anthropic",
        "claude",
        "gemini",
        "ollama",
        "langchain",
        "bedrock",
        "mistral",
        "cohere",
        "groq",
        "vllm",
        "openrouter",
    }
)

_FORBIDDEN_TOOL_CALL_SYMBOL_PREFIXES = (
    "tool_calls_from_openai",
    "tool_calls_from_langchain",
    "from_openai",
)

_FORBIDDEN_NATIVE_TOOL_CHOICE_SYMBOLS = frozenset(
    {
        "project_native_tool_choice_for_provider",
    }
)


def _contract_modules() -> list[Path]:
    return sorted(_CONTRACTS_ROOT.rglob("*.py"))


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def test_ebh_2e_r2_r1_native_tool_choice_module_has_no_provider_projection() -> None:
    path = _CONTRACTS_ROOT / "native_tool_choice.py"
    tree = _module_ast(path)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _FORBIDDEN_NATIVE_TOOL_CHOICE_SYMBOLS:
                offenders.append(f"forbidden symbol {node.name}")
            for arg in node.args.args:
                if arg.arg == "provider":
                    offenders.append(f"{node.name} accepts provider parameter")
        if isinstance(node, ast.Compare):
            for operand in ast.walk(node):
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                    if operand.value.strip().lower() in _PROVIDER_LEAK_TERMS:
                        offenders.append(f"provider literal compare: {operand.value!r}")
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r2_r1_tool_call_module_has_no_sdk_interop_entrypoints() -> None:
    path = _CONTRACTS_ROOT / "tool_call.py"
    tree = _module_ast(path)
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(node.name.startswith(prefix) for prefix in _FORBIDDEN_TOOL_CALL_SYMBOL_PREFIXES):
                offenders.append(node.name)
        if isinstance(node, ast.ClassDef) and node.name == "LLMToolCall":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("from_openai"):
                    offenders.append(f"LLMToolCall.{item.name}")
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r2_r1_contracts_no_provider_named_public_functions() -> None:
    offenders: list[str] = []
    for path in _contract_modules():
        rel = path.relative_to(_CONTRACTS_ROOT).as_posix()
        if rel == "provider_extensions.py":
            continue
        tree = _module_ast(path)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lowered = node.name.lower()
                if any(term in lowered for term in _PROVIDER_LEAK_TERMS):
                    offenders.append(f"{rel}:{node.name}")
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r2_r1_external_provider_projects_tool_choice_locally() -> None:
    captured: list[NativeToolChoice | None] = []

    class _ExternalProjectionAdapter:
        provider = "external_r2_r1_proof"
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
            return True

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
            captured.append(tool_choice)
            wire = project_openai_compatible_tool_choice(tool_choice)
            assert wire == {"type": "function", "name": "demo.tool"}
            return build_adapter_response(content="")

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
            provider_id="external_r2_r1_proof",
            factory=lambda **kwargs: _ExternalProjectionAdapter(),
        ),
        override=True,
    )
    adapter = LLMAdapterRegistry.create("external_r2_r1_proof", model="proof-model")
    assert isinstance(adapter, AdapterPort)
    choice = NativeForcedFunctionChoice(function_name="demo.tool")
    adapter.generate_with_tools(
        [ChatMessage(role="user", content="hi")],
        [],
        tool_choice=choice,
    )
    assert captured == [choice]


def test_ebh_2e_r2_r1_tool_call_interop_builds_canonical_dto() -> None:
    from intergrax.llm_adapters._shared.openai_tool_call_interop import (
        tool_calls_from_openai_dicts,
    )

    calls = tool_calls_from_openai_dicts(
        [
            {
                "id": "tc-1",
                "type": "function",
                "function": {"name": "demo.tool", "arguments": '{"k": 1}'},
            }
        ]
    )
    assert calls == (
        LLMToolCall(id="tc-1", name="demo.tool", arguments_json='{"k": 1}'),
    )
