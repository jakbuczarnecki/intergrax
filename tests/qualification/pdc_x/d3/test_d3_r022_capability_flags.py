# © Artur Czarnecki. All rights reserved.

"""PDC-D3-R022 — capability flags match implemented methods (conformance gate)."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm_adapters._shared.conformance import (
    assert_capability_flags_match_methods,
    capability_flag_violations,
)
from intergrax.llm_adapters.contracts.capabilities import LLMFeature
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.catalog_capabilities import CatalogCapabilityAdapter
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.model_catalog import ModelRecord
from tests.qualification.pdc_x.d3.fakes import (
    AlwaysTryNextPolicy,
    BufferedOnlyAdapter,
    BufferedToolsAdapter,
    OverReportingToolStreamAdapter,
    StreamingOnlyAdapter,
    StreamingToolsAdapter,
    UnderReportingStreamAdapter,
)

pytestmark = pytest.mark.unit

_PROVIDERS_DIR = Path(__file__).resolve().parents[4] / "intergrax" / "llm_adapters" / "providers"


# --------------------------------------------------------------------------- the gate itself


def test_gate_flags_stream_method_with_false_flag() -> None:
    violations = capability_flag_violations(UnderReportingStreamAdapter())
    assert any("stream_messages is implemented but supports_streaming() is False" in v for v in violations)


def test_gate_flags_tool_streaming_declared_without_method() -> None:
    violations = capability_flag_violations(OverReportingToolStreamAdapter())
    assert any("supports_tool_streaming() is True but stream_with_tools is not implemented" in v for v in violations)


@pytest.mark.parametrize(
    "adapter",
    [BufferedOnlyAdapter(), BufferedToolsAdapter(), StreamingOnlyAdapter(), StreamingToolsAdapter()],
    ids=lambda a: type(a).__name__,
)
def test_gate_accepts_honest_declarations(adapter: Any) -> None:
    assert_capability_flags_match_methods(adapter)


def test_streaming_only_adapter_declares_no_tool_streaming() -> None:
    """Declared, not forced: streaming without tool streaming is a valid declaration."""
    adapter = StreamingOnlyAdapter()
    assert adapter.supports_streaming() is True
    assert adapter.supports_tools() is True
    assert adapter.supports_tool_streaming() is False


# --------------------------------------------------------------------------- real adapters


def _claude() -> Any:
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    return ClaudeChatAdapter(client=MagicMock(), model="claude-3-5-sonnet-latest")


def _bedrock() -> Any:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    return BedrockChatAdapter(
        client=MagicMock(),
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        use_converse=True,
        region="us-east-1",
    )


def _openai_chat() -> Any:
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import (
        OpenAIChatCompletionsAdapter,
    )

    return OpenAIChatCompletionsAdapter(client=MagicMock(), model="gpt-4o-mini", provider=LLMProvider.OPENAI)


def _openai_responses() -> Any:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    return OpenAIChatResponsesAdapter(client=MagicMock(), model="gpt-4o-mini")


def _groq_compat() -> Any:
    from intergrax.llm_adapters.providers.openai_compat_providers import GroqChatAdapter

    return GroqChatAdapter(client=MagicMock(), model="llama-3.3-70b-versatile")


def _ollama_resolver() -> Any:
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    return OllamaModelCapabilityResolver(
        show_model=lambda _model: SimpleNamespace(capabilities=["tools", "completion"])
    )


def _native_ollama() -> Any:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter

    return NativeOllamaAdapter(client=MagicMock(), model="llama3.1:latest", capability_resolver=_ollama_resolver())


def _langchain_ollama() -> Any:
    pytest.importorskip("langchain_ollama", reason="optional extra llm-langchain-ollama; covered by the source gate")
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

    return LangChainOllamaAdapter(chat=MagicMock(), model="llama3.1:latest", capability_resolver=_ollama_resolver())


_REAL_BUILDERS: dict[str, Callable[[], Any]] = {
    "claude": _claude,
    "aws_bedrock": _bedrock,
    "openai_chat_completions": _openai_chat,
    "openai_responses": _openai_responses,
    "groq_compat": _groq_compat,
    "native_ollama": _native_ollama,
    "langchain_ollama": _langchain_ollama,
}

#: Expected (streaming, tool streaming) declaration per importable shipped adapter.
_EXPECTED: dict[str, tuple[bool, bool]] = {
    "claude": (True, True),
    "aws_bedrock": (True, True),
    "openai_chat_completions": (True, True),
    "openai_responses": (True, True),
    "groq_compat": (True, True),
    "native_ollama": (True, False),
    "langchain_ollama": (True, False),
}


@pytest.mark.parametrize("name", sorted(_REAL_BUILDERS))
def test_shipped_adapter_flags_match_methods(name: str) -> None:
    adapter = _REAL_BUILDERS[name]()
    assert_capability_flags_match_methods(adapter)
    assert (adapter.supports_streaming(), adapter.supports_tool_streaming()) == _EXPECTED[name]


def test_registry_reads_declared_flags_of_shipped_adapter() -> None:
    declared = LLMAdapterRegistry.declared_capabilities_for(_claude())
    assert declared.streaming is True
    assert declared.tool_streaming is True
    assert declared.supports(LLMFeature.TOOLS)


# --------------------------------------------------------------------------- wrappers


def test_catalog_overlay_follows_inner_and_forwards_tool_streaming() -> None:
    record = ModelRecord(model_id="fake-model", context_window_tokens=8_192, supports_tools=True)
    full = CatalogCapabilityAdapter(StreamingToolsAdapter(), record)
    assert_capability_flags_match_methods(full, delegating=True)
    assert full.supports_tool_streaming() is True
    # Before B-D3 the overlay inherited the base stream_with_tools and could never forward it.
    first = next(iter(full.stream_with_tools([], [])))
    assert first.delta_content == "t"

    no_tool_stream = CatalogCapabilityAdapter(StreamingOnlyAdapter(), record)
    assert_capability_flags_match_methods(no_tool_stream, delegating=True)
    assert no_tool_stream.supports_tool_streaming() is False


def test_failover_flags_follow_the_chain() -> None:
    policy = AlwaysTryNextPolicy()
    mixed = FailoverLLMAdapter([StreamingOnlyAdapter(), StreamingToolsAdapter()], failover_policy=policy)
    assert_capability_flags_match_methods(mixed, delegating=True)
    assert mixed.supports_tools() is True  # was False before B-D3 although generate_with_tools forwards
    assert mixed.supports_tool_streaming() is True

    plain = FailoverLLMAdapter([BufferedOnlyAdapter()], failover_policy=policy)
    assert_capability_flags_match_methods(plain, delegating=True)
    assert (plain.supports_streaming(), plain.supports_tools(), plain.supports_tool_streaming()) == (
        False,
        False,
        False,
    )


# --------------------------------------------------------------------------- source gate (SDK-free)


def _adapter_classes() -> list[tuple[str, ast.ClassDef]]:
    classes: list[tuple[str, ast.ClassDef]] = []
    paths = sorted(_PROVIDERS_DIR.glob("*_adapter.py")) + [_PROVIDERS_DIR / "openai_compat_providers.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes.extend((path.name, node) for node in tree.body if isinstance(node, ast.ClassDef))
    return classes


def _methods(node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}


def _declares_true(fn: ast.FunctionDef) -> bool:
    """``return True`` or a delegating call (``self._delegate...`` / ``self.supports_tools()``)."""
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    return any(
        (isinstance(r.value, ast.Constant) and r.value.value is True) or isinstance(r.value, ast.Call) for r in returns
    )


def test_every_provider_source_class_declares_what_it_implements() -> None:
    """Covers adapters whose SDK is not installed (gemini, mistral, cohere): static class-level gate."""
    problems: list[str] = []
    checked = 0
    for filename, node in _adapter_classes():
        methods = _methods(node)
        if "stream_messages" in methods:
            checked += 1
            flag = methods.get("supports_streaming")
            if flag is None or not _declares_true(flag):
                problems.append(f"{filename}:{node.name} implements stream_messages without supports_streaming()")
        if "stream_with_tools" in methods:
            checked += 1
            flag = methods.get("supports_tool_streaming")
            if flag is None or not _declares_true(flag):
                problems.append(
                    f"{filename}:{node.name} implements stream_with_tools without supports_tool_streaming()"
                )
        if "supports_tool_streaming" in methods and "stream_with_tools" not in methods:
            body = ast.unparse(methods["supports_tool_streaming"])
            if "_delegate" not in body and "supports_tools" not in body:
                problems.append(f"{filename}:{node.name} declares tool streaming without stream_with_tools")
    assert checked >= 12, "gate must actually inspect the shipped adapters"
    assert not problems, "; ".join(problems)


def test_the_formerly_misreporting_adapters_now_declare_streaming() -> None:
    expected = {
        "claude_adapter.py": "ClaudeChatAdapter",
        "gemini_adapter.py": "GeminiChatAdapter",
        "mistral_adapter.py": "MistralChatAdapter",
        "aws_bedrock_adapter.py": "BedrockChatAdapter",
        "cohere_native_adapter.py": "CohereNativeChatAdapter",
        "openai_chat_completions_adapter.py": "OpenAIChatCompletionsAdapter",
        "ollama_adapter.py": "LangChainOllamaAdapter",
    }
    by_key = {(f, n.name): _methods(n) for f, n in _adapter_classes()}
    for filename, class_name in expected.items():
        methods = by_key[(filename, class_name)]
        assert "stream_messages" in methods
        assert "supports_streaming" in methods, f"{class_name} still reports supports_streaming()=False"
