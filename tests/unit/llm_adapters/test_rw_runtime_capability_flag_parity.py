# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-D: capability-flag parity (NT-ROW-036, FRZ-CTR-01, basis C) and the VBN classification of the
provider ``DEFAULT_MODEL`` constants (NT-RW-RUN-06, NT-ROW-108, PDC-D3-R019, basis S).

Parity: a declared streaming or tool-streaming capability must have the method behind it, and an implemented
method must be declared, unless the method fails closed with the typed capability error when the flag is false.
Register: every provider adapter that hard-codes a default model is a row of the EBH-6 hardcoding register, so the
VBN counters account for it. Fake clients only (no network, no credential).
"""

from __future__ import annotations

import ast
import io
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PROVIDERS = _REPO / "intergrax" / "llm_adapters" / "providers"
_USER = [ChatMessage(role="user", content="hello")]
_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "noop", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
}


@pytest.fixture(autouse=True)
def _isolate_provider_modules() -> Iterator[None]:
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("intergrax.llm_adapters.providers.") and name.endswith("_adapter"):
            sys.modules.pop(name, None)


# ------------------------------------------------------------------------------------------------ parity


def _resolver(*capabilities: str) -> Any:
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    return OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=list(capabilities)))


def _openai_chat() -> Any:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    return OpenAIChatCompletionsAdapter(client=MagicMock(), model="gpt-4o-mini", provider=LLMProvider.OPENAI)


def _openai_responses() -> Any:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    return OpenAIChatResponsesAdapter(client=MagicMock(), model="gpt-5.1")


def _groq() -> Any:
    from intergrax.llm_adapters.providers.openai_compat_providers import GroqChatAdapter

    return GroqChatAdapter(client=MagicMock(), model="llama-3.1-8b-instant")


def _claude() -> Any:
    pytest.importorskip("anthropic", reason="optional extra llm-anthropic")
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    return ClaudeChatAdapter(client=MagicMock(), model="claude-3-5-sonnet-20241022")


def _mistral() -> Any:
    pytest.importorskip("mistralai", reason="optional extra llm-mistral")
    from intergrax.llm_adapters.providers.mistral_adapter import MistralChatAdapter

    return MistralChatAdapter(client=MagicMock(), model="mistral-large-latest")


def _gemini() -> Any:
    pytest.importorskip("google.genai", reason="optional extra llm-gemini")
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    return GeminiChatAdapter(client=MagicMock(), model="gemini-2.5-flash")


def _cohere() -> Any:
    pytest.importorskip("cohere", reason="optional extra llm-cohere-native")
    from intergrax.llm_adapters.providers.cohere_native_adapter import CohereNativeChatAdapter

    return CohereNativeChatAdapter(client=MagicMock(), model="command-r-plus")


def _bedrock(model_id: str, *, use_converse: bool) -> Callable[[], Any]:
    def build() -> Any:
        from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

        client = SimpleNamespace(
            converse=MagicMock(),
            converse_stream=MagicMock(),
            invoke_model=MagicMock(return_value={"body": io.BytesIO(json.dumps({}).encode("utf-8"))}),
            invoke_model_with_response_stream=MagicMock(),
        )
        if not use_converse:
            del client.converse
            del client.converse_stream
        return BedrockChatAdapter(client=client, model_id=model_id, region="us-east-1", use_converse=use_converse)

    return build


def _native_ollama() -> Any:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter

    return NativeOllamaAdapter(client=MagicMock(), model="qwen2.5:14b", capability_resolver=_resolver("completion"))


def _langchain_ollama() -> Any:
    pytest.importorskip("langchain_core", reason="optional extra llm-langchain-ollama")
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

    chat = MagicMock()
    chat.model = "qwen2.5:14b"
    return LangChainOllamaAdapter(chat=chat, model="qwen2.5:14b", capability_resolver=_resolver("completion"))


_ADAPTERS: dict[str, Callable[[], Any]] = {
    "openai-chat": _openai_chat,
    "openai-responses": _openai_responses,
    "groq (openai-compatible family)": _groq,
    "claude": _claude,
    "mistral": _mistral,
    "gemini": _gemini,
    "cohere": _cohere,
    "bedrock-converse": _bedrock("anthropic.claude-3-haiku-20240307-v1:0", use_converse=True),
    "bedrock-invoke-anthropic": _bedrock("anthropic.claude-3-haiku-20240307-v1:0", use_converse=False),
    "bedrock-invoke-meta": _bedrock("meta.llama3-1-8b-instruct-v1:0", use_converse=False),
    "native-ollama": _native_ollama,
    "langchain-ollama": _langchain_ollama,
}


def _implements(adapter: Any, method: str) -> bool:
    return getattr(type(adapter), method) is not getattr(BaseLLMAdapter, method)


def _fails_closed(call: Callable[[], Any]) -> bool:
    try:
        result = call()
        if not isinstance(result, (str, bytes)):
            list(result)
    except LLMCapabilityMismatchError:
        return True
    except Exception:
        return False
    return False


@pytest.mark.parametrize("case", sorted(_ADAPTERS))
def test_streaming_flag_matches_the_stream_messages_method(case: str) -> None:
    adapter = _ADAPTERS[case]()
    declared = adapter.supports_streaming() is True
    implemented = _implements(adapter, "stream_messages")
    if declared:
        assert implemented, f"{case} declares streaming without stream_messages"
    elif implemented:
        assert _fails_closed(lambda: adapter.stream_messages(_USER)), (
            f"{case} implements stream_messages but declares no streaming"
        )


@pytest.mark.parametrize("case", sorted(_ADAPTERS))
def test_tool_streaming_flag_matches_the_stream_with_tools_method(case: str) -> None:
    adapter = _ADAPTERS[case]()
    declared = adapter.supports_tool_streaming() is True
    implemented = _implements(adapter, "stream_with_tools")
    if declared:
        assert implemented, f"{case} declares tool streaming without stream_with_tools"
    elif implemented:
        assert _fails_closed(lambda: adapter.stream_with_tools(_USER, [_TOOL])), (
            f"{case} implements stream_with_tools but declares no tool streaming"
        )


def _adapter_classes() -> dict[str, str]:
    """Concrete adapter class -> its direct base name, over the provider package (AST)."""
    classes: dict[str, str] = {}
    for path in sorted(_PROVIDERS.glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.bases and isinstance(node.bases[0], ast.Name):
                classes[node.name] = node.bases[0].id
    adapters: set[str] = {"BaseLLMAdapter"}
    changed = True
    while changed:
        changed = False
        for name, base in classes.items():
            if base in adapters and name not in adapters:
                adapters.add(name)
                changed = True
    return {name: classes[name] for name in adapters - {"BaseLLMAdapter"} if not name.startswith("_")}


#: Shipped adapters that inherit their flags and stream methods from a base covered above.
_INHERITING = {
    "AzureOpenAIChatAdapter": "OpenAIChatCompletionsAdapter",
    "VertexGeminiChatAdapter": "GeminiChatAdapter",
    "GroqChatAdapter": "_CompatAdapterBase",
    "VllmChatAdapter": "_CompatAdapterBase",
    "TogetherChatAdapter": "_CompatAdapterBase",
    "FireworksChatAdapter": "_CompatAdapterBase",
    "OpenRouterChatAdapter": "_CompatAdapterBase",
    "DeepSeekChatAdapter": "_CompatAdapterBase",
    "XaiChatAdapter": "_CompatAdapterBase",
    "LlamaCppChatAdapter": "_CompatAdapterBase",
    "CohereChatAdapter": "_CompatAdapterBase",
    "AzureAiInferenceChatAdapter": "_CompatAdapterBase",
}


def test_the_parity_cases_cover_every_shipped_adapter() -> None:
    classes = _adapter_classes()
    direct = {name for name, base in classes.items() if base == "BaseLLMAdapter"}
    assert direct == {
        "OpenAIChatCompletionsAdapter",
        "OpenAIChatResponsesAdapter",
        "ClaudeChatAdapter",
        "MistralChatAdapter",
        "GeminiChatAdapter",
        "CohereNativeChatAdapter",
        "BedrockChatAdapter",
        "NativeOllamaAdapter",
        "LangChainOllamaAdapter",
    }
    assert {name for name in classes if name not in direct} == set(_INHERITING)
    for name, base in _INHERITING.items():
        assert classes[name] == base
    stream_overrides = {"supports_streaming", "supports_tool_streaming", "stream_messages", "stream_with_tools"}
    for path in sorted(_PROVIDERS.glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.name in _INHERITING and node.name != "GroqChatAdapter":
                defined = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
                assert not defined & stream_overrides, f"{node.name} overrides a stream flag; add a parity case"


# -------------------------------------------------------------------------- VBN register: DEFAULT_MODEL rows


def _default_model_constants() -> dict[str, tuple[str, str]]:
    """Adapter class -> (repo-relative file, DEFAULT_MODEL value) over ``providers/*.py``."""
    found: dict[str, tuple[str, str]] = {}
    for path in sorted(_PROVIDERS.glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "DEFAULT_MODEL" for t in item.targets)
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    found[node.name] = (path.relative_to(_REPO).as_posix(), item.value.value)
    return found


def test_the_provider_default_model_inventory_is_the_known_set() -> None:
    assert set(_default_model_constants()) == {
        "ClaudeChatAdapter",
        "CohereNativeChatAdapter",
        "GeminiChatAdapter",
        "MistralChatAdapter",
        "NativeOllamaAdapter",
        "LangChainOllamaAdapter",
        "OpenAIChatResponsesAdapter",
        "VertexGeminiChatAdapter",
    }


def test_every_provider_default_model_has_a_vbn_register_row() -> None:
    from tests.qualification.ebh_6._findings_register import (
        HardcodingFindingClassification,
        all_hardcoding_findings,
        count_prohibited_unresolved,
        count_unclassified,
    )

    rows = all_hardcoding_findings()
    for class_name, (relative_path, value) in _default_model_constants().items():
        matching = [row for row in rows if relative_path in row.location and class_name in row.location]
        assert len(matching) == 1, f"no single register row for {class_name}.DEFAULT_MODEL ({relative_path})"
        row = matching[0]
        assert row.classification is HardcodingFindingClassification.ALLOWED_ADAPTER
        assert value in row.location
    assert count_unclassified() == 0
    assert count_prohibited_unresolved() == 0
