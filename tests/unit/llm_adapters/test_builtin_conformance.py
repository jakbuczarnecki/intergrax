# © Artur Czarnecki. All rights reserved.

"""Parametric conformance for every builtin LLM provider (mocked SDK)."""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.conformance import run_adapter_conformance
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

pytestmark = pytest.mark.unit


def _openai_chat_mock() -> MagicMock:
    client = MagicMock()
    usage = MagicMock(prompt_tokens=3, completion_tokens=2)
    msg = MagicMock(content="ok", tool_calls=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    res = MagicMock(usage=usage, choices=[choice])
    client.chat.completions.create.return_value = res
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="x", tool_calls=None))]
    client.chat.completions.create.side_effect = None
    stream_iter = iter([chunk])
    client.chat.completions.create.return_value = res

    def _stream(**_kwargs: Any):
        if _kwargs.get("stream"):
            return iter([chunk])
        return res

    client.chat.completions.create.side_effect = lambda **kw: _stream(**kw)
    return client


def _build_adapter(provider: LLMProvider) -> LLMAdapter:
    env: dict[str, str] = {
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.CLAUDE: "ANTHROPIC_API_KEY",
        LLMProvider.GEMINI: "GOOGLE_API_KEY",
        LLMProvider.VERTEX_GEMINI: "INTERGRAX_VERTEX_PROJECT",
        LLMProvider.MISTRAL: "MISTRAL_API_KEY",
        LLMProvider.GROQ: "GROQ_API_KEY",
        LLMProvider.TOGETHER: "TOGETHER_API_KEY",
        LLMProvider.FIREWORKS: "FIREWORKS_API_KEY",
        LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
        LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
        LLMProvider.XAI: "XAI_API_KEY",
        LLMProvider.COHERE: "COHERE_API_KEY",
        LLMProvider.COHERE_NATIVE: "COHERE_API_KEY",
        LLMProvider.AZURE_AI_INFERENCE: "AZURE_AI_INFERENCE_API_KEY",
    }
    kwargs: dict[str, Any] = {"client": MagicMock(), "model": "test-model"}

    if provider in {
        LLMProvider.GROQ,
        LLMProvider.VLLM,
        LLMProvider.TOGETHER,
        LLMProvider.FIREWORKS,
        LLMProvider.OPENROUTER,
        LLMProvider.DEEPSEEK,
        LLMProvider.XAI,
        LLMProvider.LLAMA_CPP,
        LLMProvider.COHERE,
        LLMProvider.AZURE_AI_INFERENCE,
    }:
        kwargs["client"] = _openai_chat_mock()
        if provider == LLMProvider.AZURE_AI_INFERENCE:
            kwargs["base_url"] = "https://example.services.ai.azure.com/openai/v1"

    if provider == LLMProvider.OPENAI:
        client = MagicMock()
        usage = MagicMock(input_tokens=2, output_tokens=1)
        response = MagicMock(usage=usage, output_text="ok", output=[], status="completed")
        client.responses.create.return_value = response
        kwargs["client"] = client

    if provider == LLMProvider.CLAUDE:
        client = MagicMock()
        block = MagicMock(type="text", text="ok")
        response = MagicMock(content=[block], usage=MagicMock(input_tokens=1, output_tokens=1))
        client.messages.create.return_value = response
        kwargs["client"] = client

    if provider in {LLMProvider.GEMINI, LLMProvider.VERTEX_GEMINI}:
        client = MagicMock()
        response = MagicMock()
        response.text = "ok"
        response.candidates = []
        chat_session = MagicMock()
        chat_session.send_message.return_value = response
        client.chats.create.return_value = chat_session
        client.models.generate_content.return_value = response
        client.models.generate_content_stream.return_value = [response]
        kwargs["client"] = client
        if provider == LLMProvider.VERTEX_GEMINI:
            kwargs["project"] = "demo-project"

    if provider == LLMProvider.MISTRAL:
        client = MagicMock()
        msg = MagicMock(content="ok", tool_calls=None)
        choice = MagicMock(message=msg)
        res = MagicMock(usage=MagicMock(prompt_tokens=1, completion_tokens=1), choices=[choice])
        client.chat.complete.return_value = res
        kwargs["client"] = client

    if provider == LLMProvider.AWS_BEDROCK:
        client = MagicMock()
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
        }
        client.converse_stream.return_value = {
            "stream": [{"contentBlockDelta": {"delta": {"text": "ok"}}}],
        }
        kwargs = {
            "client": client,
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "use_converse": True,
            "region": "us-east-1",
        }

    if provider == LLMProvider.AZURE_OPENAI:
        kwargs = {"client": _openai_chat_mock(), "deployment": "gpt-test"}

    if provider == LLMProvider.OLLAMA:
        from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

        chat = MagicMock()
        chat.invoke.return_value = MagicMock(content="ok")
        bound_chat = MagicMock()
        bound_chat.invoke.return_value = MagicMock(
            content="ok",
            tool_calls=[],
            invalid_tool_calls=[],
        )
        chat.bind_tools.return_value = bound_chat
        return LangChainOllamaAdapter(chat=chat, model="llama3.1:latest")

    if provider == LLMProvider.COHERE_NATIVE:
        client = MagicMock()
        content = MagicMock(text="ok", type="text")
        client.chat.return_value = MagicMock(message=MagicMock(content=[content]))
        event = MagicMock(type="content-delta")
        event.delta.message.content.text = "x"
        client.chat_stream.return_value = [event]
        kwargs["client"] = client

    patch_env: dict[str, str] = {env.get(provider, "TEST_KEY"): "test"}
    if provider == LLMProvider.VERTEX_GEMINI:
        patch_env["INTERGRAX_VERTEX_PROJECT"] = "demo"
    if provider == LLMProvider.AWS_BEDROCK:
        patch_env["INTERGRAX_DEFAULT_AWS_REGION"] = "us-east-1"
        patch_env["INTERGRAX_DEFAULT_BEDROCK_MODEL_ID"] = kwargs.get(
            "model_id", "anthropic.claude-3-haiku-20240307-v1:0"
        )
    if provider == LLMProvider.AZURE_OPENAI:
        patch_env.update(
            {
                "INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
                "INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
                "INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT": "gpt-test",
            }
        )

    with patch.dict("os.environ", patch_env, clear=False):
        return LLMAdapterRegistry.create(provider, **kwargs)


@pytest.fixture()
def _restore_registry_state():
    snapshot = dict(LLMAdapterRegistry._factories)
    try:
        yield snapshot
    finally:
        LLMAdapterRegistry._factories = snapshot


@pytest.mark.no_ci
@pytest.mark.parametrize("provider", list(LLMProvider))
def test_builtin_provider_conformance(provider: LLMProvider, _restore_registry_state) -> None:
    LLMAdapterRegistry._factories.clear()
    adapter = _build_adapter(provider)
    tools_stream = provider in {
        LLMProvider.CLAUDE,
        LLMProvider.GEMINI,
        LLMProvider.VERTEX_GEMINI,
        LLMProvider.AWS_BEDROCK,
        LLMProvider.OPENAI,
        LLMProvider.MISTRAL,
    }
    run_adapter_conformance(adapter, check_tools_stream=tools_stream)
