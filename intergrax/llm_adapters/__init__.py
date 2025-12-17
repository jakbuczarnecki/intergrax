# intergrax/llm_adapters/__init__.py

from INTERGRAX_ENGINE_BUNDLE import GeminiChatAdapter, LangChainOllamaAdapter, OpenAIChatResponsesAdapter
from intergrax.llm_adapters.claude_adapter import ClaudeChatAdapter
from intergrax.llm_adapters.mistral_adapter import MistralChatAdapter
from .base import (
    LLMAdapter,
    LLMAdapterRegistry,
    BaseModel,
    LLMProvider,
    _extract_json_object,
    _model_json_schema,
    _validate_with_model,
    _map_messages_to_openai,
)

__all__ = [
    "LLMAdapter",
    "LLMAdapterRegistry",
    "BaseModel",
    "OpenAIChatResponsesAdapter",
    "GeminiChatAdapter",
    "LangChainOllamaAdapter",
    "ClaudeChatAdapter",
    "MistralChatAdapter",
    "_extract_json_object",
    "_model_json_schema",
    "_validate_with_model",
    "_map_messages_to_openai",
]

# Default adapter registrations
LLMAdapterRegistry.register(LLMProvider.OPENAI, lambda **kw: OpenAIChatResponsesAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.GEMINI, lambda **kw: GeminiChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.OLLAMA, lambda **kw: LangChainOllamaAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.CLAUDE, lambda **kw: ClaudeChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.MISTRAL, lambda **kw: MistralChatAdapter(**kw))
