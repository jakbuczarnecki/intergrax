# intergrax/llm_adapters/__init__.py

from .base import (
    LLMAdapter,
    LLMAdapterRegistry,
    BaseModel,
    _extract_json_object,
    _model_json_schema,
    _validate_with_model,
    _map_messages_to_openai,
)
from .openai_responses_adapter import OpenAIChatResponsesAdapter
from .gemini_adapter import GeminiChatAdapter
from .ollama_adapter import LangChainOllamaAdapter

__all__ = [
    "LLMAdapter",
    "LLMAdapterRegistry",
    "BaseModel",
    "OpenAIChatResponsesAdapter",
    "GeminiChatAdapter",
    "LangChainOllamaAdapter",
    "_extract_json_object",
    "_model_json_schema",
    "_validate_with_model",
    "_map_messages_to_openai",
]

# Default adapter registrations
LLMAdapterRegistry.register("openai", lambda **kw: OpenAIChatResponsesAdapter(**kw))
LLMAdapterRegistry.register("gemini", lambda **kw: GeminiChatAdapter(**kw))
LLMAdapterRegistry.register("ollama", lambda **kw: LangChainOllamaAdapter(**kw))
