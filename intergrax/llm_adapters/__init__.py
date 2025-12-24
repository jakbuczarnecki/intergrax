# intergrax/llm_adapters/__init__.py

from intergrax.llm_adapters.aws_bedrock_adapter import BedrockChatAdapter
from intergrax.llm_adapters.azure_openai_adapter import AzureOpenAIChatAdapter
from intergrax.llm_adapters.claude_adapter import ClaudeChatAdapter
from intergrax.llm_adapters.gemini_adapter import GeminiChatAdapter
from intergrax.llm_adapters.mistral_adapter import MistralChatAdapter
from intergrax.llm_adapters.ollama_adapter import LangChainOllamaAdapter
from intergrax.llm_adapters.openai_responses_adapter import OpenAIChatResponsesAdapter
from .base import (
    LLMAdapterRegistry,
    LLMProvider
)


# Default adapter registrations
LLMAdapterRegistry.register(LLMProvider.OPENAI, lambda **kw: OpenAIChatResponsesAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.GEMINI, lambda **kw: GeminiChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.OLLAMA, lambda **kw: LangChainOllamaAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.CLAUDE, lambda **kw: ClaudeChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.MISTRAL, lambda **kw: MistralChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.AZURE_OPENAI, lambda **kw: AzureOpenAIChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.AWS_BEDROCK, lambda **kw: BedrockChatAdapter(**kw))
