# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from enum import Enum


def llm_provider_slug(provider: LLMProvider | str) -> str:
    """Canonical provider identity slug for profile, routing, registry, and failover."""
    if isinstance(provider, LLMProvider):
        return provider.value
    if not isinstance(provider, str):
        raise TypeError(f"provider must be str or LLMProvider, got {type(provider)!r}")
    key = provider.strip()
    if not key:
        raise ValueError("provider must not be empty")
    return key.lower()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    CLAUDE = "claude"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    GROQ = "groq"
    VLLM = "vllm"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    LLAMA_CPP = "llama_cpp"
    COHERE = "cohere"
    COHERE_NATIVE = "cohere_native"
    VERTEX_GEMINI = "vertex_gemini"
    AZURE_AI_INFERENCE = "azure_ai_inference"


