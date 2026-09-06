# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.llm_adapters.providers.registrations.google_providers import (
    register_gemini,
    register_vertex_gemini,
)
from intergrax.llm_adapters.providers.registrations.openai_compat_registrations import (
    register_azure_ai_inference,
    register_cohere,
    register_deepseek,
    register_fireworks,
    register_groq,
    register_llama_cpp,
    register_openrouter,
    register_together,
    register_vllm,
    register_xai,
)
from intergrax.llm_adapters.providers.registrations.openai_providers import (
    register_azure_openai,
    register_openai,
)
from intergrax.llm_adapters.providers.registrations.standalone_providers import (
    register_aws_bedrock,
    register_claude,
    register_cohere_native,
    register_mistral,
    register_ollama,
)


def register_builtin_llm_adapters(registry: type) -> None:
    register_openai(registry)
    register_gemini(registry)
    register_vertex_gemini(registry)
    register_ollama(registry)
    register_claude(registry)
    register_mistral(registry)
    register_azure_openai(registry)
    register_aws_bedrock(registry)
    register_groq(registry)
    register_vllm(registry)
    register_together(registry)
    register_fireworks(registry)
    register_openrouter(registry)
    register_deepseek(registry)
    register_xai(registry)
    register_llama_cpp(registry)
    register_cohere(registry)
    register_cohere_native(registry)
    register_azure_ai_inference(registry)
