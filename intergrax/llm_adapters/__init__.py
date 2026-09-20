# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 LLM adapter layer — public re-exports (Phase Q-L.2, M-LLM-R)."""

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent, LLMStreamEventKind
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import (
    LLMProfile,
    create_adapter,
    create_adapter_from_secrets_store,
    create_adapter_with_failover,
    llm_profile_from_env,
    validate_runtime,
)

__all__ = [
    "LLMAdapter",
    "LLMAdapterResponse",
    "LLMAdapterRegistry",
    "LLMFinishReason",
    "LLMProfile",
    "LLMStreamEvent",
    "LLMStreamEventKind",
    "LLMStructuredResult",
    "LLMTokenUsage",
    "LLMToolCall",
    "LLMProvider",
    "create_adapter",
    "create_adapter_from_secrets_store",
    "create_adapter_with_failover",
    "llm_profile_from_env",
    "validate_runtime",
]
