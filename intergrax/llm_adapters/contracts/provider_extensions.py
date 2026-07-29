# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


UsageSource = Literal["sdk", "estimate"]


@dataclass(frozen=True, slots=True)
class OpenAIProviderExtensions:
    system_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class AnthropicProviderExtensions:
    stop_sequence: str | None = None


@dataclass(frozen=True, slots=True)
class GeminiProviderExtensions:
    candidate_index: int | None = None


@dataclass(frozen=True, slots=True)
class BedrockProviderExtensions:
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class VllmProviderExtensions:
    """Safe vLLM prefix-cache reporting metadata for proof and attribution."""

    prompt_tokens_details_reported: bool
    server_version: str | None = None


@dataclass(frozen=True, slots=True)
class LLMProviderExtensions:
    """Optional provider-specific metadata without open dict bags."""

    usage_source: UsageSource = "sdk"
    openai: OpenAIProviderExtensions | None = None
    anthropic: AnthropicProviderExtensions | None = None
    gemini: GeminiProviderExtensions | None = None
    bedrock: BedrockProviderExtensions | None = None
    vllm: VllmProviderExtensions | None = None
