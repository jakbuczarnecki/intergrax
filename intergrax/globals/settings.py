# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class GlobalSettings:
    """
    Global framework-wide configuration.

    This is the single source of truth for defaults such as:
      - default language,
      - default locale,
      - default timezone,
      - default LLM model,
      - default memory/consolidation intervals, etc.

    Local modules (runtime, memory, tools) should import and use these
    defaults instead of hardcoding values.
    """

    # ------------------------------------------------------------------
    # Language, locale, timezone defaults
    # ------------------------------------------------------------------

    # Default language for prompts, instructions, extracted content, etc.
    # May be overridden by environment variable.
    default_language: str = os.getenv("INTERGRAX_DEFAULT_LANGUAGE", "pl")

    # Default locale (used when no user/org override is available).
    default_locale: str = os.getenv("INTERGRAX_DEFAULT_LOCALE", "pl-PL")

    # Default region (country/market context), used only as a fallback.
    default_region: str = os.getenv("INTERGRAX_DEFAULT_REGION", "pl-PL")

    # Default timezone (used if no user/org-specific value is configured).
    default_timezone: str = os.getenv("INTERGRAX_DEFAULT_TIMEZONE", "Europe/Warsaw")

    # ------------------------------------------------------------------
    # Default model identifiers (fallbacks for runtime components)
    # ------------------------------------------------------------------

    # Default local Ollama chat model.
    default_ollama_model: str = os.getenv(
        "INTERGRAX_DEFAULT_OLLAMA_MODEL", "llama3.1:latest"
    )

    # Default local Ollama embedding model.
    default_ollama_embed_model: str = os.getenv(
        "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL",
        "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest",
    )

    # Default OpenAI chat model.
    default_openai_model: str = os.getenv("INTERGRAX_DEFAULT_OPENAI_MODEL", "gpt-5-mini")

    # Default OpenAI embedding model.
    default_openai_embed_model: str = os.getenv(
        "INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL", "text-embedding-3-small"
    )

    # Default HuggingFace embedding model (fallback).
    default_hf_embed_model: str = os.getenv(
        "INTERGRAX_DEFAULT_HG_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Default Google Gemini chat model.
    default_gemini_model: str = os.getenv("INTERGRAX_DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")

    # Default Anthropic Claude chat model.
    default_claude_model: str = os.getenv("INTERGRAX_DEFAULT_CLAUDE_MODEL", "claude-3-5-sonnet-latest")

    # Default Mistral chat model.
    default_mistral_model: str = os.getenv("INTERGRAX_DEFAULT_MISTRAL_MODEL", "mistral-large-latest")


    azure_openai_endpoint: str = os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT", "")
    
    azure_openai_api_version: str = os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION", "")

    default_azure_openai_deployment: str = os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT", "")


    aws_region: str = os.getenv("INTERGRAX_DEFAULT_AWS_REGION", "")


    default_bedrock_model_id: str = os.getenv("INTERGRAX_DEFAULT_BEDROCK_MODEL_ID", "")

    # ------------------------------------------------------------------
    # Session memory / consolidation defaults
    # ------------------------------------------------------------------

    # Default interval (in user turns) for mid-session consolidation.
    # If None or non-integer, the runtime components should treat it as disabled
    # or fall back to a module-local constant. Since os.getenv ALWAYS returns
    # strings, we cast to int when possible; otherwise None.
    raw_interval = os.getenv("INTERGRAX_DEFAULT_USER_TURNS_CONSOLIDATION_INTERVAL", None)

    default_user_turns_consolidation_interval: Optional[int] = (
        int(raw_interval) if raw_interval and raw_interval.isdigit() else None
    )

    raw_cooldown_sec = os.getenv("INTERGRAX_DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS", None)

    default_consolidation_cooldown_seconds: Optional[int] = (
        int(raw_cooldown_sec) if raw_cooldown_sec and raw_cooldown_sec.isdigit() else None
    )


# Singleton-style global settings instance used across the framework.
GLOBAL_SETTINGS = GlobalSettings()
