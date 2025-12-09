# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class GlobalSettings:
    """
    Global framework-wide configuration.

    This is the single source of truth for defaults such as:
      - default language,
      - default locale,
      - default timezone,
      - default LLM model (if needed),
      - etc.

    Local modules (runtime, memory, tools) should import and use these
    defaults instead of hardcoding values.
    """

    # Default language for prompts, instructions and extracted content.
    # Can be overridden via environment variable.
    default_language: str = os.getenv("INTERGRAX_DEFAULT_LANGUAGE", "pl")    

    # Default locale (used only if no more specific user/org-level value).
    default_locale: str = os.getenv("INTERGRAX_DEFAULT_LOCALE", "pl-PL")

    # Default region (used only if no more specific user/org-level value).
    default_region: str = os.getenv("INTERGRAX_DEFAULT_REGION", "pl-PL")

    # Default timezone (used only if no more specific user/org-level value).
    default_timezone: str = os.getenv("INTERGRAX_DEFAULT_TIMEZONE", "Europe/Warsaw")

    default_ollama_model: str = os.getenv("INTERGRAX_DEFAULT_OLLAMA_MODEL", "llama3.1:latest")

    default_ollama_embed_model: str = os.getenv("INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL", "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest")

    default_openai_model: str = os.getenv("INTERGRAX_DEFAULT_OPENAI_MODEL", "gpt-5-mini")

    default_openai_embed_model: str = os.getenv("INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL", "text-embedding-3-small")

    default_hf_embed_model: str = os.getenv("INTERGRAX_DEFAULT_HG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# Singleton-style global settings instance used across the framework.
GLOBAL_SETTINGS = GlobalSettings()